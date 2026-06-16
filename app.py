import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from config import APP_SECRET_KEY, DEFAULT_PREDICT_LEN, DEFAULT_TICKER
from predictors import (
    predict_direction_with_lgbm,
    predict_with_return_lstm,
    predict_with_ridge,
)
from services import (
    get_db_connection,
    get_gemini_description,
    get_price_snapshots,
    get_stock_data,
)


BASE_DIR = Path(__file__).resolve().parent
SHARE_EPSILON = 1e-9
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=APP_SECRET_KEY)


def add_flash_message(request: Request, message: str, category: str) -> None:
    flashes = request.session.get("_flashes", [])
    flashes.append({"category": category, "message": message})
    request.session["_flashes"] = flashes


def pop_flash_messages(request: Request) -> list[dict[str, str]]:
    return request.session.pop("_flashes", [])


def parse_predict_len(predict_len: Optional[str]) -> int:
    try:
        return max(1, int(predict_len or DEFAULT_PREDICT_LEN))
    except (ValueError, TypeError):
        return DEFAULT_PREDICT_LEN


def parse_share_amount(shares: Optional[str]) -> float:
    amount = float(shares)
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("Invalid share amount")
    return round(amount, 4)


def format_shares(shares: float) -> str:
    formatted = f"{float(shares):.4f}".rstrip("0").rstrip(".")
    return formatted or "0"


def build_chart_data(stock_ticker: str, current_price: float, df: pd.DataFrame) -> dict:
    labels = df.index.strftime("%Y-%m-%d").tolist()
    price_data = df["Close"].tolist()
    empty_prediction_data = [None] * len(labels)

    return {
        "labels": labels,
        "data": price_data,
        "prediction_data_lstm": empty_prediction_data.copy(),
        "prediction_data_ridge": empty_prediction_data.copy(),
        "ticker": stock_ticker,
        "current_price": round(current_price, 2),
    }


def build_prediction_response(
    stock_ticker: str,
    current_price: float,
    df: pd.DataFrame,
    predict_len: int,
) -> dict:
    lstm_prediction_series = predict_with_return_lstm(df, predict_len=predict_len)
    ridge_prediction_series = predict_with_ridge(df, predict_len=predict_len)

    predicted_price_lstm_val = (
        lstm_prediction_series.iloc[0] if not lstm_prediction_series.empty else current_price
    )
    predicted_price_ridge_val = (
        ridge_prediction_series.iloc[0] if not ridge_prediction_series.empty else current_price
    )

    if not lstm_prediction_series.empty:
        prediction_labels = lstm_prediction_series.index.strftime("%Y-%m-%d").tolist()
    elif not ridge_prediction_series.empty:
        prediction_labels = ridge_prediction_series.index.strftime("%Y-%m-%d").tolist()
    else:
        last_date_for_label = df.index[-1] if not df.empty else pd.Timestamp.today().normalize()
        prediction_labels = pd.date_range(
            start=last_date_for_label + pd.Timedelta(days=1),
            periods=predict_len,
            freq="B",
        ).strftime("%Y-%m-%d").tolist()

    prediction_data_lstm = lstm_prediction_series.tolist()
    prediction_data_ridge = ridge_prediction_series.tolist()
    historical_labels = df.index.strftime("%Y-%m-%d").tolist()
    historical_data = df["Close"].tolist()
    if historical_data:
        prediction_data_points_lstm = [None] * (len(historical_data) - 1) + [
            historical_data[-1]
        ] + prediction_data_lstm
        prediction_data_points_ridge = [None] * (len(historical_data) - 1) + [
            historical_data[-1]
        ] + prediction_data_ridge
    else:
        prediction_data_points_lstm = prediction_data_lstm
        prediction_data_points_ridge = prediction_data_ridge

    chart_data = {
        "labels": historical_labels + prediction_labels,
        "data": historical_data + [None] * len(prediction_labels),
        "prediction_data_lstm": prediction_data_points_lstm,
        "prediction_data_ridge": prediction_data_points_ridge,
        "ticker": stock_ticker,
        "current_price": round(current_price, 2),
    }

    return {
        "chart_data": chart_data,
        "predicted_price_lstm": (
            round(predicted_price_lstm_val, 2)
            if predicted_price_lstm_val is not None
            else "N/A"
        ),
        "predicted_price_ridge": (
            round(predicted_price_ridge_val, 2)
            if predicted_price_ridge_val is not None
            else "N/A"
        ),
        "predict_len": predict_len,
    }


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return sanitize_for_json(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def build_cost_basis_map(transactions) -> dict[str, dict[str, float]]:
    cost_basis_map: dict[str, dict[str, float]] = {}

    for transaction in transactions:
        ticker = transaction["ticker"]
        shares = float(transaction["shares"])
        price = float(transaction["price"])
        position = cost_basis_map.setdefault(
            ticker,
            {"shares": 0.0, "cost_basis_total": 0.0},
        )

        if transaction["type"] == "BUY":
            position["shares"] += shares
            position["cost_basis_total"] += shares * price
            continue

        if position["shares"] <= SHARE_EPSILON:
            position["shares"] = 0.0
            position["cost_basis_total"] = 0.0
            continue

        sold_shares = min(shares, position["shares"])
        average_cost = position["cost_basis_total"] / position["shares"]
        position["shares"] -= sold_shares
        position["cost_basis_total"] -= average_cost * sold_shares

        if position["shares"] <= SHARE_EPSILON:
            position["shares"] = 0.0
            position["cost_basis_total"] = 0.0

    return cost_basis_map


def build_portfolio_snapshot(user_balance: float, holdings_db, transactions):
    holdings_rows = [
        {
            "ticker": holding["ticker"],
            "shares": float(holding["shares"]),
        }
        for holding in holdings_db
        if float(holding["shares"]) > SHARE_EPSILON
    ]
    tickers = [holding["ticker"] for holding in holdings_rows]
    price_snapshots = get_price_snapshots(tickers)
    cost_basis_map = build_cost_basis_map(transactions)

    holding_details = []
    total_holdings_value = 0.0
    for holding in holdings_rows:
        ticker = holding["ticker"]
        shares = holding["shares"]
        snapshot = price_snapshots.get(
            ticker,
            {"current_price": None, "previous_close": None},
        )
        current_price = snapshot.get("current_price")
        previous_close = snapshot.get("previous_close")

        cost_state = cost_basis_map.get(
            ticker,
            {"shares": shares, "cost_basis_total": 0.0},
        )
        basis_shares = cost_state["shares"] if cost_state["shares"] > SHARE_EPSILON else shares
        average_cost = (
            cost_state["cost_basis_total"] / basis_shares
            if basis_shares > SHARE_EPSILON
            else 0.0
        )

        market_value = shares * current_price if current_price is not None else 0.0
        unrealized_pl = (
            (current_price - average_cost) * shares
            if current_price is not None
            else None
        )
        unrealized_pl_pct = (
            ((current_price - average_cost) / average_cost) * 100
            if current_price is not None and average_cost > SHARE_EPSILON
            else None
        )
        day_change_value = (
            (current_price - previous_close) * shares
            if current_price is not None and previous_close is not None
            else None
        )
        day_change_pct = (
            ((current_price - previous_close) / previous_close) * 100
            if current_price is not None
            and previous_close is not None
            and abs(previous_close) > SHARE_EPSILON
            else None
        )

        total_holdings_value += market_value
        holding_details.append(
            {
                "ticker": ticker,
                "shares": shares,
                "shares_display": format_shares(shares),
                "current_price": round(current_price, 2) if current_price is not None else None,
                "average_cost": round(average_cost, 2) if average_cost > SHARE_EPSILON else 0.0,
                "market_value": round(market_value, 2),
                "unrealized_pl": round(unrealized_pl, 2) if unrealized_pl is not None else None,
                "unrealized_pl_pct": (
                    round(unrealized_pl_pct, 2) if unrealized_pl_pct is not None else None
                ),
                "day_change_value": (
                    round(day_change_value, 2) if day_change_value is not None else None
                ),
                "day_change_pct": round(day_change_pct, 2) if day_change_pct is not None else None,
            }
        )

    total_asset = float(user_balance) + total_holdings_value
    for detail in holding_details:
        detail["holding_ratio"] = (
            round((detail["market_value"] / total_asset) * 100, 2)
            if total_asset > SHARE_EPSILON
            else 0.0
        )

    holding_details.sort(key=lambda detail: detail["market_value"], reverse=True)

    pie_labels = ["Cash"]
    pie_values = [round(float(user_balance), 2)]
    for detail in holding_details:
        if detail["market_value"] <= 0:
            continue
        pie_labels.append(detail["ticker"])
        pie_values.append(detail["market_value"])

    return (
        {
            "balance": round(float(user_balance), 2),
            "holdings_value": round(total_holdings_value, 2),
            "total_asset": round(total_asset, 2),
        },
        holding_details,
        {
            "labels": pie_labels,
            "data": pie_values,
        },
    )


def build_asset_history_chart(transactions, current_balance: float) -> Optional[str]:
    if not transactions:
        return None

    start_date = datetime.strptime(
        transactions[0]["timestamp"].split(" ")[0],
        "%Y-%m-%d",
    ).date()
    end_date = datetime.now().date()
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    tickers = sorted({transaction["ticker"] for transaction in transactions})
    hist_data = pd.DataFrame()
    if tickers:
        try:
            hist_data = yf.download(
                " ".join(tickers),
                start=start_date,
                end=end_date + timedelta(days=1),
                progress=False,
                auto_adjust=False,
            )["Close"]
            if isinstance(hist_data, pd.Series):
                hist_data = hist_data.to_frame(name=tickers[0])
        except Exception:
            hist_data = pd.DataFrame()

    cash = float(current_balance)
    for transaction in reversed(transactions):
        shares = float(transaction["shares"])
        price = float(transaction["price"])
        if transaction["type"] == "BUY":
            cash += shares * price
        else:
            cash -= shares * price

    holdings = {ticker: 0.0 for ticker in tickers}
    asset_history = []
    tx_idx = 0

    for day in date_range:
        while tx_idx < len(transactions):
            transaction = transactions[tx_idx]
            tx_date = datetime.strptime(
                transaction["timestamp"].split(" ")[0],
                "%Y-%m-%d",
            ).date()
            if tx_date > day.date():
                break

            ticker = transaction["ticker"]
            shares = float(transaction["shares"])
            price = float(transaction["price"])
            if transaction["type"] == "BUY":
                cash -= shares * price
                holdings[ticker] = holdings.get(ticker, 0.0) + shares
            else:
                cash += shares * price
                holdings[ticker] = max(0.0, holdings.get(ticker, 0.0) - shares)
            tx_idx += 1

        holdings_value = 0.0
        for ticker, shares in holdings.items():
            if shares <= SHARE_EPSILON or hist_data.empty or ticker not in hist_data.columns:
                continue

            try:
                price_series = hist_data.loc[:day, ticker].ffill().dropna()
                if not price_series.empty:
                    holdings_value += shares * float(price_series.iloc[-1])
            except (KeyError, IndexError):
                continue

        asset_history.append(round(cash + holdings_value, 2))

    return json.dumps(
        {
            "labels": [day.strftime("%Y-%m-%d") for day in date_range],
            "data": asset_history,
        }
    )


@app.get("/", name="search")
def search():
    return RedirectResponse(
        url=f"/stock/{DEFAULT_TICKER}",
        status_code=303,
    )


@app.get("/stock/{ticker}", response_class=HTMLResponse, name="index")
def index(request: Request, ticker: str, predict_len: Optional[str] = None):
    stock_ticker = ticker.upper()
    stock, current_price = get_stock_data(stock_ticker)
    if stock is None:
        add_flash_message(request, f'Ticker "{stock_ticker}" not found.', "danger")
        return RedirectResponse(url="/", status_code=303)

    stock_info = stock.info
    gemini_description = get_gemini_description(stock_ticker, stock_info)
    company_name = stock_info.get("longName", stock_ticker)

    conn = get_db_connection()
    user = conn.execute("SELECT * FROM user WHERE id = 1").fetchone()
    holdings_db = conn.execute("SELECT * FROM holdings ORDER BY ticker ASC").fetchall()
    transactions = conn.execute("SELECT * FROM transactions ORDER BY timestamp ASC").fetchall()
    conn.close()

    portfolio, holding_details, _ = build_portfolio_snapshot(
        float(user["balance"]),
        holdings_db,
        transactions,
    )

    df = stock.history(period="1y")
    predict_len_value = parse_predict_len(predict_len)
    chart_data = build_chart_data(stock_ticker, current_price, df)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "chart_data_py": chart_data,
            "chart_data_json": json.dumps(chart_data),
            "portfolio": portfolio,
            "holdings": holding_details,
            "predicted_price_lstm": "N/A",
            "predicted_price_ridge": "N/A",
            "model_type": "both",
            "gemini_description": gemini_description,
            "company_name": company_name,
            "flash_messages": pop_flash_messages(request),
            "predict_len": predict_len_value,
        },
    )


@app.get("/api/stock/{ticker}/predict", name="predict_stock")
def predict_stock(ticker: str, predict_len: Optional[str] = None):
    stock_ticker = ticker.upper()

    try:
        stock, current_price = get_stock_data(stock_ticker)
        if stock is None:
            return JSONResponse(
                status_code=404,
                content={"error": f'Ticker "{stock_ticker}" not found.'},
            )

        predict_len_value = parse_predict_len(predict_len)
        df = stock.history(period="1y")
        prediction_response = build_prediction_response(
            stock_ticker,
            current_price,
            df,
            predict_len_value,
        )

        return JSONResponse(content=sanitize_for_json(prediction_response))
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": f"予測の生成に失敗しました: {exc}"},
        )


@app.get("/api/stock/{ticker}/direction", name="predict_stock_direction")
def predict_stock_direction(ticker: str, predict_len: Optional[str] = None):
    stock_ticker = ticker.upper()

    try:
        stock, _ = get_stock_data(stock_ticker)
        if stock is None:
            return JSONResponse(
                status_code=404,
                content={"error": f'Ticker "{stock_ticker}" not found.'},
            )

        predict_len_value = parse_predict_len(predict_len)
        df = stock.history(period="2y")
        direction_response = predict_direction_with_lgbm(
            df,
            predict_len=predict_len_value,
        )
        return JSONResponse(content=sanitize_for_json(direction_response))
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": f"方向予測の生成に失敗しました: {exc}"},
        )


@app.get("/portfolio", response_class=HTMLResponse, name="portfolio_page")
def portfolio_page(request: Request, refresh: Optional[str] = None):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM user WHERE id = 1").fetchone()
    holdings_db = conn.execute("SELECT * FROM holdings ORDER BY ticker ASC").fetchall()
    transactions = conn.execute("SELECT * FROM transactions ORDER BY timestamp ASC").fetchall()
    conn.close()

    portfolio, holding_details, pie_chart = build_portfolio_snapshot(
        float(user["balance"]),
        holdings_db,
        transactions,
    )
    chart_data = build_asset_history_chart(transactions, float(user["balance"]))

    return templates.TemplateResponse(
        request=request,
        name="portfolio.html",
        context={
            "request": request,
            "portfolio": portfolio,
            "holdings": holding_details,
            "chart_data": chart_data,
            "pie_chart_data": json.dumps(pie_chart),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "refreshed": refresh is not None,
            "flash_messages": pop_flash_messages(request),
        },
    )


@app.post("/buy/{ticker}", name="buy")
def buy(request: Request, ticker: str, shares: str = Form(...)):
    stock_ticker = ticker.upper()

    try:
        shares_to_buy = parse_share_amount(shares)
    except (TypeError, ValueError):
        add_flash_message(request, "数量は 0 より大きい数値で入力してください。", "danger")
        return RedirectResponse(url=f"/stock/{stock_ticker}", status_code=303)

    _, current_price = get_stock_data(stock_ticker)
    if current_price is None:
        add_flash_message(request, f"Could not retrieve price for {stock_ticker}.", "danger")
        return RedirectResponse(url=f"/stock/{stock_ticker}", status_code=303)

    cost = shares_to_buy * float(current_price)
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM user WHERE id = 1").fetchone()

    if float(user["balance"]) + SHARE_EPSILON >= cost:
        conn.execute("UPDATE user SET balance = balance - ? WHERE id = 1", (cost,))
        conn.execute(
            """
            INSERT INTO holdings (ticker, shares) VALUES (?, ?)
            ON CONFLICT(ticker) DO UPDATE SET shares = shares + excluded.shares
            """,
            (stock_ticker, shares_to_buy),
        )
        conn.execute(
            "INSERT INTO transactions (ticker, shares, price, type) VALUES (?, ?, ?, ?)",
            (stock_ticker, shares_to_buy, float(current_price), "BUY"),
        )
        conn.commit()
        add_flash_message(
            request,
            f"Successfully bought {format_shares(shares_to_buy)} shares of {stock_ticker}.",
            "success",
        )
    else:
        add_flash_message(request, "Insufficient balance.", "danger")

    conn.close()
    return RedirectResponse(url=f"/stock/{stock_ticker}", status_code=303)


@app.post("/sell/{ticker}", name="sell")
def sell(request: Request, ticker: str, shares: str = Form(...)):
    stock_ticker = ticker.upper()

    try:
        shares_to_sell = parse_share_amount(shares)
    except (TypeError, ValueError):
        add_flash_message(request, "数量は 0 より大きい数値で入力してください。", "danger")
        return RedirectResponse(url=f"/stock/{stock_ticker}", status_code=303)

    _, current_price = get_stock_data(stock_ticker)
    if current_price is None:
        add_flash_message(request, f"Could not retrieve price for {stock_ticker}.", "danger")
        return RedirectResponse(url=f"/stock/{stock_ticker}", status_code=303)

    conn = get_db_connection()
    holding = conn.execute(
        "SELECT * FROM holdings WHERE ticker = ?",
        (stock_ticker,),
    ).fetchone()

    current_shares = float(holding["shares"]) if holding else 0.0
    if holding and current_shares + SHARE_EPSILON >= shares_to_sell:
        proceeds = shares_to_sell * float(current_price)
        conn.execute("UPDATE user SET balance = balance + ? WHERE id = 1", (proceeds,))
        new_shares = current_shares - shares_to_sell
        if new_shares <= SHARE_EPSILON:
            conn.execute("DELETE FROM holdings WHERE ticker = ?", (stock_ticker,))
        else:
            conn.execute(
                "UPDATE holdings SET shares = ? WHERE ticker = ?",
                (round(new_shares, 4), stock_ticker),
            )
        conn.execute(
            "INSERT INTO transactions (ticker, shares, price, type) VALUES (?, ?, ?, ?)",
            (stock_ticker, shares_to_sell, float(current_price), "SELL"),
        )
        conn.commit()
        add_flash_message(
            request,
            f"Successfully sold {format_shares(shares_to_sell)} shares of {stock_ticker}.",
            "success",
        )
    else:
        add_flash_message(request, "You do not own enough shares to sell.", "danger")

    conn.close()
    return RedirectResponse(url=f"/stock/{stock_ticker}", status_code=303)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
