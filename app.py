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
from services import get_db_connection, get_gemini_description, get_stock_data


BASE_DIR = Path(__file__).resolve().parent
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


@app.get("/", name="search")
def search():
    """Homepage, redirects to a default stock (e.g., AAPL)."""
    return RedirectResponse(
        url=f"/stock/{DEFAULT_TICKER}",
        status_code=303,
    )


@app.get("/stock/{ticker}", response_class=HTMLResponse, name="index")
def index(request: Request, ticker: str, predict_len: Optional[str] = None):
    """Main page displaying the stock chart and stock info."""
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
    holdings_db = conn.execute("SELECT * FROM holdings").fetchall()
    conn.close()

    total_holdings_value = 0
    holdings_with_value = []
    for holding in holdings_db:
        _, price = get_stock_data(holding["ticker"])
        value = price * holding["shares"] if price else 0
        total_holdings_value += value
        holdings_with_value.append(
            {
                "ticker": holding["ticker"],
                "shares": holding["shares"],
                "value": round(value, 2),
            }
        )

    portfolio = {
        "balance": round(user["balance"], 2),
        "holdings_value": round(total_holdings_value, 2),
        "total_asset": round(user["balance"] + total_holdings_value, 2),
    }

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
            "holdings": holdings_with_value,
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
def portfolio_page(request: Request):
    """Displays the historical asset value chart."""
    conn = get_db_connection()
    transactions = conn.execute("SELECT * FROM transactions ORDER BY timestamp ASC").fetchall()
    user = conn.execute("SELECT * FROM user WHERE id = 1").fetchone()
    conn.close()

    if not transactions:
        return templates.TemplateResponse(
            request=request,
            name="portfolio.html",
            context={
                "request": request,
                "chart_data": None,
                "flash_messages": pop_flash_messages(request),
            },
        )

    start_date = datetime.strptime(transactions[0]["timestamp"].split(" ")[0], "%Y-%m-%d").date()
    end_date = datetime.now().date()
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    tickers_str = " ".join(set([transaction["ticker"] for transaction in transactions]))
    hist_data = yf.download(
        tickers_str,
        start=start_date,
        end=end_date + timedelta(days=1),
        progress=False,
    )["Close"]
    if isinstance(hist_data, pd.Series):
        hist_data = hist_data.to_frame(name=tickers_str)

    asset_history = []
    cash = user["balance"]
    holdings = {ticker: 0 for ticker in tickers_str.split(" ")}

    holdings_conn = get_db_connection()
    all_holdings = holdings_conn.execute("SELECT * FROM holdings").fetchall()
    holdings_conn.close()
    for holding in all_holdings:
        holdings[holding["ticker"]] = holding["shares"]

    for transaction in reversed(transactions):
        if transaction["type"] == "BUY":
            cash += transaction["shares"] * transaction["price"]
        else:
            cash -= transaction["shares"] * transaction["price"]

    tx_idx = 0
    for day in date_range:
        while (
            tx_idx < len(transactions)
            and datetime.strptime(transactions[tx_idx]["timestamp"].split(" ")[0], "%Y-%m-%d").date()
            <= day.date()
        ):
            transaction = transactions[tx_idx]
            if transaction["type"] == "BUY":
                cash -= transaction["shares"] * transaction["price"]
            else:
                cash += transaction["shares"] * transaction["price"]
            tx_idx += 1

        holdings_value = 0
        for ticker, shares in holdings.items():
            if shares > 0:
                try:
                    price_on_day = hist_data.loc[:day, ticker].ffill().iloc[-1]
                    if not pd.isna(price_on_day):
                        holdings_value += shares * price_on_day
                except (KeyError, IndexError):
                    pass

        asset_history.append(cash + holdings_value)

    chart_data = json.dumps(
        {
            "labels": [day.strftime("%Y-%m-%d") for day in date_range],
            "data": asset_history,
        }
    )
    return templates.TemplateResponse(
        request=request,
        name="portfolio.html",
        context={
            "request": request,
            "chart_data": chart_data,
            "flash_messages": pop_flash_messages(request),
        },
    )


@app.post("/buy/{ticker}", name="buy")
def buy(request: Request, ticker: str, shares: str = Form(...)):
    """Processes a stock purchase."""
    try:
        shares_to_buy = int(shares)
        if shares_to_buy <= 0:
            add_flash_message(request, "Please enter a positive number of shares.", "danger")
            return RedirectResponse(url=f"/stock/{ticker}", status_code=303)
    except (TypeError, ValueError):
        add_flash_message(request, "Invalid number of shares.", "danger")
        return RedirectResponse(url=f"/stock/{ticker}", status_code=303)

    _, current_price = get_stock_data(ticker)
    if not current_price:
        add_flash_message(request, f"Could not retrieve price for {ticker}.", "danger")
        return RedirectResponse(url=f"/stock/{ticker}", status_code=303)

    cost = shares_to_buy * current_price
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM user WHERE id = 1").fetchone()

    if user["balance"] >= cost:
        conn.execute("UPDATE user SET balance = balance - ? WHERE id = 1", (cost,))
        conn.execute(
            """
            INSERT INTO holdings (ticker, shares) VALUES (?, ?)
            ON CONFLICT(ticker) DO UPDATE SET shares = shares + excluded.shares
            """,
            (ticker, shares_to_buy),
        )
        conn.execute(
            "INSERT INTO transactions (ticker, shares, price, type) VALUES (?, ?, ?, ?)",
            (ticker, shares_to_buy, current_price, "BUY"),
        )
        conn.commit()
        add_flash_message(request, f"Successfully bought {shares_to_buy} shares of {ticker}.", "success")
    else:
        add_flash_message(request, "Insufficient balance.", "danger")

    conn.close()
    return RedirectResponse(url=f"/stock/{ticker}", status_code=303)


@app.post("/sell/{ticker}", name="sell")
def sell(request: Request, ticker: str, shares: str = Form(...)):
    """Processes a stock sale."""
    try:
        shares_to_sell = int(shares)
        if shares_to_sell <= 0:
            add_flash_message(request, "Please enter a positive number of shares.", "danger")
            return RedirectResponse(url=f"/stock/{ticker}", status_code=303)
    except (TypeError, ValueError):
        add_flash_message(request, "Invalid number of shares.", "danger")
        return RedirectResponse(url=f"/stock/{ticker}", status_code=303)

    _, current_price = get_stock_data(ticker)
    if not current_price:
        add_flash_message(request, f"Could not retrieve price for {ticker}.", "danger")
        return RedirectResponse(url=f"/stock/{ticker}", status_code=303)

    conn = get_db_connection()
    holding = conn.execute("SELECT * FROM holdings WHERE ticker = ?", (ticker,)).fetchone()

    if holding and holding["shares"] >= shares_to_sell:
        proceeds = shares_to_sell * current_price
        conn.execute("UPDATE user SET balance = balance + ? WHERE id = 1", (proceeds,))
        new_shares = holding["shares"] - shares_to_sell
        if new_shares == 0:
            conn.execute("DELETE FROM holdings WHERE ticker = ?", (ticker,))
        else:
            conn.execute(
                "UPDATE holdings SET shares = ? WHERE ticker = ?",
                (new_shares, ticker),
            )
        conn.execute(
            "INSERT INTO transactions (ticker, shares, price, type) VALUES (?, ?, ?, ?)",
            (ticker, shares_to_sell, current_price, "SELL"),
        )
        conn.commit()
        add_flash_message(request, f"Successfully sold {shares_to_sell} shares of {ticker}.", "success")
    else:
        add_flash_message(request, "You do not own enough shares to sell.", "danger")

    conn.close()
    return RedirectResponse(url=f"/stock/{ticker}", status_code=303)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
