import json
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from flask import Flask, flash, redirect, render_template, request, url_for

from config import AppConfig, DEFAULT_PREDICT_LEN, DEFAULT_TICKER
from predictors import predict_with_lstm, predict_with_ridge
from services import get_db_connection, get_gemini_description, get_stock_data


app = Flask(__name__)
app.config.from_object(AppConfig)


@app.route("/")
def search():
    """Homepage, redirects to a default stock (e.g., AAPL)."""
    return redirect(url_for("index", ticker=DEFAULT_TICKER))


@app.route("/stock/<ticker>")
def index(ticker):
    """Main page displaying the stock chart, info, and prediction."""
    stock_ticker = ticker.upper()
    stock, current_price = get_stock_data(stock_ticker)
    if stock is None:
        flash(f'Ticker "{stock_ticker}" not found.', "danger")
        return redirect(url_for("search"))

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

    try:
        predict_len = int(request.args.get("predict_len", DEFAULT_PREDICT_LEN))
    except (ValueError, TypeError):
        predict_len = DEFAULT_PREDICT_LEN

    lstm_prediction_series = predict_with_lstm(df, predict_len=predict_len)
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
        last_date_for_label = df.index[-1]
        prediction_labels = pd.date_range(
            start=last_date_for_label + pd.Timedelta(days=1),
            periods=predict_len,
            freq="B",
        ).strftime("%Y-%m-%d").tolist()

    prediction_data_lstm = lstm_prediction_series.tolist()
    prediction_data_ridge = ridge_prediction_series.tolist()
    prediction_data_points_lstm = [None] * (len(df) - 1) + [df["Close"].iloc[-1]] + prediction_data_lstm
    prediction_data_points_ridge = [None] * (len(df) - 1) + [df["Close"].iloc[-1]] + prediction_data_ridge

    labels = df.index.strftime("%Y-%m-%d").tolist() + prediction_labels
    data = df["Close"].tolist() + [None] * len(prediction_labels)

    chart_data = {
        "labels": labels,
        "data": data,
        "prediction_data_lstm": prediction_data_points_lstm,
        "prediction_data_ridge": prediction_data_points_ridge,
        "ticker": stock_ticker,
        "current_price": round(current_price, 2),
    }

    predicted_price_lstm_rounded = (
        round(predicted_price_lstm_val, 2) if predicted_price_lstm_val is not None else "N/A"
    )
    predicted_price_ridge_rounded = (
        round(predicted_price_ridge_val, 2) if predicted_price_ridge_val is not None else "N/A"
    )

    return render_template(
        "index.html",
        chart_data_py=chart_data,
        chart_data_json=json.dumps(chart_data),
        portfolio=portfolio,
        holdings=holdings_with_value,
        predicted_price_lstm=predicted_price_lstm_rounded,
        predicted_price_ridge=predicted_price_ridge_rounded,
        model_type="both",
        gemini_description=gemini_description,
        company_name=company_name,
    )


@app.route("/portfolio")
def portfolio_page():
    """Displays the historical asset value chart."""
    conn = get_db_connection()
    transactions = conn.execute("SELECT * FROM transactions ORDER BY timestamp ASC").fetchall()
    user = conn.execute("SELECT * FROM user WHERE id = 1").fetchone()
    conn.close()

    if not transactions:
        return render_template("portfolio.html", chart_data=None)

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
    return render_template("portfolio.html", chart_data=chart_data)


@app.route("/buy/<ticker>", methods=["POST"])
def buy(ticker):
    """Processes a stock purchase."""
    try:
        shares_to_buy = int(request.form["shares"])
        if shares_to_buy <= 0:
            flash("Please enter a positive number of shares.", "danger")
            return redirect(url_for("index", ticker=ticker))
    except ValueError:
        flash("Invalid number of shares.", "danger")
        return redirect(url_for("index", ticker=ticker))

    _, current_price = get_stock_data(ticker)
    if not current_price:
        flash(f"Could not retrieve price for {ticker}.", "danger")
        return redirect(url_for("index", ticker=ticker))

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
        flash(f"Successfully bought {shares_to_buy} shares of {ticker}.", "success")
    else:
        flash("Insufficient balance.", "danger")

    conn.close()
    return redirect(url_for("index", ticker=ticker))


@app.route("/sell/<ticker>", methods=["POST"])
def sell(ticker):
    """Processes a stock sale."""
    try:
        shares_to_sell = int(request.form["shares"])
        if shares_to_sell <= 0:
            flash("Please enter a positive number of shares.", "danger")
            return redirect(url_for("index", ticker=ticker))
    except ValueError:
        flash("Invalid number of shares.", "danger")
        return redirect(url_for("index", ticker=ticker))

    _, current_price = get_stock_data(ticker)
    if not current_price:
        flash(f"Could not retrieve price for {ticker}.", "danger")
        return redirect(url_for("index", ticker=ticker))

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
        flash(f"Successfully sold {shares_to_sell} shares of {ticker}.", "success")
    else:
        flash("You do not own enough shares to sell.", "danger")

    conn.close()
    return redirect(url_for("index", ticker=ticker))


if __name__ == "__main__":
    app.run(debug=True)
