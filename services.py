import logging
import os
import sqlite3
from typing import Optional

import pandas as pd
import yfinance as yf

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from config import (
    DATABASE_PATH,
    GEMINI_API_KEY_ENV,
    GEMINI_MODEL_NAME,
    GEMINI_SAFETY_SETTINGS,
)


def _create_gemini_model():
    try:
        if genai is None:
            logging.warning("google-generativeai が未インストールのため Gemini は無効化されます。")
            return None

        api_key = os.environ.get(GEMINI_API_KEY_ENV)
        if not api_key:
            logging.warning("GEMINI_API_KEY が環境変数に設定されていません。")
            return None

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL_NAME,
            safety_settings=GEMINI_SAFETY_SETTINGS,
        )
        logging.info("Geminiモデルの初期化完了。")
        return model
    except Exception as error:
        logging.error(f"Geminiの初期化に失敗しました: {error}")
        return None


gemini_model = _create_gemini_model()
_schema_checked = False


def _table_info(conn: sqlite3.Connection, table_name: str) -> dict[str, str]:
    return {
        row[1]: str(row[2]).upper()
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


def ensure_portfolio_schema() -> None:
    global _schema_checked
    if _schema_checked:
        return

    conn = sqlite3.connect(DATABASE_PATH)
    try:
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        required_tables = {"user", "holdings", "transactions"}

        if not required_tables.issubset(table_names):
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS user (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    balance REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS holdings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    shares REAL NOT NULL,
                    UNIQUE(ticker)
                );

                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    shares REAL NOT NULL,
                    price REAL NOT NULL,
                    type TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            user_exists = conn.execute("SELECT 1 FROM user WHERE id = 1").fetchone()
            if user_exists is None:
                conn.execute("INSERT INTO user (balance) VALUES (10000.00)")
            conn.commit()
            _schema_checked = True
            return

        holdings_info = _table_info(conn, "holdings")
        transactions_info = _table_info(conn, "transactions")
        needs_migration = (
            holdings_info.get("shares") != "REAL"
            or transactions_info.get("shares") != "REAL"
        )

        if needs_migration:
            conn.executescript(
                """
                BEGIN;
                ALTER TABLE holdings RENAME TO holdings_old;
                CREATE TABLE holdings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    shares REAL NOT NULL,
                    UNIQUE(ticker)
                );
                INSERT INTO holdings (id, ticker, shares)
                SELECT id, ticker, CAST(shares AS REAL)
                FROM holdings_old;
                DROP TABLE holdings_old;

                ALTER TABLE transactions RENAME TO transactions_old;
                CREATE TABLE transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    shares REAL NOT NULL,
                    price REAL NOT NULL,
                    type TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                INSERT INTO transactions (id, ticker, shares, price, type, timestamp)
                SELECT id, ticker, CAST(shares AS REAL), price, type, timestamp
                FROM transactions_old;
                DROP TABLE transactions_old;
                COMMIT;
                """
            )
            conn.commit()
    finally:
        conn.close()

    _schema_checked = True


def get_db_connection():
    """Gets a connection to the SQLite database."""
    ensure_portfolio_schema()
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_stock_data(ticker):
    """Fetches the latest stock price and yfinance object with error handling."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if hist.empty:
            return None, None
        return stock, hist["Close"].iloc[-1]
    except Exception:
        return None, None


def get_price_snapshots(
    tickers: list[str],
) -> dict[str, dict[str, Optional[float]]]:
    unique_tickers = sorted({ticker.upper() for ticker in tickers if ticker})
    snapshots = {
        ticker: {"current_price": None, "previous_close": None}
        for ticker in unique_tickers
    }
    if not unique_tickers:
        return snapshots

    try:
        close_data = yf.download(
            " ".join(unique_tickers),
            period="7d",
            progress=False,
            auto_adjust=False,
        )["Close"]
        if isinstance(close_data, pd.Series):
            close_data = close_data.to_frame(name=unique_tickers[0])

        for ticker in unique_tickers:
            if ticker not in close_data.columns:
                continue
            series = close_data[ticker].dropna()
            if series.empty:
                continue

            current_price = float(series.iloc[-1])
            previous_close = float(series.iloc[-2]) if len(series) > 1 else None
            snapshots[ticker] = {
                "current_price": current_price,
                "previous_close": previous_close,
            }
    except Exception as error:
        logging.warning(f"価格一括取得に失敗しました: {error}")

    for ticker, snapshot in snapshots.items():
        if snapshot["current_price"] is not None:
            continue

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="7d")
            if hist.empty:
                continue

            close_series = hist["Close"].dropna()
            if close_series.empty:
                continue

            snapshot["current_price"] = float(close_series.iloc[-1])
            snapshot["previous_close"] = (
                float(close_series.iloc[-2]) if len(close_series) > 1 else None
            )
        except Exception:
            continue

    return snapshots


def get_gemini_description(ticker, stock_info):
    """Gemini APIを使用して銘柄の簡潔な説明を取得する"""
    if not gemini_model:
        logging.warning("Geminiモデルが利用できません。説明取得をスキップします。")
        return "説明の取得に失敗しました（Geminiモデル未初期化）。"

    company_name = stock_info.get("longName", stock_info.get("shortName", ticker))

    try:
        industry = stock_info.get("industry", "不明")
        sector = stock_info.get("sector", "不明")
        prompt = f"""
        以下の企業について、投資家向けに日本語で簡潔に説明してください。
        企業名（ティッカー）: {company_name} ({ticker})
        業種: {industry} ({sector})

        説明には以下の点を含めてください：
        1. 主な事業内容
        2. 企業の強みや市場での位置づけ

        また以下の制約を必ず守ること
        1. 銘柄名から始めて、次の行から銘柄を説明すること
        2. 銘柄に関連する「2025年以降」のニュースを日付とともに述べること。その際【2025年以降のニュース】から始めること。
        説明文は全体で300文字程度の簡潔な文章にまとめてください。
        """

        response = gemini_model.generate_content(prompt)
        if response.parts:
            return response.text

        logging.warning(f"Geminiからのレスポンスが空です: {response.prompt_feedback}")
        return f"{company_name} ({ticker}) の説明は現在取得できません。"
    except Exception as error:
        logging.error(f"Gemini API呼び出しエラー ({ticker}): {error}")
        return f"{company_name} ({ticker}) の説明取得中にエラーが発生しました。"
