import logging
import os
import sqlite3

import google.generativeai as genai
import yfinance as yf

from config import (
    DATABASE_PATH,
    GEMINI_API_KEY_ENV,
    GEMINI_MODEL_NAME,
    GEMINI_SAFETY_SETTINGS,
)


def _create_gemini_model():
    try:
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


def get_db_connection():
    """Gets a connection to the SQLite database."""
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
