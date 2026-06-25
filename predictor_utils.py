from typing import Any, Optional

import pandas as pd

from config import DIRECTION_CONFIG


def fallback_prediction(df: pd.DataFrame, predict_len: int) -> pd.Series:
    last_price = float(df["Close"].iloc[-1]) if not df.empty else 0.0
    last_date = df.index[-1] if not df.empty else pd.Timestamp.today().normalize()
    prediction_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=predict_len,
        freq="B",
    )
    return pd.Series([last_price] * predict_len, index=prediction_dates)


def format_last_date(index_value: Any) -> Optional[str]:
    if index_value is None:
        return None
    if hasattr(index_value, "strftime"):
        return index_value.strftime("%Y-%m-%d")
    return str(index_value)


def direction_fallback(df: pd.DataFrame, predict_len: int, reason: str) -> dict[str, Any]:
    last_close = None
    last_date = None
    if not df.empty and "Close" in df.columns:
        close_series = df["Close"].dropna()
        if not close_series.empty:
            last_close = float(close_series.iloc[-1])
            last_date = format_last_date(close_series.index[-1])

    return {
        "model": "LightGBM Direction Classifier",
        "horizon_days": predict_len,
        "last_date": last_date,
        "last_close": last_close,
        "probability_up": 0.5,
        "probability_down": 0.5,
        "decision_threshold": 0.5,
        "predicted_direction": "NEUTRAL",
        "signal": "HOLD",
        "target_return_threshold": DIRECTION_CONFIG.get("target_return_threshold", 0.0),
        "buy_probability_threshold": DIRECTION_CONFIG.get("buy_probability_threshold", 0.60),
        "sell_probability_threshold": DIRECTION_CONFIG.get("sell_probability_threshold", 0.40),
        "metrics": {},
        "top_features": [],
        "reason": reason,
    }
