"""Intraday market-data preparation and short-horizon forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


INTRADAY_INTERVALS = {
    "1m": {"label": "1分足", "period": "8d", "minutes": 1},
    "2m": {"label": "2分足", "period": "60d", "minutes": 2},
    "5m": {"label": "5分足", "period": "60d", "minutes": 5},
    "15m": {"label": "15分足", "period": "60d", "minutes": 15},
    "30m": {"label": "30分足", "period": "60d", "minutes": 30},
}

FEATURE_LAGS = 12
MIN_TRAINING_ROWS = 40


@dataclass
class IntradayForecast:
    prices: list[float]
    timestamps: list[pd.Timestamp]
    training_rows: int
    training_days: int


def validate_interval(value: str | None) -> str:
    return value if value in INTRADAY_INTERVALS else "5m"


def parse_steps(value: str | None, default: int = 12, maximum: int = 120) -> int:
    try:
        return min(maximum, max(1, int(value or default)))
    except (TypeError, ValueError):
        return default


def clean_intraday_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    result = df.copy().sort_index()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(column not in result.columns for column in needed):
        return pd.DataFrame(columns=needed)
    result = result[needed].apply(pd.to_numeric, errors="coerce")
    return result.dropna(subset=["Open", "High", "Low", "Close"])


def _session_key(index: pd.DatetimeIndex) -> pd.Index:
    # yfinance returns an exchange-local, timezone-aware index. Normalizing it
    # preserves the exchange trading day, including for Japanese/US tickers.
    return index.normalize()


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    features = pd.DataFrame(index=df.index)
    returns = close.pct_change()
    for lag in range(1, FEATURE_LAGS + 1):
        features[f"return_lag_{lag}"] = returns.shift(lag - 1)
    features["range"] = (df["High"] - df["Low"]) / close.replace(0, np.nan)
    features["body"] = (close - df["Open"]) / df["Open"].replace(0, np.nan)
    volume = np.log1p(df["Volume"].clip(lower=0))
    features["volume_change"] = volume.diff()

    minutes = df.index.hour * 60 + df.index.minute
    features["time_sin"] = np.sin(2 * np.pi * minutes / (24 * 60))
    features["time_cos"] = np.cos(2 * np.pi * minutes / (24 * 60))
    features["target"] = returns.shift(-1)
    return features.replace([np.inf, -np.inf], np.nan)


def _training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, int]:
    sessions = _session_key(df.index)
    parts: list[pd.DataFrame] = []
    for _, day in df.groupby(sessions):
        if len(day) <= FEATURE_LAGS + 1:
            continue
        featured = _feature_frame(day).dropna()
        if not featured.empty:
            parts.append(featured)
    if not parts:
        return pd.DataFrame(), pd.Series(dtype=float), 0
    training = pd.concat(parts).sort_index()
    return training.drop(columns="target"), training["target"], len(parts)


def _next_timestamp(timestamp: pd.Timestamp, minutes: int) -> pd.Timestamp:
    return timestamp + pd.Timedelta(minutes=minutes)


def forecast_intraday(df: pd.DataFrame, interval: str, steps: int) -> IntradayForecast:
    interval = validate_interval(interval)
    steps = parse_steps(str(steps))
    clean = clean_intraday_frame(df)
    if clean.empty:
        raise ValueError("日中足データがありません。")

    x_train, y_train, training_days = _training_data(clean)
    if len(x_train) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"学習データが不足しています（{len(x_train)}件）。別の足幅または取引時間中に再試行してください。"
        )

    scaler = StandardScaler()
    model = Ridge(alpha=4.0)
    model.fit(scaler.fit_transform(x_train), y_train)

    minutes = INTRADAY_INTERVALS[interval]["minutes"]
    current_session = clean[_session_key(clean.index) == _session_key(clean.index)[-1]].copy()
    context = current_session if len(current_session) > FEATURE_LAGS else clean.tail(FEATURE_LAGS + 3).copy()
    prices: list[float] = []
    timestamps: list[pd.Timestamp] = []

    for _ in range(steps):
        next_time = _next_timestamp(context.index[-1], minutes)
        previous_close = float(context["Close"].iloc[-1])
        synthetic = pd.DataFrame(
            {
                "Open": [previous_close], "High": [previous_close],
                "Low": [previous_close], "Close": [previous_close],
                "Volume": [float(context["Volume"].tail(5).median())],
            },
            index=pd.DatetimeIndex([next_time]),
        )
        probe = pd.concat([context, synthetic])
        row = _feature_frame(probe).drop(columns="target").iloc[-1]
        row = row.reindex(x_train.columns).fillna(0.0)
        predicted_return = float(model.predict(scaler.transform(row.to_frame().T))[0])
        predicted_return = float(np.clip(predicted_return, -0.05, 0.05))
        price = max(0.01, previous_close * (1.0 + predicted_return))
        probe.loc[next_time, ["Open", "High", "Low", "Close"]] = price
        context = probe.tail(max(FEATURE_LAGS + 3, 20))
        prices.append(round(price, 4))
        timestamps.append(next_time)

    return IntradayForecast(prices, timestamps, len(x_train), training_days)


def timestamp_label(value: pd.Timestamp) -> str:
    return value.isoformat()


def build_intraday_payload(
    ticker: str, df: pd.DataFrame, interval: str, steps: int
) -> dict[str, Any]:
    clean = clean_intraday_frame(df)
    if clean.empty:
        raise ValueError("指定した銘柄の日中足データが見つかりませんでした。")
    forecast = forecast_intraday(clean, interval, steps)
    latest_session = _session_key(clean.index)[-1]
    display = clean[_session_key(clean.index) == latest_session].copy()
    last_price = float(display["Close"].iloc[-1])
    prediction_values = [None] * (len(display) - 1) + [round(last_price, 4)] + forecast.prices
    labels = [timestamp_label(value) for value in display.index]
    forecast_labels = [timestamp_label(value) for value in forecast.timestamps]
    return {
        "ticker": ticker,
        "interval": interval,
        "interval_label": INTRADAY_INTERVALS[interval]["label"],
        "steps": steps,
        "horizon_minutes": steps * INTRADAY_INTERVALS[interval]["minutes"],
        "last_price": round(last_price, 4),
        "predicted_price": forecast.prices[-1],
        "as_of": timestamp_label(display.index[-1]),
        "training_rows": forecast.training_rows,
        "training_days": forecast.training_days,
        "labels": labels + forecast_labels,
        "ohlc": [
            {
                "x": timestamp_label(index), "o": round(float(row.Open), 4),
                "h": round(float(row.High), 4), "l": round(float(row.Low), 4),
                "c": round(float(row.Close), 4),
            }
            for index, row in display.iterrows()
        ],
        "actual_close": display["Close"].round(4).tolist() + [None] * steps,
        "volume": display["Volume"].fillna(0).astype(float).tolist() + [None] * steps,
        "prediction": prediction_values,
    }
