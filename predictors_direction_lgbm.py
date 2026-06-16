from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from services import get_stock_data
from config import DIRECTION_CONFIG
from predictor_utils import direction_fallback, format_last_date


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _build_direction_features(
    df: pd.DataFrame,
    predict_len: int,
) -> tuple[pd.DataFrame, list[str]]:
    feature_df = df.copy().sort_index()
    feature_cols: list[str] = []

    if "Close" not in feature_df.columns:
        return feature_df, feature_cols

    threshold = DIRECTION_CONFIG.get("target_return_threshold", 0.0)

    feature_df["log_return_1"] = np.log(feature_df["Close"] / feature_df["Close"].shift(1))
    feature_df["return_2"] = feature_df["Close"].pct_change(2)
    feature_df["return_5"] = feature_df["Close"].pct_change(5)
    feature_df["return_10"] = feature_df["Close"].pct_change(10)
    feature_df["return_20"] = feature_df["Close"].pct_change(20)
    feature_df["return_60"] = feature_df["Close"].pct_change(60)
    feature_df["volatility_5"] = feature_df["log_return_1"].rolling(5).std()
    feature_df["volatility_10"] = feature_df["log_return_1"].rolling(10).std()
    feature_df["volatility_20"] = feature_df["log_return_1"].rolling(20).std()
    feature_df["volatility_60"] = feature_df["log_return_1"].rolling(60).std()
    feature_df["ma_gap_5"] = feature_df["Close"] / feature_df["Close"].rolling(5).mean() - 1
    feature_df["ma_gap_20"] = feature_df["Close"] / feature_df["Close"].rolling(20).mean() - 1
    feature_df["ma_gap_60"] = feature_df["Close"] / feature_df["Close"].rolling(60).mean() - 1
    feature_df["ma_gap_120"] = feature_df["Close"] / feature_df["Close"].rolling(120).mean() - 1
    feature_df["momentum_5"] = feature_df["Close"] / feature_df["Close"].shift(5) - 1
    feature_df["momentum_20"] = feature_df["Close"] / feature_df["Close"].shift(20) - 1
    feature_df["momentum_60"] = feature_df["Close"] / feature_df["Close"].shift(60) - 1

    feature_cols.extend(
        [
            "log_return_1",
            "return_2",
            "return_5",
            "return_10",
            "return_20",
            "return_60",
            "volatility_5",
            "volatility_10",
            "volatility_20",
            "volatility_60",
            "ma_gap_5",
            "ma_gap_20",
            "ma_gap_60",
            "ma_gap_120",
            "momentum_5",
            "momentum_20",
            "momentum_60",
        ]
    )

    if "Volume" in feature_df.columns:
        feature_df["volume_change_5"] = feature_df["Volume"].pct_change(5)
        feature_df["volume_change_20"] = feature_df["Volume"].pct_change(20)
        feature_df["volume_z_20"] = (
            feature_df["Volume"] - feature_df["Volume"].rolling(20).mean()
        ) / feature_df["Volume"].rolling(20).std()
        feature_cols.extend(["volume_change_5", "volume_change_20", "volume_z_20"])

    if {"Open", "High", "Low"}.issubset(feature_df.columns):
        feature_df["intraday_return"] = feature_df["Close"] / feature_df["Open"] - 1
        feature_df["high_low_range"] = feature_df["High"] / feature_df["Low"] - 1
        feature_df["close_to_high"] = feature_df["Close"] / feature_df["High"] - 1
        feature_df["close_to_low"] = feature_df["Close"] / feature_df["Low"] - 1
        feature_cols.extend(
            ["intraday_return", "high_low_range", "close_to_high", "close_to_low"]
        )

    dow = pd.to_datetime(feature_df.index).dayofweek
    feature_df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feature_df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    feature_df["rsi_14"] = _compute_rsi(feature_df["Close"], period=14)
    feature_df["ema_12"] = feature_df["Close"].ewm(span=12, adjust=False).mean()
    feature_df["ema_26"] = feature_df["Close"].ewm(span=26, adjust=False).mean()
    feature_df["macd"] = feature_df["ema_12"] - feature_df["ema_26"]
    feature_df["macd_signal"] = feature_df["macd"].ewm(span=9, adjust=False).mean()
    feature_df["macd_hist"] = feature_df["macd"] - feature_df["macd_signal"]
    feature_cols.extend(
        [
            "dow_sin",
            "dow_cos",
            "rsi_14",
            "ema_12",
            "ema_26",
            "macd",
            "macd_signal",
            "macd_hist",
        ]
    )

    feature_df["future_log_return"] = np.log(
        feature_df["Close"].shift(-predict_len) / feature_df["Close"]
    )
    feature_df["target"] = np.where(
        feature_df["future_log_return"].notna(),
        (feature_df["future_log_return"] > threshold).astype(int),
        np.nan,
    )

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.dropna(subset=feature_cols).copy()
    return feature_df, feature_cols


def _build_direction_model() -> Any:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError(
            "lightgbm is required for direction classification. Install it with `pip install lightgbm`."
        ) from exc

    return LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=DIRECTION_CONFIG.get("random_state", 42),
        n_jobs=-1,
        verbose=-1,
    )


def predict_direction_with_lgbm(
    df: pd.DataFrame,
    predict_len: int = 10,
) -> dict[str, Any]:
    if df.empty:
        return direction_fallback(df, predict_len, "Not enough data for direction classification")
    if "Close" not in df.columns:
        return direction_fallback(df, predict_len, "Close column is required for direction classification")

    min_samples = DIRECTION_CONFIG.get("min_samples", 120)
    if len(df) < min_samples:
        return direction_fallback(df, predict_len, "Not enough data for direction classification")

    model_factory = _build_direction_model()
    feature_df, feature_cols = _build_direction_features(df, predict_len)
    if not feature_cols or feature_df.empty:
        return direction_fallback(
            df,
            predict_len,
            "Not enough valid feature rows for direction classification",
        )

    latest_features = feature_df[feature_cols].iloc[[-1]]
    if latest_features.isna().any().any():
        return direction_fallback(
            df,
            predict_len,
            "Latest feature row contains NaN values",
        )

    training_df = feature_df[feature_df["target"].notna()].copy()
    if len(training_df) < min_samples:
        return direction_fallback(df, predict_len, "Not enough data for direction classification")

    y_all = training_df["target"].astype(int)
    if y_all.nunique() < 2:
        return direction_fallback(
            df,
            predict_len,
            "Training data contains only one direction class",
        )

    train_split = DIRECTION_CONFIG.get("train_split", 0.8)
    train_len = int(len(training_df) * train_split)
    if train_len <= 0 or train_len >= len(training_df):
        return direction_fallback(
            df,
            predict_len,
            "Unable to create a valid time-ordered train/test split",
        )

    train_df = training_df.iloc[:train_len].copy()
    test_df = training_df.iloc[train_len:].copy()

    x_train = train_df[feature_cols]
    y_train = train_df["target"].astype(int)
    x_test = test_df[feature_cols]
    y_test = test_df["target"].astype(int)

    if y_train.nunique() < 2:
        return direction_fallback(
            df,
            predict_len,
            "Training data contains only one direction class",
        )

    model = model_factory
    model.fit(x_train, y_train)

    metrics: dict[str, Any] = {}
    if test_df.empty:
        print("Warning: Test data is empty for direction classification.")
    elif y_test.nunique() < 2:
        print("Warning: Test target contains a single class for direction classification.")
    else:
        y_prob = model.predict_proba(x_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        majority_class = int(y_test.mode().iloc[0])
        baseline_predictions = np.full_like(y_test.to_numpy(), majority_class)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "log_loss": round(float(log_loss(y_test, y_prob)), 4),
            "baseline_accuracy": round(
                float(accuracy_score(y_test, baseline_predictions)),
                4,
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        }

        print("\n--- Direction LGBM Test Metrics ---")
        print(f"Horizon: {predict_len} business days")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"LogLoss: {metrics['log_loss']:.4f}")
        print(f"Baseline Accuracy: {metrics['baseline_accuracy']:.4f}")
        print(f"Confusion Matrix: {metrics['confusion_matrix']}")
        print("-------------------------------------------------")

    model_full = _build_direction_model()
    model_full.fit(training_df[feature_cols], y_all)

    latest_probabilities = model_full.predict_proba(latest_features)[0]
    prob_up = float(latest_probabilities[1])
    prob_down = float(1.0 - prob_up)

    buy_threshold = DIRECTION_CONFIG.get("buy_probability_threshold", 0.60)
    sell_threshold = DIRECTION_CONFIG.get("sell_probability_threshold", 0.40)

    if prob_up >= buy_threshold:
        signal = "BUY"
        direction = "UP"
    elif prob_up <= sell_threshold:
        signal = "SELL"
        direction = "DOWN"
    else:
        signal = "HOLD"
        direction = "NEUTRAL"

    importances = model_full.feature_importances_
    sorted_pairs = sorted(
        zip(feature_cols, importances),
        key=lambda item: item[1],
        reverse=True,
    )
    top_features = [
        {"feature": feature, "importance": int(importance)}
        for feature, importance in sorted_pairs[:10]
    ]

    last_close = float(feature_df["Close"].iloc[-1])
    last_date = format_last_date(feature_df.index[-1])

    return {
        "model": "LightGBM Direction Classifier",
        "horizon_days": predict_len,
        "last_date": last_date,
        "last_close": round(last_close, 4),
        "probability_up": round(prob_up, 4),
        "probability_down": round(prob_down, 4),
        "predicted_direction": direction,
        "signal": signal,
        "target_return_threshold": DIRECTION_CONFIG.get("target_return_threshold", 0.0),
        "buy_probability_threshold": buy_threshold,
        "sell_probability_threshold": sell_threshold,
        "metrics": metrics,
        "top_features": top_features,
    }

def  main():
    ticker = "AAPL"
    stock_ticker = ticker.upper()

    try:
        stock, _ = get_stock_data(stock_ticker)
        if stock is None:
            print("stock is None")
            exit()
        predict_len_value = 10
        df = stock.history(period="5y")
        direction_response = predict_direction_with_lgbm(
            df, 
            predict_len=predict_len_value,
        )
    except Exception as exc:
        print(exc)

if __name__ == "__main__":
    main()