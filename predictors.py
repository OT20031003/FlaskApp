import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Optional

from config import DIRECTION_CONFIG, LSTM_CONFIG, RIDGE_CONFIG


class ReturnLSTMPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=3, layer_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden_state, _ = self.lstm(x)
        hidden_state = hidden_state[:, -1, :]
        return self.fc(hidden_state)


def _fallback_prediction(df, predict_len):
    last_price = float(df["Close"].iloc[-1]) if not df.empty else 0.0
    last_date = df.index[-1] if not df.empty else pd.Timestamp.today().normalize()
    prediction_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=predict_len,
        freq="B",
    )
    return pd.Series([last_price] * predict_len, index=prediction_dates)


def _build_return_dataset(df):
    df_local = df.copy().sort_index()
    df_local["log_return"] = np.log(df_local["Close"] / df_local["Close"].shift(1))
    df_local["volatility_5"] = df_local["log_return"].rolling(5).std()
    df_local["volatility_20"] = df_local["log_return"].rolling(20).std()
    df_local["momentum_5"] = df_local["Close"].pct_change(5)
    df_local["momentum_20"] = df_local["Close"].pct_change(20)
    df_local["ma_gap_20"] = df_local["Close"] / df_local["Close"].rolling(20).mean() - 1
    continuous_feature_cols = [
        "log_return",
        "volatility_5",
        "volatility_20",
        "momentum_5",
        "momentum_20",
        "ma_gap_20",
    ]
    df_local = df_local.replace([np.inf, -np.inf], np.nan).dropna(
        subset=continuous_feature_cols
    ).copy()

    if df_local.empty:
        return df_local, [], []

    weekday_dummies = pd.get_dummies(
        pd.Series(df_local.index.dayofweek, index=df_local.index),
        prefix="wd",
    )
    df_local = pd.concat([df_local, weekday_dummies], axis=1)

    weekday_feature_cols = ["wd_0", "wd_1", "wd_2", "wd_3", "wd_4"]
    feature_cols = continuous_feature_cols + weekday_feature_cols
    for column in weekday_feature_cols:
        if column not in df_local:
            df_local[column] = 0.0

    return df_local, feature_cols, continuous_feature_cols


def _scale_return_features(
    x_train_np,
    x_test_np,
    y_train_np,
    y_test_np,
    continuous_feature_count,
):
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    x_train_continuous = x_train_np[:, :, :continuous_feature_count].reshape(
        -1,
        continuous_feature_count,
    )
    feature_scaler.fit(x_train_continuous)
    target_scaler.fit(y_train_np.reshape(-1, 1))

    x_train_scaled = x_train_np.copy()
    x_test_scaled = x_test_np.copy()
    y_train_scaled = y_train_np.copy()
    y_test_scaled = y_test_np.copy()

    x_train_scaled[:, :, :continuous_feature_count] = feature_scaler.transform(
        x_train_scaled[:, :, :continuous_feature_count].reshape(-1, continuous_feature_count)
    ).reshape(x_train_scaled[:, :, :continuous_feature_count].shape)
    x_test_scaled[:, :, :continuous_feature_count] = feature_scaler.transform(
        x_test_scaled[:, :, :continuous_feature_count].reshape(-1, continuous_feature_count)
    ).reshape(x_test_scaled[:, :, :continuous_feature_count].shape)
    y_train_scaled = target_scaler.transform(y_train_scaled.reshape(-1, 1)).reshape(
        y_train_scaled.shape
    )
    y_test_scaled = target_scaler.transform(y_test_scaled.reshape(-1, 1)).reshape(
        y_test_scaled.shape
    )

    return x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler


def _log_returns_to_prices(last_close, log_returns):
    last_close_value = float(last_close)
    return last_close_value * np.exp(np.cumsum(log_returns))


def predict_with_return_lstm(df, predict_len=10):
    settings = {**LSTM_CONFIG, "predict_len": predict_len}

    try:
        if df.empty:
            return _fallback_prediction(df, settings["predict_len"])

        df_local, feature_cols, continuous_feature_cols = _build_return_dataset(df)
        if df_local.empty or len(df_local) <= settings["past_len"] + settings["predict_len"]:
            return _fallback_prediction(df, settings["predict_len"])

        x_seq = []
        y_seq = []
        actual_price_seq = []
        base_prices = []
        past_len = settings["past_len"]

        for i in range(len(df_local) - past_len - settings["predict_len"] + 1):
            x_window = df_local.iloc[i : i + past_len][feature_cols].values.astype(np.float32)
            x_seq.append(x_window)
            y_target = df_local["log_return"].iloc[
                i + past_len : i + past_len + settings["predict_len"]
            ].values.astype(np.float32)
            y_seq.append(y_target)
            actual_price_seq.append(
                df_local["Close"].iloc[
                    i + past_len : i + past_len + settings["predict_len"]
                ].values.astype(np.float32)
            )
            base_prices.append(float(df_local["Close"].iloc[i + past_len - 1]))

        if not x_seq:
            return _fallback_prediction(df_local, settings["predict_len"])

        x_data = np.stack(x_seq)
        y_data = np.stack(y_seq)
        actual_price_data = np.stack(actual_price_seq)
        base_prices_np = np.array(base_prices, dtype=np.float32)
        train_len = int(len(x_data) * settings["train_split"])

        if train_len <= 0 or train_len >= len(x_data):
            return _fallback_prediction(df_local, settings["predict_len"])

        x_train_np, x_test_np = x_data[:train_len], x_data[train_len:]
        y_train_np, y_test_np = y_data[:train_len], y_data[train_len:]
        actual_test_prices_np = actual_price_data[train_len:]
        base_test_prices_np = base_prices_np[train_len:]

        (
            x_train_scaled_np,
            x_test_scaled_np,
            y_train_scaled_np,
            y_test_scaled_np,
            feature_scaler,
            target_scaler,
        ) = _scale_return_features(
            x_train_np,
            x_test_np,
            y_train_np,
            y_test_np,
            len(continuous_feature_cols),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_train = torch.from_numpy(x_train_scaled_np).float().to(device)
        y_train = torch.from_numpy(y_train_scaled_np).float().to(device)
        x_test = torch.from_numpy(x_test_scaled_np).float().to(device)
        y_test = torch.from_numpy(y_test_scaled_np).float().to(device)

        train_dataloader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=settings["batch_size"],
            shuffle=True,
        )
        test_dataloader = DataLoader(
            TensorDataset(x_test, y_test),
            batch_size=settings["batch_size"],
            shuffle=False,
        )

        criterion = nn.MSELoss()
        model = ReturnLSTMPredictor(
            input_size=len(feature_cols),
            hidden_size=settings["hidden_size"],
            output_size=settings["predict_len"],
            layer_size=settings["num_layers"],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])

        for epoch in range(settings["epochs"]):
            model.train()
            total_loss = 0.0
            for x_batch, y_batch in train_dataloader:
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch [{epoch + 1}/{settings['epochs']}], "
                    f"Train Loss: {total_loss / len(train_dataloader):.6f}"
                )

        model.eval()
        total_test_loss = 0.0
        all_predicted_prices = []
        all_actual_prices = []
        test_sample_offset = 0

        with torch.no_grad():
            for x_batch, y_batch in test_dataloader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                total_test_loss += loss.item()

                batch_predictions_scaled = outputs.cpu().numpy()
                batch_predictions = target_scaler.inverse_transform(
                    batch_predictions_scaled.reshape(-1, 1)
                ).reshape(batch_predictions_scaled.shape)

                batch_size = batch_predictions.shape[0]
                for i in range(batch_size):
                    base_price = base_test_prices_np[test_sample_offset + i]
                    predicted_prices = _log_returns_to_prices(
                        base_price,
                        batch_predictions[i],
                    )
                    all_predicted_prices.extend(predicted_prices.tolist())
                    all_actual_prices.extend(
                        actual_test_prices_np[test_sample_offset + i].tolist()
                    )

                test_sample_offset += batch_size

        if len(test_dataloader) > 0:
            avg_loss = total_test_loss / len(test_dataloader)
            print(f"Average Test Loss (Scaled Log Return MSE): {avg_loss:.6f}")

        if all_actual_prices:
            try:
                targets_np = np.array(all_actual_prices)
                predictions_np = np.array(all_predicted_prices)
                epsilon = 1e-8

                print("\n--- Return LSTM Test Data Evaluation Metrics (Reconstructed Prices) ---")
                mae = mean_absolute_error(targets_np, predictions_np)
                print(f"MAE (Mean Absolute Error): ${mae:.4f}")
                print(
                    f"  (Interpretation: On average, the prediction was off by ${mae:.4f} "
                    "across the test data.)"
                )

                rmse = np.sqrt(mean_squared_error(targets_np, predictions_np))
                print(f"RMSE (Root Mean Squared Error): ${rmse:.4f}")
                print(
                    "  (Interpretation: Similar to MAE in dollar terms, but gives more "
                    "weight to large errors.)"
                )

                mape = np.mean(
                    np.abs((targets_np - predictions_np) / (targets_np + epsilon))
                ) * 100
                print(f"MAPE (Mean Absolute Percentage Error): {mape:.4f}%")
                print(
                    f"  (Interpretation: On average, the prediction was off by {mape:.4f}% "
                    "across the test data.)"
                )
                print("-------------------------------------------------")
            except Exception as error:
                print(f"Error during calculation of evaluation metrics: {error}")

        with torch.no_grad():
            last_sequence_df = df_local.iloc[-past_len:][feature_cols]
            last_sequence_np = last_sequence_df.values.astype(np.float32)
            last_sequence_np[:, : len(continuous_feature_cols)] = feature_scaler.transform(
                last_sequence_np[:, : len(continuous_feature_cols)]
            )
            input_tensor = torch.from_numpy(last_sequence_np).unsqueeze(0).float().to(device)

            prediction_scaled = model(input_tensor).cpu().numpy().reshape(-1, 1)
            predicted_log_returns = target_scaler.inverse_transform(prediction_scaled).flatten()
            predicted_prices = _log_returns_to_prices(
                df["Close"].iloc[-1],
                predicted_log_returns,
            )

            last_date = df.index[-1]
            prediction_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=settings["predict_len"],
                freq="B",
            )
            prediction_series = pd.Series(
                predicted_prices,
                index=prediction_dates,
                name="Predicted Close",
            )
            print("\n--- Predicted Stock Prices (Return LSTM) ---")
            print(prediction_series)
            return prediction_series
    except Exception as error:
        print(f"Return LSTM Prediction Error: {error}")
        return _fallback_prediction(df, settings["predict_len"])


predict_with_lstm = predict_with_return_lstm


def _format_last_date(index_value: Any) -> Optional[str]:
    if index_value is None:
        return None
    if hasattr(index_value, "strftime"):
        return index_value.strftime("%Y-%m-%d")
    return str(index_value)


def _direction_fallback(df: pd.DataFrame, predict_len: int, reason: str) -> dict[str, Any]:
    last_close = None
    last_date = None
    if not df.empty and "Close" in df.columns:
        close_series = df["Close"].dropna()
        if not close_series.empty:
            last_close = float(close_series.iloc[-1])
            last_date = _format_last_date(close_series.index[-1])

    return {
        "model": "LightGBM Direction Classifier",
        "horizon_days": predict_len,
        "last_date": last_date,
        "last_close": last_close,
        "probability_up": 0.5,
        "probability_down": 0.5,
        "predicted_direction": "NEUTRAL",
        "signal": "HOLD",
        "target_return_threshold": DIRECTION_CONFIG.get("target_return_threshold", 0.0),
        "buy_probability_threshold": DIRECTION_CONFIG.get("buy_probability_threshold", 0.60),
        "sell_probability_threshold": DIRECTION_CONFIG.get("sell_probability_threshold", 0.40),
        "metrics": {},
        "top_features": [],
        "reason": reason,
    }


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
        return _direction_fallback(df, predict_len, "Not enough data for direction classification")
    if "Close" not in df.columns:
        return _direction_fallback(df, predict_len, "Close column is required for direction classification")

    min_samples = DIRECTION_CONFIG.get("min_samples", 120)
    if len(df) < min_samples:
        return _direction_fallback(df, predict_len, "Not enough data for direction classification")

    model_factory = _build_direction_model()
    feature_df, feature_cols = _build_direction_features(df, predict_len)
    if not feature_cols or feature_df.empty:
        return _direction_fallback(
            df,
            predict_len,
            "Not enough valid feature rows for direction classification",
        )

    latest_features = feature_df[feature_cols].iloc[[-1]]
    if latest_features.isna().any().any():
        return _direction_fallback(
            df,
            predict_len,
            "Latest feature row contains NaN values",
        )

    # Use only rows whose target is known. The final `predict_len` rows are excluded to avoid leakage.
    training_df = feature_df[feature_df["target"].notna()].copy()
    if len(training_df) < min_samples:
        return _direction_fallback(df, predict_len, "Not enough data for direction classification")

    y_all = training_df["target"].astype(int)
    if y_all.nunique() < 2:
        return _direction_fallback(
            df,
            predict_len,
            "Training data contains only one direction class",
        )

    train_split = DIRECTION_CONFIG.get("train_split", 0.8)
    train_len = int(len(training_df) * train_split)
    if train_len <= 0 or train_len >= len(training_df):
        return _direction_fallback(
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
        return _direction_fallback(
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
    last_date = _format_last_date(feature_df.index[-1])

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


# Smoke test
# result = predict_direction_with_lgbm(df, predict_len=10)
# print(result["probability_up"], result["predicted_direction"], result["signal"])


def predict_with_ridge(df, predict_len=10):
    settings = {**RIDGE_CONFIG, "predict_len": predict_len}

    try:
        df_local = df.copy().sort_index()
        if settings["use_days"]:
            df_local = df_local.tail(settings["use_days"])

        if len(df_local) < settings["n_lags"] + 1:
            return _fallback_prediction(df_local, settings["predict_len"])

        data = pd.DataFrame(index=df_local.index)
        data["Close"] = df_local["Close"]

        for lag in range(1, settings["n_lags"] + 1):
            data[f"lag_{lag}"] = data["Close"].shift(lag)

        for window in settings["moving_windows"]:
            data[f"ma_{window}"] = data["Close"].rolling(window=window).mean()
            data[f"std_{window}"] = data["Close"].rolling(window=window).std()

        dow = data.index.dayofweek
        data["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        data["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        data = data.bfill().dropna()

        train_len = int(len(data) * settings["train_split"])
        data_train = data.iloc[:train_len]
        data_test = data.iloc[train_len:]

        if data_train.empty:
            return _fallback_prediction(df_local, settings["predict_len"])
        if data_test.empty:
            print("Not enough data to perform train/test split for Ridge.")
            data_train = data

        feature_cols = [column for column in data.columns if column != "Close"]

        x_train = data_train[feature_cols].values
        y_train = data_train["Close"].values

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        model = Ridge(alpha=settings["alpha"])
        model.fit(x_train_scaled, y_train)

        if not data_test.empty:
            x_test = data_test[feature_cols].values
            y_test_actual = data_test["Close"].values
            x_test_scaled = scaler.transform(x_test)
            y_test_pred = model.predict(x_test_scaled)

            print("\n--- Ridge Test Data Evaluation Metrics ---")
            try:
                epsilon = 1e-8
                mae = mean_absolute_error(y_test_actual, y_test_pred)
                print(f"MAE (Mean Absolute Error): ${mae:.4f}")
                print(f"  (Interpretation: On average, the prediction was off by ${mae:.4f}.)")

                rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
                print(f"RMSE (Root Mean Squared Error): ${rmse:.4f}")
                print("  (Interpretation: Similar to MAE, but gives more weight to large errors.)")

                mape = np.mean(np.abs((y_test_actual - y_test_pred) / (y_test_actual + epsilon))) * 100
                print(f"MAPE (Mean Absolute Percentage Error): {mape:.4f}%")
                print(f"  (Interpretation: On average, the prediction was off by {mape:.4f}%.)")
                print("-------------------------------------------------")
            except Exception as error:
                print(f"Error during Ridge evaluation: {error}")

        x_full = data[feature_cols].values
        y_full = data["Close"].values
        scaler_full = StandardScaler()
        x_full_scaled = scaler_full.fit_transform(x_full)
        model_full = Ridge(alpha=settings["alpha"])
        model_full.fit(x_full_scaled, y_full)

        predictions = []
        temp_df = df_local.copy()

        for _ in range(settings["predict_len"]):
            last_day_features = {}
            closes = temp_df["Close"].tolist()

            for lag in range(1, settings["n_lags"] + 1):
                last_day_features[f"lag_{lag}"] = closes[-lag]

            for window in settings["moving_windows"]:
                ma = temp_df["Close"].rolling(window=window).mean().iloc[-1]
                std = temp_df["Close"].rolling(window=window).std().iloc[-1]
                last_day_features[f"ma_{window}"] = ma if not pd.isna(ma) else closes[-1]
                last_day_features[f"std_{window}"] = std if not pd.isna(std) else 0.0

            next_day = temp_df.index[-1] + pd.Timedelta(days=1)
            dow_next = next_day.dayofweek
            last_day_features["dow_sin"] = np.sin(2 * np.pi * dow_next / 7)
            last_day_features["dow_cos"] = np.cos(2 * np.pi * dow_next / 7)

            x_next = np.array([last_day_features[column] for column in feature_cols]).reshape(1, -1)
            x_next_scaled = scaler_full.transform(x_next)

            pred = model_full.predict(x_next_scaled)[0]
            pred = float(pred) if pred > 0 else float(temp_df["Close"].iloc[-1])
            predictions.append(pred)

            new_row = pd.DataFrame({"Close": [pred]}, index=[next_day])
            temp_df = pd.concat([temp_df, new_row])

        prediction_dates = pd.date_range(
            start=df_local.index[-1] + pd.Timedelta(days=1),
            periods=settings["predict_len"],
            freq="B",
        )
        return pd.Series(predictions, index=prediction_dates)
    except Exception as error:
        print(f"Ridge Prediction Error: {error}")
        return _fallback_prediction(df, settings["predict_len"])
