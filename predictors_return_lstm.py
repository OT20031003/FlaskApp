import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import LSTM_CONFIG
from predictor_utils import fallback_prediction


class ReturnLSTMPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=3, layer_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden_state, _ = self.lstm(x)
        hidden_state = hidden_state[:, -1, :]
        return self.fc(hidden_state)


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
            return fallback_prediction(df, settings["predict_len"])

        df_local, feature_cols, continuous_feature_cols = _build_return_dataset(df)
        if df_local.empty or len(df_local) <= settings["past_len"] + settings["predict_len"]:
            return fallback_prediction(df, settings["predict_len"])

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
            return fallback_prediction(df_local, settings["predict_len"])

        x_data = np.stack(x_seq)
        y_data = np.stack(y_seq)
        actual_price_data = np.stack(actual_price_seq)
        base_prices_np = np.array(base_prices, dtype=np.float32)
        train_len = int(len(x_data) * settings["train_split"])

        if train_len <= 0 or train_len >= len(x_data):
            return fallback_prediction(df_local, settings["predict_len"])

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
        return fallback_prediction(df, settings["predict_len"])


predict_with_lstm = predict_with_return_lstm
