import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import LSTM_CONFIG, RIDGE_CONFIG


class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class StockLongPredictor(nn.Module):
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


def predict_with_lstm(df, predict_len=10):
    settings = {**LSTM_CONFIG, "predict_len": predict_len}

    try:
        if df.empty or len(df) <= settings["past_len"] + settings["predict_len"]:
            return _fallback_prediction(df, settings["predict_len"])

        df_local = df.copy()
        scaler = MinMaxScaler(feature_range=settings["feature_range"])
        train_cutoff = int(len(df_local) * settings["train_split"])

        scaler.fit_transform(df_local.iloc[:train_cutoff]["Close"].values.reshape(-1, 1))
        df_local["Close_s"] = scaler.transform(df_local["Close"].values.reshape(-1, 1))
        df_local["Date"] = df_local.index
        df_local["weekdays"] = df_local["Date"].dt.dayofweek

        weekday_dummies = pd.get_dummies(df_local["weekdays"], prefix="wd")
        df_local = pd.concat([df_local, weekday_dummies], axis=1)

        feature_cols = ["Close_s", "wd_0", "wd_1", "wd_2", "wd_3", "wd_4"]
        for column in feature_cols[1:]:
            if column not in df_local:
                df_local[column] = 0

        x_seq = []
        y_seq = []
        past_len = settings["past_len"]

        for i in range(len(df_local) - past_len - settings["predict_len"]):
            x_window = df_local.iloc[i : i + past_len][feature_cols].values.astype(np.float32)
            x_seq.append(x_window)
            y_target = df_local["Close_s"].iloc[
                i + past_len : i + past_len + settings["predict_len"]
            ].values.astype(np.float32)
            y_seq.append(y_target)

        if not x_seq:
            return _fallback_prediction(df_local, settings["predict_len"])

        x_data = np.stack(x_seq)
        y_data = np.stack(y_seq)
        train_len = int(len(x_data) * settings["train_split"])

        if train_len <= 0 or train_len >= len(x_data):
            return _fallback_prediction(df_local, settings["predict_len"])

        x_train_np, x_test_np = x_data[:train_len], x_data[train_len:]
        y_train_np, y_test_np = y_data[:train_len], y_data[train_len:]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_train = torch.from_numpy(x_train_np).float().to(device)
        y_train = torch.from_numpy(y_train_np).float().to(device)
        x_test = torch.from_numpy(x_test_np).float().to(device)
        y_test = torch.from_numpy(y_test_np).float().to(device)

        train_dataloader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=settings["batch_size"],
            shuffle=True,
        )
        test_dataloader = DataLoader(
            TensorDataset(x_test, y_test),
            batch_size=settings["batch_size"],
            shuffle=True,
        )

        criterion = nn.MSELoss()
        model = StockLongPredictor(
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
        all_predictions_unscaled = []
        all_targets_unscaled = []

        with torch.no_grad():
            for x_batch, y_batch in test_dataloader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                total_test_loss += loss.item()

                batch_predictions_scaled = outputs.cpu().numpy()
                batch_targets_scaled = y_batch.cpu().numpy()

                for i in range(batch_predictions_scaled.shape[0]):
                    pred_unscaled = scaler.inverse_transform(
                        batch_predictions_scaled[i].reshape(-1, 1)
                    ).flatten()
                    target_unscaled = scaler.inverse_transform(
                        batch_targets_scaled[i].reshape(-1, 1)
                    ).flatten()
                    all_predictions_unscaled.extend(pred_unscaled)
                    all_targets_unscaled.extend(target_unscaled)

        if len(test_dataloader) > 0:
            avg_loss = total_test_loss / len(test_dataloader)
            print(f"Average Test Loss (Scaled MSE): {avg_loss:.6f}")

        if all_targets_unscaled:
            try:
                targets_np = np.array(all_targets_unscaled)
                predictions_np = np.array(all_predictions_unscaled)
                epsilon = 1e-8

                print("\n--- Test Data Evaluation Metrics (Unscaled) ---")
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
            input_tensor = torch.from_numpy(last_sequence_np).unsqueeze(0).to(device)

            prediction_scaled = model(input_tensor)
            prediction_np = prediction_scaled.cpu().numpy().reshape(-1, 1)
            prediction_unscaled = scaler.inverse_transform(prediction_np)

            last_date = df_local.index[-1]
            prediction_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=settings["predict_len"],
                freq="B",
            )
            prediction_series = pd.Series(
                prediction_unscaled.flatten(),
                index=prediction_dates,
                name="Predicted Close",
            )
            print("\n--- Predicted Stock Prices ---")
            print(prediction_series)
            return prediction_series
    except Exception as error:
        print(f"LSTM Prediction Error: {error}")
        return _fallback_prediction(df, settings["predict_len"])


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
