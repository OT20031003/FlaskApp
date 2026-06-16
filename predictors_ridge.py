import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from config import RIDGE_CONFIG
from predictor_utils import fallback_prediction


def predict_with_ridge(df, predict_len=10):
    settings = {**RIDGE_CONFIG, "predict_len": predict_len}

    try:
        df_local = df.copy().sort_index()
        if settings["use_days"]:
            df_local = df_local.tail(settings["use_days"])

        if len(df_local) < settings["n_lags"] + 1:
            return fallback_prediction(df_local, settings["predict_len"])

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
            return fallback_prediction(df_local, settings["predict_len"])
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
        return fallback_prediction(df, settings["predict_len"])
