import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, render_template, request, redirect, url_for, flash
import yfinance as yf
import pandas as pd
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

# (★追加) Gemini関連のライブラリ
import google.generativeai as genai
import os
import logging

# --- PyTorch LSTM Model Definition ---
# The StockPredictor class you provided.
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM returns all hidden states and the final hidden/cell state
        # We only need the final hidden state for our prediction
        _, (h_n, _) = self.lstm(x)
        # Pass the hidden state of the last layer to the fully connected layer
        output = self.fc(h_n[-1])
        return output
class StockLongPredictor(nn.Module):
  def __init__(self, input_size=6, hidden_size=64, output_size=3, layer_size=2):
    super().__init__()

    self.lstm = nn.LSTM(input_size, hidden_size, layer_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)
  def forward(self, x):
    h, c = self.lstm(x)
    h= h[:,-1,:]
    x = self.fc(h)
    return x
# --- Flask App Initialization ---
app = Flask(__name__)
# IMPORTANT: Change this secret key in a real application
app.config['SECRET_KEY'] = 'a_very_secret_and_complex_key_12345'

# --- (★追加) Gemini APIのセットアップ ---
try:
    # 環境変数からAPIキーを取得
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        logging.warning("GEMINI_API_KEY が環境変数に設定されていません。")
        # ここにAPIキーを直接書くのは非推奨です
        # GEMINI_API_KEY = "YOUR_API_KEY" 
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    # 安全設定
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    # 高速なモデル (gemini-1.5-flash-latest) を使用
    gemini_model = genai.GenerativeModel(
        model_name="models/gemini-pro-latest",
        safety_settings=safety_settings
    )
    logging.info("Geminiモデルの初期化完了。")

except Exception as e:
    logging.error(f"Geminiの初期化に失敗しました: {e}")
    gemini_model = None

# --- Database Functions ---
def get_db_connection():
    """Gets a connection to the SQLite database."""
    conn = sqlite3.connect('portfolio.db')
    conn.row_factory = sqlite3.Row  # Allows accessing columns by name
    return conn

# --- Stock Data Functions ---
def get_stock_data(ticker):
    """
    Fetches the latest stock price and yfinance object with error handling.
    """
    try:
        stock = yf.Ticker(ticker)
        # Use a short period to get the most recent closing price
        hist = stock.history(period="5d")
        if hist.empty:
            return None, None
        # Safely access the last available closing price
        return stock, hist['Close'].iloc[-1]
    except Exception:
        return None, None

# --- (★追加) Gemini 銘柄説明取得関数 ---
def get_gemini_description(ticker, stock_info):
    """Gemini APIを使用して銘柄の簡潔な説明を取得する"""
    if not gemini_model:
        logging.warning("Geminiモデルが利用できません。説明取得をスキップします。")
        return "説明の取得に失敗しました（Geminiモデル未初期化）。"
        
    try:
        # yfinanceから取得した情報を活用
        company_name = stock_info.get('longName', stock_info.get('shortName', ticker))
        industry = stock_info.get('industry', '不明')
        sector = stock_info.get('sector', '不明')

        # Geminiへのプロンプト
        prompt = f"""
        以下の企業について、投資家向けに日本語で簡潔に説明してください。
        企業名（ティッカー）: {company_name} ({ticker})
        業種: {industry} ({sector})
        
        説明には以下の点を含めてください：
        1. 主な事業内容
        2. 企業の強みや市場での位置づけ
        
        また以下の制約を必ず守ること
        1. 銘柄名から始めて、次の行から銘柄を説明すること
        2. 銘柄に関連する最近のニュース(トランプ関税、ドル安などの影響)も述べること
        説明文は全体で200文字程度の簡潔な文章にまとめてください。
        """

        response = gemini_model.generate_content(prompt)
        
        if response.parts:
            description = response.text
        else:
            description = f"{company_name} ({ticker}) の説明は現在取得できません。"
            logging.warning(f"Geminiからのレスポンスが空です: {response.prompt_feedback}")

        return description

    except Exception as e:
        logging.error(f"Gemini API呼び出しエラー ({ticker}): {e}")
        return f"{company_name} ({ticker}) の説明取得中にエラーが発生しました。"


# --- Prediction Models ---
def predict_with_lstm(df, sequence_length=40, epochs=100, predict_len=10):
    try:
        df.reset_index()
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_len = int(len(df) * 0.7)
        ds = scaler.fit_transform(df.iloc[0:train_len ]["Close"].values.reshape(-1, 1))
        df["Close_s"] = scaler.transform(df["Close"].values.reshape(-1, 1))
        df["Date"] = df.index
        df["weekdays"] = df["Date"].dt.dayofweek
        wd_dummies = pd.get_dummies(df["weekdays"], prefix="wd")
        df = pd.concat([df, wd_dummies], axis=1)
        feature_cols = ['Close_s',  'wd_0', 'wd_1', 'wd_2', 'wd_3', 'wd_4']
        train_len = int(len(df) * 0.7)
        x_seq = []
        y_seq = []
        past_len = 50
        input_size = len(feature_cols)
        batch_size = 32
        for i in range(len(df) - past_len-predict_len):
            x_window = df.iloc[i:i+past_len][feature_cols].values.astype(np.float32)
            x_seq.append(x_window)
            y_target = df["Close_s"].iloc[i + past_len:i + past_len + predict_len].values.astype(np.float32)
            #y_target = df["Close_s"].iloc[i + past_len:i + past_len + predict_len
            y_seq.append(y_target)
        print(y_seq[0])

        X = np.stack(x_seq)
        Y = np.stack(y_seq)
        print(X.shape, Y.shape)
        X_train_np, X_test_np = X[:train_len], X[train_len:]
        Y_train_np, Y_test_np = Y[:train_len], Y[train_len:]
        print(X_train_np.shape, X_test_np.shape)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train = torch.from_numpy(X_train_np).float().to(device)
        Y_train = torch.from_numpy(Y_train_np).float().to(device)
        X_test = torch.from_numpy(X_test_np).float().to(device)
        Y_test = torch.from_numpy(Y_test_np).float().to(device)
        train_dataloader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=True)


        criterion = nn.MSELoss()
        model = StockLongPredictor(input_size=input_size, hidden_size=128, output_size=predict_len).to(device)
        optimizer =torch.optim.Adam(model.parameters(), lr=1e-4)
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for X_batch, Y_batch in train_dataloader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch+1) % 5 == 0 or epoch == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss / len(train_dataloader):.6f}')
        model.eval()
        total_test_loss = 0
        # (★追加) 評価指標計算用のリスト
        all_predictions_unscaled = []
        all_targets_unscaled = []

        with torch.no_grad():
            for X_batch, Y_batch in test_dataloader:
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                total_test_loss += loss.item()

                # (★変更) バッチの予測結果と実測値を逆正規化
                batch_predictions_scaled = outputs.cpu().numpy()
                batch_targets_scaled = Y_batch.cpu().numpy()

                # scaler.inverse_transform は (n_samples, n_features=1) を期待する
                for i in range(batch_predictions_scaled.shape[0]): # バッチ内の各サンプル
                    # 予測値: (predict_len,) -> (predict_len, 1) -> 逆変換 -> (predict_len, 1) -> flatten -> (predict_len,)
                    pred_unscaled = scaler.inverse_transform(batch_predictions_scaled[i].reshape(-1, 1)).flatten()
                    # 実測値: (predict_len,) -> (predict_len, 1) -> 逆変換 -> (predict_len, 1) -> flatten -> (predict_len,)
                    target_unscaled = scaler.inverse_transform(batch_targets_scaled[i].reshape(-1, 1)).flatten()
                    
                    all_predictions_unscaled.extend(pred_unscaled)
                    all_targets_unscaled.extend(target_unscaled)

        avg_loss = total_test_loss / len(test_dataloader)
        print(f'Average Test Loss (Scaled MSE): {avg_loss:.6f}') # 元の損失
        
        # 最新株価を予測
        print("\nPredicting the next", predict_len, "days...")
        if all_targets_unscaled:
            try:
                targets_np = np.array(all_targets_unscaled)
                predictions_np = np.array(all_predictions_unscaled)
                
                # ゼロ割を避けるための微小値（MAPE計算用）
                epsilon = 1e-8 

                print(f'\n--- Test Data Evaluation Metrics (Unscaled) ---')
                
                # MAE (Mean Absolute Error) / 平均絶対誤差 (ドル単位)
                mae = mean_absolute_error(targets_np, predictions_np)
                print(f'MAE (Mean Absolute Error): ${mae:.4f}')
                # (↓ 英語に変更)
                print(f'  (Interpretation: On average, the prediction was off by ${mae:.4f} across the test data.)')

                # RMSE (Root Mean Squared Error) / 二乗平均平方根誤差 (ドル単位)
                rmse = np.sqrt(mean_squared_error(targets_np, predictions_np))
                print(f'RMSE (Root Mean Squared Error): ${rmse:.4f}')
                # (↓ 英語に変更)
                print(f'  (Interpretation: Similar to MAE in dollar terms, but gives more weight to large errors.)')
                
                # MAPE (Mean Absolute Percentage Error) / 平均絶対パーセント誤差 (%)
                # ゼロ割を避けるため、(targets_np + epsilon) で割ります。
                mape = np.mean(np.abs((targets_np - predictions_np) / (targets_np + epsilon))) * 100
                print(f'MAPE (Mean Absolute Percentage Error): {mape:.4f}%')
                # (↓ 英語に変更)
                print(f'  (Interpretation: On average, the prediction was off by {mape:.4f}% across the test data.)')
                print(f'-------------------------------------------------')

            except Exception as e:
                # (↓ 英語に変更)
                print(f"Error during calculation of evaluation metrics: {e}")
        with torch.no_grad():
            last_sequence_df = df.iloc[-past_len:][feature_cols]
            last_sequence_np = last_sequence_df.values.astype(np.float32)
            input_tensor = torch.from_numpy(last_sequence_np).unsqueeze(0).to(device)

            prediction_scaled = model(input_tensor)

            prediction_np = prediction_scaled.cpu().numpy().reshape(-1, 1)
            prediction_unscaled = scaler.inverse_transform(prediction_np)

            last_date = df.index[-1]
            prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=predict_len, freq='B')

            prediction_series = pd.Series(prediction_unscaled.flatten(), index=prediction_dates, name="Predicted Close")
            print("\n--- Predicted Stock Prices ---")
            print(prediction_series)
    except Exception as e:
        print(f"LSTM Prediction Error: {e}")
        # Fallback to returning a simple series with the last price repeated
        last_price = float(df['Close'].iloc[-1]) if not df.empty else 0
        prediction_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=predict_len, freq='B')
        return pd.Series([last_price] * predict_len, index=prediction_dates)


    return prediction_series


def predict_with_ridge(df, n_lags=5, use_days=180, predict_len=10):
    """
    Predicts the next N days' closing price using Ridge Regression with feature engineering.
    (★ MODIFIED to include train/test split for evaluation)
    """
    try:
        df_local = df.copy().sort_index()
        if len(df_local) < n_lags + 1:
            # Not enough data, return a simple series with the last price repeated
            last_price = float(df_local['Close'].iloc[-1]) if not df_local.empty else 0
            prediction_dates = pd.date_range(start=df_local.index[-1] + pd.Timedelta(days=1), periods=predict_len, freq='B')
            return pd.Series([last_price] * predict_len, index=prediction_dates)

        # Feature Engineering (Same as original)
        data = pd.DataFrame(index=df_local.index)
        data['Close'] = df_local['Close']
        for lag in range(1, n_lags + 1):
            data[f'lag_{lag}'] = data['Close'].shift(lag)
        for w in (5, 10, 20):
            data[f'ma_{w}'] = data['Close'].rolling(window=w).mean()
            data[f'std_{w}'] = data['Close'].rolling(window=w).std()
        dow = data.index.dayofweek
        data['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        data['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        data = data.fillna(method='bfill').dropna()

        # (★ NEW) Train/Test Split for Evaluation (70/30 split)
        train_len = int(len(data) * 0.7)
        data_train = data.iloc[:train_len]
        data_test = data.iloc[train_len:]

        if data_train.empty or data_test.empty:
            print("Not enough data to perform train/test split for Ridge.")
            # Fallback to original prediction logic without evaluation
            # (Note: This skips evaluation)
            data_train = data # Use all data if split failed
        
        feature_cols = [c for c in data.columns if c != 'Close']
        
        # --- 1. Model Training (on 70% train data) ---
        X_train = data_train[feature_cols].values
        y_train = data_train['Close'].values
        
        scaler = StandardScaler()
        Xs_train = scaler.fit_transform(X_train) # Fit scaler ONLY on train data
        model = Ridge(alpha=1.0)
        model.fit(Xs_train, y_train)

        # --- (★ NEW) 2. Model Evaluation (on 30% test data) ---
        if not data_test.empty:
            X_test = data_test[feature_cols].values
            y_test_actual = data_test['Close'].values # Ground truth
            
            Xs_test = scaler.transform(X_test) # Use SAME scaler from training
            y_test_pred = model.predict(Xs_test)

            print(f'\n--- Ridge Test Data Evaluation Metrics ---')
            try:
                epsilon = 1e-8
                mae = mean_absolute_error(y_test_actual, y_test_pred)
                print(f'MAE (Mean Absolute Error): ${mae:.4f}')
                print(f'  (Interpretation: On average, the prediction was off by ${mae:.4f}.)')

                rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
                print(f'RMSE (Root Mean Squared Error): ${rmse:.4f}')
                print(f'  (Interpretation: Similar to MAE, but gives more weight to large errors.)')

                mape = np.mean(np.abs((y_test_actual - y_test_pred) / (y_test_actual + epsilon))) * 100
                print(f'MAPE (Mean Absolute Percentage Error): {mape:.4f}%')
                print(f'  (Interpretation: On average, the prediction was off by {mape:.4f}%.)')
                print(f'-------------------------------------------------')
            except Exception as e:
                print(f"Error during Ridge evaluation: {e}")

        # --- 3. Retrain Model on ALL Data for Future Prediction ---
        # (This uses the full 'data' DataFrame)
        X_full = data[feature_cols].values
        y_full = data['Close'].values
        
        scaler_full = StandardScaler()
        Xs_full = scaler_full.fit_transform(X_full)
        model_full = Ridge(alpha=1.0)
        model_full.fit(Xs_full, y_full)

        # --- 4. Iterative Prediction (Original logic, but using 'model_full' and 'scaler_full') ---
        predictions = []
        temp_df = df_local.copy() # Start with the full history

        for _ in range(predict_len):
            # Prepare features for the next day's prediction (using 'temp_df')
            last_day_features = {}
            closes = temp_df['Close'].tolist()
            
            for lag in range(1, n_lags + 1):
                last_day_features[f'lag_{lag}'] = closes[-lag]
            for w in (5, 10, 20):
                # Calculate rolling features from the *end* of the temp_df
                ma = temp_df['Close'].rolling(window=w).mean().iloc[-1]
                std = temp_df['Close'].rolling(window=w).std().iloc[-1]
                # Handle NaNs if window is larger than available data
                last_day_features[f'ma_{w}'] = ma if not pd.isna(ma) else closes[-1]
                last_day_features[f'std_{w}'] = std if not pd.isna(std) else 0.0
            
            next_day = temp_df.index[-1] + pd.Timedelta(days=1)
            dow_next = next_day.dayofweek
            last_day_features['dow_sin'] = np.sin(2 * np.pi * dow_next / 7)
            last_day_features['dow_cos'] = np.cos(2 * np.pi * dow_next / 7)

            # Create the feature vector for prediction
            X_next_list = [last_day_features[col] for col in feature_cols]
            X_next = np.array(X_next_list).reshape(1, -1)
            
            # Use the scaler trained on ALL data
            X_next_s = scaler_full.transform(X_next) 
            
            # Predict using the model trained on ALL data
            pred = model_full.predict(X_next_s)[0] 
            pred = float(pred) if pred > 0 else float(temp_df['Close'].iloc[-1]) # Ensure non-negative
            predictions.append(pred)

            # Add the new prediction to temp_df for the next iteration
            new_row = pd.DataFrame({'Close': [pred]}, index=[next_day])
            temp_df = pd.concat([temp_df, new_row])

        # Create the final Series to return
        last_date = df_local.index[-1]
        prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=predict_len, freq='B')
        
        return pd.Series(predictions, index=prediction_dates)

    except Exception as e:
        print(f"Ridge Prediction Error: {e}")
        # Fallback to returning a simple series with the last price repeated
        last_price = float(df['Close'].iloc[-1]) if not df.empty else 0
        prediction_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=predict_len, freq='B')
        return pd.Series([last_price] * predict_len, index=prediction_dates)

@app.route('/')
def search():
    """Homepage, redirects to a default stock (e.g., AAPL)."""
    return redirect(url_for('index', ticker='AAPL'))

@app.route('/stock/<ticker>')
def index(ticker):
    """Main page displaying the stock chart, info, and prediction."""
    stock_ticker = ticker.upper()
    stock, current_price = get_stock_data(stock_ticker)
    if stock is None:
        flash(f'Ticker "{stock_ticker}" not found.', 'danger')
        return redirect(url_for('search'))

    # --- (★変更) Gemini 銘柄説明と会社名の取得 ---
    stock_info = stock.info
    gemini_description = get_gemini_description(stock_ticker, stock_info)
    company_name = stock_info.get('longName', stock_ticker) # テンプレートのタイトル用

    # --- Portfolio Info ---
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM user WHERE id = 1').fetchone()
    holdings_db = conn.execute('SELECT * FROM holdings').fetchall()
    conn.close()

    total_holdings_value = 0
    holdings_with_value = []
    for holding in holdings_db:
        _, price = get_stock_data(holding['ticker'])
        value = price * holding['shares'] if price else 0
        total_holdings_value += value
        holdings_with_value.append({
            'ticker': holding['ticker'], 'shares': holding['shares'], 'value': round(value, 2)
        })

    portfolio = {
        'balance': round(user['balance'], 2),
        'holdings_value': round(total_holdings_value, 2),
        'total_asset': round(user['balance'] + total_holdings_value, 2)
    }
    
    # --- Chart and Prediction ---
    df = stock.history(period="1y")
    
    # URLからpredict_len（予測日数）を取得。デフォルトは10
    try:
        predict_len = int(request.args.get('predict_len', 10))
    except (ValueError, TypeError):
        predict_len = 10 # エラーの場合はデフォルト値に戻す

    # --- (★変更) 両方のモデルで予測 ---
    lstm_prediction_series = predict_with_lstm(df, predict_len=predict_len)
    ridge_prediction_series = predict_with_ridge(df, predict_len=predict_len) # (★バグ修正)
    
    # (★変更) 両モデルの予測値を取得
    if not lstm_prediction_series.empty:
        predicted_price_lstm_val = lstm_prediction_series.iloc[0]
    else:
        predicted_price_lstm_val = current_price # フォールバック

    if not ridge_prediction_series.empty:
        predicted_price_ridge_val = ridge_prediction_series.iloc[0] # (★追加)
    else:
        predicted_price_ridge_val = current_price # フォールバック

    # ... (chart_dataの準備ロジックは前回の修正通り) ...
    # --- (★↓ ここから不足していたブロックを挿入 ↓) ---
    
    # 予測用のラベル（両モデルで共通のはず）
    if not lstm_prediction_series.empty:
        prediction_labels = lstm_prediction_series.index.strftime('%Y-%m-%d').tolist()
    elif not ridge_prediction_series.empty:
        prediction_labels = ridge_prediction_series.index.strftime('%Y-%m-%d').tolist()
    else:
        # 両方空の場合のフォールバック
        last_date_for_label = df.index[-1]
        prediction_labels = pd.date_range(start=last_date_for_label + pd.Timedelta(days=1), periods=predict_len, freq='B').strftime('%Y-%m-%d').tolist()

    
    # (★変更) 両モデルの予測データをリストに変換
    prediction_data_lstm = lstm_prediction_series.tolist()
    prediction_data_ridge = ridge_prediction_series.tolist()

    # Prepare data for Chart.js
    last_date = df.index[-1]
    
    # (★変更) 両モデルの予測ポイントを作成
    prediction_data_points_lstm = [None] * (len(df)-1) + [df['Close'].iloc[-1]] + prediction_data_lstm
    prediction_data_points_ridge = [None] * (len(df)-1) + [df['Close'].iloc[-1]] + prediction_data_ridge
    
    labels = df.index.strftime('%Y-%m-%d').tolist() + prediction_labels
    data = df['Close'].tolist() + [None] * len(prediction_labels) # 実績データ
    
    chart_data = {
        'labels': labels, 
        'data': data, 
        'prediction_data_lstm': prediction_data_points_lstm,
        'prediction_data_ridge': prediction_data_points_ridge,
        'ticker': stock_ticker, 
        'current_price': round(current_price, 2)
    }
    # --- (★↑ ここまで不足していたブロックを挿入 ↑) ---
    # (★変更) 両方の予測値を丸める
    predicted_price_lstm_rounded = round(predicted_price_lstm_val, 2) if predicted_price_lstm_val is not None else "N/A"
    predicted_price_ridge_rounded = round(predicted_price_ridge_val, 2) if predicted_price_ridge_val is not None else "N/A"

    # (★変更) render_template に渡す変数を修正
    return render_template(
        'index.html',
        chart_data_py=chart_data,
        chart_data_json=json.dumps(chart_data),
        portfolio=portfolio,
        holdings=holdings_with_value,
        predicted_price_lstm=predicted_price_lstm_rounded,  # (★変更)
        predicted_price_ridge=predicted_price_ridge_rounded, # (★追加)
        model_type='both',
        gemini_description=gemini_description, 
        company_name=company_name
    )


@app.route('/portfolio')
def portfolio_page():
    """Displays the historical asset value chart."""
    conn = get_db_connection()
    transactions = conn.execute('SELECT * FROM transactions ORDER BY timestamp ASC').fetchall()
    user = conn.execute('SELECT * FROM user WHERE id = 1').fetchone()
    conn.close()

    if not transactions:
        return render_template('portfolio.html', chart_data=None)

    # Calculate daily asset history
    start_date = datetime.strptime(transactions[0]['timestamp'].split(' ')[0], '%Y-%m-%d').date()
    end_date = datetime.now().date()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    tickers_str = ' '.join(set([t['ticker'] for t in transactions]))
    hist_data = yf.download(tickers_str, start=start_date, end=end_date + timedelta(days=1), progress=False)['Close']
    if isinstance(hist_data, pd.Series):
        hist_data = hist_data.to_frame(name=tickers_str)
    
    asset_history = []
    # Start with initial cash balance by reversing all transactions
    cash = user['balance'] 
    holdings = {ticker: 0 for ticker in tickers_str.split(' ')}
    
    all_holdings = get_db_connection().execute('SELECT * FROM holdings').fetchall()
    for h in all_holdings:
        holdings[h['ticker']] = h['shares']

    for t in reversed(transactions):
        if t['type'] == 'BUY':
            cash += t['shares'] * t['price']
        else: # SELL
            cash -= t['shares'] * t['price']

    tx_idx = 0
    for day in date_range:
        # Apply transactions that happened up to and including 'day'
        while tx_idx < len(transactions) and datetime.strptime(transactions[tx_idx]['timestamp'].split(' ')[0], '%Y-%m-%d').date() <= day.date():
            t = transactions[tx_idx]
            if t['type'] == 'BUY':
                cash -= t['shares'] * t['price']
            else: # SELL
                cash += t['shares'] * t['price']
            tx_idx += 1
            
        holdings_value = 0
        for ticker, shares in holdings.items():
            if shares > 0:
                try:
                    # Get the most recent price up to 'day'
                    price_on_day = hist_data.loc[:day, ticker].ffill().iloc[-1]
                    if not pd.isna(price_on_day):
                      holdings_value += shares * price_on_day
                except (KeyError, IndexError):
                    pass # Stock data might not be available for that day
            
        asset_history.append(cash + holdings_value)

    chart_data = json.dumps({
        'labels': [d.strftime('%Y-%m-%d') for d in date_range],
        'data': asset_history
    })
    
    return render_template('portfolio.html', chart_data=chart_data)


@app.route('/buy/<ticker>', methods=['POST'])
def buy(ticker):
    """Processes a stock purchase."""
    try:
        shares_to_buy = int(request.form['shares'])
        if shares_to_buy <= 0:
            flash('Please enter a positive number of shares.', 'danger')
            return redirect(url_for('index', ticker=ticker))
    except ValueError:
        flash('Invalid number of shares.', 'danger')
        return redirect(url_for('index', ticker=ticker))

    _, current_price = get_stock_data(ticker)
    if not current_price:
        flash(f'Could not retrieve price for {ticker}.', 'danger')
        return redirect(url_for('index', ticker=ticker))

    cost = shares_to_buy * current_price
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM user WHERE id = 1').fetchone()

    if user['balance'] >= cost:
        conn.execute('UPDATE user SET balance = balance - ? WHERE id = 1', (cost,))
        conn.execute('''
            INSERT INTO holdings (ticker, shares) VALUES (?, ?)
            ON CONFLICT(ticker) DO UPDATE SET shares = shares + excluded.shares
        ''', (ticker, shares_to_buy))
        conn.execute('INSERT INTO transactions (ticker, shares, price, type) VALUES (?, ?, ?, ?)',
                     (ticker, shares_to_buy, current_price, 'BUY'))
        conn.commit()
        flash(f'Successfully bought {shares_to_buy} shares of {ticker}.', 'success')
    else:
        flash('Insufficient balance.', 'danger')
    conn.close()
    return redirect(url_for('index', ticker=ticker))


@app.route('/sell/<ticker>', methods=['POST'])
def sell(ticker):
    """Processes a stock sale."""
    try:
        shares_to_sell = int(request.form['shares'])
        if shares_to_sell <= 0:
            flash('Please enter a positive number of shares.', 'danger')
            return redirect(url_for('index', ticker=ticker))
    except ValueError:
        flash('Invalid number of shares.', 'danger')
        return redirect(url_for('index', ticker=ticker))

    _, current_price = get_stock_data(ticker)
    if not current_price:
        flash(f'Could not retrieve price for {ticker}.', 'danger')
        return redirect(url_for('index', ticker=ticker))
        
    conn = get_db_connection()
    holding = conn.execute('SELECT * FROM holdings WHERE ticker = ?', (ticker,)).fetchone()

    if holding and holding['shares'] >= shares_to_sell:
        proceeds = shares_to_sell * current_price
        conn.execute('UPDATE user SET balance = balance + ? WHERE id = 1', (proceeds,))
        new_shares = holding['shares'] - shares_to_sell
        if new_shares == 0:
            conn.execute('DELETE FROM holdings WHERE ticker = ?', (ticker,))
        else:
            conn.execute('UPDATE holdings SET shares = ? WHERE ticker = ?', (new_shares, ticker))
        conn.execute('INSERT INTO transactions (ticker, shares, price, type) VALUES (?, ?, ?, ?)',
                     (ticker, shares_to_sell, current_price, 'SELL'))
        conn.commit()
        flash(f'Successfully sold {shares_to_sell} shares of {ticker}.', 'success')
    else:
        flash('You do not own enough shares to sell.', 'danger')
    conn.close()
    return redirect(url_for('index', ticker=ticker))


if __name__ == '__main__':
    # (★追加) loggingの基本設定
    logging.basicConfig(level=logging.INFO)
    
    # Note: This app requires a 'portfolio.db' file created by 'init_db.py'
    # and templates like 'index.html' and 'portfolio.html' to be in a 'templates' folder.
    app.run(debug=True)