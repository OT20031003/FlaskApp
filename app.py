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

# --- Flask App Initialization ---
app = Flask(__name__)
# IMPORTANT: Change this secret key in a real application
app.config['SECRET_KEY'] = 'a_very_secret_and_complex_key_12345'

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

# --- Prediction Models ---

def predict_with_lstm(df, sequence_length=30, epochs=50):
    """
    Predicts the next day's closing price using the PyTorch LSTM model.
    The model is trained on-the-fly with the provided historical data.
    """
    try:
        # 1. Data Preparation
        close_prices = df['Close'].values.reshape(-1, 1)
        if len(close_prices) < sequence_length + 1:
            # Not enough data to create sequences, fallback to last price
            return float(df['Close'].iloc[-1])

        # Scale data to be between 0 and 1 for the neural network
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(close_prices)

        # Create sequences of data (e.g., use 30 days to predict the 31st)
        X, y = [], []
        for i in range(len(scaled_prices) - sequence_length):
            X.append(scaled_prices[i:i + sequence_length])
            y.append(scaled_prices[i + sequence_length])
        
        X = torch.FloatTensor(np.array(X))
        y = torch.FloatTensor(np.array(y))

        # 2. Model Training
        # Define model parameters
        input_size = 1
        hidden_size = 50
        output_size = 1
        num_layers = 2
        
        model = StockPredictor(input_size, hidden_size, output_size, num_layers)
        criterion = nn.MSELoss()  # Mean Squared Error for regression
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # 3. Prediction
        model.eval()
        with torch.no_grad():
            # Get the last `sequence_length` days from the original data
            last_sequence_scaled = scaled_prices[-sequence_length:]
            # Reshape for the model: [batch_size, sequence_length, features]
            input_tensor = torch.FloatTensor(last_sequence_scaled).view(1, sequence_length, 1)
            
            prediction_scaled = model(input_tensor)

        # 4. Inverse Transform
        # The prediction is scaled, so we revert it back to the original price scale
        predicted_price = scaler.inverse_transform(prediction_scaled.numpy())[0, 0]
        
        # Reality check: ensure prediction is a positive number
        return float(predicted_price) if predicted_price > 0 else float(df['Close'].iloc[-1])

    except Exception as e:
        print(f"LSTM Prediction Error: {e}")
        # Fallback to a simple prediction if anything goes wrong
        return float(df['Close'].iloc[-1]) if not df.empty else None


def predict_with_ridge(df, n_lags=5, use_days=180):
    """
    Predicts the next day's closing price using Ridge Regression with feature engineering.
    """
    try:
        df_local = df.copy().sort_index()
        if len(df_local) < n_lags + 1:
            return float(df_local['Close'].iloc[-1]) if not df_local.empty else None

        # Feature Engineering
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

        data_train = data.iloc[-use_days:] if len(data) > use_days else data
        
        feature_cols = [c for c in data_train.columns if c != 'Close']
        X = data_train[feature_cols].values
        y = data_train['Close'].values

        # Model Training
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = Ridge(alpha=1.0)
        model.fit(Xs, y)

        # Prepare features for the next day's prediction
        last_day_features = {}
        closes = df_local['Close'].tolist()
        for lag in range(1, n_lags + 1):
            last_day_features[f'lag_{lag}'] = closes[-lag]
        for w in (5, 10, 20):
            last_day_features[f'ma_{w}'] = df_local['Close'].rolling(window=w).mean().iloc[-1]
            last_day_features[f'std_{w}'] = df_local['Close'].rolling(window=w).std().iloc[-1]
        next_day = df_local.index[-1] + pd.Timedelta(days=1)
        dow_next = next_day.dayofweek
        last_day_features['dow_sin'] = np.sin(2 * np.pi * dow_next / 7)
        last_day_features['dow_cos'] = np.cos(2 * np.pi * dow_next / 7)

        X_next = np.array([last_day_features[col] for col in feature_cols]).reshape(1, -1)
        X_next_s = scaler.transform(X_next)
        
        pred = model.predict(X_next_s)[0]

        return float(pred) if pred > 0 else float(df_local['Close'].iloc[-1])

    except Exception as e:
        print(f"Ridge Prediction Error: {e}")
        # Fallback to simple linear regression on failure
        try:
            df_pred = df.copy()
            df_pred['days'] = (df_pred.index - df_pred.index[0]).days
            model_lr = LinearRegression()
            model_lr.fit(df_pred[['days']], df_pred['Close'])
            tomorrow = np.array([[df_pred['days'].iloc[-1] + 1]])
            return float(model_lr.predict(tomorrow)[0])
        except Exception:
            return float(df['Close'].iloc[-1]) if not df.empty else None

# --- Flask Routes ---

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
    
    # Select prediction model based on URL query parameter
    model_type = request.args.get('model', 'ridge').lower()
    print(f"model_type = {model_type}")
    if model_type == 'lstm':
        predicted_price = predict_with_lstm(df)
    else:
        # Default to Ridge model
        model_type = 'ridge'
        predicted_price = predict_with_ridge(df)

    # Prepare data for Chart.js
    last_date = df.index[-1]
    next_day_label = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    # The prediction line connects today's close with tomorrow's prediction
    prediction_data_points = [None] * (len(df) - 1) + [df['Close'].iloc[-1], predicted_price]
    labels = df.index.strftime('%Y-%m-%d').tolist() + [next_day_label]
    data = df['Close'].tolist() + [None] # Historical data points
    
    chart_data = {
        'labels': labels, 'data': data, 'prediction_data': prediction_data_points,
        'ticker': stock_ticker, 'current_price': round(current_price, 2)
    }

    predicted_price_rounded = round(predicted_price, 2) if predicted_price is not None else "N/A"

    return render_template(
        'index.html',
        chart_data_py=chart_data,
        chart_data_json=json.dumps(chart_data),
        portfolio=portfolio,
        holdings=holdings_with_value,
        predicted_price=predicted_price_rounded,
        model_type=model_type # Pass model type to the template
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
    # Note: This app requires a 'portfolio.db' file created by 'init_db.py'
    # and templates like 'index.html' and 'portfolio.html' to be in a 'templates' folder.
    app.run(debug=True)

