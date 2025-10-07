# app.py

from flask import Flask, render_template, request, redirect, url_for, flash
import yfinance as yf
import pandas as pd
import json
import sqlite3
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_should_be_more_complex' # 実際にはもっと複雑なキーにしてください

def get_db_connection():
    """データベースへの接続を取得する"""
    conn = sqlite3.connect('portfolio.db')
    conn.row_factory = sqlite3.Row # カラム名でアクセスできるようにする
    return conn

def get_stock_data(ticker):
    """最新の株価とyfinanceオブジェクトを取得する（エラーハンドリング付き）"""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    if hist.empty:
        return None, None
    # .iloc[0] を使って位置で安全にアクセス
    return stock, hist['Close'].iloc[0]

@app.route('/')
def search():
    """トップページ。デフォルトでAAPLにリダイレクトする"""
    return redirect(url_for('index', ticker='AAPL'))

@app.route('/stock/<ticker>')
def index(ticker):
    """指定された銘柄のチャートや情報を表示するメインページ"""
    stock, current_price = get_stock_data(ticker.upper())
    if stock is None:
        flash(f'銘柄コード "{ticker}" が見つかりませんでした。', 'danger')
        return redirect(url_for('search'))

    # --- ポートフォリオ情報取得 ---
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
            'ticker': holding['ticker'],
            'shares': holding['shares'],
            'value': round(value, 2)
        })

    portfolio = {
        'balance': round(user['balance'], 2),
        'holdings_value': round(total_holdings_value, 2),
        'total_asset': round(user['balance'] + total_holdings_value, 2)
    }
    
    # --- チャートと予測 ---
    df = stock.history(period="1y")
    predicted_price = predict_tomorrow_price(df)
    last_date = df.index[-1]
    next_day_label = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    prediction_data_points = [None] * (len(df) - 1) + [df['Close'].iloc[-1], predicted_price]
    labels = df.index.strftime('%Y-%m-%d').tolist() + [next_day_label]
    data = df['Close'].tolist() + [None]
    
    chart_data = {
        'labels': labels, 'data': data, 'prediction_data': prediction_data_points,
        'ticker': ticker.upper(), 'current_price': round(current_price, 2)
    }

    return render_template('index.html', 
                           chart_data_py=chart_data,               # HTML用 (Python辞書)
                           chart_data_json=json.dumps(chart_data), # JavaScript用 (JSON文字列)
                           portfolio=portfolio, 
                           holdings=holdings_with_value)
@app.route('/portfolio')
def portfolio_page():
    """資産推移グラフを表示するページ"""
    conn = get_db_connection()
    transactions = conn.execute('SELECT * FROM transactions ORDER BY timestamp ASC').fetchall()
    user = conn.execute('SELECT * FROM user WHERE id = 1').fetchone()
    conn.close()

    if not transactions:
        return render_template('portfolio.html', chart_data=None)

    # 日々の資産推移を計算
    start_date = datetime.strptime(transactions[0]['timestamp'].split(' ')[0], '%Y-%m-%d').date()
    end_date = datetime.now().date()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    tickers_str = ' '.join(set([t['ticker'] for t in transactions]))
    hist_data = yf.download(tickers_str, start=start_date, end=end_date + timedelta(days=1))['Close']
    if isinstance(hist_data, pd.Series):
        hist_data = hist_data.to_frame(name=tickers_str)
    
    asset_history = []
    cash = user['balance']
    holdings = {ticker: 0 for ticker in tickers_str.split(' ')}

    # 取引履歴を逆算して、計算開始時点の現金と保有株を割り出す
    for t in reversed(transactions):
        if t['type'] == 'BUY':
            cash += t['shares'] * t['price']
            holdings[t['ticker']] -= t['shares']
        else: # SELL
            cash -= t['shares'] * t['price']
            holdings[t['ticker']] += t['shares']

    tx_idx = 0
    for day in date_range:
        while tx_idx < len(transactions) and datetime.strptime(transactions[tx_idx]['timestamp'].split(' ')[0], '%Y-%m-%d').date() <= day.date():
            t = transactions[tx_idx]
            if datetime.strptime(t['timestamp'].split(' ')[0], '%Y-%m-%d').date() == day.date():
                if t['type'] == 'BUY':
                    cash -= t['shares'] * t['price']
                    holdings[t['ticker']] += t['shares']
                else: # SELL
                    cash += t['shares'] * t['price']
                    holdings[t['ticker']] -= t['shares']
            tx_idx += 1
            
        holdings_value = 0
        for ticker, shares in holdings.items():
            if shares > 0:
                try:
                    price_on_day = hist_data.loc[:day, ticker].ffill().iloc[-1]
                    holdings_value += shares * price_on_day
                except (KeyError, IndexError):
                    pass
        
        asset_history.append(cash + holdings_value)

    chart_data = json.dumps({
        'labels': [d.strftime('%Y-%m-%d') for d in date_range],
        'data': asset_history
    })
    
    return render_template('portfolio.html', chart_data=chart_data)

@app.route('/buy/<ticker>', methods=['POST'])
def buy(ticker):
    """株を購入する処理"""
    shares_to_buy = int(request.form['shares'])
    _, current_price = get_stock_data(ticker)
    if not current_price:
        flash(f'銘柄 {ticker} の価格を取得できませんでした。', 'danger')
        return redirect(url_for('index', ticker=ticker))

    cost = shares_to_buy * current_price
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM user WHERE id = 1').fetchone()

    if user['balance'] >= cost:
        conn.execute('UPDATE user SET balance = balance - ? WHERE id = 1', (cost,))
        conn.execute('INSERT INTO holdings (ticker, shares) VALUES (?, ?) ON CONFLICT(ticker) DO UPDATE SET shares = shares + ?', (ticker, shares_to_buy, shares_to_buy))
        conn.execute('INSERT INTO transactions (ticker, shares, price, type) VALUES (?, ?, ?, ?)', (ticker, shares_to_buy, current_price, 'BUY'))
        conn.commit()
        flash(f'{ticker}を{shares_to_buy}株購入しました。', 'success')
    else:
        flash('残高が不足しています。', 'danger')
    conn.close()
    return redirect(url_for('index', ticker=ticker))

@app.route('/sell/<ticker>', methods=['POST'])
def sell(ticker):
    """株を売却する処理"""
    shares_to_sell = int(request.form['shares'])
    _, current_price = get_stock_data(ticker)
    if not current_price:
        flash(f'銘柄 {ticker} の価格を取得できませんでした。', 'danger')
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
        conn.execute('INSERT INTO transactions (ticker, shares, price, type) VALUES (?, ?, ?, ?)', (ticker, shares_to_sell, current_price, 'SELL'))
        conn.commit()
        flash(f'{ticker}を{shares_to_sell}株売却しました。', 'success')
    else:
        flash('保有株数が不足しています。', 'danger')
    conn.close()
    return redirect(url_for('index', ticker=ticker))

def predict_tomorrow_price(df):
    """線形回帰モデルで翌日の株価を予測する"""
    df_pred = df.copy()
    df_pred['days'] = (df_pred.index - df_pred.index[0]).days
    X = df_pred[['days']]
    y = df_pred['Close']
    model = LinearRegression()
    model.fit(X, y)
    last_day = df_pred['days'].iloc[-1]
    tomorrow = np.array([[last_day + 1]])
    prediction = model.predict(tomorrow)
    return prediction[0]

if __name__ == '__main__':
    app.run(debug=True)