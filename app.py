# app.py

from flask import Flask, render_template, request, redirect, url_for, flash
import yfinance as yf
import pandas as pd
import json
import sqlite3
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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

def predict_tomorrow_price(df, n_lags=5, use_days=180):
    """
    過去データからラグ特徴・移動平均・ボラティリティ・曜日特徴を作り
    Ridge回帰で翌日終値を予測する。
    df: yfinanceで取得した履歴データ（indexがDatetimeIndex, 列に 'Close' を含む）
    n_lags: 使用するラグの数（例: 5 -> 1日〜5日の終値を特徴に）
    use_days: 学習に使う直近の日数（len(df) が小さい時は調整）
    """
    try:
        df_local = df.copy().sort_index()
        if df_local.empty or 'Close' not in df_local.columns or len(df_local) < 5:
            # データ不足なら既存の簡単な手法でフォールバック
            return df_local['Close'].iloc[-1] if not df_local.empty else None

        # 特徴量作成
        data = pd.DataFrame({'Close': df_local['Close']})
        # ラグ特徴
        for lag in range(1, n_lags + 1):
            data[f'lag_{lag}'] = data['Close'].shift(lag)
        # リターンと移動平均・ボラ
        for w in (5, 10, 20):
            data[f'ma_{w}'] = data['Close'].rolling(window=w, min_periods=1).mean()
            data[f'std_{w}'] = data['Close'].rolling(window=w, min_periods=1).std().fillna(0)
        data['return_1'] = data['Close'].pct_change().fillna(0)
        # 曜日（周期特徴 sin/cos）
        dow = data.index.dayofweek
        data['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        data['dow_cos'] = np.cos(2 * np.pi * dow / 7)

        # 学習用に NaN を落とす
        data = data.dropna()

        # 学習データの期間を限定（直近 use_days 日）
        if use_days is not None and len(data) > use_days:
            data_train = data.iloc[-use_days:]
        else:
            data_train = data

        # 特徴量/ターゲット分離
        feature_cols = [c for c in data_train.columns if c != 'Close']
        X = data_train[feature_cols].values
        y = data_train['Close'].values

        # 標準化 + Ridge 回帰
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = Ridge(alpha=1.0)  # 正則化強さは必要に応じて調整
        model.fit(Xs, y)

        # --- 翌日の特徴量を作る ---
        last = data.iloc[-1:].copy()  # 最終日行（shift済みのlagではないので注意）
        # ラグ値は直前のClose等を使って作る
        last_row = {}
        last_close = df_local['Close'].iloc[-1]
        # ラグ1は今日の終値、ラグ2は昨日の終値... とする
        closes = df_local['Close'].tolist()
        for lag in range(1, n_lags + 1):
            if len(closes) >= lag:
                last_row[f'lag_{lag}'] = closes[-lag]
            else:
                last_row[f'lag_{lag}'] = closes[-1]  # データ不足時フォールバック

        for w in (5, 10, 20):
            last_row[f'ma_{w}'] = df_local['Close'].rolling(window=w, min_periods=1).mean().iloc[-1]
            last_row[f'std_{w}'] = df_local['Close'].rolling(window=w, min_periods=1).std().fillna(0).iloc[-1]
        last_row['return_1'] = df_local['Close'].pct_change().fillna(0).iloc[-1]
        next_day = df_local.index[-1] + pd.Timedelta(days=1)
        dow_next = next_day.dayofweek
        last_row['dow_sin'] = np.sin(2 * np.pi * dow_next / 7)
        last_row['dow_cos'] = np.cos(2 * np.pi * dow_next / 7)

        X_next = np.array([last_row[col] for col in feature_cols]).reshape(1, -1)
        X_next_s = scaler.transform(X_next)
        pred = model.predict(X_next_s)[0]

        # 現実性チェック（価格は負にならないように）
        if np.isnan(pred) or pred <= 0:
            pred = last_close

        return float(pred)
    except Exception as e:
        # 何か問題が起きたら既存の単純線形回帰風にフォールバック（安全策）
        try:
            df_pred = df.copy()
            df_pred['days'] = (df_pred.index - df_pred.index[0]).days
            X = df_pred[['days']]
            y = df_pred['Close']
            model = LinearRegression()
            model.fit(X, y)
            last_day = df_pred['days'].iloc[-1]
            tomorrow = np.array([[last_day + 1]])
            return float(model.predict(tomorrow)[0])
        except Exception:
            # それでもダメなら直近の終値を返す
            if not df.empty:
                return float(df['Close'].iloc[-1])
            return None


if __name__ == '__main__':
    app.run(debug=True)