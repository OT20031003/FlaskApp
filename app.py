# app.py

from flask import Flask, render_template
import yfinance as yf
import pandas as pd
import json
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

def get_stock_data(ticker_symbol):
    """指定された銘柄コードの過去1年間の株価データを取得する"""
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="1y")
    return hist

def predict_tomorrow_price(df):
    """線形回帰モデルを使って翌日の終値を予測する"""
    # 目的変数（y）を終値、説明変数（X）を日付からの経過日数とする
    df_pred = df.copy()
    df_pred['days'] = (df_pred.index - df_pred.index[0]).days

    X = df_pred[['days']]
    y = df_pred['Close']

    # 線形回帰モデルの学習
    model = LinearRegression()
    model.fit(X, y)

    # 最後の日の次の日（翌日）を予測
    last_day = df_pred['days'].iloc[-1]
    tomorrow = np.array([[last_day + 1]])
    prediction = model.predict(tomorrow)
    
    return prediction[0]

@app.route('/')
def index():
    """トップページを表示し、株価チャートと予測値を描画する"""
    ticker = 'AAPL'
    df = get_stock_data(ticker)
    
    # 翌日の株価を予測
    predicted_price = predict_tomorrow_price(df)

    # 過去のデータに予測値を追加してチャートで表示する準備
    # 予測用のラベル（最後の日付の次）
    last_date = df.index[-1]
    next_day_label = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    # 過去データと予測データを結合するためのリストを作成
    # 過去データはそのまま、予測データは過去データの最後と予測値をつなぐ線として描画
    prediction_data_points = [None] * (len(df) - 1) + [df['Close'].iloc[-1], predicted_price]

    # Chart.jsで扱いやすいようにデータを整形
    labels = df.index.strftime('%Y-%m-%d').tolist() + [next_day_label]
    data = df['Close'].tolist() + [None] # 実績データの最後はNoneにする
    
    chart_data = {
        'labels': labels,
        'data': data,
        'prediction_data': prediction_data_points, # 予測データ用のリストを追加
        'ticker': ticker
    }

    return render_template('index.html', chart_data=json.dumps(chart_data))

if __name__ == '__main__':
    app.run(debug=True)