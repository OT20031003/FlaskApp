-- schema.sql

DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS holdings;
DROP TABLE IF EXISTS user;

CREATE TABLE user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    balance REAL NOT NULL
);

CREATE TABLE holdings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    shares INTEGER NOT NULL,
    UNIQUE(ticker) -- 同じ銘柄は一行にまとめる
);

CREATE TABLE transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    shares INTEGER NOT NULL,
    price REAL NOT NULL,
    type TEXT NOT NULL, -- 'BUY' or 'SELL'
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ユーザーの初期残高を設定
INSERT INTO user (balance) VALUES (10000.00);