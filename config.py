import os


APP_SECRET_KEY = os.environ.get(
    "APP_SECRET_KEY",
    os.environ.get("FLASK_SECRET_KEY", "a_very_secret_and_complex_key_12345"),
)

DATABASE_PATH = os.environ.get("PORTFOLIO_DB_PATH", "portfolio.db")
DEFAULT_TICKER = "AAPL"
DEFAULT_PREDICT_LEN = 10

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "models/gemini-pro-latest")
GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

LSTM_CONFIG = {
    "train_split": 0.7,
    "past_len": 30,
    "batch_size": 32,
    "hidden_size": 128,
    "num_layers": 3,
    "epochs": 50,
    "learning_rate": 1e-4,
}

RIDGE_CONFIG = {
    "train_split": 0.7,
    "n_lags": 5,
    "use_days": 180,
    "alpha": 1.0,
    "moving_windows": (5, 10, 20),
}
