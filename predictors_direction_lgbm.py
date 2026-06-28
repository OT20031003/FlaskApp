from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from services import get_stock_data
from config import DIRECTION_CONFIG
from predictor_utils import direction_fallback, format_last_date


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _score_direction_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    metric_name: str,
) -> float:
    y_pred = (y_prob >= threshold).astype(int)

    if metric_name == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric_name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    return float(balanced_accuracy_score(y_true, y_pred))


def _find_optimal_direction_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    metric_name: str,
    default_threshold: float = 0.5,
) -> tuple[float, Optional[float]]:
    if len(y_true) == 0 or y_true.nunique() < 2:
        return default_threshold, None

    candidate_thresholds = np.unique(np.round(y_prob, 4))
    candidate_thresholds = np.concatenate(
        [
            np.array([0.01, default_threshold, 0.99]),
            candidate_thresholds,
        ]
    )
    candidate_thresholds = np.unique(np.clip(candidate_thresholds, 0.01, 0.99))

    best_threshold = default_threshold
    best_score: Optional[float] = None

    for threshold in candidate_thresholds:
        score = _score_direction_threshold(y_true, y_prob, float(threshold), metric_name)
        if best_score is None or score > best_score:
            best_threshold = float(threshold)
            best_score = score
        elif best_score is not None and np.isclose(score, best_score):
            if abs(float(threshold) - default_threshold) < abs(best_threshold - default_threshold):
                best_threshold = float(threshold)

    return best_threshold, best_score


def _build_direction_features(
    df: pd.DataFrame,
    predict_len: int,
) -> tuple[pd.DataFrame, list[str]]:
    feature_df = df.copy().sort_index()
    feature_cols: list[str] = []

    if "Close" not in feature_df.columns:
        return feature_df, feature_cols

    threshold = DIRECTION_CONFIG.get("target_return_threshold", 0.0)

    feature_df["log_return_1"] = np.log(feature_df["Close"] / feature_df["Close"].shift(1))
    feature_df["return_2"] = feature_df["Close"].pct_change(2, fill_method=None)
    feature_df["return_5"] = feature_df["Close"].pct_change(5, fill_method=None)
    feature_df["return_10"] = feature_df["Close"].pct_change(10, fill_method=None)
    feature_df["return_20"] = feature_df["Close"].pct_change(20, fill_method=None)
    feature_df["return_60"] = feature_df["Close"].pct_change(60, fill_method=None)
    feature_df["volatility_5"] = feature_df["log_return_1"].rolling(5).std()
    feature_df["volatility_10"] = feature_df["log_return_1"].rolling(10).std()
    feature_df["volatility_20"] = feature_df["log_return_1"].rolling(20).std()
    feature_df["volatility_60"] = feature_df["log_return_1"].rolling(60).std()
    feature_df["ma_gap_5"] = feature_df["Close"] / feature_df["Close"].rolling(5).mean() - 1
    feature_df["ma_gap_20"] = feature_df["Close"] / feature_df["Close"].rolling(20).mean() - 1
    feature_df["ma_gap_60"] = feature_df["Close"] / feature_df["Close"].rolling(60).mean() - 1
    feature_df["ma_gap_120"] = feature_df["Close"] / feature_df["Close"].rolling(120).mean() - 1

    feature_cols.extend(
        [
            "log_return_1",
            "return_2",
            "return_5",
            "return_10",
            "return_20",
            "return_60",
            "volatility_5",
            "volatility_10",
            "volatility_20",
            "volatility_60",
            "ma_gap_5",
            "ma_gap_20",
            "ma_gap_60",
            "ma_gap_120",
        ]
    )

    if "Volume" in feature_df.columns:
        feature_df["volume_change_5"] = feature_df["Volume"].pct_change(5, fill_method=None)
        feature_df["volume_change_20"] = feature_df["Volume"].pct_change(20, fill_method=None)
        feature_df["volume_z_20"] = (
            feature_df["Volume"] - feature_df["Volume"].rolling(20).mean()
        ) / feature_df["Volume"].rolling(20).std()
        feature_cols.extend(["volume_change_5", "volume_change_20", "volume_z_20"])

    if {"Open", "High", "Low"}.issubset(feature_df.columns):
        feature_df["intraday_return"] = feature_df["Close"] / feature_df["Open"] - 1
        feature_df["high_low_range"] = feature_df["High"] / feature_df["Low"] - 1
        feature_df["close_to_high"] = feature_df["Close"] / feature_df["High"] - 1
        feature_df["close_to_low"] = feature_df["Close"] / feature_df["Low"] - 1
        feature_cols.extend(
            ["intraday_return", "high_low_range", "close_to_high", "close_to_low"]
        )

    dow = pd.to_datetime(feature_df.index).dayofweek
    feature_df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feature_df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    feature_df["rsi_14"] = _compute_rsi(feature_df["Close"], period=14)
    feature_df["ema_12"] = feature_df["Close"].ewm(span=12, adjust=False).mean()
    feature_df["ema_26"] = feature_df["Close"].ewm(span=26, adjust=False).mean()
    feature_df["macd"] = feature_df["ema_12"] - feature_df["ema_26"]
    feature_df["macd_signal"] = feature_df["macd"].ewm(span=9, adjust=False).mean()
    feature_df["macd_hist"] = feature_df["macd"] - feature_df["macd_signal"]
    feature_cols.extend(
        [
            "dow_sin",
            "dow_cos",
            "rsi_14",
            "ema_12",
            "ema_26",
            "macd",
            "macd_signal",
            "macd_hist",
        ]
    )

    feature_df["future_log_return"] = np.log(
        feature_df["Close"].shift(-predict_len) / feature_df["Close"]
    )
    feature_df["target"] = np.where(
        feature_df["future_log_return"].notna(),
        (feature_df["future_log_return"] > threshold).astype(int),
        np.nan,
    )

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.dropna(subset=feature_cols).copy()
    return feature_df, feature_cols


def _build_direction_model() -> Any:
    try:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=DIRECTION_CONFIG.get("random_state", 42),
            n_jobs=-1,
            verbose=-1,
        )
        setattr(model, "_direction_model_name", "LightGBM Direction Classifier")
        return model
    except Exception as exc:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=DIRECTION_CONFIG.get("random_state", 42),
            n_jobs=-1,
        )
        setattr(model, "_direction_model_name", "RandomForest Direction Classifier")
        setattr(model, "_direction_model_warning", str(exc))
        return model


def _round_float(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _summarize_probabilities(y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    return {
        "min": _round_float(float(np.min(y_prob))),
        "p25": _round_float(float(np.quantile(y_prob, 0.25))),
        "median": _round_float(float(np.quantile(y_prob, 0.5))),
        "p75": _round_float(float(np.quantile(y_prob, 0.75))),
        "max": _round_float(float(np.max(y_prob))),
        "mean": _round_float(float(np.mean(y_prob))),
        "std": _round_float(float(np.std(y_prob))),
        "count_ge_threshold": int(np.sum(y_prob >= threshold)),
        "count_lt_threshold": int(np.sum(y_prob < threshold)),
        "sample_size": int(len(y_prob)),
    }


def _iter_rolling_window_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    step_size: int,
    purge_size: int,
) -> Iterator[dict[str, int]]:
    if min(n_samples, train_size, test_size, step_size) <= 0 or purge_size < 0:
        return

    first_test_start = train_size + purge_size
    last_test_start = n_samples - test_size
    if first_test_start > last_test_start:
        return

    fold_number = 1
    for test_start in range(first_test_start, last_test_start + 1, step_size):
        train_end = test_start - purge_size
        train_start = train_end - train_size
        test_end = test_start + test_size
        if train_start < 0 or test_end > n_samples:
            continue

        yield {
            "fold_number": fold_number,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "purge_size": purge_size,
        }
        fold_number += 1


def _fit_direction_model_with_threshold(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    calibration_ratio: float,
    threshold_metric: str,
) -> dict[str, Any]:
    y_train = train_df["target"].astype(int)
    if y_train.nunique() < 2:
        raise ValueError("Training data contains only one direction class")

    calibration_len = int(len(train_df) * calibration_ratio)
    fit_df = train_df.copy()
    calibration_df = train_df.iloc[0:0].copy()
    optimal_threshold = 0.5
    calibration_score: Optional[float] = None
    threshold_search_status = "disabled"

    threshold_search_enabled = calibration_len > 0 and calibration_len < len(train_df)
    if threshold_search_enabled:
        fit_df = train_df.iloc[:-calibration_len].copy()
        calibration_df = train_df.iloc[-calibration_len:].copy()
        y_fit = fit_df["target"].astype(int)
        y_calibration = calibration_df["target"].astype(int)

        if y_fit.nunique() >= 2 and y_calibration.nunique() >= 2:
            threshold_model = _build_direction_model()
            threshold_model.fit(fit_df[feature_cols], y_fit)
            calibration_probabilities = threshold_model.predict_proba(calibration_df[feature_cols])[:, 1]
            optimal_threshold, calibration_score = _find_optimal_direction_threshold(
                y_calibration,
                calibration_probabilities,
                threshold_metric,
            )
            threshold_search_status = "ok"
        else:
            threshold_search_status = "insufficient_class_variation"
    else:
        threshold_search_status = "insufficient_samples"

    model = _build_direction_model()
    model.fit(train_df[feature_cols], y_train)
    return {
        "model": model,
        "decision_threshold": float(optimal_threshold),
        "calibration_score": calibration_score,
        "threshold_search_status": threshold_search_status,
        "model_backend": getattr(model, "_direction_model_name", type(model).__name__),
        "model_warning": getattr(model, "_direction_model_warning", None),
    }


def _evaluate_direction_fold(
    training_df: pd.DataFrame,
    feature_cols: list[str],
    split: dict[str, int],
    calibration_ratio: float,
    threshold_metric: str,
) -> dict[str, Any]:
    train_df = training_df.iloc[split["train_start"]:split["train_end"]].copy()
    test_df = training_df.iloc[split["test_start"]:split["test_end"]].copy()

    fold_result: dict[str, Any] = {
        "fold_number": split["fold_number"],
        "train_start": split["train_start"],
        "train_end": split["train_end"],
        "test_start": split["test_start"],
        "test_end": split["test_end"],
        "purge_size": split["purge_size"],
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "train_start_date": format_last_date(train_df.index[0]) if not train_df.empty else None,
        "train_end_date": format_last_date(train_df.index[-1]) if not train_df.empty else None,
        "test_start_date": format_last_date(test_df.index[0]) if not test_df.empty else None,
        "test_end_date": format_last_date(test_df.index[-1]) if not test_df.empty else None,
    }

    if train_df.empty or test_df.empty:
        fold_result.update(
            {
                "status": "insufficient_samples",
                "accuracy": None,
                "balanced_accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
                "log_loss": None,
                "baseline_accuracy": None,
                "confusion_matrix": None,
                "decision_threshold": None,
                "threshold_search_metric": threshold_metric,
                "threshold_search_status": "insufficient_samples",
                "calibration_score": None,
                "model_backend": None,
                "model_warning": None,
                "y_prob_summary": None,
            }
        )
        return fold_result

    y_test = test_df["target"].astype(int)

    try:
        fit_result = _fit_direction_model_with_threshold(
            train_df,
            feature_cols,
            calibration_ratio,
            threshold_metric,
        )
    except ValueError as exc:
        fold_result.update(
            {
                "status": str(exc),
                "accuracy": None,
                "balanced_accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
                "log_loss": None,
                "baseline_accuracy": None,
                "confusion_matrix": None,
                "decision_threshold": None,
                "threshold_search_metric": threshold_metric,
                "threshold_search_status": "insufficient_class_variation",
                "calibration_score": None,
                "model_backend": None,
                "model_warning": None,
                "y_prob_summary": None,
            }
        )
        return fold_result

    model = fit_result["model"]
    optimal_threshold = float(fit_result["decision_threshold"])
    y_prob = model.predict_proba(test_df[feature_cols])[:, 1]
    y_pred = (y_prob >= optimal_threshold).astype(int)
    majority_class = int(y_test.mode().iloc[0])
    baseline_predictions = np.full_like(y_test.to_numpy(), majority_class)
    test_has_both_classes = y_test.nunique() >= 2

    balanced_accuracy_value = (
        float(balanced_accuracy_score(y_test, y_pred)) if test_has_both_classes else None
    )
    roc_auc_value = float(roc_auc_score(y_test, y_prob)) if test_has_both_classes else None

    fold_result.update(
        {
            "status": "ok",
            "accuracy": _round_float(float(accuracy_score(y_test, y_pred))),
            "balanced_accuracy": _round_float(balanced_accuracy_value),
            "precision": _round_float(float(precision_score(y_test, y_pred, zero_division=0))),
            "recall": _round_float(float(recall_score(y_test, y_pred, zero_division=0))),
            "f1": _round_float(float(f1_score(y_test, y_pred, zero_division=0))),
            "roc_auc": _round_float(roc_auc_value),
            "log_loss": _round_float(float(log_loss(y_test, y_prob, labels=[0, 1]))),
            "baseline_accuracy": _round_float(
                float(accuracy_score(y_test, baseline_predictions))
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
            "decision_threshold": _round_float(optimal_threshold),
            "threshold_search_metric": threshold_metric,
            "threshold_search_status": fit_result["threshold_search_status"],
            "calibration_score": _round_float(fit_result["calibration_score"]),
            "model_backend": fit_result["model_backend"],
            "model_warning": fit_result["model_warning"],
            "y_prob_summary": _summarize_probabilities(y_prob, optimal_threshold),
        }
    )
    return fold_result


def _aggregate_rolling_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "log_loss",
        "baseline_accuracy",
        "decision_threshold",
        "calibration_score",
    ]
    aggregate: dict[str, Any] = {
        "fold_count": int(sum(1 for fold in fold_metrics if fold.get("status") == "ok")),
        "total_fold_count": int(len(fold_metrics)),
        "beats_baseline_fold_ratio": 0.0,
        "confusion_matrix_sum": [[0, 0], [0, 0]],
    }

    for metric_name in scalar_metric_names:
        values = [fold[metric_name] for fold in fold_metrics if fold.get(metric_name) is not None]
        if not values:
            aggregate[metric_name] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
            continue

        values_np = np.asarray(values, dtype=float)
        aggregate[metric_name] = {
            "mean": _round_float(float(np.mean(values_np))),
            "std": _round_float(float(np.std(values_np))),
            "min": _round_float(float(np.min(values_np))),
            "max": _round_float(float(np.max(values_np))),
        }

    beats_baseline_flags = [
        fold["accuracy"] > fold["baseline_accuracy"]
        for fold in fold_metrics
        if fold.get("accuracy") is not None and fold.get("baseline_accuracy") is not None
    ]
    if beats_baseline_flags:
        aggregate["beats_baseline_fold_ratio"] = _round_float(
            float(np.mean(beats_baseline_flags))
        )

    confusion_matrices = [
        np.asarray(fold["confusion_matrix"], dtype=int)
        for fold in fold_metrics
        if fold.get("confusion_matrix") is not None
    ]
    if confusion_matrices:
        confusion_sum = np.sum(confusion_matrices, axis=0)
        aggregate["confusion_matrix_sum"] = confusion_sum.astype(int).tolist()

    return aggregate


def validate_direction_with_rolling_window(
    training_df: pd.DataFrame,
    feature_cols: list[str],
    predict_len: int,
) -> dict[str, Any]:
    calibration_ratio = DIRECTION_CONFIG.get("threshold_calibration_split", 0.25)
    threshold_metric = DIRECTION_CONFIG.get("threshold_search_metric", "balanced_accuracy")
    train_size = int(DIRECTION_CONFIG.get("rolling_train_size", 504))
    test_size = int(DIRECTION_CONFIG.get("rolling_test_size", 63))
    step_size = int(DIRECTION_CONFIG.get("rolling_step_size", test_size))
    configured_purge_size = DIRECTION_CONFIG.get("rolling_purge_size")
    purge_size = predict_len if configured_purge_size is None else int(configured_purge_size)
    purge_size = max(purge_size, predict_len)

    rolling_config = {
        "train_size": train_size,
        "test_size": test_size,
        "step_size": step_size,
        "purge_size": purge_size,
        "calibration_ratio": calibration_ratio,
        "threshold_metric": threshold_metric,
    }

    splits = list(
        _iter_rolling_window_splits(
            n_samples=len(training_df),
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
            purge_size=purge_size,
        )
    )
    fold_metrics = [
        _evaluate_direction_fold(
            training_df,
            feature_cols,
            split,
            calibration_ratio,
            threshold_metric,
        )
        for split in splits
    ]
    metrics = _aggregate_rolling_metrics(fold_metrics)

    min_rolling_folds = int(DIRECTION_CONFIG.get("min_rolling_folds", 3))
    min_balanced_accuracy_mean = float(DIRECTION_CONFIG.get("min_balanced_accuracy_mean", 0.52))
    min_roc_auc_mean = float(DIRECTION_CONFIG.get("min_roc_auc_mean", 0.52))
    min_beats_baseline_fold_ratio = float(
        DIRECTION_CONFIG.get("min_beats_baseline_fold_ratio", 0.5)
    )

    balanced_accuracy_mean = metrics["balanced_accuracy"]["mean"]
    roc_auc_mean = metrics["roc_auc"]["mean"]
    beats_baseline_fold_ratio = metrics["beats_baseline_fold_ratio"]
    fold_count = metrics["fold_count"]

    model_valid = bool(
        fold_count >= min_rolling_folds
        and balanced_accuracy_mean is not None
        and balanced_accuracy_mean >= min_balanced_accuracy_mean
        and roc_auc_mean is not None
        and roc_auc_mean >= min_roc_auc_mean
        and beats_baseline_fold_ratio >= min_beats_baseline_fold_ratio
    )

    metrics["validation_thresholds"] = {
        "min_rolling_folds": min_rolling_folds,
        "min_balanced_accuracy_mean": min_balanced_accuracy_mean,
        "min_roc_auc_mean": min_roc_auc_mean,
        "min_beats_baseline_fold_ratio": min_beats_baseline_fold_ratio,
    }

    return {
        "validation_method": "purged_rolling_walk_forward",
        "rolling_config": rolling_config,
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "model_valid": model_valid,
    }


def predict_direction_with_lgbm(
    df: pd.DataFrame,
    predict_len: int = 10,
) -> dict[str, Any]:
    if df.empty:
        return direction_fallback(df, predict_len, "Not enough data for direction classification")
    if "Close" not in df.columns:
        return direction_fallback(df, predict_len, "Close column is required for direction classification")

    min_samples = DIRECTION_CONFIG.get("min_samples", 120)
    if len(df) < min_samples:
        return direction_fallback(df, predict_len, "Not enough data for direction classification")

    feature_df, feature_cols = _build_direction_features(df, predict_len)
    if not feature_cols or feature_df.empty:
        return direction_fallback(
            df,
            predict_len,
            "Not enough valid feature rows for direction classification",
        )

    latest_features = feature_df[feature_cols].iloc[[-1]]
    if latest_features.isna().any().any():
        return direction_fallback(
            df,
            predict_len,
            "Latest feature row contains NaN values",
        )

    training_df = feature_df[feature_df["target"].notna()].copy()
    if len(training_df) < min_samples:
        return direction_fallback(df, predict_len, "Not enough data for direction classification")

    y_all = training_df["target"].astype(int)
    if y_all.nunique() < 2:
        return direction_fallback(
            df,
            predict_len,
            "Training data contains only one direction class",
        )

    rolling_validation = validate_direction_with_rolling_window(
        training_df,
        feature_cols,
        predict_len,
    )
    rolling_config = rolling_validation["rolling_config"]
    train_size = int(rolling_config["train_size"])
    purge_size = int(rolling_config["purge_size"])
    test_size = int(rolling_config["test_size"])

    if len(training_df) < train_size:
        return direction_fallback(
            df,
            predict_len,
            "Not enough data for final rolling training window",
        )
    if len(training_df) < train_size + purge_size + test_size:
        return direction_fallback(
            df,
            predict_len,
            "Not enough data for rolling validation",
        )

    final_train_df = training_df.iloc[-train_size:].copy()
    try:
        final_fit = _fit_direction_model_with_threshold(
            final_train_df,
            feature_cols,
            float(rolling_config["calibration_ratio"]),
            str(rolling_config["threshold_metric"]),
        )
    except ValueError as exc:
        return direction_fallback(df, predict_len, str(exc))

    model_full = final_fit["model"]
    optimal_threshold = float(final_fit["decision_threshold"])
    full_model_backend = final_fit["model_backend"]

    latest_probabilities = model_full.predict_proba(latest_features)[0]
    prob_up = float(latest_probabilities[1])
    prob_down = float(1.0 - prob_up)

    buy_threshold = DIRECTION_CONFIG.get("buy_probability_threshold", 0.60)
    sell_threshold = DIRECTION_CONFIG.get("sell_probability_threshold", 0.40)
    model_valid = bool(rolling_validation["model_valid"])

    if model_valid:
        direction = "UP" if prob_up >= optimal_threshold else "DOWN"
        if prob_up >= buy_threshold:
            signal = "BUY"
        elif prob_up <= sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"
    else:
        direction = "MODEL_INVALID"
        signal = "HOLD"

    split_importances = np.asarray(getattr(model_full, "feature_importances_", np.zeros(len(feature_cols))))
    if hasattr(model_full, "booster_"):
        gain_importances = model_full.booster_.feature_importance(importance_type="gain")
    else:
        gain_importances = split_importances.astype(float)
    total_gain = float(np.sum(gain_importances))
    sorted_importances = sorted(
        zip(feature_cols, split_importances, gain_importances),
        key=lambda item: item[2],
        reverse=True,
    )
    top_features = [
        {
            "feature": feature,
            "split_importance": int(split_importance),
            "gain_importance": round(float(gain_importance), 4),
            "gain_share": round(float(gain_importance) / total_gain, 4) if total_gain > 0 else 0.0,
        }
        for feature, split_importance, gain_importance in sorted_importances[:10]
    ]

    last_close = float(feature_df["Close"].iloc[-1])
    last_date = format_last_date(feature_df.index[-1])
    metrics = rolling_validation["metrics"]
    fold_metrics = rolling_validation["fold_metrics"]

    print("\n--- Direction Rolling Validation Metrics ---")
    print(f"Model Backend: {full_model_backend}")
    if final_fit["model_warning"]:
        print(f"Model Warning: {final_fit['model_warning']}")
    print(f"Horizon: {predict_len} business days")
    print(f"Validation Method: {rolling_validation['validation_method']}")
    print(f"Rolling Config: {rolling_config}")
    print(f"Model Valid: {model_valid}")
    print(f"Fold Count: {metrics['fold_count']} / {metrics['total_fold_count']}")
    print(f"Balanced Accuracy Mean: {metrics['balanced_accuracy']['mean']}")
    print(f"ROC-AUC Mean: {metrics['roc_auc']['mean']}")
    print(f"Beats Baseline Fold Ratio: {metrics['beats_baseline_fold_ratio']}")
    print(f"Decision Threshold (Final Window): {_round_float(optimal_threshold)}")
    print("\n--- Fold Details ---")
    for fold in fold_metrics:
        print(
            f"Fold {fold['fold_number']}: "
            f"{fold['test_start_date']} to {fold['test_end_date']} | "
            f"Acc={fold['accuracy']} | "
            f"BalAcc={fold['balanced_accuracy']} | "
            f"AUC={fold['roc_auc']} | "
            f"Baseline={fold['baseline_accuracy']} | "
            f"Threshold={fold['decision_threshold']} | "
            f"CM={fold['confusion_matrix']}"
        )
    print("-------------------------------------------------")

    return {
        "model": full_model_backend,
        "horizon_days": predict_len,
        "last_date": last_date,
        "last_close": round(last_close, 4),
        "probability_up": round(prob_up, 4),
        "probability_down": round(prob_down, 4),
        "decision_threshold": round(float(optimal_threshold), 4),
        "predicted_direction": direction,
        "signal": signal,
        "target_return_threshold": DIRECTION_CONFIG.get("target_return_threshold", 0.0),
        "buy_probability_threshold": buy_threshold,
        "sell_probability_threshold": sell_threshold,
        "validation_method": rolling_validation["validation_method"],
        "rolling_config": rolling_config,
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "model_valid": model_valid,
        "top_features": top_features,
    }

def main():
    ticker = "^N225"
    stock_ticker = ticker.upper()

    try:
        stock, _ = get_stock_data(stock_ticker)
        if stock is None:
            print("stock is None")
            exit()
        predict_len_value = 10
        

        df = stock.history(period="5y")
        
        direction_response = predict_direction_with_lgbm(
            df, 
            predict_len=predict_len_value,
        )
    except Exception as exc:
        print(exc)

if __name__ == "__main__":
    main()
