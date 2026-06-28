from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DIRECTION_CONFIG
from services import get_stock_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direction classification data diagnostics.",
    )
    parser.add_argument("--ticker", default="^GSPC", help="Ticker symbol")
    parser.add_argument(
        "--predict-len",
        type=int,
        default=10,
        help="Prediction horizon in business days",
    )
    parser.add_argument(
        "--period",
        default="10y",
        help="yfinance history period, e.g. 2y, 5y, 10y, max",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=252,
        help="Rolling window for class balance plots",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Histogram bins",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional custom output directory",
    )
    return parser.parse_args()


def safe_ticker_name(ticker: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in ticker.upper()).strip("_") or "TICKER"


def load_history(ticker: str, period: str) -> pd.DataFrame:
    stock, _ = get_stock_data(ticker)
    if stock is None:
        raise ValueError(f"Ticker not found: {ticker}")

    df = stock.history(period=period)
    if df.empty:
        raise ValueError(f"No history returned for {ticker} with period={period}")

    if "Close" not in df.columns:
        raise ValueError("Close column is missing from downloaded history")

    return df.sort_index()


def build_analysis_frame(df: pd.DataFrame, predict_len: int, threshold: float) -> pd.DataFrame:
    analysis_df = df.copy().sort_index()
    analysis_df["future_close"] = analysis_df["Close"].shift(-predict_len)
    analysis_df["future_log_return"] = np.log(analysis_df["future_close"] / analysis_df["Close"])
    analysis_df["future_return"] = analysis_df["future_close"] / analysis_df["Close"] - 1
    analysis_df["future_return_pct"] = analysis_df["future_return"] * 100.0
    analysis_df["is_up_zero"] = (analysis_df["future_log_return"] > 0.0).astype(float)
    analysis_df["is_up_threshold"] = (analysis_df["future_log_return"] > threshold).astype(float)
    analysis_df["between_zero_and_threshold"] = (
        (analysis_df["future_log_return"] > 0.0)
        & (analysis_df["future_log_return"] <= threshold)
    ).astype(float)
    analysis_df["year"] = pd.to_datetime(analysis_df.index).year
    analysis_df["month"] = pd.to_datetime(analysis_df.index).to_period("M").astype(str)
    analysis_df = analysis_df.dropna(
        subset=["future_close", "future_log_return", "future_return", "future_return_pct"]
    ).copy()
    return analysis_df


def build_summary_frame(analysis_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    future_log_return = analysis_df["future_log_return"]
    future_return = analysis_df["future_return"]
    summary_items = [
        ("sample_count", int(len(analysis_df))),
        ("start_date", str(analysis_df.index.min().date())),
        ("end_date", str(analysis_df.index.max().date())),
        ("mean_future_log_return", float(future_log_return.mean())),
        ("median_future_log_return", float(future_log_return.median())),
        ("std_future_log_return", float(future_log_return.std())),
        ("min_future_log_return", float(future_log_return.min())),
        ("max_future_log_return", float(future_log_return.max())),
        ("mean_future_return", float(future_return.mean())),
        ("median_future_return", float(future_return.median())),
        ("std_future_return", float(future_return.std())),
        ("share_up_zero", float((future_log_return > 0.0).mean())),
        ("share_up_threshold", float((future_log_return > threshold).mean())),
        (
            "share_between_zero_and_threshold",
            float(((future_log_return > 0.0) & (future_log_return <= threshold)).mean()),
        ),
        ("q01_future_return", float(future_return.quantile(0.01))),
        ("q05_future_return", float(future_return.quantile(0.05))),
        ("q25_future_return", float(future_return.quantile(0.25))),
        ("q50_future_return", float(future_return.quantile(0.50))),
        ("q75_future_return", float(future_return.quantile(0.75))),
        ("q95_future_return", float(future_return.quantile(0.95))),
        ("q99_future_return", float(future_return.quantile(0.99))),
        ("skew_future_return", float(future_return.skew())),
        ("threshold_log_return", float(threshold)),
        ("threshold_simple_return_approx", float(np.expm1(threshold))),
    ]
    return pd.DataFrame(summary_items, columns=["metric", "value"])


def build_yearly_summary_frame(analysis_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    grouped = analysis_df.groupby("year")
    return (
        grouped.apply(
            lambda frame: pd.Series(
                {
                    "sample_count": int(len(frame)),
                    "mean_future_return": float(frame["future_return"].mean()),
                    "median_future_return": float(frame["future_return"].median()),
                    "std_future_return": float(frame["future_return"].std()),
                    "share_up_zero": float((frame["future_log_return"] > 0.0).mean()),
                    "share_up_threshold": float(
                        (frame["future_log_return"] > threshold).mean()
                    ),
                    "share_between_zero_and_threshold": float(
                        (
                            (frame["future_log_return"] > 0.0)
                            & (frame["future_log_return"] <= threshold)
                        ).mean()
                    ),
                }
            )
        )
        .reset_index()
    )


def build_monthly_summary_frame(analysis_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    grouped = analysis_df.groupby("month")
    return (
        grouped.apply(
            lambda frame: pd.Series(
                {
                    "sample_count": int(len(frame)),
                    "mean_future_return": float(frame["future_return"].mean()),
                    "share_up_zero": float((frame["future_log_return"] > 0.0).mean()),
                    "share_up_threshold": float(
                        (frame["future_log_return"] > threshold).mean()
                    ),
                }
            )
        )
        .reset_index()
    )


def save_histogram(
    analysis_df: pd.DataFrame,
    threshold: float,
    bins: int,
    output_path: Path,
) -> None:
    future_return_pct = analysis_df["future_return_pct"]
    threshold_pct = np.expm1(threshold) * 100.0

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(future_return_pct, bins=bins, color="#3B82F6", alpha=0.80, edgecolor="white")
    ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.5, label="0%")
    ax.axvline(
        threshold_pct,
        color="#DC2626",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold {threshold_pct:.2f}%",
    )
    ax.axvline(
        float(future_return_pct.mean()),
        color="#059669",
        linestyle="-",
        linewidth=1.5,
        label=f"Mean {future_return_pct.mean():.2f}%",
    )
    ax.set_title("Future Return Histogram")
    ax.set_xlabel("Future return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_return_timeseries(
    analysis_df: pd.DataFrame,
    threshold: float,
    output_path: Path,
) -> None:
    threshold_pct = np.expm1(threshold) * 100.0

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(
        analysis_df.index,
        analysis_df["future_return_pct"],
        color="#2563EB",
        linewidth=1.0,
        label="Future return (%)",
    )
    ax.axhline(0.0, color="#111827", linestyle="--", linewidth=1.2, label="0%")
    ax.axhline(
        threshold_pct,
        color="#DC2626",
        linestyle="--",
        linewidth=1.2,
        label=f"Threshold {threshold_pct:.2f}%",
    )
    ax.set_title("Future Return Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Future return (%)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_rolling_class_balance(
    analysis_df: pd.DataFrame,
    threshold: float,
    rolling_window: int,
    output_path: Path,
) -> None:
    rolling_up_zero = analysis_df["is_up_zero"].rolling(rolling_window).mean()
    rolling_up_threshold = analysis_df["is_up_threshold"].rolling(rolling_window).mean()
    rolling_between = analysis_df["between_zero_and_threshold"].rolling(rolling_window).mean()

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(
        analysis_df.index,
        rolling_up_zero,
        color="#2563EB",
        linewidth=1.3,
        label="P(future_log_return > 0)",
    )
    ax.plot(
        analysis_df.index,
        rolling_up_threshold,
        color="#DC2626",
        linewidth=1.3,
        label=f"P(future_log_return > {threshold:.4f})",
    )
    ax.plot(
        analysis_df.index,
        rolling_between,
        color="#059669",
        linewidth=1.3,
        label="P(0 < return <= threshold)",
    )
    ax.axhline(0.5, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_title(f"Rolling Class Balance ({rolling_window} samples)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_yearly_class_balance(
    yearly_summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    if yearly_summary_df.empty:
        return

    x = np.arange(len(yearly_summary_df))
    width = 0.26

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(
        x - width,
        yearly_summary_df["share_up_zero"],
        width=width,
        color="#2563EB",
        label="share_up_zero",
    )
    ax.bar(
        x,
        yearly_summary_df["share_up_threshold"],
        width=width,
        color="#DC2626",
        label="share_up_threshold",
    )
    ax.bar(
        x + width,
        yearly_summary_df["share_between_zero_and_threshold"],
        width=width,
        color="#059669",
        label="between_zero_and_threshold",
    )
    ax.axhline(0.5, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(yearly_summary_df["year"].astype(str).tolist(), rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Yearly Class Balance")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rate")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_markdown_report(
    ticker: str,
    predict_len: int,
    period: str,
    rolling_window: int,
    threshold: float,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> str:
    summary_rows = "\n".join(
        f"| {row.metric} | {row.value} |"
        for row in summary_df.itertuples(index=False)
    )
    threshold_pct = np.expm1(threshold) * 100.0
    return "\n".join(
        [
            f"# Data Check: {ticker}",
            "",
            "## Configuration",
            "",
            "| Item | Value |",
            "| --- | --- |",
            f"| ticker | {ticker} |",
            f"| predict_len | {predict_len} |",
            f"| period | {period} |",
            f"| rolling_window | {rolling_window} |",
            f"| threshold_log_return | {threshold:.6f} |",
            f"| threshold_simple_return_approx | {threshold_pct:.4f}% |",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            summary_rows,
            "",
            "## Output Files",
            "",
            f"- `{output_dir / 'future_return_dataset.csv'}`",
            f"- `{output_dir / 'summary_metrics.csv'}`",
            f"- `{output_dir / 'yearly_summary.csv'}`",
            f"- `{output_dir / 'monthly_summary.csv'}`",
            f"- `{output_dir / 'future_return_histogram.png'}`",
            f"- `{output_dir / 'future_return_timeseries.png'}`",
            f"- `{output_dir / 'rolling_class_balance.png'}`",
            f"- `{output_dir / 'yearly_class_balance.png'}`",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()
    threshold = float(DIRECTION_CONFIG.get("target_return_threshold", 0.0))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            PROJECT_ROOT
            / "Analysis"
            / "output"
            / f"{safe_ticker_name(ticker)}_h{args.predict_len}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_history(ticker, args.period)
    analysis_df = build_analysis_frame(df, args.predict_len, threshold)
    if analysis_df.empty:
        raise ValueError("No valid rows after building future return analysis frame")

    summary_df = build_summary_frame(analysis_df, threshold)
    yearly_summary_df = build_yearly_summary_frame(analysis_df, threshold)
    monthly_summary_df = build_monthly_summary_frame(analysis_df, threshold)

    analysis_df.to_csv(output_dir / "future_return_dataset.csv", encoding="utf-8")
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False, encoding="utf-8")
    yearly_summary_df.to_csv(output_dir / "yearly_summary.csv", index=False, encoding="utf-8")
    monthly_summary_df.to_csv(output_dir / "monthly_summary.csv", index=False, encoding="utf-8")

    save_histogram(
        analysis_df,
        threshold,
        args.bins,
        output_dir / "future_return_histogram.png",
    )
    save_return_timeseries(
        analysis_df,
        threshold,
        output_dir / "future_return_timeseries.png",
    )
    save_rolling_class_balance(
        analysis_df,
        threshold,
        args.rolling_window,
        output_dir / "rolling_class_balance.png",
    )
    save_yearly_class_balance(
        yearly_summary_df,
        output_dir / "yearly_class_balance.png",
    )

    report_text = build_markdown_report(
        ticker=ticker,
        predict_len=args.predict_len,
        period=args.period,
        rolling_window=args.rolling_window,
        threshold=threshold,
        summary_df=summary_df,
        output_dir=output_dir,
    )
    (output_dir / "README.md").write_text(report_text, encoding="utf-8")

    print(f"Saved analysis outputs to: {output_dir}")


if __name__ == "__main__":
    main()
