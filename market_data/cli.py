"""Command-line interface for the market-data pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run_benchmark
from .features import add_features, summarize_backtest
from .ingest import download_klines, load_raw_klines
from .quality import validate_ohlcv
from .settings import DEFAULT_DAYS, DEFAULT_INTERVAL, DEFAULT_SYMBOL, RAW_DIR, REPORTS_DIR
from .storage import analyze_with_duckdb, features_path, parquet_path, save_json, save_parquet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Educational OHLCV data engineering pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download raw candles from Binance Spot")
    download_parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    download_parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    download_parser.add_argument("--days", type=int, default=DEFAULT_DAYS)

    validate_parser = subparsers.add_parser("validate", help="Validate raw data and save Parquet")
    validate_parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    validate_parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    validate_parser.add_argument("--raw", type=Path, default=None)

    analyze_parser = subparsers.add_parser("analyze", help="Run DuckDB SQL analytics on Parquet")
    analyze_parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    analyze_parser.add_argument("--interval", default=DEFAULT_INTERVAL)

    backtest_parser = subparsers.add_parser("backtest", help="Build features and run a lagged demo backtest")
    backtest_parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    backtest_parser.add_argument("--interval", default=DEFAULT_INTERVAL)

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark Pandas, NumPy, and Numba")
    benchmark_parser.add_argument("--rows", type=int, default=1_000_000)
    benchmark_parser.add_argument("--window", type=int, default=60)
    benchmark_parser.add_argument("--repeats", type=int, default=5)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    symbol = getattr(args, "symbol", DEFAULT_SYMBOL).upper()
    interval = getattr(args, "interval", DEFAULT_INTERVAL)

    if args.command == "download":
        raw_path, row_count = download_klines(symbol, interval, args.days)
        print(f"Downloaded {row_count:,} candles.")
        print(f"Raw JSON saved to: {raw_path}")
        print("Next step: run the validate command.")

    elif args.command == "validate":
        raw_path = args.raw or (RAW_DIR / f"{symbol}_{interval}.json")
        frame = load_raw_klines(raw_path)
        cleaned, report = validate_ohlcv(frame, interval)
        report_path = REPORTS_DIR / f"{symbol}_{interval}_data_quality.json"
        save_json(report, report_path)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if report["invalid_rows"]:
            raise SystemExit("Data contains invalid rows. Review the quality report; Parquet was not saved.")
        output_path = parquet_path(symbol, interval)
        save_parquet(cleaned, output_path)
        print(f"Cleaned Parquet saved to: {output_path}")
        print(f"Quality report saved to: {report_path}")

    elif args.command == "analyze":
        path = parquet_path(symbol, interval)
        if not path.exists():
            raise SystemExit(f"Parquet file not found: {path}. Run download and validate first.")
        summary = analyze_with_duckdb(path)
        output_path = REPORTS_DIR / f"{symbol}_{interval}_duckdb_summary.json"
        save_json(summary, output_path)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Summary saved to: {output_path}")

    elif args.command == "backtest":
        path = parquet_path(symbol, interval)
        if not path.exists():
            raise SystemExit(f"Parquet file not found: {path}. Run download and validate first.")
        import pandas as pd
        frame = pd.read_parquet(path)
        features = add_features(frame)
        output_path = features_path(symbol, interval)
        save_parquet(features, output_path)
        summary = summarize_backtest(features)
        report_path = REPORTS_DIR / f"{symbol}_{interval}_backtest.json"
        save_json(summary, report_path)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Feature dataset saved to: {output_path}")
        print(f"Backtest report saved to: {report_path}")

    elif args.command == "benchmark":
        results = run_benchmark(args.rows, args.window, args.repeats)
        output_path = REPORTS_DIR / "sma_benchmark.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(results.to_string(index=False))
        print(f"Benchmark saved to: {output_path}")


if __name__ == "__main__":
    main()
