"""Parquet storage and DuckDB analysis helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .settings import PROCESSED_DIR, REPORTS_DIR


def parquet_path(symbol: str = "BTCUSDT", interval: str = "1m") -> Path:
    return PROCESSED_DIR / f"{symbol.upper()}_{interval}.parquet"


def features_path(symbol: str = "BTCUSDT", interval: str = "1m") -> Path:
    return PROCESSED_DIR / f"{symbol.upper()}_{interval}_features.parquet"


def save_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="snappy")
    return path


def save_json(data: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _normalize_result_value(value):
    """Convert timestamps to UTC and NumPy scalars to Python values."""

    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)

        return value.isoformat()

    if hasattr(value, "item"):
        return value.item()

    return value


def analyze_with_duckdb(path: Path) -> dict:
    """Run SQL directly against Parquet without loading the entire file in pandas."""

    import duckdb

    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    connection = duckdb.connect(database=":memory:")

    try:
        connection.execute("SET TimeZone = 'UTC'")

        query = """
            SELECT
                COUNT(*) AS candle_count,
                MIN(open_time) AS first_candle_utc,
                MAX(open_time) AS last_candle_utc,
                AVG(close) AS average_close,
                STDDEV_SAMP(close) AS close_stddev,
                MIN(low) AS period_low,
                MAX(high) AS period_high,
                SUM(volume) AS total_base_volume,
                SUM(number_of_trades) AS total_trades
            FROM read_parquet(?)
        """

        row = connection.execute(
            query,
            [str(path)],
        ).fetchone()

        columns = [
            item[0] for item in connection.description
        ]

        result = dict(zip(columns, row))

        for key, value in list(result.items()):
            result[key] = _normalize_result_value(value)

        return result

    finally:
        connection.close()
