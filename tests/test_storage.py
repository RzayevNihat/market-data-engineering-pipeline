from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from market_data.storage import (
    _normalize_result_value,
    analyze_with_duckdb,
    save_parquet,
)


def test_timezone_aware_datetime_is_normalized_to_utc():
    local_time = datetime(
        2026, 5, 3, 12, 53,
        tzinfo=timezone(timedelta(hours=4)),
    )

    result = _normalize_result_value(local_time)

    assert result == "2026-05-03T08:53:00+00:00"


def test_naive_datetime_is_interpreted_as_utc():
    naive_time = datetime(2026, 5, 3, 8, 53)

    result = _normalize_result_value(naive_time)

    assert result == "2026-05-03T08:53:00+00:00"


def test_pandas_timestamp_is_normalized_to_utc():
    timestamp = pd.Timestamp(
        "2026-05-03T12:53:00+04:00"
    )

    result = _normalize_result_value(timestamp)

    assert result == "2026-05-03T08:53:00+00:00"


def test_duckdb_reads_parquet_and_returns_utc_timestamps(tmp_path):
    frame = pd.DataFrame({
        "open_time": pd.date_range(
            "2026-10-01T08:00:00Z",
            periods=3,
            freq="min",
        ),
        "high": [102.0, 103.0, 104.0],
        "low": [99.0, 100.0, 101.0],
        "close": [101.0, 102.0, 103.0],
        "volume": [10.0, 20.0, 30.0],
        "number_of_trades": [100, 200, 300],
    })

    parquet_file = tmp_path / "sample.parquet"

    save_parquet(frame, parquet_file)

    result = analyze_with_duckdb(parquet_file)

    assert result["candle_count"] == 3
    assert result["first_candle_utc"] == (
        "2026-10-01T08:00:00+00:00"
    )
    assert result["last_candle_utc"] == (
        "2026-10-01T08:02:00+00:00"
    )
    assert result["total_base_volume"] == pytest.approx(60.0)
    assert result["total_trades"] == 600


def test_duckdb_rejects_missing_parquet_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_with_duckdb(
            tmp_path / "missing.parquet"
        )
