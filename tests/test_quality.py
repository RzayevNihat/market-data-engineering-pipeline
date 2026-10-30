import pandas as pd
import pytest

from market_data.quality import validate_ohlcv


def sample_frame():
    """Sample data containing one duplicate and one missing candle."""

    return pd.DataFrame({
        "open_time": pd.to_datetime([
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:01:00Z",
            "2026-01-01T00:01:00Z",
            "2026-01-01T00:03:00Z",
        ], utc=True),

        "open": [100.0, 101.0, 101.0, 103.0],
        "high": [101.0, 102.0, 102.0, 104.0],
        "low": [99.0, 100.0, 100.0, 102.0],
        "close": [100.5, 101.5, 101.5, 103.5],
        "volume": [10.0, 11.0, 11.0, 13.0],
    })


def regular_frame():
    """Three consecutive valid one-minute candles."""

    return pd.DataFrame({
        "open_time": pd.date_range(
            "2026-01-01T00:00:00Z",
            periods=3,
            freq="min",
        ),
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "close": [100.5, 101.5, 102.5],
        "volume": [10.0, 11.0, 12.0],
    })


def test_quality_detects_duplicate_and_missing_minute():
    cleaned, report = validate_ohlcv(
        sample_frame(),
        "1m",
    )

    assert len(cleaned) == 3
    assert report["duplicate_timestamp_rows"] == 1
    assert report["missing_candle_count"] == 1
    assert report["invalid_rows"] == 0
    assert report["status"] == "WARNING"


def test_quality_detects_invalid_ohlc():
    frame = regular_frame()
    frame.loc[0, "high"] = 90.0

    _, report = validate_ohlcv(frame, "1m")

    assert report["invalid_price_or_ohlc_rows"] == 1
    assert report["invalid_rows"] == 1
    assert report["status"] == "FAIL"


def test_quality_detects_unaligned_timestamp_and_irregular_gaps():
    frame = regular_frame()

    frame.loc[1, "open_time"] = pd.Timestamp(
        "2026-01-01T00:01:30Z"
    )

    _, report = validate_ohlcv(frame, "1m")

    assert report["unaligned_timestamp_rows"] == 1
    assert report["irregular_timestamp_gap_count"] == 2
    assert report["missing_candle_count"] == 0
    assert report["invalid_rows"] == 1
    assert report["status"] == "FAIL"


def test_quality_detects_nat_timestamp():
    frame = regular_frame()
    frame.loc[1, "open_time"] = pd.NaT

    _, report = validate_ohlcv(frame, "1m")

    assert report["invalid_timestamp_rows"] == 1
    assert report["invalid_rows"] == 1
    assert report["first_open_time_utc"] is not None
    assert report["last_open_time_utc"] is not None
    assert report["status"] == "FAIL"


def test_quality_detects_negative_quote_volume():
    frame = regular_frame()

    frame["quote_asset_volume"] = [
        1000.0,
        -10.0,
        2000.0,
    ]

    _, report = validate_ohlcv(frame, "1m")

    assert report["negative_volume_rows"] == 1
    assert report["invalid_rows"] == 1
    assert report["status"] == "FAIL"


def test_quality_detects_invalid_trade_counts():
    frame = regular_frame()

    frame["number_of_trades"] = [
        10,
        1.5,
        -1,
    ]

    _, report = validate_ohlcv(frame, "1m")

    assert report["invalid_trade_count_rows"] == 2
    assert report["invalid_rows"] == 2
    assert report["status"] == "FAIL"


def test_quality_detects_unparseable_numeric_values():
    frame = regular_frame()

    frame["volume"] = [
        10.0,
        "not-a-number",
        12.0,
    ]

    _, report = validate_ohlcv(frame, "1m")

    assert report["non_finite_numeric_rows"] == 1
    assert report["invalid_rows"] == 1
    assert report["status"] == "FAIL"


def test_quality_warns_on_out_of_order_input_and_sorts_output():
    frame = (
        regular_frame()
        .iloc[[0, 2, 1]]
        .reset_index(drop=True)
    )

    cleaned, report = validate_ohlcv(frame, "1m")

    assert report["out_of_order_rows_before_sort"] == 1
    assert cleaned["open_time"].is_monotonic_increasing
    assert report["invalid_rows"] == 0
    assert report["status"] == "WARNING"


def test_quality_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        validate_ohlcv(
            pd.DataFrame(),
            "1m",
        )


def test_quality_rejects_missing_required_columns():
    frame = pd.DataFrame({
        "open_time": [
            pd.Timestamp("2026-01-01T00:00:00Z")
        ]
    })

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        validate_ohlcv(frame, "1m")
