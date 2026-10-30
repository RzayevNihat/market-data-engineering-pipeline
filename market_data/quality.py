"""Data-quality checks for time-series OHLCV data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .settings import INTERVAL_TO_MS, NUMERIC_COLUMNS


VOLUME_COLUMNS = (
    "volume",
    "quote_asset_volume",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
)

PRICE_COLUMNS = ("open", "high", "low", "close")


def _numeric_array(series: pd.Series) -> np.ndarray:
    """Convert a series to float64, representing invalid values as NaN."""
    return pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )


def validate_ohlcv(
    frame: pd.DataFrame,
    interval: str,
) -> tuple[pd.DataFrame, dict]:
    """Validate, sort and deduplicate time-series OHLCV data.

    Duplicate valid timestamps are removed, keeping the last input row.
    Invalid timestamps are retained and reported.
    Missing candles are reported but never automatically filled.

    Status:
        FAIL    - at least one invalid row remains.
        WARNING - duplicates, ordering issues or timestamp gaps exist.
        PASS    - no detected quality issues.
    """

    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"Unsupported interval: {interval}")

    if frame.empty:
        raise ValueError("Cannot validate an empty DataFrame")

    required = {
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing_columns = sorted(required.difference(frame.columns))

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = frame.copy().reset_index(drop=True)

    # Parse timestamps consistently as UTC.
    # Naive timestamps are interpreted as UTC.
    parsed_times = pd.to_datetime(
        data["open_time"],
        utc=True,
        errors="coerce",
    )

    valid_input_times = parsed_times.notna()
    previous_times = parsed_times.shift(1)

    comparable_pairs = (
        valid_input_times
        & previous_times.notna()
    )

    out_of_order_rows = int(
        (
            comparable_pairs
            & parsed_times.lt(previous_times)
        ).sum()
    )

    # Deduplicate only valid timestamps.
    # Keep NaT rows so malformed records remain visible.
    duplicate_mask = pd.Series(
        False,
        index=data.index,
    )

    duplicate_mask.loc[valid_input_times] = (
        parsed_times.loc[valid_input_times]
        .duplicated(keep="first")
        .to_numpy()
    )

    duplicate_rows = int(duplicate_mask.sum())

    data["open_time"] = parsed_times

    data = (
        data.loc[~duplicate_mask]
        .sort_values(
            "open_time",
            na_position="last",
        )
        .reset_index(drop=True)
    )

    # Convert all available numeric columns.
    numeric_present = [
        column
        for column in NUMERIC_COLUMNS
        if column in data.columns
    ]

    for column in numeric_present:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    numeric_arrays = {
        column: _numeric_array(data[column])
        for column in numeric_present
    }

    finite_mask = np.ones(len(data), dtype=bool)

    for values in numeric_arrays.values():
        finite_mask &= np.isfinite(values)

    non_finite_numeric_rows = int(
        (~finite_mask).sum()
    )

    # Price validation.
    price_arrays = {
        column: numeric_arrays[column]
        for column in PRICE_COLUMNS
    }

    open_values = price_arrays["open"]
    high_values = price_arrays["high"]
    low_values = price_arrays["low"]
    close_values = price_arrays["close"]

    positive_prices = np.ones(
        len(data),
        dtype=bool,
    )

    for values in price_arrays.values():
        positive_prices &= (
            np.isfinite(values)
            & (values > 0)
        )

    # OHLC consistency.
    ohlc_consistent = (
        (high_values >= np.maximum(open_values, close_values))
        & (low_values <= np.minimum(open_values, close_values))
        & (high_values >= low_values)
    )

    invalid_price_or_ohlc_mask = (
        ~(positive_prices & ohlc_consistent)
    )

    invalid_price_or_ohlc_rows = int(
        invalid_price_or_ohlc_mask.sum()
    )

    # All available volume columns must be non-negative.
    negative_volume_mask = np.zeros(
        len(data),
        dtype=bool,
    )

    for column in VOLUME_COLUMNS:
        if column in numeric_arrays:
            values = numeric_arrays[column]

            negative_volume_mask |= (
                np.isfinite(values)
                & (values < 0)
            )

    negative_volume_rows = int(
        negative_volume_mask.sum()
    )

    # Trade counts must be finite, non-negative integers.
    invalid_trade_count_mask = np.zeros(
        len(data),
        dtype=bool,
    )

    if "number_of_trades" in numeric_arrays:
        trades = numeric_arrays["number_of_trades"]

        finite_trades = np.isfinite(trades)

        safe_trades = np.where(
            finite_trades,
            trades,
            0.0,
        )

        invalid_trade_count_mask = (
            ~finite_trades
            | (trades < 0)
            | (safe_trades != np.floor(safe_trades))
        )

    invalid_trade_count_rows = int(
        invalid_trade_count_mask.sum()
    )

    # Timestamp validation.
    invalid_timestamp_mask = (
        data["open_time"]
        .isna()
        .to_numpy(dtype=bool)
    )

    invalid_timestamp_rows = int(
        invalid_timestamp_mask.sum()
    )

    interval_ns = (
        int(INTERVAL_TO_MS[interval])
        * 1_000_000
    )

    valid_row_indices = np.flatnonzero(
        ~invalid_timestamp_mask
    )

    timestamp_ns = (
        data.loc[
            ~invalid_timestamp_mask,
            "open_time",
        ]
        .astype("int64")
        .to_numpy()
    )

    # Every candle must start on its UTC interval boundary.
    unaligned_valid_mask = (
        timestamp_ns % interval_ns != 0
    )

    unaligned_row_mask = np.zeros(
        len(data),
        dtype=bool,
    )

    unaligned_row_mask[
        valid_row_indices
    ] = unaligned_valid_mask

    unaligned_timestamp_rows = int(
        unaligned_row_mask.sum()
    )

    # Missing and irregular timestamp gaps.
    missing_candle_count = 0
    irregular_timestamp_gap_count = 0
    missing_examples = []

    if len(timestamp_ns) > 1:

        for left_ns, right_ns in zip(
            timestamp_ns[:-1],
            timestamp_ns[1:],
        ):

            gap_ns = int(right_ns - left_ns)

            if gap_ns <= 0:
                continue

            # A gap not divisible by the interval is irregular.
            # Do not guess the number of missing candles.
            if gap_ns % interval_ns != 0:
                irregular_timestamp_gap_count += 1
                continue

            if gap_ns > interval_ns:

                missing_count = (
                    gap_ns // interval_ns
                ) - 1

                missing_candle_count += missing_count

                next_missing_ns = (
                    int(left_ns) + interval_ns
                )

                while (
                    next_missing_ns < int(right_ns)
                    and len(missing_examples) < 10
                ):

                    missing_examples.append(
                        pd.to_datetime(
                            next_missing_ns,
                            unit="ns",
                            utc=True,
                        ).isoformat()
                    )

                    next_missing_ns += interval_ns

    # Count each invalid row only once.
    invalid_rows_mask = (
        ~finite_mask
        | invalid_price_or_ohlc_mask
        | negative_volume_mask
        | invalid_trade_count_mask
        | invalid_timestamp_mask
        | unaligned_row_mask
    )

    invalid_rows = int(
        invalid_rows_mask.sum()
    )

    valid_times = data.loc[
        ~invalid_timestamp_mask,
        "open_time",
    ]

    first_time = (
        valid_times.iloc[0].isoformat()
        if not valid_times.empty
        else None
    )

    last_time = (
        valid_times.iloc[-1].isoformat()
        if not valid_times.empty
        else None
    )

    has_warnings = any((
        duplicate_rows > 0,
        out_of_order_rows > 0,
        missing_candle_count > 0,
        irregular_timestamp_gap_count > 0,
    ))

    if invalid_rows > 0:
        status = "FAIL"
    elif has_warnings:
        status = "WARNING"
    else:
        status = "PASS"

    report = {
        "row_count_input": int(len(frame)),
        "row_count_after_deduplication": int(len(data)),
        "duplicate_timestamp_rows": duplicate_rows,
        "out_of_order_rows_before_sort": out_of_order_rows,
        "invalid_timestamp_rows": invalid_timestamp_rows,
        "unaligned_timestamp_rows": unaligned_timestamp_rows,
        "missing_candle_count": int(missing_candle_count),
        "irregular_timestamp_gap_count": (
            irregular_timestamp_gap_count
        ),
        "missing_timestamp_examples_utc": missing_examples,
        "non_finite_numeric_rows": non_finite_numeric_rows,
        "invalid_price_or_ohlc_rows": (
            invalid_price_or_ohlc_rows
        ),
        "negative_volume_rows": negative_volume_rows,
        "invalid_trade_count_rows": (
            invalid_trade_count_rows
        ),
        "invalid_rows": invalid_rows,
        "first_open_time_utc": first_time,
        "last_open_time_utc": last_time,
        "interval": interval,
        "status": status,
    }

    return data, report
