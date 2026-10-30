"""Expanding-window walk-forward research for BTCUSDT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .costs import (
    DEFAULT_ONE_WAY_FEE_RATE,
    apply_transaction_costs,
)


DATA_PATH = Path("data/processed/BTCUSDT_1m.parquet")
REPORT_PATH = Path("reports/walk_forward_30_120_confirm10.csv")

FAST_WINDOW = 30
SLOW_WINDOW = 120
CONFIRMATION_BARS = 10

FEE_RATE = DEFAULT_ONE_WAY_FEE_RATE

BARS_PER_DAY = 1440
TRAIN_WINDOWS_DAYS = [60, 90, 120, 150]
TEST_WINDOW_DAYS = 30


def build_positions(
    entry_signal: np.ndarray,
    exit_signal: np.ndarray,
    confirmation: int,
) -> np.ndarray:
    """Build confirmed long-only signals.

    Every test window starts flat.
    Signals generated at candle close are applied on the next candle.
    """

    if confirmation < 1:
        raise ValueError("confirmation must be >= 1")

    entry_signal = np.asarray(
        entry_signal,
        dtype=bool,
    )

    exit_signal = np.asarray(
        exit_signal,
        dtype=bool,
    )

    if entry_signal.ndim != 1 or exit_signal.ndim != 1:
        raise ValueError("Signals must be one-dimensional")

    if len(entry_signal) != len(exit_signal):
        raise ValueError(
            "Entry and exit signals must have equal lengths"
        )

    n = len(entry_signal)

    state = 0
    entry_count = 0
    exit_count = 0

    state_at_close = np.zeros(n, dtype=np.int8)

    for i in range(n):

        if state == 0:

            if entry_signal[i]:
                entry_count += 1
            else:
                entry_count = 0

            if entry_count >= confirmation:
                state = 1
                entry_count = 0
                exit_count = 0

        else:

            if exit_signal[i]:
                exit_count += 1
            else:
                exit_count = 0

            if exit_count >= confirmation:
                state = 0
                entry_count = 0
                exit_count = 0

        state_at_close[i] = state

    positions = np.zeros(n, dtype=np.int8)

    if n > 1:
        positions[1:] = state_at_close[:-1]

    return positions


def evaluate_test_window(
    close: np.ndarray,
    returns: np.ndarray,
    position: np.ndarray,
    start: int,
    end: int,
) -> dict:
    """Evaluate an independent flat-start test window.

    Uses the shared multiplicative transaction-cost engine.
    An open position is liquidated at the final test close.
    """

    close = np.asarray(close, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)

    position = np.asarray(
        position,
        dtype=np.int8,
    )

    if close.ndim != 1 or returns.ndim != 1:
        raise ValueError("Close and returns must be one-dimensional")

    if position.ndim != 1:
        raise ValueError("Position must be one-dimensional")

    if start <= 0 or end > len(close) or end <= start:
        raise ValueError("Invalid test window")

    if len(returns) != len(close):
        raise ValueError("Close and returns lengths differ")

    if len(position) != end - start:
        raise ValueError("Position length does not match test window")

    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Close prices must be finite and positive")

    if not np.isfinite(returns).all():
        raise ValueError("Returns must be finite")

    if not np.isin(position, [0, 1]).all():
        raise ValueError("Only positions 0 and 1 are supported")

    if np.any(returns <= -1):
        raise ValueError("Returns must be greater than -100%")

    if len(position) == 0:
        raise ValueError("Test position array is empty")

    period_returns = returns[start:end]
    period_gross = position * period_returns

    # Shared cost model; flat-start and forced final liquidation.
    period_net, fee_events = apply_transaction_costs(
        gross_returns=period_gross,
        position=position,
        fee_rate=FEE_RATE,
        liquidate_at_end=True,
    )

    net_factors = 1.0 + period_net

    equity = np.concatenate(
        ([1.0], np.cumprod(net_factors))
    )

    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0

    gross_return = np.prod(
        1.0 + period_gross
    ) - 1.0

    net_return = equity[-1] - 1.0

    # Buy & Hold enters at the first test close.
    # Both entry and exit fees are charged.
    buy_hold_gross = (
        close[end - 1] / close[start] - 1.0
    )

    buy_hold_net = (
        (1.0 + buy_hold_gross)
        * (1.0 - FEE_RATE) ** 2
        - 1.0
    )

    position_changes = np.diff(
        np.concatenate(([0], position))
    )

    forced_exit = int(position[-1] == 1)

    entries = int(
        np.sum(position_changes == 1)
    )

    exits = (
        int(np.sum(position_changes == -1))
        + forced_exit
    )

    return {
        "gross_return": float(gross_return),
        "net_return": float(net_return),
        "max_drawdown": float(drawdowns.min()),
        "buy_hold_gross": float(buy_hold_gross),
        "buy_hold_net": float(buy_hold_net),
        "exposure": float(position.mean()),
        "entries": entries,
        "exits": exits,
        "round_trips": min(entries, exits),
        "turnover_events": int(fee_events.sum()),
        "simple_fee_sum": float(
            fee_events.sum() * FEE_RATE
        ),
        "ending_equity": float(equity[-1]),
    }


def main() -> None:

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    required = {"open_time", "close"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Dataset missing columns: {sorted(missing)}"
        )

    df = (
        df.sort_values("open_time")
        .reset_index(drop=True)
    )

    if df["open_time"].isna().any():
        raise ValueError("Dataset contains invalid timestamps")

    if df["open_time"].duplicated().any():
        raise ValueError("Dataset contains duplicate timestamps")

    close_series = pd.to_numeric(
        df["close"],
        errors="coerce",
    ).astype("float64")

    close = close_series.to_numpy()

    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError(
            "Close prices must be finite and positive"
        )

    returns = (
        close_series.pct_change(fill_method=None)
        .fillna(0.0)
        .to_numpy()
    )

    n = len(df)

    # Indicators use trailing historical observations only.
    fast_sma = close_series.rolling(
        FAST_WINDOW,
        min_periods=FAST_WINDOW,
    ).mean()

    slow_sma = close_series.rolling(
        SLOW_WINDOW,
        min_periods=SLOW_WINDOW,
    ).mean()

    log_returns = np.log(close_series).diff()

    volatility = (
        log_returns
        .rolling(60, min_periods=60)
        .std()
        .to_numpy()
    )

    valid_sma = (
        fast_sma.notna() & slow_sma.notna()
    ).to_numpy()

    bullish = (
        (fast_sma > slow_sma).to_numpy()
        & valid_sma
    )

    bearish = (
        (fast_sma <= slow_sma).to_numpy()
        & valid_sma
    )

    results = []

    expected_test_rows = (
        TEST_WINDOW_DAYS * BARS_PER_DAY + 1
    )

    for fold, train_days in enumerate(
        TRAIN_WINDOWS_DAYS,
        start=1,
    ):

        train_end = train_days * BARS_PER_DAY
        test_start = train_end

        # +1 includes both test-window boundary candles.
        test_end = min(
            test_start + expected_test_rows,
            n,
        )

        if test_end - test_start != expected_test_rows:
            raise ValueError(
                f"Insufficient data for complete test window "
                f"in fold {fold}"
            )

        # Training-only volatility threshold.
        train_volatility = volatility[:train_end]

        train_volatility = train_volatility[
            np.isfinite(train_volatility)
        ]

        if len(train_volatility) == 0:
            raise ValueError(
                f"No valid training volatility in fold {fold}"
            )

        q50_threshold = float(
            np.quantile(train_volatility, 0.50)
        )

        entry_no_filter = bullish[test_start:test_end]

        test_volatility = volatility[test_start:test_end]

        entry_q50 = (
            bullish[test_start:test_end]
            & np.isfinite(test_volatility)
            & (test_volatility <= q50_threshold)
        )

        exit_signal = bearish[test_start:test_end]

        variants = [
            ("no_filter", entry_no_filter),
            ("q50_train_only", entry_q50),
        ]

        for variant_name, entry_signal in variants:

            # Every test fold starts with a flat position.
            position = build_positions(
                entry_signal,
                exit_signal,
                CONFIRMATION_BARS,
            )

            metrics = evaluate_test_window(
                close=close,
                returns=returns,
                position=position,
                start=test_start,
                end=test_end,
            )

            results.append({
                "fold": fold,
                "train_days": train_days,
                "test_days": TEST_WINDOW_DAYS,
                "train_end_utc": str(
                    df["open_time"].iloc[train_end - 1]
                ),
                "test_start_utc": str(
                    df["open_time"].iloc[test_start]
                ),
                "test_end_utc": str(
                    df["open_time"].iloc[test_end - 1]
                ),
                "variant": variant_name,
                "fast_sma": FAST_WINDOW,
                "slow_sma": SLOW_WINDOW,
                "confirmation_bars": CONFIRMATION_BARS,
                "training_q50_volatility": q50_threshold,
                "previously_explored_window": fold == 4,
                **metrics,
            })

    result_df = pd.DataFrame(results)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        REPORT_PATH,
        index=False,
    )

    display_columns = [
        "fold",
        "train_days",
        "variant",
        "gross_return",
        "net_return",
        "max_drawdown",
        "buy_hold_net",
        "exposure",
        "round_trips",
        "turnover_events",
        "previously_explored_window",
    ]

    print("\n--- EXPANDING WALK-FORWARD RESULTS ---")

    print(
        result_df[display_columns].to_string(
            index=False,
            formatters={
                "gross_return": "{:.2%}".format,
                "net_return": "{:.2%}".format,
                "max_drawdown": "{:.2%}".format,
                "buy_hold_net": "{:.2%}".format,
                "exposure": "{:.2%}".format,
            },
        )
    )

    print("\n--- AGGREGATED TEST RETURNS ---")

    for variant, group in result_df.groupby("variant"):

        compounded = (
            np.prod(
                1.0 + group["net_return"].to_numpy()
            )
            - 1.0
        )

        print(
            f"{variant}: "
            f"compounded fold return = {compounded:.2%}"
        )

    print(f"\nSaved report: {REPORT_PATH}")

    print(
        "\nNote: Fold 4 overlaps the previously explored "
        "30-day period and is not an untouched holdout."
    )


if __name__ == "__main__":
    main()
