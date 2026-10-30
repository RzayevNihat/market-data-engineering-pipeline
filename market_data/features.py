"""Feature engineering and a lagged baseline backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import (
    DEFAULT_ONE_WAY_FEE_RATE,
    apply_transaction_costs,
)


FEE_RATE = DEFAULT_ONE_WAY_FEE_RATE


def add_features(
    frame: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 60,
) -> pd.DataFrame:
    """Calculate features and a long-only baseline strategy."""

    if fast_window <= 0 or slow_window <= fast_window:
        raise ValueError("Require 0 < fast_window < slow_window")

    if frame.empty:
        raise ValueError("Cannot calculate features on empty data")

    required = {"open_time", "close"}
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = (
        frame.sort_values("open_time")
        .copy()
        .reset_index(drop=True)
    )

    close = pd.to_numeric(
        data["close"],
        errors="coerce",
    ).astype("float64")

    close_values = close.to_numpy(dtype=np.float64)

    if not np.isfinite(close_values).all():
        raise ValueError("Close prices must be finite")

    if (close_values <= 0).any():
        raise ValueError("Close prices must be positive")

    data["close"] = close

    # Price returns.
    data["return_1"] = close.pct_change(fill_method=None)
    data["log_return_1"] = np.log(close).diff()

    # Moving averages.
    data[f"sma_{fast_window}"] = close.rolling(
        fast_window,
        min_periods=fast_window,
    ).mean()

    data[f"sma_{slow_window}"] = close.rolling(
        slow_window,
        min_periods=slow_window,
    ).mean()

    # Rolling volatility.
    data["volatility_60"] = (
        data["log_return_1"]
        .rolling(60, min_periods=60)
        .std()
    )

    # Generate signal at candle close.
    signal_at_close = (
        data[f"sma_{fast_window}"]
        > data[f"sma_{slow_window}"]
    ).astype("int8")

    # Apply signal on the next candle.
    data["position"] = (
        signal_at_close
        .shift(1)
        .fillna(0)
        .astype("int8")
    )

    positions = data["position"].to_numpy(
        dtype=np.int8
    )

    gross_values = (
        positions.astype(np.float64)
        * data["return_1"].fillna(0.0).to_numpy(
            dtype=np.float64
        )
    )

    # Actual position changes, excluding final liquidation.
    turnover = np.abs(
        np.diff(np.concatenate(([0], positions)))
    ).astype(np.int64)

    # Shared transaction-cost implementation.
    net_values, fee_events = apply_transaction_costs(
        gross_returns=gross_values,
        position=positions,
        fee_rate=FEE_RATE,
        liquidate_at_end=True,
    )

    data["turnover"] = turnover
    data["fee_events"] = fee_events

    data["gross_strategy_return"] = gross_values
    data["estimated_cost"] = fee_events * FEE_RATE
    data["net_strategy_return"] = net_values

    # Compound net returns.
    data["equity_curve"] = (
        1.0 + data["net_strategy_return"]
    ).cumprod()

    data["drawdown"] = (
        data["equity_curve"]
        / data["equity_curve"].cummax()
        - 1.0
    )

    return data


def summarize_backtest(features: pd.DataFrame) -> dict:
    """Generate baseline backtest summary metrics."""

    if features.empty:
        raise ValueError("Cannot summarize an empty DataFrame")

    required = {
        "net_strategy_return",
        "equity_curve",
        "drawdown",
        "position",
    }

    missing = required.difference(features.columns)

    if missing:
        raise ValueError(
            f"Missing backtest columns: {sorted(missing)}"
        )

    returns = features["net_strategy_return"].dropna()
    equity = features["equity_curve"].dropna()

    if returns.empty or equity.empty:
        raise ValueError(
            "Not enough rows to calculate backtest results"
        )

    fee_events = (
        features["fee_events"].sum()
        if "fee_events" in features.columns
        else features["turnover"].sum()
    )

    return {
        "bars": int(len(features)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(features["drawdown"].min()),
        "positive_return_bars": int((returns > 0).sum()),
        "negative_return_bars": int((returns < 0).sum()),
        "assumed_one_way_cost_bps": 5,
        "cost_model": "multiplicative_per_position_change",
        "transaction_cost_events": int(fee_events),
        "terminal_liquidation": bool(
            features["position"].iloc[-1] == 1
        ),
        "strategy_note": (
            "Educational moving-average example only; "
            "not an investment strategy or trading recommendation."
        ),
        "lookahead_note": (
            "Signal is shifted by one candle before "
            "it is applied to returns."
        ),
    }
