"""Shared transaction-cost calculations for backtesting."""

from __future__ import annotations

import numpy as np


DEFAULT_ONE_WAY_FEE_RATE = 0.0005


def apply_transaction_costs(
    gross_returns,
    position,
    fee_rate: float = DEFAULT_ONE_WAY_FEE_RATE,
    liquidate_at_end: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply multiplicative transaction costs to long-only returns.

    Positions:
        0 = flat
        1 = long

    A fee is charged for every position change.
    If requested, a final exit fee is charged when the last
    position remains open.

    Returns:
        net_returns: Net returns after transaction costs.
        fee_events: Number of one-way fee events per candle.
    """

    gross = np.asarray(
        gross_returns,
        dtype=np.float64,
    )

    raw_position = np.asarray(
        position,
        dtype=np.float64,
    )

    fee_rate = float(fee_rate)

    if gross.ndim != 1 or raw_position.ndim != 1:
        raise ValueError("Returns and positions must be one-dimensional")

    if len(gross) == 0:
        raise ValueError("Cannot calculate costs for empty arrays")

    if len(gross) != len(raw_position):
        raise ValueError("Returns and positions must have equal lengths")

    if not np.isfinite(gross).all():
        raise ValueError("Gross returns must be finite")

    if (gross <= -1.0).any():
        raise ValueError("Gross returns must be greater than -100%")

    if not np.isfinite(raw_position).all():
        raise ValueError("Positions must be finite")

    if not np.isin(raw_position, [0.0, 1.0]).all():
        raise ValueError("Only long-only positions 0 and 1 are supported")

    if not np.isfinite(fee_rate):
        raise ValueError("Fee rate must be finite")

    if not 0.0 <= fee_rate < 1.0:
        raise ValueError("Fee rate must be in the range [0, 1)")

    positions = raw_position.astype(np.int8)

    # Initial position is always assumed flat.
    fee_events = np.abs(
        np.diff(
            np.concatenate(([0], positions))
        )
    ).astype(np.int64)

    # Force liquidation at the last candle when requested.
    if liquidate_at_end and positions[-1] == 1:
        fee_events[-1] += 1

    fee_factors = np.power(
        np.float64(1.0 - fee_rate),
        fee_events,
    )

    net_returns = (
        (1.0 + gross) * fee_factors - 1.0
    )

    return net_returns, fee_events
