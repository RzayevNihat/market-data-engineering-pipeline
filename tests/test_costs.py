import numpy as np
import pytest

from market_data.costs import (
    DEFAULT_ONE_WAY_FEE_RATE,
    apply_transaction_costs,
)


FEE = DEFAULT_ONE_WAY_FEE_RATE


def test_entry_and_terminal_exit_fees():
    gross = np.array([0.0, 0.01, 0.02])
    position = np.array([0, 1, 1])

    net, events = apply_transaction_costs(
        gross,
        position,
        liquidate_at_end=True,
    )

    expected_events = np.array([0, 1, 1])

    expected_net = (
        (1.0 + gross)
        * (1.0 - FEE) ** expected_events
        - 1.0
    )

    np.testing.assert_array_equal(
        events,
        expected_events,
    )

    np.testing.assert_allclose(
        net,
        expected_net,
        rtol=1e-12,
        atol=1e-12,
    )


def test_internal_exit_is_charged_once():
    gross = np.array([0.0, 0.01, 0.0])
    position = np.array([0, 1, 0])

    net, events = apply_transaction_costs(
        gross,
        position,
        liquidate_at_end=True,
    )

    np.testing.assert_array_equal(
        events,
        np.array([0, 1, 1]),
    )

    assert net[1] == pytest.approx(
        (1.01 * (1.0 - FEE)) - 1.0
    )

    assert net[2] == pytest.approx(-FEE)


def test_flat_position_has_no_fees():
    gross = np.array([0.0, 0.01, -0.005])
    position = np.array([0, 0, 0])

    net, events = apply_transaction_costs(
        gross,
        position,
    )

    np.testing.assert_array_equal(
        events,
        np.zeros(3, dtype=np.int64),
    )

    np.testing.assert_allclose(net, gross)


def test_terminal_liquidation_can_be_disabled():
    gross = np.array([0.0, 0.01, 0.02])
    position = np.array([0, 1, 1])

    _, events = apply_transaction_costs(
        gross,
        position,
        liquidate_at_end=False,
    )

    np.testing.assert_array_equal(
        events,
        np.array([0, 1, 0]),
    )


def test_mismatched_array_lengths_are_rejected():
    with pytest.raises(ValueError, match="equal lengths"):
        apply_transaction_costs(
            np.array([0.0, 0.01]),
            np.array([0]),
        )


def test_short_positions_are_rejected():
    with pytest.raises(ValueError, match="long-only"):
        apply_transaction_costs(
            np.array([0.0, 0.01]),
            np.array([0, -1]),
        )


def test_invalid_fee_rate_is_rejected():
    with pytest.raises(ValueError, match="Fee rate"):
        apply_transaction_costs(
            np.array([0.0, 0.01]),
            np.array([0, 1]),
            fee_rate=1.0,
        )


def test_shared_cost_engine_is_used_by_features():
    import pandas as pd

    from market_data.features import add_features

    close = np.array(
        [100.0, 101.0, 102.0, 101.0, 100.0]
    )

    frame = pd.DataFrame({
        "open_time": pd.date_range(
            "2026-01-01",
            periods=len(close),
            freq="min",
            tz="UTC",
        ),
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.ones(len(close)),
    })

    features = add_features(
        frame,
        fast_window=1,
        slow_window=2,
    )

    expected_net, expected_events = apply_transaction_costs(
        features["gross_strategy_return"].to_numpy(),
        features["position"].to_numpy(),
        liquidate_at_end=True,
    )

    np.testing.assert_allclose(
        features["net_strategy_return"].to_numpy(),
        expected_net,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_array_equal(
        features["fee_events"].to_numpy(),
        expected_events,
    )
