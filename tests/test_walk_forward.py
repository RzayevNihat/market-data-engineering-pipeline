import numpy as np
import pytest

from market_data.walk_forward import (
    build_positions,
    evaluate_test_window,
    FEE_RATE,
)


def test_position_starts_flat_and_entry_is_delayed():
    entry = np.array([True, True, True, False, False])
    exit_signal = np.zeros(5, dtype=bool)

    result = build_positions(entry, exit_signal, 2)

    expected = np.array([0, 0, 1, 1, 1], dtype=np.int8)

    np.testing.assert_array_equal(result, expected)


def test_exit_requires_confirmation_and_is_delayed():
    entry = np.array([True] * 6)
    exit_signal = np.array(
        [False, False, True, True, False, False]
    )

    result = build_positions(entry, exit_signal, 2)

    expected = np.array([0, 0, 1, 1, 0, 0], dtype=np.int8)

    np.testing.assert_array_equal(result, expected)


def test_entry_and_exit_fees_are_applied():
    close = np.array([100.0, 100.0, 110.0, 110.0])
    returns = np.array([0.0, 0.0, 0.10, 0.0])

    # Flat -> Long -> Flat
    position = np.array([0, 1, 0], dtype=np.int8)

    result = evaluate_test_window(
        close, returns, position, start=1, end=4
    )

    expected_net = (
        1.10 * (1 - FEE_RATE) ** 2 - 1
    )

    assert result["gross_return"] == pytest.approx(0.10)
    assert result["net_return"] == pytest.approx(expected_net)
    assert result["entries"] == 1
    assert result["exits"] == 1
    assert result["round_trips"] == 1
    assert result["turnover_events"] == 2


def test_open_position_is_liquidated_at_test_end():
    close = np.array([100.0, 100.0, 110.0, 110.0])
    returns = np.array([0.0, 0.0, 0.10, 0.0])

    # Position remains open until the last candle.
    position = np.array([0, 1, 1], dtype=np.int8)

    result = evaluate_test_window(
        close, returns, position, start=1, end=4
    )

    expected_net = (
        1.10 * (1 - FEE_RATE) ** 2 - 1
    )

    assert result["net_return"] == pytest.approx(expected_net)
    assert result["entries"] == 1
    assert result["exits"] == 1
    assert result["turnover_events"] == 2


def test_flat_strategy_has_no_trading_costs():
    close = np.array([100.0, 100.0, 101.0, 102.0])
    returns = np.array([0.0, 0.0, 0.01, 0.0099])

    position = np.zeros(3, dtype=np.int8)

    result = evaluate_test_window(
        close, returns, position, start=1, end=4
    )

    assert result["net_return"] == pytest.approx(0.0)
    assert result["turnover_events"] == 0
    assert result["entries"] == 0
    assert result["exits"] == 0


def test_confirmation_must_be_positive():
    entry = np.array([True, True])
    exit_signal = np.array([False, False])

    with pytest.raises(ValueError):
        build_positions(entry, exit_signal, 0)


def test_entry_and_exit_signals_must_have_equal_lengths():
    entry = np.array([True, False])
    exit_signal = np.array([False])

    with pytest.raises(ValueError):
        build_positions(entry, exit_signal, 1)

def test_buy_hold_starts_at_first_test_close():
    close = np.array([100.0, 110.0, 121.0, 133.1])
    returns = np.array([0.0, 0.1, 0.1, 0.1])

    position = np.zeros(3, dtype=np.int8)

    result = evaluate_test_window(
        close, returns, position, start=1, end=4
    )

    expected_gross = 133.1 / 110.0 - 1.0

    expected_net = (
        (1 + expected_gross) * (1 - FEE_RATE) ** 2 - 1
    )

    assert result["buy_hold_gross"] == pytest.approx(
        expected_gross
    )

    assert result["buy_hold_net"] == pytest.approx(
        expected_net
    )
