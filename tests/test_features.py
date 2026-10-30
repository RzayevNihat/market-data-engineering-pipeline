import numpy as np
import pandas as pd
import pytest

from market_data.features import (
    FEE_RATE,
    add_features,
    summarize_backtest,
)


def make_frame(close):
    close = np.asarray(close, dtype=np.float64)

    return pd.DataFrame({
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


def test_signal_is_lagged_one_candle():
    close = np.concatenate([
        np.linspace(100, 90, 80),
        np.linspace(90, 120, 80),
    ])

    result = add_features(
        make_frame(close),
        fast_window=5,
        slow_window=15,
    )

    signal = (
        result["sma_5"] > result["sma_15"]
    ).astype("int8")

    expected_position = (
        signal.shift(1)
        .fillna(0)
        .astype("int8")
    )

    pd.testing.assert_series_equal(
        result["position"],
        expected_position,
        check_names=False,
    )


def test_feature_columns_exist():
    close = np.linspace(100, 110, 100)

    result = add_features(
        make_frame(close),
        fast_window=5,
        slow_window=15,
    )

    expected_columns = {
        "return_1",
        "log_return_1",
        "sma_5",
        "sma_15",
        "volatility_60",
        "position",
        "turnover",
        "gross_strategy_return",
        "estimated_cost",
        "net_strategy_return",
        "equity_curve",
        "drawdown",
    }

    assert expected_columns.issubset(result.columns)


def test_transaction_cost_uses_multiplicative_formula():
    close = [100, 101, 102, 101, 100]

    result = add_features(
        make_frame(close),
        fast_window=1,
        slow_window=2,
    )

    gross_values = result[
        "gross_strategy_return"
    ].to_numpy(dtype=np.float64)

    turnover_values = result[
        "turnover"
    ].to_numpy(dtype=np.float64)

    expected_fee_factor = np.power(
        np.float64(1.0 - FEE_RATE),
        turnover_values,
    )

    expected_net = (
        (1.0 + gross_values) * expected_fee_factor
        - 1.0
    )

    actual_net = result[
        "net_strategy_return"
    ].to_numpy(dtype=np.float64)

    np.testing.assert_allclose(
        actual_net,
        expected_net,
        rtol=1e-12,
        atol=1e-12,
    )


def test_entry_and_exit_both_generate_transaction_costs():
    close = [100, 101, 102, 101, 100]

    result = add_features(
        make_frame(close),
        fast_window=1,
        slow_window=2,
    )

    assert result["position"].tolist() == [0, 0, 1, 1, 0]

    assert result["turnover"].sum() == 2

    assert result["estimated_cost"].sum() == pytest.approx(
        2 * FEE_RATE
    )


def test_net_return_and_equity_are_consistent():
    result = add_features(
        make_frame([100, 101, 102, 103, 104]),
        fast_window=1,
        slow_window=2,
    )

    expected_equity = (
        1.0 + result["net_strategy_return"]
    ).cumprod()

    np.testing.assert_allclose(
        result["equity_curve"].to_numpy(),
        expected_equity.to_numpy(),
    )


def test_invalid_close_prices_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        add_features(
            make_frame([100, 101, 0, 103]),
            fast_window=1,
            slow_window=2,
        )


def test_non_finite_close_prices_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        add_features(
            make_frame([100, 101, np.nan, 103]),
            fast_window=1,
            slow_window=2,
        )


def test_invalid_window_configuration_is_rejected():
    with pytest.raises(ValueError):
        add_features(
            make_frame([100, 101, 102]),
            fast_window=10,
            slow_window=5,
        )


def test_summary_includes_cost_model():
    result = add_features(
        make_frame(np.linspace(100, 120, 100)),
        fast_window=5,
        slow_window=15,
    )

    summary = summarize_backtest(result)

    assert summary["cost_model"] == (
        "multiplicative_per_position_change"
    )

    assert summary["assumed_one_way_cost_bps"] == 5
