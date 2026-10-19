"""Simulated long/short and long-only returns from a ranked signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.evaluation import simulate_strategy_returns
from tradingdev.domain.ml.validation import purged_walk_forward_splits


def _seeded_returns(count: int) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(0)
    values = rng.normal(size=count)
    return pd.Series(values, name="signal"), pd.Series(values, name="fwd")


def test_perfect_signal_yields_a_positive_long_short_book() -> None:
    signal, forward = _seeded_returns(300)
    result = simulate_strategy_returns(signal, forward, n_quantiles=5)

    assert result.long_short_mean > 0
    assert result.long_only_mean > 0
    assert result.quantile_means == tuple(sorted(result.quantile_means))
    assert result.quantile_means[-1] > result.quantile_means[0]


def test_inverse_signal_yields_a_negative_long_short_book() -> None:
    rng = np.random.default_rng(1)
    forward = pd.Series(rng.normal(size=300))
    result = simulate_strategy_returns(-forward, forward, n_quantiles=5)

    assert result.long_short_mean < 0


def test_uninformative_signal_is_roughly_zero() -> None:
    rng = np.random.default_rng(2)
    signal = pd.Series(rng.normal(size=400))
    forward = pd.Series(rng.normal(size=400))
    noisy = simulate_strategy_returns(signal, forward, n_quantiles=5)

    perfect_signal, perfect_fwd = _seeded_returns(400)
    perfect = simulate_strategy_returns(perfect_signal, perfect_fwd, n_quantiles=5)

    assert abs(noisy.long_short_mean) < 0.2 * abs(perfect.long_short_mean)


def test_increasing_signal_buys_only_after_historical_warmup() -> None:
    signal = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    forward = pd.Series([0.1] * 10)
    result = simulate_strategy_returns(signal, forward, n_quantiles=2)

    assert result.long_only.to_list() == [0.0] * 2 + [0.1] * 8
    assert result.long_only_mean == pytest.approx(0.1 * (8 / 10))


def test_long_short_book_holds_the_short_leg() -> None:
    signal = pd.Series([2.0, 1.0, 0.0, 3.0])
    forward = pd.Series([0.1, 0.2, 0.3, 0.4])
    result = simulate_strategy_returns(signal, forward, n_quantiles=2)

    assert result.long_only.to_list() == [0.0, 0.0, 0.0, 0.4]
    assert result.long_short.to_list() == [0.0, 0.0, -0.3, 0.4]
    assert result.long_short_mean == pytest.approx(0.025)
    assert result.long_short_cumulative == pytest.approx(0.7 * 1.4 - 1.0)


def test_cumulative_compounds_period_returns() -> None:
    signal = pd.Series([2.0, 1.0, 0.0, 3.0])
    forward = pd.Series([0.1, 0.2, 0.3, 0.4])
    result = simulate_strategy_returns(signal, forward, n_quantiles=2)

    assert result.long_short.to_list() == [0.0, 0.0, -0.3, 0.4]
    assert result.long_short_cumulative == pytest.approx(0.7 * 1.4 - 1.0)


def test_nan_pairs_are_dropped() -> None:
    signal = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    forward = pd.Series([np.nan, 2.0, 3.0, 4.0, 5.0])
    result = simulate_strategy_returns(signal, forward, n_quantiles=2)

    assert len(result.long_short) == 4
    assert result.long_short.index.tolist() == [1, 2, 3, 4]


def test_output_index_tracks_the_input_index() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    signal = pd.Series([float(i) for i in range(10)], index=index)
    forward = pd.Series([0.1] * 10, index=index)
    result = simulate_strategy_returns(signal, forward, n_quantiles=2)

    assert result.long_short.index.equals(index)


def test_requires_at_least_two_quantiles() -> None:
    signal = pd.Series([1.0, 2.0, 3.0, 4.0])
    forward = pd.Series([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="at least 2"):
        simulate_strategy_returns(signal, forward, n_quantiles=1)


def test_collapsed_signal_is_rejected() -> None:
    signal = pd.Series([5.0] * 8)
    forward = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    with pytest.raises(ValueError, match="distinct quantiles"):
        simulate_strategy_returns(signal, forward, n_quantiles=5)


def test_discrete_signal_yields_one_stratum_per_value() -> None:
    """Tied quantile edges must preserve the -1/0/1 short/flat/long strata."""
    signal = pd.Series([-1.0, 0.0, 1.0] * 20)
    forward = pd.Series([-0.02, 0.0, 0.02] * 20)
    result = simulate_strategy_returns(signal, forward, n_quantiles=5)

    assert result.n_quantiles == 3
    assert len(result.quantile_means) == 3
    # Bottom stratum shorts the -1 bars and top stratum holds the +1 bars.
    assert result.long_short_mean > 0
    assert result.long_only_mean > 0


def test_binary_signal_still_forms_a_two_legged_book() -> None:
    signal = pd.Series([-1.0] * 25 + [1.0] * 25)
    forward = pd.Series([-0.01] * 25 + [0.03] * 25)
    result = simulate_strategy_returns(signal, forward, n_quantiles=5)

    assert result.n_quantiles == 2
    assert result.long_short_mean > 0


def test_too_few_bars_is_rejected() -> None:
    signal = pd.Series([1.0, 2.0])
    forward = pd.Series([0.1, 0.2])
    with pytest.raises(ValueError, match="finite bars"):
        simulate_strategy_returns(signal, forward, n_quantiles=5)


def test_deterministic_across_calls() -> None:
    signal, forward = _seeded_returns(120)
    first = simulate_strategy_returns(signal, forward, n_quantiles=4)
    second = simulate_strategy_returns(signal, forward, n_quantiles=4)

    pd.testing.assert_series_equal(first.long_short, second.long_short)
    assert first.quantile_means == second.quantile_means


def test_simulated_returns_on_walk_forward_folds() -> None:
    rng = np.random.default_rng(3)
    n = 400
    forward = pd.Series(rng.normal(size=n))
    signal = forward

    for split in purged_walk_forward_splits(n, horizon=5, n_splits=4):
        test_start, test_end = split.test
        test_signal = signal.iloc[test_start:test_end]
        test_forward = forward.iloc[test_start:test_end]
        result = simulate_strategy_returns(test_signal, test_forward, n_quantiles=5)

        assert len(result.long_short) == split.test_bars
        assert result.long_only_mean > 0


@pytest.mark.parametrize("horizon", [1, 3])
def test_appending_future_signals_cannot_change_past_positions(horizon: int) -> None:
    signal = pd.Series([2.0, 1.0, 0.0, 3.0, 2.5, -0.5, 4.0, -1.0])
    forward = pd.Series([0.05] * len(signal))
    prefix = simulate_strategy_returns(signal, forward, n_quantiles=2, horizon=horizon)
    extended_signal = pd.concat([signal, pd.Series([1e6] * 20)], ignore_index=True)
    extended_forward = pd.Series([0.05] * len(extended_signal))
    extended = simulate_strategy_returns(
        extended_signal, extended_forward, n_quantiles=2, horizon=horizon
    )
    pd.testing.assert_series_equal(prefix.long_short, extended.long_short.iloc[:8])
    pd.testing.assert_series_equal(prefix.long_only, extended.long_only.iloc[:8])


def test_multi_bar_returns_compound_only_non_overlapping_intervals() -> None:
    signal = pd.Series(range(10), dtype=float)
    forward = pd.Series([0.21] * 10)
    result = simulate_strategy_returns(signal, forward, n_quantiles=2, horizon=2)

    # Every eligible trade earns a two-bar holding return. Rows between entry
    # bars are zero; counting their overlapping forward returns would overstate it.
    assert result.long_only.to_list() == [
        0.0,
        0.0,
        0.21,
        0.0,
        0.21,
        0.0,
        0.21,
        0.0,
        0.21,
        0.0,
    ]
    assert result.long_only_cumulative == pytest.approx(1.21**4 - 1.0)


def test_missing_pairs_do_not_shift_the_holding_grid_or_signal_history() -> None:
    signal = pd.Series(range(10), dtype=float)
    forward = pd.Series([0.1] * 10)
    forward.iloc[1] = np.nan
    result = simulate_strategy_returns(signal, forward, n_quantiles=2, horizon=2)

    assert result.long_only.index.tolist() == [0, 2, 3, 4, 5, 6, 7, 8, 9]
    assert result.long_only.loc[2] == pytest.approx(0.1)
    assert result.long_only.loc[[3, 5, 7, 9]].eq(0.0).all()


def test_unbalanced_binary_signal_with_two_quantiles_is_tradeable() -> None:
    signal = pd.Series([-1.0] * 95 + [1.0] * 5)
    forward = pd.Series([-0.01] * 95 + [0.03] * 5)
    result = simulate_strategy_returns(signal, forward, n_quantiles=2)

    assert result.n_quantiles == 2
    # The first new value cannot be classified using a previously unknown tail.
    assert result.long_short.iloc[:96].eq(0.0).all()
    assert result.long_short.iloc[96:].eq(0.03).all()


def test_repeated_value_on_collapsed_historical_cutoffs_is_neutral() -> None:
    signal = pd.Series([-1.0, 0.0, 1.0] + [0.0] * 8 + [2.0, -2.0])
    forward = pd.Series([0.1] * len(signal))
    result = simulate_strategy_returns(signal, forward, n_quantiles=2)

    assert result.long_short.iloc[3:11].eq(0.0).all()
    assert result.long_short.iloc[-2:].tolist() == [0.1, -0.1]


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_nonfinite_signal_and_return_pairs_are_omitted(missing: float) -> None:
    signal = pd.Series([1.0, 2.0, missing, 4.0, 5.0, 6.0])
    forward = pd.Series([0.1, 0.1, 0.1, missing, 0.1, 0.1])
    result = simulate_strategy_returns(signal, forward, n_quantiles=2)
    assert result.long_short.index.tolist() == [0, 1, 4, 5]
    assert np.isfinite(result.long_short).all()


def test_nonpositive_horizon_is_rejected() -> None:
    signal, forward = _seeded_returns(10)
    with pytest.raises(ValueError, match="horizon must be at least 1"):
        simulate_strategy_returns(signal, forward, horizon=0)


def test_different_or_duplicate_bar_indexes_are_rejected() -> None:
    signal = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    forward = pd.Series([0.1, 0.2, 0.3], index=[1, 2, 3])
    with pytest.raises(ValueError, match="same unique bar index"):
        simulate_strategy_returns(signal, forward, n_quantiles=2)
    signal.index = pd.Index([0, 0, 1])
    forward.index = signal.index
    with pytest.raises(ValueError, match="same unique bar index"):
        simulate_strategy_returns(signal, forward, n_quantiles=2)
