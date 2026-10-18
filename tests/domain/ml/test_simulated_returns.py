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
    """The headline: a signal that equals the return must trade profitably."""
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
    """A signal unrelated to returns must not pretend to be profitable."""
    rng = np.random.default_rng(2)
    signal = pd.Series(rng.normal(size=400))
    forward = pd.Series(rng.normal(size=400))
    noisy = simulate_strategy_returns(signal, forward, n_quantiles=5)

    perfect_signal, perfect_fwd = _seeded_returns(400)
    perfect = simulate_strategy_returns(perfect_signal, perfect_fwd, n_quantiles=5)

    assert abs(noisy.long_short_mean) < 0.2 * abs(perfect.long_short_mean)


def test_long_only_only_holds_the_top_quantile() -> None:
    signal = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    forward = pd.Series([0.1] * 10)
    result = simulate_strategy_returns(signal, forward, n_quantiles=2)

    assert result.long_only.to_list() == [0.0] * 5 + [0.1] * 5
    # Held half the time, so the mean is half the top-quantile mean.
    assert result.long_only_mean == pytest.approx(0.1 * (5 / 10))


def test_long_short_book_holds_the_short_leg() -> None:
    signal = pd.Series([1.0, 2.0, 3.0, 4.0])
    forward = pd.Series([0.1, 0.2, 0.3, 0.4])
    result = simulate_strategy_returns(signal, forward, n_quantiles=2)

    assert result.long_only.to_list() == [0.0, 0.0, 0.3, 0.4]
    assert result.long_short.to_list() == [-0.1, -0.2, 0.3, 0.4]
    assert result.long_short_mean == pytest.approx(0.1)
    assert result.long_short_cumulative == pytest.approx(0.9 * 0.8 * 1.3 * 1.4 - 1.0)


def test_cumulative_compounds_period_returns() -> None:
    signal = pd.Series([1.0, 2.0, 3.0, 4.0])
    forward = pd.Series([0.1, 0.2, 0.3, 0.4])
    result = simulate_strategy_returns(signal, forward, n_quantiles=2)

    assert result.long_short.to_list() == [-0.1, -0.2, 0.3, 0.4]
    assert result.long_short_cumulative == pytest.approx(0.9 * 0.8 * 1.3 * 1.4 - 1.0)


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
    """A direction model emits -1/0/1; every quantile edge would be a tie.

    Before this was handled, ``duplicates="drop"`` collapsed such a signal
    into a single bin, so no direction model could ever be simulated. Now each
    distinct value is its own stratum, which is exactly the short/flat/long
    split a long-short book needs.
    """
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
    """Compose with the purged splits: evaluate the book out-of-sample."""
    rng = np.random.default_rng(3)
    n = 400
    forward = pd.Series(rng.normal(size=n))
    signal = forward  # a perfectly predictive signal

    for split in purged_walk_forward_splits(n, horizon=5, n_splits=4):
        test_start, test_end = split.test
        test_signal = signal.iloc[test_start:test_end]
        test_forward = forward.iloc[test_start:test_end]
        result = simulate_strategy_returns(test_signal, test_forward, n_quantiles=5)

        assert len(result.long_short) == split.test_bars
        assert result.long_only_mean > 0
