"""Discrete signals need one stratum per distinct value when coarser than
requested quantiles; tied edges must not collapse direction predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.evaluation import quantile_strata
from tradingdev.domain.ml.evaluation.stratified import quantile_bins


def _seeded(count: int) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(0)
    values = rng.normal(size=count)
    return pd.Series(values, name="prediction"), pd.Series(values, name="fwd")


class TestQuantileBins:
    def test_continuous_signal_is_cut_at_quantile_edges(self) -> None:
        signal, _ = _seeded(200)
        bins = quantile_bins(signal, 5)
        assert bins.nunique() == 5
        assert bins.min() == 0 and bins.max() == 4

    def test_discrete_signal_gets_one_bin_per_value(self) -> None:
        signal = pd.Series([-1.0, 0.0, 1.0] * 10)
        bins = quantile_bins(signal, 5)
        assert bins.nunique() == 3
        assert sorted(bins.unique()) == [0, 1, 2]

    def test_bins_ascend_with_the_signal(self) -> None:
        signal = pd.Series([-1.0, 0.0, 1.0] * 4)
        bins = quantile_bins(signal, 5)
        assert bins[signal == -1.0].eq(0).all()
        assert bins[signal == 1.0].eq(2).all()

    def test_constant_signal_collapses_to_a_single_bin(self) -> None:
        bins = quantile_bins(pd.Series([7.0] * 10), 5)
        assert bins.nunique() == 1

    def test_unbalanced_binary_signal_with_two_requested_bins(self) -> None:
        signal = pd.Series([-1.0] * 95 + [1.0] * 5)
        bins = quantile_bins(signal, 2)
        assert bins.tolist() == [0] * 95 + [1] * 5


class TestQuantileStrata:
    def test_continuous_signal_forms_the_requested_bins(self) -> None:
        prediction, forward = _seeded(300)
        strata = quantile_strata(prediction, forward, n_bins=5)
        assert len(strata) == 5
        assert list(strata.index) == [1, 2, 3, 4, 5]
        # A perfect signal must produce monotonically rising realised returns.
        assert strata["mean_forward_return"].is_monotonic_increasing

    def test_discrete_signal_is_not_collapsed(self) -> None:
        prediction = pd.Series([-1.0, 0.0, 1.0] * 20)
        forward = pd.Series([-0.02, 0.0, 0.02] * 20)
        strata = quantile_strata(prediction, forward, n_bins=5)
        assert len(strata) == 3
        assert strata["mean_forward_return"].is_monotonic_increasing
        assert strata["count"].tolist() == [20, 20, 20]

    def test_unbalanced_binary_strata_keep_both_signal_values(self) -> None:
        prediction = pd.Series([-1.0] * 95 + [1.0] * 5)
        forward = pd.Series([-0.02] * 95 + [0.03] * 5)
        strata = quantile_strata(prediction, forward, n_bins=2)
        assert strata["count"].tolist() == [95, 5]
        assert strata["mean_forward_return"].tolist() == pytest.approx([-0.02, 0.03])

    def test_too_few_observations_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not enough observations"):
            quantile_strata(pd.Series([1.0, 2.0]), pd.Series([0.1, 0.2]), n_bins=5)

    def test_single_bin_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            quantile_strata(pd.Series([1.0] * 10), pd.Series([0.1] * 10), n_bins=1)
