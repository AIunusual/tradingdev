"""Numeric checks for the signal-quality metrics: IC, ICIR, hit rate, turnover.

Every expectation is rebuilt here from the aligned series with numpy and pandas
primitives, never by calling ``rolling_ic`` or ``ic_statistics``. The properties
pinned are the ones a plausible "cleanup" would silently break: that IC is a
rolling time-window average rather than one full-sample correlation, that
``n_effective_periods`` divides the *bar count* rather than the coefficient
count, that ICIR uses the sample deviation, and that tied predictions get
average ranks instead of an arbitrary order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.evaluation.metrics import MetricScalar, signal_metrics

_WINDOW = 3


def _number(metrics: dict[str, MetricScalar], key: str) -> float:
    """Read a metric that must be a number, so arithmetic stays type-safe."""
    value = metrics[key]
    assert isinstance(value, (int, float)), key
    return float(value)


def _rolling_ic_values(
    prediction: list[float],
    forward_return: list[float],
    *,
    window: int,
    method: str,
) -> np.ndarray:
    """Correlate each time-ordered window on its own.

    Slicing the series by hand keeps this independent of ``sliding_window_view``
    and of the row-wise correlation the module under test implements.
    """
    left = np.asarray(prediction, dtype=float)
    right = np.asarray(forward_return, dtype=float)
    values = []
    for start in range(left.size - window + 1):
        block_left = left[start : start + window]
        block_right = right[start : start + window]
        if method == "spearman":
            block_left = pd.Series(block_left).rank().to_numpy()
            block_right = pd.Series(block_right).rank().to_numpy()
        values.append(float(np.corrcoef(block_left, block_right)[0, 1]))
    return np.asarray(values)


def _expected_statistics(
    values: np.ndarray, *, n_bars: int, window: int
) -> dict[str, float]:
    """Aggregate coefficients the way the docstring says they are aggregated."""
    mean = float(np.mean(values))
    # pandas' sample deviation (ddof=1), not the population one.
    spread = float(np.std(values, ddof=1))
    effective = max(1, n_bars // window)
    icir = mean / spread
    return {
        "ic_mean": mean,
        "ic_std": spread,
        "icir": icir,
        "ic_t_stat": icir * float(np.sqrt(effective)),
        "ic_positive_rate": float(np.mean(values > 0)),
        "n_periods": float(values.size),
        "n_effective_periods": float(effective),
    }


class TestRollingWindow:
    def test_ic_is_average_of_rolling_windows_not_one_full_sample_correlation(
        self,
    ) -> None:
        prediction = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        forward = [1.0, 2.0, 3.0, 6.0, 5.0, 4.0]

        metrics = signal_metrics(prediction, forward, window=_WINDOW)

        values = _rolling_ic_values(
            prediction, forward, window=_WINDOW, method="pearson"
        )
        expected = _expected_statistics(values, n_bars=6, window=_WINDOW)
        for key, value in expected.items():
            assert metrics[key] == pytest.approx(value, rel=1e-12), key

        # The structural point: a single full-sample correlation is a different
        # number, so a refactor that collapsed the windows would fail here.
        full_sample = float(np.corrcoef(prediction, forward)[0, 1])
        assert abs(full_sample - expected["ic_mean"]) > 0.05

    @pytest.mark.parametrize(("bars", "window"), [(6, 3), (8, 4), (10, 2), (5, 5)])
    def test_one_coefficient_per_complete_window(self, bars: int, window: int) -> None:
        prediction = [float(index) for index in range(bars)]
        forward = [float((index * 7) % bars) for index in range(bars)]

        metrics = signal_metrics(prediction, forward, window=window)

        assert metrics["n_periods"] == bars - window + 1

    def test_effective_periods_divides_the_bar_count_not_the_coefficient_count(
        self,
    ) -> None:
        """Overlap correction uses every bar, so it is smaller than ``n_periods``.

        With 6 bars and a window of 3 there are 4 coefficients but only 2
        independent stretches of time; dividing the coefficient count would
        understate the evidence.
        """
        prediction = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        forward = [1.0, 2.0, 3.0, 6.0, 5.0, 4.0]

        metrics = signal_metrics(prediction, forward, window=_WINDOW)

        assert metrics["n_periods"] == 4
        assert metrics["n_effective_periods"] == 2

    def test_icir_divides_the_mean_by_the_sample_deviation(self) -> None:
        prediction = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        forward = [1.0, 2.0, 3.0, 6.0, 5.0, 4.0]

        metrics = signal_metrics(prediction, forward, window=_WINDOW)

        values = _rolling_ic_values(
            prediction, forward, window=_WINDOW, method="pearson"
        )
        mean = float(np.mean(values))
        assert metrics["ic_std"] == pytest.approx(float(np.std(values, ddof=1)))
        assert metrics["icir"] == pytest.approx(mean / float(np.std(values, ddof=1)))
        # Population deviation would differ by the sqrt(n/(n-1)) factor.
        assert metrics["ic_std"] != pytest.approx(float(np.std(values, ddof=0)))

    def test_t_statistic_scales_icir_by_the_effective_sample_size(self) -> None:
        prediction = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        forward = [1.0, 2.0, 3.0, 6.0, 5.0, 4.0]

        metrics = signal_metrics(prediction, forward, window=_WINDOW)

        assert metrics["ic_t_stat"] == pytest.approx(
            _number(metrics, "icir")
            * float(np.sqrt(_number(metrics, "n_effective_periods")))
        )


class TestRankAgainstPearson:
    def test_rank_ic_ignores_a_monotone_rescaling_of_the_prediction(self) -> None:
        """Ranks survive a monotone transform; Pearson does not."""
        forward = [1.0, 2.0, 3.0, 5.0, 4.0]
        prediction = [1.0, 2.0, 3.0, 4.0, 100.0]
        rescaled = [value**3 for value in prediction]

        original = signal_metrics(prediction, forward, window=5)
        stretched = signal_metrics(rescaled, forward, window=5)

        # 0.9, not 1.0, so the invariance is not the trivial one.
        assert original["rank_ic_mean"] == pytest.approx(0.9)
        assert stretched["rank_ic_mean"] == pytest.approx(0.9)
        assert stretched["ic_mean"] != pytest.approx(original["ic_mean"])

    def test_an_outlier_leaves_rank_ic_perfect_and_pearson_below_it(self) -> None:
        forward = [1.0, 2.0, 3.0, 4.0, 5.0]
        prediction = [1.0, 2.0, 3.0, 4.0, 100.0]

        metrics = signal_metrics(prediction, forward, window=5)

        # Both orders are identical, so the rank correlation is exactly one.
        assert metrics["rank_ic_mean"] == pytest.approx(1.0)
        assert metrics["ic_mean"] == pytest.approx(0.725, rel=1e-3)
        assert _number(metrics, "ic_mean") < _number(metrics, "rank_ic_mean")

    def test_tied_predictions_take_average_ranks(self) -> None:
        """Arbitrary tie-breaking would rank [1, 1, 2] as [1, 2, 3] and claim 1.0."""
        prediction = [1.0, 1.0, 2.0]
        forward = [1.0, 2.0, 3.0]

        metrics = signal_metrics(prediction, forward, window=3)

        assert metrics["rank_ic_mean"] == pytest.approx(float(np.sqrt(3.0) / 2.0))
        assert metrics["rank_ic_mean"] != pytest.approx(1.0)

    def test_the_rank_family_mirrors_the_pearson_keys(self) -> None:
        prediction = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        forward = [1.0, 2.0, 3.0, 6.0, 5.0, 4.0]

        metrics = signal_metrics(prediction, forward, window=_WINDOW)

        for suffix in (
            "ic_mean",
            "ic_std",
            "icir",
            "ic_t_stat",
            "ic_positive_rate",
        ):
            assert f"rank_{suffix}" in metrics
        values = _rolling_ic_values(
            prediction, forward, window=_WINDOW, method="spearman"
        )
        expected = _expected_statistics(values, n_bars=6, window=_WINDOW)
        assert metrics["rank_icir"] == pytest.approx(expected["icir"], rel=1e-12)
        assert metrics["rank_ic_positive_rate"] == pytest.approx(
            expected["ic_positive_rate"], rel=1e-12
        )


class TestAlignmentAndDegeneracy:
    def test_non_finite_pairs_are_dropped_together_before_windowing(self) -> None:
        """A dropped row shortens the series, so fewer windows exist."""
        prediction = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0]
        forward = [1.0, 2.0, 3.0, 4.0, 5.0, np.inf, 7.0, 8.0]

        metrics = signal_metrics(prediction, forward, window=_WINDOW)

        # Two rows removed leaves six bars, hence four windows, not six.
        assert metrics["n_periods"] == 4

    def test_a_constant_benchmark_leaves_every_correlation_unavailable(self) -> None:
        prediction = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        forward = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        metrics = signal_metrics(prediction, forward, window=_WINDOW)

        for key in ("ic_mean", "ic_std", "icir", "ic_t_stat", "ic_positive_rate"):
            assert metrics[key] is None, key
            assert metrics[f"rank_{key}"] is None, key
        assert metrics["n_periods"] == 0
        # The overlap correction is about bars, so it survives an empty IC.
        assert metrics["n_effective_periods"] == 2


class TestHitRateAndTurnover:
    def test_hit_rate_counts_only_bars_where_both_sides_are_non_zero(self) -> None:
        prediction = [1.0, 1.0, 0.0, 2.0, -3.0]
        forward = [0.1, -0.2, 0.3, 0.0, -0.5]

        metrics = signal_metrics(prediction, forward, window=3)

        # Scored rows are 0, 1 and 4: two agreements and one disagreement.
        assert metrics["hit_rate"] == pytest.approx(2.0 / 3.0)

    def test_turnover_is_the_mean_absolute_bar_to_bar_change(self) -> None:
        prediction = [1.0, 3.0, 2.0, 5.0]

        metrics = signal_metrics(prediction, [1.0, 2.0, 3.0, 4.0], window=2)

        # |3-1|, |2-3|, |5-2| average to 2.0.
        assert metrics["signal_turnover"] == pytest.approx(2.0)


class TestDecay:
    def test_decay_needs_a_price_series(self) -> None:
        prediction = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        forward = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

        without_price = signal_metrics(prediction, forward, window=_WINDOW)
        assert not any(key.startswith("rank_ic_h") for key in without_price)

        with_price = signal_metrics(
            prediction,
            forward,
            window=_WINDOW,
            price=[100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
        )
        assert "rank_ic_h1" in with_price

    def test_decay_uses_one_full_sample_rank_correlation_per_horizon(self) -> None:
        """Decay is not a rolling average, unlike the headline IC."""
        prediction = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        price = [100.0, 101.0, 103.0, 102.0, 104.0, 105.0]
        forward = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

        metrics = signal_metrics(
            prediction, forward, window=_WINDOW, price=price, horizons=(1,)
        )

        realised = pd.Series(price).shift(-1) / pd.Series(price) - 1.0
        block = pd.DataFrame({"pred": prediction, "fwd": realised}).dropna()
        expected = float(np.corrcoef(block["pred"].rank(), block["fwd"].rank())[0, 1])
        assert metrics["rank_ic_h1"] == pytest.approx(expected, rel=1e-12)
