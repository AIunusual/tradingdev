"""Pin exact values at the boundaries that mutation testing found under-asserted.

Every expected value is derived by hand below, from the declared formula and
the inputs, not read from either implementation. These are the contracts that
surviving mutants showed the existing suites left implicit.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.statistics.benchmark import (
    _negligible_spread,
    _periods_per_year,
    compare_to_benchmark,
)
from tradingdev.domain.statistics.comparison import (
    _correlation,
    _crossing,
    _jackknife_influence,
    _percentile_p_value,
    difference_interval_outcome,
    interval_table_outcome,
)
from tradingdev.domain.statistics.sharpe import IntervalUnavailable


def _daily(values: list[float]) -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC"),
    )


class TestJackknifeInfluence:
    def test_mean_statistic_has_the_leave_one_out_influence(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.zeros(3)

        def mean(values: np.ndarray) -> float:
            return float(np.mean(values))

        # Leaving out period t gives mean(a)-mean(b) of the remaining two:
        # 2.5, 2.0, 1.5 -> influence = (n-1) * -left_out = [-5, -4, -3].
        result = _jackknife_influence(mean, a, b)

        assert result is not None
        assert result.tolist() == pytest.approx([-5.0, -4.0, -3.0])

    def test_a_non_finite_leave_one_out_is_unusable(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.zeros(3)

        def broken(values: np.ndarray) -> float:
            return math.nan if values.size == 2 else float(np.mean(values))

        assert _jackknife_influence(broken, a, b) is None


class TestPercentilePValue:
    def test_a_single_zero_draw_is_certainly_not_significant(self) -> None:
        assert _percentile_p_value(np.array([0.0])) == 1.0

    def test_a_single_nonzero_draw_is_certainly_significant(self) -> None:
        assert _percentile_p_value(np.array([0.5])) == 0.0

    def test_symmetric_draws_give_p_one(self) -> None:
        # ordered [-1, 1]: the interpolated zero crossing is at 0.5 on both
        # tails, so p = min(1, 2 * min(0.5, 1 - 0.5)) = 1.
        assert _percentile_p_value(np.array([-1.0, 1.0])) == pytest.approx(1.0)

    @pytest.mark.parametrize("draws", [[-2.0, -1.0], [1.0, 2.0]])
    def test_one_sided_draws_give_p_zero_and_never_negative(
        self, draws: list[float]
    ) -> None:
        result = _percentile_p_value(np.array(draws))

        assert result == pytest.approx(0.0)
        assert result >= 0.0


class TestCrossingAndCorrelation:
    def test_crossing_interpolates_linearly(self) -> None:
        # Between -1 and 1 the linear interpolation reaches zero at 0.5.
        assert _crossing(np.array([-1.0, 1.0]), 0) == pytest.approx(0.5)

    def test_crossing_of_equal_values_stays_at_the_index(self) -> None:
        # No spread means no interpolation offset; index 0 stays 0.
        assert _crossing(np.array([1.0, 1.0]), 0) == pytest.approx(0.0)

    def test_perfectly_proportional_series_have_correlation_one(self) -> None:
        assert _correlation(
            np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])
        ) == pytest.approx(1.0)

    def test_a_constant_series_has_no_correlation(self) -> None:
        assert _correlation(np.ones(3), np.array([1.0, 2.0, 3.0])) is None
        assert _correlation(np.array([1.0, 2.0, 3.0]), np.ones(3)) is None


class TestUnavailableOutcomes:
    def test_undefined_statistic_reports_zero_reps(self) -> None:
        a = _daily([0.01, 0.02, 0.03] * 20)
        b = _daily([0.02, 0.01, 0.03] * 20)

        def undefined(values: np.ndarray) -> None:
            return None

        result = difference_interval_outcome(a, b, undefined, reps=30, seed=0)

        assert isinstance(result, IntervalUnavailable)
        assert result.reason == "undefined_statistic"
        assert result.reps == 0
        assert result.estimate is None

    def test_short_overlap_reports_zero_reps_for_every_row(self) -> None:
        series = {"a": _daily([0.01] * 10)}

        table = interval_table_outcome(series, np.mean, reps=30, seed=0)

        for row in table.values():
            assert isinstance(row, IntervalUnavailable)
            assert row.reason == "insufficient_observations"
            assert row.reps == 0


class TestBenchmarkBoundaries:
    def test_two_periods_still_define_a_beta(self) -> None:
        result = compare_to_benchmark(
            _daily([0.01, 0.03]), _daily([0.01, 0.02]), periods_per_year=1.0
        )

        # A 0.02 change in strategy returns per 0.01 change in benchmark.
        assert result.beta == pytest.approx(2.0)

    def test_two_periods_still_define_a_tracking_error(self) -> None:
        result = compare_to_benchmark(
            _daily([0.01, 0.03]), _daily([0.01, 0.02]), periods_per_year=365.0
        )

        # Active = [0, 0.01]; std(ddof=1) = sqrt(5e-5); scaled by sqrt(365).
        expected = math.sqrt(5e-5) * math.sqrt(365.0)
        assert result.tracking_error_annualised == pytest.approx(expected)

    def test_spread_between_one_and_two_epsilons_is_not_noise(self) -> None:
        # magnitude = max(1.0, 0) = 1.0, so the 1e-12 floor governs;
        # 1.5e-12 exceeds it and must be treated as a real spread.
        assert not _negligible_spread(np.array([0.0, 1.5e-12]), np.array([0.0, 0.0]))

    def test_periods_per_year_of_one_half_is_valid(self) -> None:
        assert _periods_per_year(0.5) == pytest.approx(0.5)

        result = compare_to_benchmark(
            _daily([0.01, 0.02]), _daily([0.01, 0.03]), periods_per_year=0.5
        )
        assert result.n_periods == 2
