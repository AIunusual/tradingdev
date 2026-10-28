"""Annualisation, drawdown and exposure measured on a known return series."""

from __future__ import annotations

import pytest

from tradingdev.domain.ml.evaluation.performance import (
    periods_per_year,
    return_metrics,
)

# Hand-checkable four-period example, annualised at 252 periods:
#   equity  1.00 -> 1.10 -> 1.045 -> 1.0659 -> 1.0659
#   peak    1.00    1.10    1.10     1.10
#   mdd     1 - 1.045 / 1.10 = 0.05
#   total   6.59%, so annual = 1.0659 ** (252 / 4) - 1
#   mean    0.0175, std(ddof=1) 0.0623835, so sharpe = 0.0175 / 0.0623835 * sqrt(252)
#   beta    cov(0.0007833) / var_b(0.0001667) = 4.7
#   alpha   (0.0175 - 4.7 * 0.005) * 252 = -1.512
_RETURNS = [0.10, -0.05, 0.02, 0.00]
_BENCHMARK = [0.02, -0.01, 0.01, 0.00]
_PERIODS = 252.0


class TestPeriodsPerYear:
    @pytest.mark.parametrize(
        "timeframe,expected",
        [("1m", 525_600.0), ("1h", 8_760.0), ("4h", 2_190.0), ("1d", 365.0)],
    )
    def test_a_declared_bar_length_gives_a_calendar(
        self, timeframe: str, expected: float
    ) -> None:
        assert periods_per_year(timeframe) == pytest.approx(expected)

    def test_a_week_is_seven_days(self) -> None:
        assert periods_per_year("1w") == pytest.approx(365 / 7)

    def test_the_holding_interval_scales_the_year(self) -> None:
        assert periods_per_year("1h", horizon=4) == pytest.approx(2_190.0)
        assert periods_per_year("1d", horizon=7) == pytest.approx(365 / 7)

    @pytest.mark.parametrize("timeframe", ["1M", "1mo", "1y", "", "bogus", "0h"])
    def test_an_ambiguous_bar_length_refuses_to_annualise(self, timeframe: str) -> None:
        assert periods_per_year(timeframe) is None

    def test_a_non_positive_horizon_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            periods_per_year("1h", horizon=0)


class TestReturnMetrics:
    def test_every_metric_matches_the_hand_calculation(self) -> None:
        result = return_metrics(_RETURNS, _BENCHMARK, periods_per_year=_PERIODS)
        assert result.max_drawdown == pytest.approx(0.05)
        assert result.annual_return == pytest.approx(54.73616961539993)
        assert result.sharpe == pytest.approx(4.453182583026776)
        assert result.beta == pytest.approx(4.7)
        assert result.alpha == pytest.approx(-1.512)
        assert result.n_periods == 4
        assert result.periods_per_year == _PERIODS

    def test_the_projection_names_the_metrics_it_measures(self) -> None:
        metrics = return_metrics(
            _RETURNS, _BENCHMARK, periods_per_year=_PERIODS
        ).as_metrics()
        assert set(metrics) == {
            "annual_return",
            "max_drawdown",
            "sharpe",
            "alpha",
            "beta",
            "periods_per_year",
            "n_return_periods",
        }
        # n_periods belongs to the rolling IC windows; a second writer would
        # silently replace that count with the return series length.
        assert "n_periods" not in metrics

    def test_without_a_calendar_only_the_scale_free_metrics_survive(self) -> None:
        result = return_metrics(_RETURNS, _BENCHMARK, periods_per_year=None)
        assert result.max_drawdown == pytest.approx(0.05)
        assert result.beta == pytest.approx(4.7)
        assert result.annual_return is None
        assert result.sharpe is None
        assert result.alpha is None

    def test_a_flat_benchmark_leaves_beta_and_alpha_undefined(self) -> None:
        result = return_metrics(
            _RETURNS, [0.0, 0.0, 0.0, 0.0], periods_per_year=_PERIODS
        )
        assert result.beta is None
        assert result.alpha is None
        assert result.max_drawdown == pytest.approx(0.05)

    def test_a_constant_book_has_no_sharpe(self) -> None:
        result = return_metrics(
            [0.01, 0.01, 0.01, 0.01], _BENCHMARK, periods_per_year=_PERIODS
        )
        assert result.sharpe is None
        assert result.annual_return is not None

    def test_a_total_loss_is_not_annualised(self) -> None:
        result = return_metrics(
            [-1.0, 0.0, 0.0], _BENCHMARK[:3], periods_per_year=_PERIODS
        )
        assert result.annual_return is None
        assert result.max_drawdown == pytest.approx(1.0)

    def test_gaps_are_dropped_from_both_series_together(self) -> None:
        with_gap = return_metrics(
            [0.10, float("nan"), -0.05, 0.02, 0.00],
            [0.02, 0.50, -0.01, 0.01, 0.00],
            periods_per_year=_PERIODS,
        )
        clean = return_metrics(_RETURNS, _BENCHMARK, periods_per_year=_PERIODS)
        assert with_gap == clean

    def test_an_empty_series_reports_nothing_measured(self) -> None:
        result = return_metrics([], [], periods_per_year=_PERIODS)
        assert result.n_periods == 0
        assert result.annual_return is None
        assert result.max_drawdown is None
        assert result.beta is None

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            return_metrics(_RETURNS, _BENCHMARK[:3], periods_per_year=_PERIODS)

    def test_a_drawdown_is_reported_as_a_positive_fraction(self) -> None:
        """A loss must not rank above a smaller loss under ``minimize``."""
        result = return_metrics([0.10, -0.30], [0.0, 0.0], periods_per_year=_PERIODS)
        assert result.max_drawdown == pytest.approx(0.30)

    def test_an_infinite_annualisation_is_reported_as_missing(self) -> None:
        """A tiny sample compounded to a year must not report an infinity."""
        # 1.9 ** 200 is about 1e58; raising it to 525600/200 overflows a float.
        result = return_metrics([0.9] * 200, [0.0] * 200, periods_per_year=525_600.0)
        assert result.annual_return is None
        assert result.max_drawdown == pytest.approx(0.0)


@pytest.mark.parametrize("returns", [[-2.0, -2.0], [0.0, -3.0, 0.0, -3.0]])
def test_nonpositive_capital_cannot_revive_into_a_rankable_annual_return(
    returns: list[float],
) -> None:
    result = return_metrics(returns, [0.0] * len(returns), periods_per_year=365.0)
    assert result.annual_return is None
    assert result.max_drawdown is not None and result.max_drawdown >= 1.0
