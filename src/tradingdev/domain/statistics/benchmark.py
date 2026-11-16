"""Buy-and-hold benchmark returns and a strategy's regression on them.

This is a research comparison, not the product's benchmark. The two follow
different rules:

* The product benchmark, which runs store and reports show, is defined by
  :mod:`tradingdev.domain.benchmark`. Its ``BENCHMARKS`` table fixes one
  instrument per asset class; that table, not this docstring, decides which
  asset classes and instruments exist (at the time of writing it covers
  crypto, US equities and Taiwan equities; crypto, for example, uses
  BTC/USDT). ``buy_and_hold`` invests at the first close and liquidates at
  the last close; both fills pay the run's fee and slippage. The reports
  show the total-return difference, strategy minus benchmark.
* This module takes whatever prices the caller passes, usually the very
  instrument the strategy trades, charges no costs, and reports the
  annualised mean of the per-period differences as the active return.

The figures of the two benchmarks differ in three ways: the instrument,
the cost treatment, and the definition of the return difference. Between
its entry and exit fills the product's per-period returns still equal the
price returns, so its costs change the final total return, not the
intermediate per-period series; this module charges no costs at all. The
two are separate measurements and their figures are not interchangeable.

The convention this package adopts in benchmark comparisons: "excess
return" is reserved for the total-return difference, strategy minus
benchmark, which the reports show beside the product benchmark; the
annualised mean of the per-period differences is the active return
(``BenchmarkComparison.active_return_annualised``). The two answer
different questions and their figures are not interchangeable.

All functions take series supplied by the caller, such as daily returns from
:func:`tradingdev.domain.statistics.periods.period_returns`, and never read
runs, artifacts, or caches.

The headline question, "does the strategy beat buy-and-hold", is a Sharpe
difference with an interval. That is a paired comparison already served by
:func:`tradingdev.domain.statistics.comparison.difference_interval`::

    difference_interval(
        strategy, benchmark, partial(sharpe_ratio, periods_per_year=365)
    )

This module adds what a difference of one statistic cannot express: how the
strategy co-moves with the benchmark (:func:`compare_to_benchmark`).

Reading the results:

* Beta measures the fitted linear sensitivity of strategy returns to
  benchmark returns. For the same instrument and a benchmark with
  sufficient variation, beta equals a fixed exposure when strategy returns
  equal that exposure times benchmark returns plus a constant. With
  time-varying positions, even if strategy returns are simply the product
  of exposure and benchmark returns ``b``, the slope is
  ``cov(exposure * b, b) / var(b)`` and can differ from the time average
  of the positions. Alpha is the fitted intercept, scaled arithmetically
  to an annual figure. Attributing it to timing skill requires evidence
  beyond this regression.
* The benchmark series here pays no transaction costs and the strategy
  series supplied by the caller does. That is deliberate: costs are part
  of what a strategy has to overcome to beat holding the asset. The
  product's stored benchmark charges the run's costs on both entry and
  exit; its figures differ from this module's by instrument, cost
  treatment and the return difference used.
* Both series must cover the same bars. They are always aligned with
  :func:`tradingdev.domain.statistics.comparison.align_returns`, which keeps
  only the periods both cover; build the benchmark from the same bars and
  the same ``frequency`` as the strategy returns.
* Annualisation uses the caller's ``periods_per_year``, as in
  :mod:`tradingdev.domain.statistics.sharpe`.
  :func:`compare_to_benchmark` scales sample means and the regression
  intercept by ``periods_per_year`` and a sample standard deviation by
  ``sqrt(periods_per_year)``. For a stationary series, serial dependence
  does not by itself bias those sample means. They still estimate the
  per-period means, and the annualisation is a fixed arithmetic rescaling.
  Dependence changes their uncertainty, not their location. An i.i.d.
  standard error and the ``sqrt`` scaling of a standard deviation no longer
  describe the sampling variance of a mean
  when successive periods are correlated, so confidence intervals and
  significance computed under independence are miscalibrated, and the
  annualised dispersion figures inherit that scaling assumption.
  For differences of a statistic evaluated separately on each series,
  :func:`tradingdev.domain.statistics.comparison.difference_interval`
  uses paired block resampling to account for dependence. Beta and alpha
  are fitted jointly from both series; this module reports their point
  estimates only.

Conventions:

* Undefined figures (too few periods, zero benchmark dispersion, zero
  tracking error, no up or down periods) are ``None`` rather than
  infinities or zeros.
* A range of returns no wider than ``NEGLIGIBLE_SPREAD`` times
  ``max(1, largest absolute return)`` is rounding noise and counts as
  zero dispersion. A fully invested strategy computes its returns from a
  scaled price curve, so its active returns differ from buy-and-hold only
  by rounding; without this rule that noise would be reported as a
  tracking error and an arbitrary information ratio. The floor of 1
  reflects where the noise arises: a return is ``closing / opening - 1``,
  whose rounding error is about one machine epsilon of the ratio near 1
  however small the return, so a quiet instrument does not lower it.
* Both series must use the same kind of index, such as the UTC period
  starts of :mod:`tradingdev.domain.statistics.periods`. Mixing a
  timezone-naive index with a timezone-aware one raises a timezone
  mismatch rather than reporting no common period.
* Non-finite returns or duplicate index labels raise ``ValueError``.
  Calculations use binary scaling to limit overflow. With a sufficiently
  wide range of input magnitudes, scaling can round away small observations.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np

from tradingdev.domain.statistics._scaling import (
    centered_dot,
    rescaled,
    rescaled_difference,
    shift_for_moments,
)
from tradingdev.domain.statistics.comparison import align_returns
from tradingdev.domain.statistics.periods import period_returns

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd

    from tradingdev.domain.statistics.periods import Frequency

NEGLIGIBLE_SPREAD = 1e-12
"""Return range treated as zero, relative to ``max(1, largest |return|)``."""


@dataclass(frozen=True)
class BenchmarkComparison:
    """Regression and capture figures of a strategy against a benchmark.

    Attributes:
        n_periods: Number of periods both series cover.
        beta: Least-squares slope of strategy on benchmark returns, giving
            fitted linear sensitivity. See the module docstring for the
            assumptions relating it to fixed exposure. ``None`` with fewer
            than two periods or benchmark returns constant within
            ``NEGLIGIBLE_SPREAD``.
        alpha_annualised: Regression intercept times ``periods_per_year``;
            ``None`` when ``beta`` is.
        active_return_annualised: Mean strategy return minus mean benchmark
            return, times ``periods_per_year``; the active return, not the
            total-return excess return of the reports.
        tracking_error_annualised: Sample standard deviation (``ddof=1``)
            of ``strategy - benchmark`` times ``sqrt(periods_per_year)``;
            ``0.0`` when that difference is constant within
            ``NEGLIGIBLE_SPREAD``, and ``None`` with fewer than two periods.
        information_ratio: ``active_return_annualised`` over
            ``tracking_error_annualised``, the Sharpe ratio of the active
            return; ``None`` when the tracking error is zero or undefined.
        up_capture: Mean strategy return over mean benchmark return in the
            periods where the benchmark rose; ``None`` without such periods.
        down_capture: The same over the periods where the benchmark fell.
            The denominator is negative. A negative ratio means the strategy
            gained on average during those periods. ``None`` without such
            periods.
        n_up_periods: Periods where the benchmark return is positive.
        n_down_periods: Periods where the benchmark return is negative.
            Periods with a zero benchmark return count towards neither.
    """

    n_periods: int
    beta: float | None
    alpha_annualised: float | None
    active_return_annualised: float
    tracking_error_annualised: float | None
    information_ratio: float | None
    up_capture: float | None
    down_capture: float | None
    n_up_periods: int
    n_down_periods: int


def buy_and_hold_returns(
    prices: npt.ArrayLike,
    *,
    timestamps: npt.ArrayLike,
    frequency: Frequency,
) -> pd.Series:
    """Simple returns of holding the instrument from the first price.

    Holding one unit makes equity proportional to the price, so this is
    :func:`~tradingdev.domain.statistics.periods.period_returns` of the
    price series: the same UTC periods, labels, and validation as the
    strategy returns built from an equity curve over the same bars.
    In exact arithmetic, compounding the gross returns reproduces
    ``prices[-1] / prices[0]``. See
    :func:`~tradingdev.domain.statistics.periods.period_returns` for the
    floating-point limits on recovering price ratios from simple returns.

    Pass the price the strategy's equity is marked at (normally the bar
    close) and the same ``frequency`` as the strategy returns; returns of
    different frequencies share few or no index labels.

    Raises:
        ValueError: If the inputs are empty, differ in length, contain
            non-finite or non-positive prices, or timestamps are not
            strictly increasing. The messages are those of
            :func:`~tradingdev.domain.statistics.periods.period_returns`,
            which calls the prices an equity curve.
    """
    returns = period_returns(prices, timestamps, frequency)
    returns.name = "benchmark_return"
    return returns


def compare_to_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    periods_per_year: float,
) -> BenchmarkComparison:
    """Regress strategy returns on benchmark returns over their common periods.

    With ``s`` and ``b`` the aligned strategy and benchmark returns,
    ``ppy`` the periods per year, and means and sums over the ``n`` common
    periods:

    ``beta = sum((b - mean(b)) * (s - mean(s))) / sum((b - mean(b))**2)``

    ``alpha_annualised = (mean(s) - beta * mean(b)) * ppy``

    ``active_return_annualised = (mean(s) - mean(b)) * ppy``

    ``tracking_error_annualised = std(s - b, ddof=1) * sqrt(ppy)``

    ``information_ratio = active_return_annualised /
    tracking_error_annualised``

    ``up_capture = mean(s[b > 0]) / mean(b[b > 0])`` and likewise
    ``down_capture`` over ``b < 0``. These are ratios of per-period means,
    not ratios of annualised returns.

    ``beta`` is the ordinary least-squares slope. ``alpha_annualised``
    reports the fitted intercept times ``ppy``, following the arithmetic
    regression convention of ``quantstats.stats.greeks``. See the module
    docstring for interpreting the regression and the benchmark's costs.

    The information ratio divides the annualised mean active return by
    the annualised tracking error. The catalog's ``annual_return``
    reports compounded annual growth from each series' observed daily
    returns.
    ``active_return_annualised`` measures the annualised mean per-period
    difference between the two series.

    For a single series with 365 equal daily returns of ``0.168 / 365``,
    arithmetic annualisation gives ``0.168`` and compounding gives
    approximately ``0.183``. For varying daily returns, compounded growth
    depends on the individual returns.

    Raises:
        ValueError: If ``periods_per_year`` is not positive and finite, the
            series share no period, timezone-naive and timezone-aware
            indexes are mixed, an index has duplicate labels, a return
            is not finite, or a computed metric exceeds the finite float
            range. In the last case the error names the metric.
    """
    periods = _periods_per_year(periods_per_year)
    aligned = align_returns(
        {"strategy": strategy_returns, "benchmark": benchmark_returns},
        min_overlap=1,
    )
    if aligned.overlap == 0:
        raise ValueError(
            "strategy and benchmark returns share no period; check that both "
            "are indexed by the same UTC periods"
        )
    strategy = aligned.frame["strategy"].to_numpy()
    benchmark = aligned.frame["benchmark"].to_numpy()

    beta, alpha = _regression(strategy, benchmark, periods)
    active, exponent = rescaled_difference(strategy, benchmark)
    active_mean, mean_exponent = _active_mean(strategy, benchmark)
    annual_active = _annualise(
        active_mean, mean_exponent, periods, "active_return_annualised"
    )
    tracking_error = None
    if aligned.overlap >= 2:
        tracking_error = 0.0
        if not _negligible_spread(active, strategy, benchmark, exponent=exponent):
            tracking_error = _annualise(
                float(active.std(ddof=1)),
                exponent,
                math.sqrt(periods),
                "tracking_error_annualised",
            )
    up = benchmark > 0.0
    down = benchmark < 0.0
    return BenchmarkComparison(
        n_periods=aligned.overlap,
        beta=beta,
        alpha_annualised=alpha,
        active_return_annualised=annual_active,
        tracking_error_annualised=tracking_error,
        information_ratio=(
            _finite_metric(annual_active / tracking_error, 0, "information_ratio")
            if tracking_error is not None and tracking_error > 0.0
            else None
        ),
        up_capture=_capture(strategy[up], benchmark[up], metric="up_capture"),
        down_capture=_capture(strategy[down], benchmark[down], metric="down_capture"),
        n_up_periods=int(up.sum()),
        n_down_periods=int(down.sum()),
    )


def _regression(
    strategy: npt.NDArray[np.float64],
    benchmark: npt.NDArray[np.float64],
    periods: float,
) -> tuple[float | None, float | None]:
    if benchmark.size < 2 or _negligible_spread(benchmark, benchmark):
        return None, None
    s, s_exponent = rescaled(strategy)
    b, b_exponent = rescaled(benchmark)
    s_shifted, b_shifted = shift_for_moments(s), shift_for_moments(b)
    slope = centered_dot(b_shifted, s_shifted) / centered_dot(b_shifted, b_shifted)
    beta = _finite_metric(slope, s_exponent - b_exponent, "beta")
    intercept = math.fma(-slope, math.fsum(b) / b.size, math.fsum(s) / s.size)
    alpha = _annualise(intercept, s_exponent, periods, "alpha_annualised")
    return beta, alpha


def _active_mean(
    strategy: npt.NDArray[np.float64], benchmark: npt.NDArray[np.float64]
) -> tuple[float, int]:
    # Sum the two sets of terms together, before subtracting would round
    # away the smaller terms of individual active returns. Keep the exponent
    # separate so averaging can precede annualisation even at extreme scales.
    try:
        fraction, exponent = math.frexp(math.fsum(chain(strategy, -benchmark)))
    except OverflowError:
        peak = max(float(np.abs(strategy).max()), float(np.abs(benchmark).max()))
        scale = math.frexp(peak)[1]
        terms = chain(np.ldexp(strategy, -scale), np.ldexp(-benchmark, -scale))
        fraction, exponent = math.frexp(math.fsum(terms))
        exponent += scale
    return fraction / strategy.size, exponent


def _annualise(value: float, exponent: int, factor: float, metric: str) -> float:
    fraction, value_exponent = math.frexp(value)
    factor_fraction, factor_exponent = math.frexp(factor)
    return _finite_metric(
        fraction * factor_fraction, exponent + value_exponent + factor_exponent, metric
    )


def _finite_metric(value: float, exponent: int, metric: str) -> float:
    error = f"{metric} is outside the finite float range"
    try:
        result = math.ldexp(value, exponent)
    except OverflowError:
        raise ValueError(error) from None
    if not math.isfinite(result):
        raise ValueError(error)
    return result


def _negligible_spread(
    values: npt.NDArray[np.float64],
    *scales: npt.NDArray[np.float64],
    exponent: int = 0,
) -> bool:
    """Compare the spread of the scaled values with rounding noise."""
    magnitude = max(1.0, *(float(np.max(np.abs(scale))) for scale in scales))
    threshold = NEGLIGIBLE_SPREAD * magnitude
    spread = float(values.max()) - float(values.min())
    if exponent < 0:
        spread = math.ldexp(spread, exponent)
    else:
        threshold = math.ldexp(threshold, -exponent)
    return spread <= threshold


def _capture(
    strategy: npt.NDArray[np.float64],
    benchmark: npt.NDArray[np.float64],
    *,
    metric: str = "capture",
) -> float | None:
    if benchmark.size == 0:
        return None
    b, b_exponent = rescaled(benchmark)
    denominator, mean_exponent = math.frexp(math.fsum(b) / b.size)
    if denominator == 0.0:
        return None
    s, s_exponent = rescaled(strategy)
    return _finite_metric(
        (math.fsum(s) / s.size) / denominator,
        s_exponent - b_exponent - mean_exponent,
        metric,
    )


def _periods_per_year(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("periods_per_year is not numeric")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("periods_per_year must be positive and finite")
    return float(value)
