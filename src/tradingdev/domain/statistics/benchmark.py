"""Buy-and-hold benchmark returns and a strategy's regression on them.

This is a research comparison, not the product's benchmark. The two follow
different rules:

* The product benchmark, which runs store and reports show, is defined by
  :mod:`tradingdev.domain.benchmark`. Its ``BENCHMARKS`` table fixes one
  instrument per asset class; that table, not this docstring, decides which
  asset classes and instruments exist (at the time of writing it covers
  crypto, US equities and Taiwan equities; crypto, for example, uses
  BTC/USDT). The instrument is bought once with the run's fee and slippage
  and held to the end, and the reports show the total-return difference,
  strategy minus benchmark.
* This module takes whatever prices the caller passes, usually the very
  instrument the strategy trades, charges no costs, and reports the
  annualised mean of the per-period differences as the active return.

The figures of the two benchmarks differ because of the instrument and the
definition of the return difference, not because of costs: the product's
cost falls on the entry only, after which its per-period returns equal the
price returns. The cost rules therefore stay as they are.

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

* Beta here is not systematic risk. When the strategy trades the same
  instrument as the benchmark, the regression slope mostly measures the
  strategy's average net exposure to that instrument (1.0 always long, 0.0
  flat on average or balanced long and short, negative net short), not a
  market beta in the CAPM sense. Alpha is correspondingly the return not
  explained by that average exposure, which includes any timing skill. For
  a different instrument, such as the product's BTC/USDT benchmark of an
  ETH strategy, neither reading holds.
* The benchmark pays no transaction costs and the strategy does. Here that
  is deliberate: costs are part of what a strategy has to overcome to beat
  holding the asset. The product's stored benchmark charges the run's
  costs, but only on entry; its figures differ from this module's mainly
  by instrument and by the return difference used.
* Both series must cover the same bars. They are always aligned with
  :func:`tradingdev.domain.statistics.comparison.align_returns`, which keeps
  only the periods both cover; build the benchmark from the same bars and
  the same ``frequency`` as the strategy returns.
* Annualisation uses the caller's ``periods_per_year``, as in
  :mod:`tradingdev.domain.statistics.sharpe`. Point estimates scale means
  by ``periods_per_year`` and standard deviations by
  ``sqrt(periods_per_year)``, which assumes independent periods, while
  :func:`difference_interval` uses a stationary bootstrap that keeps their
  autocorrelation. The point estimate and the interval therefore rest on
  different assumptions; with autocorrelated returns the annualised point
  estimates are biased, and the interval is the better guide to precision.

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
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

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
        beta: Least-squares slope of strategy on benchmark returns; the
            average net exposure, not systematic risk (see the module
            docstring). ``None`` with fewer than two periods or benchmark
            returns constant within ``NEGLIGIBLE_SPREAD``.
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
            Both means are negative there, so a value below 1 means the
            strategy lost less than the benchmark; ``None`` without such
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
    Compounding the returns reproduces ``prices[-1] / prices[0]``.

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

    ``beta = sum((b - mean(b)) * s) / sum((b - mean(b))**2)``

    ``alpha_annualised = (mean(s) - beta * mean(b)) * ppy``

    ``active_return_annualised = (mean(s) - mean(b)) * ppy``

    ``tracking_error_annualised = std(s - b, ddof=1) * sqrt(ppy)``

    ``information_ratio = active_return_annualised /
    tracking_error_annualised``

    ``up_capture = mean(s[b > 0]) / mean(b[b > 0])`` and likewise
    ``down_capture`` over ``b < 0``. These are ratios of per-period means,
    not ratios of annualised returns.

    ``beta`` is the ordinary least-squares slope. ``alpha_annualised``
    annualises the per-period intercept arithmetically, as
    ``intercept * ppy``, consistent with the active return and the
    information ratio; both agree with ``quantstats.stats.greeks``. A
    compounded alpha, ``(1 + intercept)**ppy - 1``, differs noticeably for
    large intercepts. See the module docstring for how to read beta and why the
    benchmark pays no costs.

    Annualisation differs within this project on purpose. The catalog's
    ``annual_return`` (provided by QuantStats) compounds daily
    returns, while the active-return figures here (``alpha_annualised``,
    ``active_return_annualised``, ``tracking_error_annualised``,
    ``information_ratio``) scale arithmetically. Both are correct for what
    they measure. A return figure answers how much capital grew, which is a
    compounded quantity. The information ratio is defined on the arithmetic
    mean active return over its standard deviation, so the active return
    and alpha must be arithmetic for ``information_ratio`` to equal
    ``active_return_annualised / tracking_error_annualised``. Side by side
    in one report the two can differ substantially: in particular,
    ``active_return_annualised`` is not the difference of the two series'
    ``annual_return``, and on daily data with a modest edge an arithmetic
    0.168 corresponds to a compounded 0.183.

    Raises:
        ValueError: If ``periods_per_year`` is not positive and finite, the
            series share no period, timezone-naive and timezone-aware
            indexes are mixed, an index has duplicate labels, or a return
            is not a finite number.
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

    beta = _beta(strategy, benchmark)
    alpha = (
        (float(strategy.mean()) - beta * float(benchmark.mean())) * periods
        if beta is not None
        else None
    )
    active = (float(strategy.mean()) - float(benchmark.mean())) * periods
    tracking_error = _tracking_error(strategy, benchmark, periods)
    up = benchmark > 0.0
    down = benchmark < 0.0
    return BenchmarkComparison(
        n_periods=aligned.overlap,
        beta=beta,
        alpha_annualised=alpha,
        active_return_annualised=active,
        tracking_error_annualised=tracking_error,
        information_ratio=(
            active / tracking_error
            if tracking_error is not None and tracking_error > 0.0
            else None
        ),
        up_capture=_capture(strategy[up], benchmark[up]),
        down_capture=_capture(strategy[down], benchmark[down]),
        n_up_periods=int(up.sum()),
        n_down_periods=int(down.sum()),
    )


def _beta(
    strategy: npt.NDArray[np.float64], benchmark: npt.NDArray[np.float64]
) -> float | None:
    if benchmark.size < 2 or _negligible_spread(benchmark, benchmark):
        return None
    deviations = benchmark - benchmark.mean()
    return float(deviations @ strategy) / float(deviations @ deviations)


def _tracking_error(
    strategy: npt.NDArray[np.float64],
    benchmark: npt.NDArray[np.float64],
    periods: float,
) -> float | None:
    if strategy.size < 2:
        return None
    active = strategy - benchmark
    if _negligible_spread(active, strategy, benchmark):
        return 0.0
    return float(active.std(ddof=1)) * math.sqrt(periods)


def _negligible_spread(
    values: npt.NDArray[np.float64], *scales: npt.NDArray[np.float64]
) -> bool:
    """Whether the range of ``values`` is rounding noise relative to ``scales``."""
    magnitude = max(1.0, *(float(np.max(np.abs(scale))) for scale in scales))
    return float(np.ptp(values)) <= NEGLIGIBLE_SPREAD * magnitude


def _capture(
    strategy: npt.NDArray[np.float64], benchmark: npt.NDArray[np.float64]
) -> float | None:
    if benchmark.size == 0:
        return None
    denominator = float(benchmark.mean())
    if denominator == 0.0:
        return None
    return float(strategy.mean()) / denominator


def _periods_per_year(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("periods_per_year is not numeric")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("periods_per_year must be positive and finite")
    return float(value)
