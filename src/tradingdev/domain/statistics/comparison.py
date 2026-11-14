"""Paired bootstrap comparison of strategies evaluated on the same data.

All functions take return series supplied by the caller, such as the daily
returns from :func:`tradingdev.domain.statistics.periods.period_returns`,
and never read runs, artifacts, or caches.

Why paired: two strategies run on the same market data share its shocks, so
their returns are usually strongly correlated. The variance of a difference
is ``var(A) + var(B) - 2 cov(A, B)``, far below ``var(A) + var(B)`` when the
correlation is high. Comparing two independently bootstrapped intervals
ignores that covariance. :func:`difference_interval` therefore aligns both
series on one timeline and resamples the same block indices for both, so
each resample keeps the pairing of their returns.

Reading the results:

* Non-overlapping individual intervals (from :func:`interval_table`) imply
  a significant difference, but overlapping intervals do not imply an
  insignificant one. The individual intervals are a conservative screen;
  decide with :func:`difference_interval`.
* Compare out-of-sample returns only, such as walk-forward test folds.
  In-sample returns of tuned parameters measure which strategy overfits
  more thoroughly, not which one is better.
* Strategies run over different periods, instruments, or cost assumptions
  are not comparable. That is an experimental design problem that no
  statistic can repair; this module only guarantees the common period.
* Statistical significance is not economic significance. A tiny but
  precisely estimated edge may not survive costs or capacity, so the
  difference itself is always reported together with its interval.

Conventions:

* ``statistic`` maps a one-dimensional array of returns to a float, or
  ``None`` when undefined; for example
  ``functools.partial(sharpe_ratio, periods_per_year=365)``.
* Undefined results (insufficient overlap, undefined statistic, no valid
  resample) are ``None``. :func:`difference_interval_outcome` and
  :func:`interval_table_outcome` return an
  :class:`~tradingdev.domain.statistics.sharpe.IntervalUnavailable`
  instead, with the reason and the point estimate. The overlap threshold
  counts periods, so monthly returns need 20 common months.
* Non-finite returns or duplicate index labels raise ``ValueError``; the
  caller decides how to treat missing periods before calling.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap

from tradingdev.domain.statistics._validation import (
    positive_int,
    real_array,
    seed_value,
)
from tradingdev.domain.statistics.sharpe import (
    MIN_BOOTSTRAP_OBSERVATIONS,
    IntervalUnavailable,
    UnavailableReason,
    _block_length,
    _returns_block_size,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import numpy.typing as npt

    Statistic = Callable[[npt.NDArray[np.float64]], float | None]


@dataclass(frozen=True)
class AlignedReturns:
    """Return series restricted to the periods they all cover.

    Attributes:
        frame: One column per series, indexed by the common periods in
            ascending order.
        overlap: Number of common periods.
        lengths: Number of observations of each series before alignment;
            ``lengths[name] - overlap`` observations of that series were
            dropped.
        min_overlap: Minimum overlap required for inference.
        sufficient: Whether ``overlap`` reaches ``min_overlap``.
    """

    frame: pd.DataFrame
    overlap: int
    lengths: Mapping[str, int]
    min_overlap: int
    sufficient: bool


@dataclass(frozen=True)
class StatisticInterval:
    """Bootstrap percentile interval of one strategy's statistic.

    Attributes:
        estimate: Statistic of the aligned returns.
        lower: Lower percentile bound of the bootstrap distribution.
        upper: Upper percentile bound of the bootstrap distribution.
        confidence: Two-sided confidence level of ``[lower, upper]``.
        block_size: Mean block length of the stationary bootstrap, shared
            by every series of the table: either given by the caller or
            estimated from the series and their squares.
        overlap: Number of aligned observations.
        reps: Number of bootstrap resamples drawn.
        valid_reps: Resamples with a defined statistic.
    """

    estimate: float
    lower: float
    upper: float
    confidence: float
    block_size: int
    overlap: int
    reps: int
    valid_reps: int


@dataclass(frozen=True)
class DifferenceInterval:
    """Paired bootstrap inference on ``statistic(a) - statistic(b)``.

    Attributes:
        difference: ``estimate_a - estimate_b`` on the aligned returns.
        lower: Lower percentile bound of the bootstrap differences.
        upper: Upper percentile bound of the bootstrap differences.
        p_value: Two-sided bootstrap p-value of a zero difference, from
            inverting the percentile interval: ``[lower, upper]`` excludes
            zero exactly when ``p_value < 1 - confidence``.
        estimate_a: Statistic of ``a`` on the aligned returns.
        estimate_b: Statistic of ``b`` on the aligned returns.
        confidence: Two-sided confidence level of ``[lower, upper]``.
        block_size: Mean block length of the stationary bootstrap, either
            given by the caller or estimated from the difference.
        correlation: Pearson correlation of the aligned returns; ``None``
            when either series is constant.
        overlap: Number of aligned observations.
        reps: Number of bootstrap resamples drawn.
        valid_reps: Resamples where both statistics are defined.
    """

    difference: float
    lower: float
    upper: float
    p_value: float
    estimate_a: float
    estimate_b: float
    confidence: float
    block_size: int
    correlation: float | None
    overlap: int
    reps: int
    valid_reps: int


def align_returns(
    series_by_name: Mapping[str, pd.Series],
    *,
    min_overlap: int = MIN_BOOTSTRAP_OBSERVATIONS,
) -> AlignedReturns:
    """Restrict return series to the index labels they all share.

    The intersection is taken on index labels, so a period missing from any
    series is dropped from all of them. Nothing is truncated silently: the
    result carries each series' original length and whether the overlap
    reaches ``min_overlap``; when it does not, inference functions return
    ``None`` or an
    :class:`~tradingdev.domain.statistics.sharpe.IntervalUnavailable`.

    Timezone-naive and timezone-aware labels never compare equal, so mixing
    them would silently leave no overlap; that raises instead, because the
    cause is a timezone mismatch, not disjoint periods.

    Raises:
        ValueError: If no series is given, a value is not a
            ``pd.Series``, ``min_overlap`` is not a positive integer, an
            index has duplicate labels, a return is
            not a finite number (booleans and strings are not numbers), or
            timezone-naive and timezone-aware datetime indexes are mixed.
    """
    if not series_by_name:
        raise ValueError("at least one return series is required")
    positive_int(min_overlap, "min_overlap")
    for name, series in series_by_name.items():
        # Pairing is by index label, so an unlabelled array has nothing to
        # align on; positions would silently pair unrelated periods.
        if not isinstance(series, pd.Series):
            raise ValueError(
                f"returns {name!r} must be a pandas Series indexed by period, "
                f"got {type(series).__name__}"
            )
    _require_consistent_timezones(series_by_name)
    columns: dict[str, pd.Series] = {}
    for name, series in series_by_name.items():
        if not series.index.is_unique:
            raise ValueError(f"returns {name!r} have duplicate index labels")
        # Booleans and numeric strings are rejected, not coerced: a 0/1
        # signal passed by mistake would otherwise pair as returns.
        array = real_array(series, f"returns {name!r}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"returns {name!r} must be finite")
        columns[name] = pd.Series(array, index=series.index, name=series.name)
    frame = pd.concat(columns, axis=1, join="inner").sort_index()
    overlap = len(frame)
    return AlignedReturns(
        frame=frame,
        overlap=overlap,
        lengths={name: len(series) for name, series in columns.items()},
        min_overlap=min_overlap,
        sufficient=overlap >= min_overlap,
    )


def difference_interval(
    a: pd.Series,
    b: pd.Series,
    statistic: Statistic,
    *,
    reps: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
    block_size: int | None = None,
) -> DifferenceInterval | None:
    """Paired stationary-bootstrap interval of ``statistic(a) - statistic(b)``.

    The same as :func:`difference_interval_outcome`, with every
    :class:`~tradingdev.domain.statistics.sharpe.IntervalUnavailable`
    collapsed to ``None``. A short overlap returns ``None`` before the
    statistic is evaluated, so a statistic that rejects short arrays is
    never called on one.
    """
    positive_int(reps, "reps")
    _tail(confidence)
    seed_value(seed)
    if block_size is not None:
        positive_int(block_size, "block_size")
    if not align_returns({"a": a, "b": b}).sufficient:
        return None
    outcome = difference_interval_outcome(
        a,
        b,
        statistic,
        reps=reps,
        confidence=confidence,
        seed=seed,
        block_size=block_size,
    )
    return outcome if isinstance(outcome, DifferenceInterval) else None


def difference_interval_outcome(
    a: pd.Series,
    b: pd.Series,
    statistic: Statistic,
    *,
    reps: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
    block_size: int | None = None,
) -> DifferenceInterval | IntervalUnavailable:
    """Paired stationary-bootstrap interval of ``statistic(a) - statistic(b)``.

    Both series are aligned with :func:`align_returns`, then resampled with
    one :class:`arch.bootstrap.StationaryBootstrap` so every resample draws
    the same period indices for ``a`` and ``b``. This keeps their
    correlation, which is what makes the difference interval narrower than
    the individual intervals suggest. This is the formal test; see the
    module docstring for why overlapping individual intervals are not.

    Block size: the blocks must preserve the dependence of the estimator
    of the difference, which is not the dependence of either series. When
    both strategies share large market noise, the noise cancels in the
    difference while a small but persistent difference remains; block
    lengths fitted to ``a`` and ``b`` then follow the short-memory noise,
    the bootstrap understates the variance of the difference, and a
    difference that is not significant comes out significant. The block
    length is therefore estimated from the jackknife influence values of
    the difference: element ``t`` is ``(n - 1)`` times the change in
    ``statistic(a) - statistic(b)`` when period ``t`` is left out of both
    series, which for the mean is ``a_t - b_t`` up to a constant. The
    Politis and White (2004) automatic stationary block length with the
    Patton, Politis and White (2009) correction
    (:func:`arch.bootstrap.optimal_block_length`) of that sequence is
    rounded up to a whole period and at least one. If a leave-one-out
    statistic is undefined, the raw difference ``a - b`` is used instead.

    This costs ``2 * n`` extra evaluations of ``statistic`` on ``n - 1``
    periods, which is about the cost of the bootstrap itself for a few
    thousand daily returns. Pass ``block_size`` to skip it. A block size
    must be given whenever the jackknife does not describe the statistic:

    * Non-smooth statistics, such as a median, a quantile, or a maximum
      drawdown, whose leave-one-out values jump instead of tracking the
      statistic's sensitivity to each period.
    * Statistics too expensive to evaluate ``2 * n`` more times.

    Without one, the estimated block length may be too short, and a block
    length that is too short narrows the interval and lowers the p-value,
    which is the false positive this test is meant to prevent. A block
    size that is too long costs only some precision. An explicit
    ``block_size`` is used as given and reported back unchanged.

    p-value: the smallest ``1 - confidence`` at which the percentile
    interval excludes zero, obtained by inverting the same linearly
    interpolated quantiles that give ``lower`` and ``upper``. The interval
    at ``confidence`` therefore excludes zero exactly when ``p_value < 1 -
    confidence``, for any number of resamples, up to floating-point
    rounding when the two are equal. It is zero when every resample is on
    one side of zero. Below ``2 / (valid_reps - 1)`` it only interpolates
    between the two most extreme resamples, so with few ``reps`` a small
    p-value is no more precise than that interpolation; draw enough
    ``reps`` that the levels of interest lie well above it.

    Returns an :class:`~tradingdev.domain.statistics.sharpe.IntervalUnavailable`
    when the overlap is below ``MIN_BOOTSTRAP_OBSERVATIONS``, either
    statistic is undefined on the aligned returns, or no resample defines
    both statistics. Its ``estimate`` is the difference whenever both
    statistics are defined on the aligned returns, even a short overlap,
    so the statistic is evaluated on short arrays too (never on an empty
    one).

    Raises:
        ValueError: If ``reps`` or a given ``block_size`` is not a positive
            integer, ``seed`` is neither ``None`` nor a non-negative int,
            ``confidence`` is not strictly between 0 and 1, ``a`` or ``b``
            is not a ``pd.Series``, an index has duplicate labels, or a
            return is not a finite number.
    """
    positive_int(reps, "reps")
    tail = _tail(confidence)
    seed_value(seed)
    if block_size is not None:
        positive_int(block_size, "block_size")
    aligned = align_returns({"a": a, "b": b})
    values_a = aligned.frame["a"].to_numpy()
    values_b = aligned.frame["b"].to_numpy()
    estimate_a = _estimate(statistic, values_a)
    estimate_b = _estimate(statistic, values_b)
    difference = (
        estimate_a - estimate_b
        if estimate_a is not None and estimate_b is not None
        else None
    )
    if not aligned.sufficient:
        return _unavailable("insufficient_observations", difference, aligned, 0)
    if estimate_a is None or estimate_b is None:
        return _unavailable("undefined_statistic", None, aligned, 0)

    def resample(
        x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return np.array([_evaluate(statistic, x) - _evaluate(statistic, y)])

    if block_size is None:
        block_size = _difference_block_size(statistic, values_a, values_b)
    bootstrap = StationaryBootstrap(
        block_size, values_a, values_b, seed=np.random.default_rng(seed)
    )
    draws = bootstrap.apply(resample, reps)[:, 0]
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return _unavailable("no_valid_resample", difference, aligned, reps)
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    return DifferenceInterval(
        difference=estimate_a - estimate_b,
        lower=float(lower),
        upper=float(upper),
        p_value=_percentile_p_value(draws),
        estimate_a=estimate_a,
        estimate_b=estimate_b,
        confidence=float(confidence),
        block_size=block_size,
        correlation=_correlation(values_a, values_b),
        overlap=aligned.overlap,
        reps=reps,
        valid_reps=int(draws.size),
    )


def interval_table(
    series_by_name: Mapping[str, pd.Series],
    statistic: Statistic,
    *,
    reps: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
    block_size: int | None = None,
) -> dict[str, StatisticInterval | None] | None:
    """Individual stationary-bootstrap intervals for a forest plot.

    The same as :func:`interval_table_outcome`, except that an insufficient
    overlap gives ``None`` for the whole table and every other
    :class:`~tradingdev.domain.statistics.sharpe.IntervalUnavailable` row
    is ``None``. A short overlap returns ``None`` before the statistic is
    evaluated, as in :func:`difference_interval`.
    """
    positive_int(reps, "reps")
    _tail(confidence)
    seed_value(seed)
    if block_size is not None:
        positive_int(block_size, "block_size")
    if not align_returns(series_by_name).sufficient:
        return None
    table = interval_table_outcome(
        series_by_name,
        statistic,
        reps=reps,
        confidence=confidence,
        seed=seed,
        block_size=block_size,
    )
    return {
        name: row if isinstance(row, StatisticInterval) else None
        for name, row in table.items()
    }


def interval_table_outcome(
    series_by_name: Mapping[str, pd.Series],
    statistic: Statistic,
    *,
    reps: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
    block_size: int | None = None,
) -> dict[str, StatisticInterval | IntervalUnavailable]:
    """Individual stationary-bootstrap intervals for a forest plot.

    The series are aligned with :func:`align_returns` first, so every row
    covers the same periods, and resampled jointly with one block length:
    the largest Politis and White estimate over the series and their
    squares, so neither a series' own dependence nor the persistence of its
    volatility is cut short. This is the rule of
    :func:`~tradingdev.domain.statistics.sharpe.sharpe_interval_outcome`,
    whose docstring gives its measured coverage: better than blocks fitted
    to the returns alone when volatility clusters, still below the nominal
    level. Pass ``block_size`` to use a fixed mean block length instead,
    under the same conditions as there, and for statistics whose dependence
    neither the returns nor their squares describe, such as a maximum
    drawdown.

    These intervals answer "how precise is each estimate", not "do two
    strategies differ". Non-overlapping rows imply a significant difference;
    overlapping rows do not imply an insignificant one, because the
    intervals ignore the correlation between strategies. Use
    :func:`difference_interval` for the comparison itself.

    The result maps each name, in input order, to its interval or to an
    :class:`~tradingdev.domain.statistics.sharpe.IntervalUnavailable`. The
    overlap is shared, so when it is below ``MIN_BOOTSTRAP_OBSERVATIONS``
    every row is ``"insufficient_observations"``, still carrying its point
    estimate; otherwise a row is unavailable when its statistic is
    undefined on the aligned returns or on every resample.

    Raises:
        ValueError: If no series is given, ``reps`` or a given
            ``block_size`` is not a positive integer, ``seed`` is neither
            ``None`` nor a non-negative int, ``confidence`` is not strictly
            between 0 and 1, a value is not a ``pd.Series``, an index has
            duplicate labels, or a return is not a finite number.
    """
    positive_int(reps, "reps")
    tail = _tail(confidence)
    seed_value(seed)
    if block_size is not None:
        positive_int(block_size, "block_size")
    aligned = align_returns(series_by_name)
    columns = [aligned.frame[name].to_numpy() for name in aligned.frame.columns]
    if not aligned.sufficient:
        return {
            name: _unavailable(
                "insufficient_observations",
                _estimate(statistic, columns[position]),
                aligned,
                0,
            )
            for position, name in enumerate(aligned.frame.columns)
        }

    def resample(*values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([_evaluate(statistic, column) for column in values])

    if block_size is None:
        block_size = _returns_block_size(np.column_stack(columns))
    bootstrap = StationaryBootstrap(
        block_size, *columns, seed=np.random.default_rng(seed)
    )
    draws = bootstrap.apply(resample, reps)

    table: dict[str, StatisticInterval | IntervalUnavailable] = {}
    for position, name in enumerate(aligned.frame.columns):
        estimate = _estimate(statistic, columns[position])
        valid = draws[:, position][np.isfinite(draws[:, position])]
        if estimate is None:
            table[name] = _unavailable("undefined_statistic", None, aligned, 0)
            continue
        if valid.size == 0:
            table[name] = _unavailable("no_valid_resample", estimate, aligned, reps)
            continue
        lower, upper = np.quantile(valid, [tail, 1.0 - tail])
        table[name] = StatisticInterval(
            estimate=estimate,
            lower=float(lower),
            upper=float(upper),
            confidence=float(confidence),
            block_size=block_size,
            overlap=aligned.overlap,
            reps=reps,
            valid_reps=int(valid.size),
        )
    return table


def _evaluate(statistic: Statistic, values: npt.NDArray[np.float64]) -> float:
    result = statistic(values)
    if result is None:
        return math.nan
    return float(result)


def _estimate(statistic: Statistic, values: npt.NDArray[np.float64]) -> float | None:
    # An empty overlap has no estimate; a statistic need not handle it.
    if values.size == 0:
        return None
    value = _evaluate(statistic, values)
    return value if math.isfinite(value) else None


def _unavailable(
    reason: UnavailableReason,
    estimate: float | None,
    aligned: AlignedReturns,
    reps: int,
) -> IntervalUnavailable:
    return IntervalUnavailable(
        reason=reason,
        estimate=estimate,
        observations=aligned.overlap,
        min_observations=aligned.min_overlap,
        reps=reps,
    )


def _difference_block_size(
    statistic: Statistic,
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> int:
    influence = _jackknife_influence(statistic, a, b)
    if influence is None:
        influence = a - b
    return _block_length(influence[:, np.newaxis])


def _jackknife_influence(
    statistic: Statistic,
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64] | None:
    keep = np.ones(a.size, dtype=bool)
    left_out = np.empty(a.size)
    for period in range(a.size):
        keep[period] = False
        left_out[period] = _evaluate(statistic, a[keep]) - _evaluate(statistic, b[keep])
        keep[period] = True
    if not np.all(np.isfinite(left_out)):
        return None
    # The block length estimator ignores location and scale, so the usual
    # centring on the full-sample difference is left out.
    return (a.size - 1) * -left_out


def _percentile_p_value(draws: npt.NDArray[np.float64]) -> float:
    # Inverts np.quantile's default linear interpolation: the quantile at
    # level p is the sorted draws interpolated at position (m - 1) * p.
    ordered = np.sort(draws)
    last = ordered.size - 1
    if last == 0:
        return 1.0 if ordered[0] == 0.0 else 0.0
    # Highest level whose quantile is still <= 0; the lower bound excludes
    # zero exactly when the lower tail level is above it.
    below = int(np.searchsorted(ordered, 0.0, side="right")) - 1
    if below < 0:
        lower_level = 0.0
    elif below == last:
        lower_level = 1.0
    else:
        lower_level = _crossing(ordered, below) / last
    # Lowest level whose quantile is already >= 0; the upper bound excludes
    # zero exactly when the upper level 1 - tail is below it.
    above = int(np.searchsorted(ordered, 0.0, side="left"))
    if above == 0:
        upper_level = 0.0
    elif above > last:
        upper_level = 1.0
    else:
        upper_level = _crossing(ordered, above - 1) / last
    return min(1.0, 2.0 * min(lower_level, 1.0 - upper_level))


def _crossing(ordered: npt.NDArray[np.float64], index: int) -> float:
    # Position between ordered[index] <= 0 and ordered[index + 1] >= 0 where
    # the linear interpolation reaches zero.
    low, high = float(ordered[index]), float(ordered[index + 1])
    return index + (-low / (high - low) if high > low else 0.0)


def _correlation(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> float | None:
    # Equal values can still yield a rounding-error std, so test spread exactly.
    if np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _tail(confidence: float) -> float:
    if isinstance(confidence, bool) or not isinstance(confidence, numbers.Real):
        raise ValueError("confidence must be strictly between 0 and 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    return (1.0 - float(confidence)) / 2.0


def _require_consistent_timezones(series_by_name: Mapping[str, pd.Series]) -> None:
    naive = [
        name
        for name, series in series_by_name.items()
        if isinstance(series.index, pd.DatetimeIndex) and series.index.tz is None
    ]
    aware = [
        name
        for name, series in series_by_name.items()
        if isinstance(series.index, pd.DatetimeIndex) and series.index.tz is not None
    ]
    if naive and aware:
        raise ValueError(
            "returns mix timezone-naive and timezone-aware indexes "
            f"(naive: {naive}, aware: {aware}); naive labels never match aware "
            "ones, so this is a timezone mismatch, not non-overlapping periods. "
            "Localize the naive index first, e.g. to UTC."
        )
