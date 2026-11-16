"""Sharpe ratio, its bootstrap interval, and probabilistic and deflated Sharpe.

All functions take a return series supplied by the caller, such as the
daily returns from :func:`tradingdev.domain.statistics.periods.period_returns`,
and never read runs, artifacts, or caches.

Frequency: use daily returns for daily-return analysis and choose
``periods_per_year`` for the caller's annualisation basis. Square-root
annualisation does not remove frequency dependence: serial correlation and
compounding can change the ratio, and the direction depends on the return
process. Report the frequency and compare ratios at matching frequencies
and annualisation bases.

Conventions:

* The Sharpe ratio is the mean over the sample standard deviation
  (``ddof=1``) of the supplied returns. Supply excess returns when using
  a nonzero risk-free benchmark; these functions do not subtract a rate.
  The performance summary applies its fixed risk-free setting before
  calling :func:`sharpe_ratio`.
* Undefined scalar figures (fewer than two returns or zero dispersion)
  are ``None`` rather than infinities or zeros. The interval entry point,
  :func:`sharpe_interval_outcome`, returns an :class:`IntervalUnavailable`
  saying why the interval is missing; monthly returns
  need ``MIN_BOOTSTRAP_OBSERVATIONS`` months, so one year never has one.
* Non-finite returns raise ``ValueError``; the caller decides how to treat
  missing periods before calling.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING, Literal

import numpy as np
from arch.bootstrap import StationaryBootstrap, optimal_block_length

from tradingdev.domain.statistics._scaling import rescaled, rescaled_columns
from tradingdev.domain.statistics._validation import (
    positive_int,
    real_array,
    seed_value,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt

MIN_BOOTSTRAP_OBSERVATIONS = 20
EULER_MASCHERONI = 0.5772156649015329
_DISPERSED_TRIALS = "the trial Sharpes are too dispersed for a finite hurdle"

UnavailableReason = Literal[
    "insufficient_observations", "undefined_statistic", "no_valid_resample"
]


@dataclass(frozen=True)
class IntervalUnavailable:
    """Why a bootstrap interval is unavailable and what could be computed.

    The reasons are checked in this order, so the first that applies wins:

    * ``"insufficient_observations"``: the sample has fewer than
      ``min_observations`` returns or aligned periods.
    * ``"undefined_statistic"``: the observed statistic or paired
      difference has no usable point estimate.
    * ``"no_valid_resample"``: none of the draws has a usable value
      for the requested statistic or difference.

    Attributes:
        reason: Which of the cases above applies.
        estimate: Point estimate when available, even on a sample too
            short to bootstrap. Otherwise ``None``.
        observations: Number of input returns or common periods.
        min_observations: Number of observations the bootstrap requires.
        reps: Resamples drawn for this result; ``0`` when none were drawn.
    """

    reason: UnavailableReason
    estimate: float | None
    observations: int
    min_observations: int
    reps: int

    @property
    def message(self) -> str:
        """One-sentence explanation for a report or tool response."""
        if self.reason == "insufficient_observations":
            return (
                f"{self.observations} observations, fewer than the "
                f"{self.min_observations} needed to bootstrap an interval"
            )
        if self.reason == "undefined_statistic":
            return "the statistic is undefined on these returns"
        return f"none of the {self.reps} resamples has a defined statistic"


@dataclass(frozen=True)
class SharpeInterval:
    """Bootstrap confidence interval of an annualised Sharpe ratio.

    Attributes:
        sharpe: Annualised Sharpe ratio of the original returns.
        lower: Lower percentile bound of the bootstrap distribution.
        upper: Upper percentile bound of the bootstrap distribution.
        confidence: Two-sided confidence level of ``[lower, upper]``.
        block_size: Mean block length of the stationary bootstrap, either
            given by the caller or estimated from the returns and their
            squares.
        reps: Number of bootstrap resamples drawn.
        valid_reps: Resamples with a defined Sharpe ratio; resamples whose
            returns are all equal are discarded.
    """

    sharpe: float
    lower: float
    upper: float
    confidence: float
    block_size: int
    reps: int
    valid_reps: int


@dataclass(frozen=True)
class DeflatedSharpe:
    """Deflated Sharpe ratio of a strategy selected from several trials.

    Attributes:
        dsr: Probability that the true Sharpe ratio exceeds the selection
            hurdle ``SR*_0``.
        psr_vs_zero: Probabilistic Sharpe ratio against a zero benchmark,
            for comparison; ``psr_vs_zero - dsr`` is what the
            selection-bias correction costs.
        hurdle_annualised: The hurdle ``SR*_0`` annualised, the expected
            maximum Sharpe of ``n_trials`` skill-less trials. This can
            overflow to infinity even when the validated per-period
            hurdle is finite; this output is not separately validated.
        n_trials: Number of trials ``N`` used for the hurdle.
        trial_sharpe_std_annualised: Sample standard deviation
            (``ddof=1``) of the annualised trial Sharpe ratios; ``None``
            when ``N`` is 1 and fewer than two trial Sharpes are given.
    """

    dsr: float
    psr_vs_zero: float
    hurdle_annualised: float
    n_trials: int
    trial_sharpe_std_annualised: float | None


def sharpe_ratio(returns: Iterable[float], *, periods_per_year: float) -> float | None:
    """Annualised Sharpe ratio: mean over sample std times ``sqrt(ppy)``.

    Returns ``None`` with fewer than two returns or zero dispersion.

    Raises:
        ValueError: If ``periods_per_year`` is not positive and finite or a
            return is not a finite number.
    """
    scale = _annualisation(periods_per_year)
    values = _returns(returns)
    sharpe = _period_sharpe(values)
    return sharpe * scale if sharpe is not None else None


def sharpe_interval_outcome(
    returns: Iterable[float],
    *,
    periods_per_year: float,
    confidence: float = 0.95,
    reps: int = 1000,
    seed: int | None = None,
    block_size: int | None = None,
) -> SharpeInterval | IntervalUnavailable:
    """Stationary-bootstrap percentile interval of the annualised Sharpe.

    Strategy returns can exhibit serial dependence. An iid bootstrap that
    shuffles single periods removes that dependence and may misstate the
    sampling variability. The stationary bootstrap of Politis and Romano
    (1994) resamples blocks with geometrically distributed lengths,
    preserving local order within each block. This does not make inference
    valid for an arbitrary non-stationary return process.

    Block size: the blocks must keep the dependence the Sharpe ratio
    depends on, which is that of the mean and of the variance. Returns can
    be nearly uncorrelated while their volatility persists for long
    stretches, and blocks fitted to the returns alone can cut those
    stretches apart and understate the variability. The mean block length
    is therefore the larger of two Politis and White (2004) automatic
    estimates, with the correction of Patton, Politis and White (2009), as
    computed by :func:`arch.bootstrap.optimal_block_length`: one for the
    returns and one for the squared returns, rounded up to a whole number
    of periods and at least one. Finite-sample block estimates can vary even
    for independent returns. This API requires at least
    ``MIN_BOOTSTRAP_OBSERVATIONS`` returns: 20 periods at any frequency,
    such as 20 days of daily returns or 20 months of monthly ones. Meeting
    that threshold does not establish that the sample is adequate for
    reliable inference.

    Automatic block selection does not guarantee nominal coverage. When
    regimes persist for much of the sample, too few transitions may be
    observed to estimate the sampling variability reliably, regardless
    of block length.

    Pass ``block_size`` to use a fixed mean block length instead, reported
    back unchanged. Give one whenever the estimate is known to be too
    short: dependence that neither the returns nor their squares show
    (regimes in higher moments, or returns whose persistence spans much of
    the sample), or a sample long enough to bootstrap but too short for a
    reliable estimate. A given ``block_size`` does not lift the minimum:
    fewer than ``MIN_BOOTSTRAP_OBSERVATIONS`` returns still give no
    interval. Blocks that are too short can omit dependence; excessive
    block lengths can leave too few block restarts and produce unreliable
    or nearly degenerate intervals. Either choice can impair coverage.

    Returns an :class:`IntervalUnavailable` when the sample is too short,
    the Sharpe ratio itself is undefined, or no resample has a defined
    Sharpe ratio; its ``estimate`` is the annualised Sharpe ratio whenever
    that is defined.

    Raises:
        ValueError: If ``periods_per_year`` is not positive and finite,
            ``confidence`` is not strictly between 0 and 1, ``reps`` or a
            given ``block_size`` is not a positive integer, ``seed`` is
            neither ``None`` nor a non-negative int, or a return is not a
            finite number.
    """
    scale = _annualisation(periods_per_year)
    if not (isinstance(confidence, numbers.Real) and 0.0 < confidence < 1.0):
        raise ValueError("confidence must be strictly between 0 and 1")
    positive_int(reps, "reps")
    seed_value(seed)
    if block_size is not None:
        positive_int(block_size, "block_size")
    values = _returns(returns)
    sharpe = _period_sharpe(values)

    def unavailable(reason: UnavailableReason, drawn: int) -> IntervalUnavailable:
        return IntervalUnavailable(
            reason=reason,
            estimate=sharpe * scale if sharpe is not None else None,
            observations=int(values.size),
            min_observations=MIN_BOOTSTRAP_OBSERVATIONS,
            reps=drawn,
        )

    if values.size < MIN_BOOTSTRAP_OBSERVATIONS:
        return unavailable("insufficient_observations", 0)
    if sharpe is None:
        return unavailable("undefined_statistic", 0)

    if block_size is None:
        block_size = _returns_block_size(values[:, np.newaxis])
    bootstrap = StationaryBootstrap(
        block_size, values, seed=np.random.default_rng(seed)
    )
    draws = bootstrap.apply(_resample_sharpe, reps)[:, 0] * scale
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return unavailable("no_valid_resample", reps)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    return SharpeInterval(
        sharpe=sharpe * scale,
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        block_size=block_size,
        reps=reps,
        valid_reps=int(draws.size),
    )


def probabilistic_sharpe_ratio(
    returns: Iterable[float], *, benchmark_sr: float = 0.0
) -> float | None:
    """Probability that the true Sharpe ratio exceeds ``benchmark_sr``.

    Bailey and Lopez de Prado (2012):

    ``PSR = Phi((SR - SR*) * sqrt(n - 1) / sqrt(1 - g3 * SR + (g4 - 1) / 4
    * SR**2))``

    where ``SR`` is the per-period (not annualised) Sharpe ratio, ``g3`` the
    skewness and ``g4`` the (non-excess) kurtosis of the returns, both
    computed from population moments. For a positive ``SR``, more negative
    skew and greater kurtosis increase the standard error, holding the
    other terms fixed. A larger standard error moves the probability
    towards 0.5: down when ``SR > SR*`` and up when ``SR < SR*``. The skew
    term reverses direction for negative ``SR``.

    ``benchmark_sr`` is per period as well; divide an annual benchmark by
    ``sqrt(periods_per_year)``.

    Assumption: the returns are independent. The standard error uses an
    asymptotic variance expression for iid returns; it accounts for skew
    and fat tails but has no term for serial correlation.
    Serial correlation can invalidate the reported probability; this
    function does not estimate an effective sample size or adjust for
    serial correlation.

    Returns ``None`` with fewer than two returns, zero dispersion, or a
    non-positive variance term.

    Raises:
        ValueError: If ``benchmark_sr`` or a return is not a finite number.
    """
    benchmark = _finite(benchmark_sr, "benchmark_sr")
    values = _returns(returns)
    sharpe = _period_sharpe(values)
    if sharpe is None:
        return None
    # Skewness and kurtosis are scale-free too; the fourth moment overflows
    # above about 1e77 without rescaling.
    values = rescaled(values)[0]
    deviations = values - values.mean()
    variance = float(np.mean(deviations**2))
    skewness = float(np.mean(deviations**3)) / variance**1.5
    kurtosis = float(np.mean(deviations**4)) / variance**2
    spread = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if not spread > 0.0:
        return None
    z = (sharpe - benchmark) * math.sqrt(values.size - 1) / math.sqrt(spread)
    return NormalDist().cdf(z)


def deflated_sharpe_ratio(
    returns: Iterable[float],
    annualised_trial_sharpes: Iterable[float],
    *,
    periods_per_year: float,
    n_trials: int | None = None,
) -> DeflatedSharpe | None:
    """Probabilistic Sharpe against the best Sharpe expected from luck alone.

    Bailey and Lopez de Prado (2014). Selecting the best of ``N`` trials
    inflates its Sharpe ratio even without any edge, so the benchmark of
    :func:`probabilistic_sharpe_ratio` is raised from 0 to the expected
    maximum Sharpe of ``N`` skill-less trials:

    ``SR*_0 = std(SR_n) * ((1 - gamma) * Z^-1(1 - 1/N)
    + gamma * Z^-1(1 - 1/(N * e)))``

    ``DSR = PSR(benchmark_sr=SR*_0)``

    where ``std(SR_n)`` is the sample standard deviation (``ddof=1``) of
    the trials' per-period Sharpe ratios, ``gamma`` the Euler-Mascheroni
    constant, and ``Z^-1`` the standard normal quantile function.

    Units: ``returns`` are the selected strategy's per-period returns, as
    for :func:`sharpe_ratio`. ``annualised_trial_sharpes`` are annualised
    Sharpe ratios, as :func:`sharpe_ratio` returns them, all computed with
    the same ``periods_per_year``; they are converted to per-period values
    here, so the hurdle is never inflated by ``sqrt(periods_per_year)``.

    ``n_trials`` must be the total number of trials searched, including
    failed trials that were dropped because their metric was undefined. It
    defaults to the number of trial Sharpes, which is correct only when no
    trial was dropped. Track all attempted trials before filtering out
    undefined Sharpes and pass that total as ``n_trials``. Passing only
    the surviving count underestimates ``N`` and under-corrects the
    selection bias, without any error. With ``N = 1`` there is no
    selection, and with equal trial Sharpes there is no dispersion; both
    give ``SR*_0 = 0``, so the DSR equals the PSR against zero.

    Limitation: the formula assumes ``N`` independent trials. Grid-search
    trials are strongly correlated (a lookback of 20 and of 21 is almost
    the same strategy), and the correlation has two opposite effects on
    the hurdle. The effective number of trials is smaller than the actual
    ``N``, so using the actual ``N`` raises the ``Z^-1`` multiplier, which
    pushes towards over-correction. But correlated trials also have Sharpe
    ratios that cluster together, so the measured ``std(SR_n)`` is small,
    which lowers the hurdle. Which effect dominates has not been measured,
    so the net direction of the error is unknown: the result may under- or
    over-correct for the selection. Using the actual ``N`` is a deliberate
    design choice meant to lean conservative, preferring a missed edge to
    a false positive; it is not a demonstrated property of the result.
    Estimating the effective ``N`` requires clustering the trials by the
    correlation of their returns and is out of scope. The DSR is also a
    :func:`probabilistic_sharpe_ratio` against a raised benchmark, so it
    inherits that function's assumption of independent returns unchanged:
    serial correlation can invalidate its probability, separately from any
    error in the trial correction.

    Two selection corrections: this project corrects for multiple
    comparisons in two ways, one per metric family.

    * This function corrects the Sharpe family: strategy return series of
      backtest or optimisation trials. Its hurdle is the expected maximum
      Sharpe of ``N`` skill-less trials, and the DSR is the probability
      that the true Sharpe clears it.
    * ``required_t_stat`` in the ML leaderboard
      (``tradingdev.domain.ml.leaderboard``)
      corrects the IC family:
      t-statistics of signal ICs. It is a two-sided Bonferroni threshold,
      ``Phi^-1(1 - alpha / (2 N))``, controlling the family-wise error
      rate, with a normal approximation the leaderboard itself flags as
      optimistic below 30 effective periods.

    They answer different questions: the DSR asks how likely the selected
    strategy's Sharpe beats what luck alone would produce in ``N`` trials;
    Bonferroni asks whether any of ``N`` IC tests would reject at
    family-wise ``alpha``. Their candidate populations also differ: this
    function requires all attempted trials in ``n_trials``, while
    ``rank_entries`` in the ML leaderboard passes only ``ranked_total``
    to its Bonferroni caveat. That count includes rankable candidates
    before ``top_k`` truncation, but excludes skipped entries and any
    candidates filtered out before ranking. The caveat alone does not
    establish coverage of the complete research search history.

    The corrections are not interchangeable. When reporting them side by
    side, label the metric, correction and candidate population; do not
    apply a second correction to the same selection without defining the
    statistical procedure. They also differ in what they assume about
    independence, in two dimensions. Across trials or tests: Bonferroni
    rests on the union bound, which holds whatever the dependence between
    the ``N`` tests, so correlated candidates only make it conservative;
    this function's hurdle
    is the expected maximum of ``N`` independent trials, as the limitation
    above describes. Within one series: each IC t-statistic Bonferroni
    compares (``ic_statistics`` in ``tradingdev.domain.ml.evaluation.metrics``)
    approximates effective sample size as ``max(1, bars // window)`` to account
    for overlap, without estimating remaining serial dependence. This function
    uses the independent-return variance through its PSR. Bonferroni's
    robustness to dependence between tests does not repair miscalibrated
    individual IC tests.

    Returns ``None`` when :func:`probabilistic_sharpe_ratio` is undefined
    for ``returns``, or when ``N >= 2`` but fewer than two trial Sharpes
    are given, so their dispersion cannot be estimated.

    Raises:
        ValueError: If ``periods_per_year`` is not positive and finite, a
            return or trial Sharpe is not a finite number, ``n_trials`` is
            not a positive integer or too large for a float, ``n_trials``
            is smaller than the number of trial Sharpes, no trial
            Sharpes are given without ``n_trials``, the dispersion
            calculation overflows, or the per-period hurdle is nonfinite.
    """
    scale = _annualisation(periods_per_year)
    trial_sharpes = np.array(
        [
            _finite(value, f"trial Sharpe {position}")
            for position, value in enumerate(annualised_trial_sharpes)
        ],
        dtype=np.float64,
    )
    if n_trials is None:
        if trial_sharpes.size == 0:
            raise ValueError("annualised_trial_sharpes is empty; give n_trials")
        n = trial_sharpes.size
    else:
        n = positive_int(n_trials, "n_trials")
    if n < trial_sharpes.size:
        raise ValueError("n_trials must not be smaller than the trial Sharpes")
    try:
        tail = 1.0 / n
    except OverflowError:
        raise ValueError("n_trials is too large for a float") from None
    values = _returns(returns)

    std_annualised: float | None = None
    if trial_sharpes.size >= 2:
        # Equal Sharpes can still yield a rounding-error std, so test spread exactly.
        spread = float(trial_sharpes.max()) != float(trial_sharpes.min())
        scaled, exponent = rescaled(trial_sharpes)
        try:
            std_annualised = (
                math.ldexp(float(scaled.std(ddof=1)), exponent) if spread else 0.0
            )
        except OverflowError:
            raise ValueError(_DISPERSED_TRIALS) from None
    if n == 1:
        hurdle = 0.0
    elif std_annualised is None:
        return None
    else:
        # Z^-1(1 - p) as -Z^-1(p) keeps precision when 1 - 1/N rounds to 1;
        # dividing 1/N by e avoids N * e overflowing to infinity.
        normal = NormalDist()
        expected_max = -(1.0 - EULER_MASCHERONI) * normal.inv_cdf(
            tail
        ) - EULER_MASCHERONI * normal.inv_cdf(tail / math.e)
        hurdle = std_annualised / scale * expected_max
        if not math.isfinite(hurdle):
            raise ValueError(_DISPERSED_TRIALS)

    psr_vs_zero = probabilistic_sharpe_ratio(values)
    dsr = probabilistic_sharpe_ratio(values, benchmark_sr=hurdle)
    if psr_vs_zero is None or dsr is None:
        return None
    return DeflatedSharpe(
        dsr=dsr,
        psr_vs_zero=psr_vs_zero,
        hurdle_annualised=hurdle * scale,
        n_trials=n,
        trial_sharpe_std_annualised=std_annualised,
    )


def _block_length(columns: npt.NDArray[np.float64]) -> int:
    """Largest automatic stationary block length over the columns.

    Constant columns have no dependence to preserve and break the
    estimator, so they are left out; without a finite estimate the block
    length is one.
    """
    varying = columns[:, columns.max(axis=0) > columns.min(axis=0)]
    if varying.shape[1] == 0:
        return 1
    # The estimate is invariant to each column's scale, but its numerics are
    # not: far from unit spread it silently falls back to one.
    varying = rescaled_columns(varying)
    estimates = optimal_block_length(varying)["stationary"].to_numpy()
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return 1
    return max(1, math.ceil(float(finite.max())))


def _returns_block_size(columns: npt.NDArray[np.float64]) -> int:
    """Block length that keeps the dependence of returns and of volatility.

    ``columns`` holds one return series per column. Their squares carry
    the persistence of volatility that the returns themselves may not show.
    """
    # Rescale before squaring, so returns near 1e200 do not square to inf.
    columns = rescaled_columns(columns)
    return _block_length(np.hstack([columns, columns**2]))


def _period_sharpe(values: npt.NDArray[np.float64]) -> float | None:
    # Equal values can still yield a rounding-error std, so test spread exactly.
    if values.size < 2 or float(values.max()) == float(values.min()):
        return None
    # The ratio is scale-invariant in exact arithmetic. Power-of-two
    # rescaling reduces overflow/underflow risk, within _scaling's floating
    # point limits. Each resample gets its own scale: a resample without an
    # outlier must not inherit that outlier's scale.
    values = rescaled(values)[0]
    std = float(values.std(ddof=1))
    return float(values.mean()) / std


def _resample_sharpe(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    sharpe = _period_sharpe(values)
    return np.array([sharpe if sharpe is not None else math.nan])


def _returns(returns: Iterable[float]) -> npt.NDArray[np.float64]:
    # An array or Series keeps its dtype for real_array to judge; any other
    # iterable, such as a generator, is materialised first.
    values = real_array(
        returns if getattr(returns, "dtype", None) is not None else list(returns),
        "returns",
    )
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must be finite")
    return values


def _annualisation(periods_per_year: float) -> float:
    value = _finite(periods_per_year, "periods_per_year")
    if value <= 0:
        raise ValueError("periods_per_year must be positive")
    return math.sqrt(value)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} is not numeric")
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite")
    return float(value)
