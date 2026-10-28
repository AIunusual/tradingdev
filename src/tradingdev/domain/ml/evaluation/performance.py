"""Return, drawdown and market-exposure statistics for a simulated book.

These describe money, not signal quality: they are computed from the same
non-overlapping holding-interval returns the simulation produces, so a zero
return on an inactive row means the book held cash, not that the signal was
right. Annualisation needs a declared bar frequency, because a 1h bar and a
1d bar compound to the same total over different spans; anything that cannot
be annualised is reported as ``None`` rather than assuming 252 or 365.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tradingdev.domain.performance.sampling import bar_duration_ns
from tradingdev.domain.research_policy import CRYPTO_DAILY_PERIODS_PER_YEAR

if TYPE_CHECKING:
    from collections.abc import Sequence

_NS_PER_YEAR = CRYPTO_DAILY_PERIODS_PER_YEAR * 86_400 * 10**9


def periods_per_year(timeframe: str, horizon: int = 1) -> float | None:
    """Periods in a continuous crypto year for ``horizon``-bar intervals.

    The caller must verify the market and consecutive interval boundaries.

    ``None`` when the bar frequency has no fixed length (months, years) or is
    unparseable, so no caller can mistake a guess for a measurement.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    duration = bar_duration_ns(timeframe)
    if duration is None:
        return None
    return _NS_PER_YEAR / (duration * horizon)


@dataclass(frozen=True)
class ReturnMetrics:
    """Performance of one book, with the annualisation it assumed."""

    annual_return: float | None
    """Geometric: the per-period total compounded to a year."""
    max_drawdown: float | None
    """Largest peak-to-trough loss as a positive fraction of the peak."""
    sharpe: float | None
    """Mean per-period return over its standard deviation, annualised.

    The risk-free rate is zero; a positive rate would change the number.
    """
    alpha: float | None
    """Annualised mean return not explained by the benchmark's own return."""
    beta: float | None
    """Sensitivity of the book's per-period return to the benchmark's."""
    periods_per_year: float | None
    n_periods: int

    def as_metrics(self) -> dict[str, float | int | None]:
        """Metric-catalog shaped projection for evaluation and storage."""
        return {
            "annual_return": self.annual_return,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "alpha": self.alpha,
            "beta": self.beta,
            "periods_per_year": self.periods_per_year,
            # Named apart from the IC window count, which owns ``n_periods``.
            "n_return_periods": self.n_periods,
        }


def _finite(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def return_metrics(
    returns: pd.Series | Sequence[float],
    benchmark: pd.Series | Sequence[float],
    *,
    periods_per_year: float | None = None,
) -> ReturnMetrics:
    """Measure one book's returns against the benchmark it was traded on.

    ``returns`` and ``benchmark`` must be aligned period by period: the book's
    return over a holding interval and the asset's own return over the same
    interval. Non-finite pairs are dropped together, so a gap cannot shift the
    regression. ``beta`` and ``max_drawdown`` need no calendar and are reported
    even when ``periods_per_year`` is unknown; the annualised figures are not.
    """
    book = np.asarray(pd.Series(returns, dtype=float), dtype=float)
    market = np.asarray(pd.Series(benchmark, dtype=float), dtype=float)
    if book.shape != market.shape:
        raise ValueError("returns and benchmark must have equal length")
    keep = np.isfinite(book) & np.isfinite(market)
    book, market = book[keep], market[keep]
    n_periods = int(book.size)
    if n_periods == 0:
        return ReturnMetrics(None, None, None, None, None, periods_per_year, 0)

    # Drawdown is measured from a starting equity of 1.0, so the first period
    # can itself be the peak; compounding keeps it on the book's own scale.
    equity = np.concatenate(([1.0], np.cumprod(1.0 + book)))
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peak > 0, 1.0 - equity / peak, 0.0)
    max_drawdown = _finite(float(np.max(drawdown))) if drawdown.size else None

    # Once capital is nonpositive, later negative factors cannot revive it
    # into a valid compounded investment return.
    total = float(np.prod(1.0 + book) - 1.0)
    annual_return: float | None = None
    if periods_per_year is not None and np.all(equity > 0.0) and total > -1.0:
        with np.errstate(over="ignore", invalid="ignore"):
            annual_return = _finite(
                float(np.power(1.0 + total, periods_per_year / n_periods) - 1.0)
            )

    sharpe: float | None = None
    beta: float | None = None
    alpha: float | None = None
    if n_periods > 1:
        spread = float(np.std(book, ddof=1))
        if periods_per_year is not None and spread > 0.0:
            sharpe = _finite(float(np.mean(book)) / spread * np.sqrt(periods_per_year))
        market_variance = float(np.var(market, ddof=1))
        if market_variance > 0.0:
            covariance = float(np.cov(book, market, ddof=1)[0, 1])
            beta = _finite(covariance / market_variance)
            if beta is not None and periods_per_year is not None:
                alpha = _finite(
                    (float(np.mean(book)) - beta * float(np.mean(market)))
                    * periods_per_year
                )

    return ReturnMetrics(
        annual_return=annual_return,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        alpha=alpha,
        beta=beta,
        periods_per_year=periods_per_year,
        n_periods=n_periods,
    )
