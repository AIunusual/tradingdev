"""Long/short and long-only simulated returns from a ranked signal.

:mod:`tradingdev.domain.ml.evaluation.metrics` answers *does the signal rank
right?* This module answers the next question: *if I had actually traded on
that rank, what would the return stream look like?*

Bars are sorted into prediction quantiles. The top quantile is held long and,
for the long/short book, the bottom quantile is held short; every period the
book's return is the realised forward return of the bars it holds. The output
is a per-period return series (plus the mean return of each quantile), so it
can be compounded, Sharpe'd or drawdown'd downstream rather than being reduced
to a single average.

Nothing here looks ahead: the return at bar ``t`` is always paired with the
signal at the same bar, exactly as the signal would have been available then.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradingdev.domain.ml.evaluation.stratified import quantile_bins


@dataclass(frozen=True)
class SimulatedReturns:
    """Per-period returns of a signal-driven book and its quantile table."""

    long_short: pd.Series
    long_only: pd.Series
    quantile_means: tuple[float, ...]
    n_quantiles: int

    @property
    def long_short_cumulative(self) -> float:
        """Compounded return of the long/short book: prod(1 + r) - 1."""
        return float(np.prod(1.0 + self.long_short.to_numpy()) - 1.0)

    @property
    def long_only_cumulative(self) -> float:
        """Compounded return of the long-only book: prod(1 + r) - 1."""
        return float(np.prod(1.0 + self.long_only.to_numpy()) - 1.0)

    @property
    def long_short_mean(self) -> float:
        """Mean per-period return of the long/short book."""
        return float(self.long_short.mean())

    @property
    def long_only_mean(self) -> float:
        """Mean per-period return of the long-only book."""
        return float(self.long_only.mean())


def simulate_strategy_returns(
    signal: pd.Series,
    forward_returns: pd.Series,
    *,
    n_quantiles: int = 5,
) -> SimulatedReturns:
    """Build long/short and long-only return streams from a ranked signal.

    Args:
        signal: Predicted value per bar; higher means stronger conviction.
        forward_returns: Realised return over the label horizon per bar,
            sharing ``signal``'s index so the two align by bar.
        n_quantiles: Requested number of prediction quantiles; the upper bound
            on what is formed, since a discrete signal yields one stratum per
            distinct value. Must allow at least two distinct quantiles so the
            long/short book has both a long and a short leg.

    Returns:
        The per-period long/short and long-only returns, the mean forward
        return of each quantile (ascending by signal), and the number of
        quantiles actually formed.
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2 for a long/short book")

    frame = pd.concat(
        [pd.Series(signal).rename("signal"), pd.Series(forward_returns).rename("fwd")],
        axis=1,
    ).dropna()
    if len(frame) < n_quantiles:
        raise ValueError(
            f"need at least {n_quantiles} finite bars to form {n_quantiles} "
            f"quantiles, got {len(frame)}"
        )

    labels = quantile_bins(frame["signal"], n_quantiles)
    distinct = int(labels.nunique())
    if distinct < 2:
        raise ValueError(
            f"signal collapsed to {distinct} quantile(s) after dropping ties; "
            "a long/short book needs at least two distinct quantiles"
        )

    top, bottom = distinct - 1, 0
    forward = frame["fwd"].to_numpy()
    is_top = labels.to_numpy() == top
    is_bottom = labels.to_numpy() == bottom

    long_only = np.where(is_top, forward, 0.0)
    long_short = np.where(is_top, forward, np.where(is_bottom, -forward, 0.0))

    quantile_means = tuple(
        float(frame.loc[labels == quantile, "fwd"].mean())
        for quantile in range(distinct)
    )

    return SimulatedReturns(
        long_short=pd.Series(long_short, index=frame.index, name="long_short"),
        long_only=pd.Series(long_only, index=frame.index, name="long_only"),
        quantile_means=quantile_means,
        n_quantiles=distinct,
    )
