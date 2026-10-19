"""Causal long/short and long-only simulations over label holding intervals.

At a decision bar, position thresholds use only earlier signal observations.
After a warmup, positions are opened on a fixed ``horizon``-bar grid, so their
forward-return intervals never overlap. Each realised holding-interval return
is stored at its decision bar; intervening rows contain zero. These are not
mark-to-market bar returns, and their means are not annualised performance.

The quantile table is a separate full-sample descriptive statistic. It does
not choose positions and must not be used as evidence of a causal trade rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradingdev.domain.ml.evaluation.stratified import quantile_bins


@dataclass(frozen=True)
class SimulatedReturns:
    """Non-overlapping holding-interval returns, indexed by decision bar."""

    long_short: pd.Series
    long_only: pd.Series
    quantile_means: tuple[float, ...]
    n_quantiles: int

    @property
    def long_short_cumulative(self) -> float:
        """Compound the non-overlapping long/short returns: prod(1 + r) - 1."""
        return float(np.prod(1.0 + self.long_short.to_numpy()) - 1.0)

    @property
    def long_only_cumulative(self) -> float:
        """Compound the non-overlapping long-only returns: prod(1 + r) - 1."""
        return float(np.prod(1.0 + self.long_only.to_numpy()) - 1.0)

    @property
    def long_short_mean(self) -> float:
        """Mean return across decision rows, including inactive zero rows."""
        return float(self.long_short.mean())

    @property
    def long_only_mean(self) -> float:
        """Mean return across decision rows, including inactive zero rows."""
        return float(self.long_only.mean())


def _causal_positions(signal: pd.Series, n_quantiles: int) -> pd.Series:
    """Choose positions using expanding thresholds fitted strictly before t."""
    history = signal.shift(1).expanding(min_periods=n_quantiles)
    seen_distinct = (
        (signal.notna() & ~signal.duplicated()).cumsum().shift(1, fill_value=0)
    )
    discrete = seen_distinct <= n_quantiles
    lower = history.quantile(1.0 / n_quantiles).where(~discrete, history.min())
    upper = history.quantile(1.0 - 1.0 / n_quantiles).where(~discrete, history.max())
    ready = (seen_distinct >= 2) & lower.notna() & upper.notna()
    # A repeated value exactly on coincident cutoffs is neutral, not both legs.
    long = ready & (signal >= upper) & ((lower < upper) | (signal > upper))
    short = ready & (signal <= lower) & ((lower < upper) | (signal < lower))
    return pd.Series(np.select([long, short], [1, -1], default=0), index=signal.index)


def simulate_strategy_returns(
    signal: pd.Series,
    forward_returns: pd.Series,
    *,
    n_quantiles: int = 5,
    horizon: int = 1,
) -> SimulatedReturns:
    """Build causal, non-overlapping long/short and long-only return streams.

    Args:
        signal: Chronologically ordered predictions with a unique bar index.
            Thresholds use only earlier finite signals. The first
            ``n_quantiles`` observations are warmup, and at least two historical
            signal values are needed. With at most ``n_quantiles`` known values,
            the historical minimum/maximum select the short/long tails;
            otherwise expanding lower/upper quantiles select them. A signal
            exactly on coincident quantile cutoffs is neutral.
        forward_returns: Realised holding-interval return at each decision bar,
            sharing exactly the signal index. Non-finite pairs are omitted from
            the output, but never compress the grid or the signal history.
        n_quantiles: Requested number of quantiles, at least two. The returned
            quantile means describe the full sample and do not drive positions.
            Tied quantile edges can collapse this descriptive table to one bin
            even when the signal has multiple distinct values.
        horizon: Holding-interval length in bars, at least one. Only original
            positions ``0, horizon, 2 * horizon, ...`` may open a trade; other
            rows return zero. Warmup and missing values do not shift this grid.

    Returns:
        Interval returns indexed by their decision bars and full-sample
        quantile means. Compounding is valid for the non-overlapping intervals;
        the indexed series is not a mark-to-market equity curve and excludes
        transaction costs. Inputs must represent consecutive bars of one asset.
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2 for a long/short book")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if not signal.index.equals(forward_returns.index) or not signal.index.is_unique:
        raise ValueError(
            "signal and forward_returns must have the same unique bar index"
        )

    frame = pd.concat(
        [signal.rename("signal"), forward_returns.rename("fwd")], axis=1
    ).replace([np.inf, -np.inf], np.nan)
    finite = frame.dropna()
    if len(finite) < n_quantiles:
        raise ValueError(
            f"need at least {n_quantiles} finite bars to form {n_quantiles} "
            f"quantiles, got {len(finite)}"
        )

    labels = quantile_bins(finite["signal"], n_quantiles)
    distinct = int(labels.nunique())
    if finite["signal"].nunique() < 2:
        raise ValueError(
            "a long/short book needs at least two distinct signal values "
            "to define distinct quantiles"
        )

    positions = _causal_positions(frame["signal"], n_quantiles)
    on_grid = np.arange(len(frame)) % horizon == 0
    positions = positions.where(on_grid, 0).loc[finite.index]
    long_short = positions * finite["fwd"]
    long_only = positions.clip(lower=0) * finite["fwd"]
    quantile_means = tuple(
        float(finite.loc[labels == quantile, "fwd"].mean())
        for quantile in range(distinct)
    )

    return SimulatedReturns(
        long_short=long_short.rename("long_short"),
        long_only=long_only.rename("long_only"),
        quantile_means=quantile_means,
        n_quantiles=distinct,
    )
