"""Quantile (stratified) analysis of a signal against realised returns.

A single average IC can hide a signal that only works in the tails. Splitting
bars into prediction quantiles exposes whether the realised return actually
rises monotonically across them, which is the property a long/short rule
depends on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tradingdev.domain.ml.evaluation.metrics import MetricScalar, _aligned_frame
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = setup_logger(__name__)


def quantile_bins(prediction: pd.Series, n_bins: int) -> pd.Series:
    """Assign every bar a 0-based stratum, ascending by signal.

    A continuous signal is cut at quantile edges. A discrete signal - a
    direction model emits only ``-1``/``0``/``1`` - has at most as many distinct
    values as the requested bins, so quantile edges can land on ties and
    ``duplicates="drop"`` would collapse the entire series into one bin. Such
    signals are grouped by distinct value instead, so ``-1``/``0``/``1`` still
    yields the short / flat / long strata a long-short book needs.

    Args:
        prediction: Signal values, already aligned and free of nulls.
        n_bins: Requested number of strata; the upper bound on what is formed.

    Returns:
        Integer Series of stratum numbers sharing ``prediction``'s index.
    """
    distinct = int(prediction.nunique())
    if distinct <= n_bins:
        # One stratum per distinct value; ties are the signal, not noise.
        return (prediction.rank(method="dense") - 1).astype(int)
    return pd.qcut(prediction, n_bins, labels=False, duplicates="drop").astype(int)


def quantile_strata(
    prediction: pd.Series | Sequence[float],
    forward_return: pd.Series | Sequence[float],
    *,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Group aligned bars into prediction quantiles.

    Args:
        prediction: Signal values, one per bar.
        forward_return: Realised return over the label horizon.
    n_bins: Requested number of quantiles; the upper bound on what is
        formed. A discrete signal yields one stratum per distinct value.

    Returns:
        DataFrame indexed by 1-based quantile number with columns
        ``count``, ``mean_prediction``, ``mean_forward_return`` and
        ``std_forward_return``.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    frame = _aligned_frame(prediction, forward_return)
    if len(frame) < n_bins:
        raise ValueError("not enough observations for the requested quantiles")
    bins = quantile_bins(frame["prediction"], n_bins)
    frame = frame.assign(_bin=bins + 1)
    grouped = frame.groupby("_bin")
    strata = pd.DataFrame(
        {
            "count": grouped.size().astype(int),
            "mean_prediction": grouped["prediction"].mean(),
            "mean_forward_return": grouped["forward_return"].mean(),
            "std_forward_return": grouped["forward_return"].std(ddof=1),
        }
    )
    strata.index.name = "bin"
    logger.info("Signal stratified into %d quantiles", len(strata))
    return strata


def strata_statistics(strata: pd.DataFrame) -> dict[str, MetricScalar]:
    """Reduce a quantile table to monotonicity and spread.

    Returns:
        ``strata_monotonicity`` (Spearman correlation between quantile rank
        and that quantile's mean forward return) and ``long_short_spread``
        (top quantile mean return minus bottom quantile mean return).
    """
    if strata.empty:
        return {"strata_monotonicity": None, "long_short_spread": None}
    means = strata["mean_forward_return"]
    if len(means) < 2 or means.isna().any():
        return {"strata_monotonicity": None, "long_short_spread": None}
    # Spearman: rank both sides, so any monotone ordering scores +/-1.
    observed = means.to_numpy(dtype=float)
    bin_ranks = pd.Series(means.index, dtype=float).rank().to_numpy()
    value_ranks = pd.Series(observed).rank().to_numpy()
    if np.nanstd(bin_ranks) == 0.0 or np.nanstd(value_ranks) == 0.0:
        monotonicity: float | None = None
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            monotonicity = float(np.corrcoef(bin_ranks, value_ranks)[0, 1])
        if not np.isfinite(monotonicity):
            monotonicity = None
    # Spread stays in return units; ranking it would destroy the magnitude.
    spread = float(observed[-1] - observed[0])
    return {
        "strata_monotonicity": monotonicity,
        "long_short_spread": spread if np.isfinite(spread) else None,
    }
