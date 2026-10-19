"""Descriptive prediction strata and their realised forward returns."""

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
    """Assign ascending, zero-based strata while preserving the input index.

    ``n_bins`` is an upper bound. With no more distinct values than bins,
    group by value so tied direction signals retain separate -1/0/1 strata;
    otherwise cut at quantile edges. Input must be aligned and non-null.
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
    """Summarize aligned bars in ascending, one-based prediction strata.

    ``n_bins`` bounds the count; discrete signals group by distinct value.
    Return count, mean_prediction, mean_forward_return and std_forward_return.
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
    """Return Spearman rank/mean-return monotonicity and top-minus-bottom spread."""
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
