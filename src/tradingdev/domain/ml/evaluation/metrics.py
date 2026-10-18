"""Signal quality metrics computed from a prediction and realised returns.

The functions here never touch a portfolio. They answer a single question:
*does this prediction carry information about the next move?* That question
can be answered before, and independently of, any backtest.

Single-asset note
-----------------
qlib computes IC cross-sectionally (one value per trading date, across many
instruments). With one instrument there is no cross-section, so the IC is
computed inside a **rolling window of bars**. The windows overlap, which
inflates a naive t-statistic; :func:`ic_statistics` therefore divides the bar
count by the window length to obtain an effective, non-overlapping sample
before scaling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import NDArray

logger = setup_logger(__name__)

IC_METHODS = ("spearman", "pearson")

type MetricScalar = int | float | None


def _aligned_frame(
    prediction: pd.Series | Sequence[float],
    forward_return: pd.Series | Sequence[float] | None = None,
    *,
    price: pd.Series | Sequence[float] | None = None,
) -> pd.DataFrame:
    """Stack equal-length series into a finite, row-aligned frame.

    Every column participates in the same ``dropna``, so the returned frame
    keeps positional alignment between the signal, its realised return and
    the price used for horizon decay.
    """
    pred_values = np.asarray(prediction, dtype=float)
    if pred_values.ndim != 1:
        raise ValueError("prediction must be one-dimensional")
    index = None
    if isinstance(prediction, pd.Series) and len(prediction.index) == len(pred_values):
        index = prediction.index
    columns: dict[str, NDArray[np.float64]] = {"prediction": pred_values}
    if forward_return is not None:
        fwd_values = np.asarray(forward_return, dtype=float)
        if fwd_values.ndim != 1 or fwd_values.shape != pred_values.shape:
            raise ValueError(
                "forward_return must be one-dimensional and match prediction length"
            )
        columns["forward_return"] = fwd_values
    if price is not None:
        price_values = np.asarray(price, dtype=float)
        if price_values.ndim != 1 or price_values.shape != pred_values.shape:
            raise ValueError(
                "price must be one-dimensional and match prediction length"
            )
        columns["price"] = price_values
    frame = pd.DataFrame(columns, index=index)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        raise ValueError("aligned signal inputs have no finite overlap")
    return frame


def _rowwise_rank(matrix: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Average ranks within each row, with a tie-safe fallback."""
    width = matrix.shape[1]
    order = np.argsort(matrix, axis=1, kind="mergesort")
    base = np.broadcast_to(np.arange(1, width + 1, dtype=float), matrix.shape).copy()
    ranks = np.empty(matrix.shape, dtype=float)
    np.put_along_axis(ranks, order, base, axis=1)
    sorted_values = np.take_along_axis(matrix, order, axis=1)
    has_ties = (sorted_values[:, 1:] == sorted_values[:, :-1]).any(axis=1)
    if has_ties.any():
        # Arbitrary tie-breaking is not the Spearman definition; fix those rows.
        for row in np.flatnonzero(has_ties):
            ranks[row] = pd.Series(matrix[row]).rank().to_numpy()
    return ranks


def _rowwise_corr(
    left: NDArray[np.floating[Any]], right: NDArray[np.floating[Any]]
) -> NDArray[np.float64]:
    """Row-wise Pearson correlation of two (rows, width) blocks."""
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    numerator = (left * right).sum(axis=1)
    denominator = np.sqrt((left * left).sum(axis=1) * (right * right).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denominator > 0, numerator / denominator, np.nan)
    return result


def _window_correlation(values: NDArray[np.float64], method: str) -> float:
    """Correlation of a (window, 2) block; NaN when a side is constant."""
    pred = values[:, 0]
    fwd = values[:, 1]
    if pred.size < 2:
        return float("nan")
    if np.nanstd(pred) == 0.0 or np.nanstd(fwd) == 0.0:
        return float("nan")
    left = pd.Series(pred).rank().to_numpy() if method == "spearman" else pred
    right = pd.Series(fwd).rank().to_numpy() if method == "spearman" else fwd
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.corrcoef(left, right)
    value = float(matrix[0, 1])
    return value if np.isfinite(value) else float("nan")


def rolling_ic(
    prediction: pd.Series | Sequence[float],
    forward_return: pd.Series | Sequence[float],
    *,
    window: int = 60,
    method: str = "spearman",
) -> pd.Series:
    """Correlate prediction and forward return inside a rolling window.

    Args:
        prediction: Signal values, one per bar.
        forward_return: Realised return over the label horizon, one per bar.
        window: Number of bars per correlation window.
        method: ``"spearman"`` (Rank IC) or ``"pearson"`` (IC).

    Returns:
        Series of per-window correlation coefficients with one entry per
        aligned bar; the first ``window - 1`` entries are NaN.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    if method not in IC_METHODS:
        raise ValueError(f"method must be one of {IC_METHODS}, got {method!r}")
    frame = _aligned_frame(prediction, forward_return)
    n_bars = len(frame)
    if n_bars < window:
        raise ValueError("not enough observations for the requested window")
    pred_windows = sliding_window_view(
        frame["prediction"].to_numpy(dtype=float), window
    )
    fwd_windows = sliding_window_view(
        frame["forward_return"].to_numpy(dtype=float), window
    )
    if method == "spearman":
        pred_windows = _rowwise_rank(pred_windows)
        fwd_windows = _rowwise_rank(fwd_windows)
    coefficients = _rowwise_corr(pred_windows, fwd_windows)
    return pd.Series(
        np.concatenate([np.full(window - 1, np.nan), coefficients]),
        index=frame.index,
    )


def ic_statistics(ic: pd.Series, *, window: int) -> dict[str, MetricScalar]:
    """Summarise a rolling IC series, correcting for window overlap.

    The t-statistic uses the *effective* sample size ``bars // window``
    instead of the number of windows, because overlapping windows are not
    independent observations.
    """
    series = pd.Series(ic).replace([np.inf, -np.inf], np.nan)
    n_bars = len(series)
    n_effective = max(1, n_bars // window)
    values = series.dropna()
    count = int(len(values))
    if count == 0:
        return {
            "ic_mean": None,
            "ic_std": None,
            "icir": None,
            "ic_t_stat": None,
            "ic_positive_rate": None,
            "n_periods": 0,
            "n_effective_periods": int(n_effective),
        }
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if count > 1 else 0.0
    icir = float(mean / std) if std > 0 else None
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": icir,
        "ic_t_stat": float(icir * np.sqrt(n_effective)) if icir is not None else None,
        "ic_positive_rate": float((values > 0).mean()),
        "n_periods": count,
        "n_effective_periods": int(n_effective),
    }


def _prefixed_statistics(
    ic: pd.Series, *, window: int, prefix: str
) -> dict[str, MetricScalar]:
    """Rename :func:`ic_statistics` keys for the Rank IC family."""
    raw = ic_statistics(ic, window=window)
    mapped: dict[str, MetricScalar] = {}
    for key, value in raw.items():
        mapped[key if key.startswith("n_") else f"{prefix}_{key}"] = value
    return mapped


def hit_rate(
    prediction: pd.Series | Sequence[float],
    forward_return: pd.Series | Sequence[float],
) -> float | None:
    """Fraction of bars where prediction and realised return share a sign."""
    frame = _aligned_frame(prediction, forward_return)
    scored = frame[(frame["prediction"] != 0) & (frame["forward_return"] != 0)]
    if scored.empty:
        return None
    agreement = np.sign(scored["prediction"]) == np.sign(scored["forward_return"])
    return float(agreement.mean())


def signal_turnover(prediction: pd.Series | Sequence[float]) -> float | None:
    """Mean absolute bar-to-bar change of the prediction."""
    values = pd.Series(np.asarray(prediction, dtype=float))
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        return None
    return float(values.diff().abs().mean())


def forward_returns(price: pd.Series | Sequence[float], horizon: int) -> pd.Series:
    """Realised simple return ``horizon`` bars ahead."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    series = pd.Series(np.asarray(price, dtype=float))
    return series.shift(-horizon) / series - 1.0


def ic_decay(
    prediction: pd.Series | Sequence[float],
    price: pd.Series | Sequence[float],
    horizons: Sequence[int] = (1, 3, 5, 10),
    *,
    method: str = "spearman",
) -> dict[str, MetricScalar]:
    """Full-sample correlation of the prediction at increasing horizons.

    A signal that decays slowly keeps a usable correlation at larger
    horizons, which widens the range of executable holding periods.
    """
    if method not in IC_METHODS:
        raise ValueError(f"method must be one of {IC_METHODS}, got {method!r}")
    frame = _aligned_frame(prediction, price=price)
    label = "rank_ic" if method == "spearman" else "ic"
    decay: dict[str, MetricScalar] = {}
    for horizon in horizons:
        key = f"{label}_h{horizon}"
        fwd = forward_returns(frame["price"], horizon)
        block = np.column_stack(
            [frame["prediction"].to_numpy(), fwd.to_numpy(dtype=float)]
        )
        block = block[~np.isnan(block).any(axis=1)]
        if block.shape[0] < 3:
            decay[key] = None
            continue
        value = _window_correlation(block, method)
        decay[key] = float(value) if np.isfinite(value) else None
    return decay


def signal_metrics(
    prediction: pd.Series | Sequence[float],
    forward_return: pd.Series | Sequence[float],
    *,
    window: int = 60,
    price: pd.Series | Sequence[float] | None = None,
    horizons: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, MetricScalar]:
    """Compute the full signal-evaluation metric set in one call.

    Both the Pearson (IC) and Spearman (Rank IC) families are always
    produced; they answer slightly different questions and cost one extra
    rolling pass.
    """
    frame = _aligned_frame(prediction, forward_return, price=price)
    pred = frame["prediction"]
    fwd = frame["forward_return"]

    metrics: dict[str, MetricScalar] = {}
    metrics.update(
        ic_statistics(
            rolling_ic(pred, fwd, window=window, method="pearson"), window=window
        )
    )
    metrics.update(
        _prefixed_statistics(
            rolling_ic(pred, fwd, window=window, method="spearman"),
            window=window,
            prefix="rank",
        )
    )
    metrics["hit_rate"] = hit_rate(pred, fwd)
    metrics["signal_turnover"] = signal_turnover(pred)
    if price is not None:
        metrics.update(ic_decay(pred, frame["price"], horizons, method="spearman"))
    logger.info(
        "Signal evaluated: %d bars, window=%d, rank_icir=%s",
        len(frame),
        window,
        metrics.get("rank_icir"),
    )
    return metrics
