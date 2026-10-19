"""Feature processors with separate learning and transformation phases.

Learn processors retain only fitted statistics; infer processors are
stateless. Transform never learns and raises ProcessorNotFittedError when
required statistics are missing. Callers must fit on training data only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

_DEFAULT_EPS = 1e-12


class ProcessorNotFittedError(RuntimeError):
    """Raised when a stateful processor transforms data it never learned from."""


def _numeric_fields(
    frame: pd.DataFrame,
    fields: Sequence[str] | None,
) -> list[str]:
    """Resolve the requested fields, defaulting to every numeric column."""
    if fields is None:
        return [
            str(column)
            for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column])
        ]
    missing = [name for name in fields if name not in frame.columns]
    if missing:
        raise KeyError(f"frame is missing requested field(s): {', '.join(missing)}")
    return list(fields)


def _safe_scale(scale: float, eps: float) -> float:
    """Collapse degenerate scales to 1.0 so constant columns cannot explode."""
    return scale if scale > eps else 1.0


class Processor(ABC):
    """Feature transformation with optional fitting; default fit is a no-op."""

    name: str = "processor"
    requires_fit: bool = False

    @property
    def fields(self) -> tuple[str, ...]:
        """Fields this processor touches; empty means "every numeric column"."""
        return ()

    @abstractmethod
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a transformed copy of ``frame``, learning nothing."""

    def fit(self, frame: pd.DataFrame) -> None:
        """Learn parameters from ``frame``; a no-op for infer processors."""
        return None

    def learned(self) -> dict[str, float]:
        """Parameters captured during :meth:`fit`, for audits and reports."""
        return {}


class _StatsProcessor(Processor, ABC):
    """Learn two scalars per field, exposed under _stat_names for inspection."""

    requires_fit = True
    _stat_names: tuple[str, str] = ("center", "scale")

    def __init__(
        self,
        fields: Sequence[str] | None = None,
        *,
        eps: float = _DEFAULT_EPS,
    ) -> None:
        self._requested = None if fields is None else tuple(fields)
        self._eps = eps
        self._stats: dict[str, tuple[float, float]] = {}
        self._fitted = False

    @property
    def fields(self) -> tuple[str, ...]:
        if self._stats:
            return tuple(self._stats)
        return self._requested or ()

    def fit(self, frame: pd.DataFrame) -> None:
        """Capture the statistics of ``frame``; ``frame`` must be training data."""
        if len(frame) == 0:
            raise ValueError(f"{self.name}: cannot fit on an empty frame")
        self._stats = {
            column: self._learn(frame[column].astype(float))
            for column in _numeric_fields(frame, self._requested)
        }
        self._fitted = True

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned statistics without recomputing them."""
        if not self._fitted:
            raise ProcessorNotFittedError(
                f"{self.name}: fit() on the training segment must precede transform()"
            )
        out = frame.copy()
        for column, (first, second) in self._stats.items():
            if column not in out.columns:
                raise KeyError(f"{self.name}: frame is missing field {column!r}")
            out[column] = self._apply(out[column].astype(float), first, second)
        return out

    def learned(self) -> dict[str, float]:
        """Expose the learned scalars as ``field__stat`` entries."""
        first_name, second_name = self._stat_names
        learned: dict[str, float] = {}
        for column, (first, second) in self._stats.items():
            learned[f"{column}__{first_name}"] = first
            learned[f"{column}__{second_name}"] = second
        return learned

    @abstractmethod
    def _learn(self, series: pd.Series) -> tuple[float, float]:
        """Return the two scalars that describe ``series``."""

    @abstractmethod
    def _apply(self, series: pd.Series, first: float, second: float) -> pd.Series:
        """Return ``series`` transformed by the scalars from :meth:`_learn`."""


class ZScoreNorm(_StatsProcessor):
    """Centre on the training mean and scale by the training standard deviation."""

    name = "zscore_norm"
    _stat_names = ("mean", "std")

    def _learn(self, series: pd.Series) -> tuple[float, float]:
        return float(series.mean()), float(series.std(ddof=0))

    def _apply(self, series: pd.Series, first: float, second: float) -> pd.Series:
        return (series - first) / _safe_scale(second, self._eps)


class RobustZScoreNorm(_StatsProcessor):
    """Centre on the training median and scale by IQR to limit outlier influence."""

    name = "robust_zscore_norm"
    _stat_names = ("median", "iqr")

    def _learn(self, series: pd.Series) -> tuple[float, float]:
        interquartile = series.quantile(0.75) - series.quantile(0.25)
        return float(series.median()), float(interquartile)

    def _apply(self, series: pd.Series, first: float, second: float) -> pd.Series:
        return (series - first) / _safe_scale(second, self._eps)


class MinMaxNorm(_StatsProcessor):
    """Map the training minimum and maximum onto ``feature_range``."""

    name = "minmax_norm"
    _stat_names = ("min", "max")

    def __init__(
        self,
        fields: Sequence[str] | None = None,
        *,
        feature_range: tuple[float, float] = (0.0, 1.0),
        eps: float = _DEFAULT_EPS,
    ) -> None:
        super().__init__(fields, eps=eps)
        self._low, self._high = feature_range

    def _learn(self, series: pd.Series) -> tuple[float, float]:
        return float(series.min()), float(series.max())

    def _apply(self, series: pd.Series, first: float, second: float) -> pd.Series:
        scaled = (series - first) / _safe_scale(second - first, self._eps)
        return scaled * (self._high - self._low) + self._low


class ClipOutliers(_StatsProcessor):
    """Clip using training mean/std bounds, never the inference distribution."""

    name = "clip_outliers"
    _stat_names = ("center", "scale")

    def __init__(
        self,
        fields: Sequence[str] | None = None,
        *,
        n_sigmas: float = 3.0,
        eps: float = _DEFAULT_EPS,
    ) -> None:
        super().__init__(fields, eps=eps)
        self._n_sigmas = n_sigmas

    def _learn(self, series: pd.Series) -> tuple[float, float]:
        return float(series.mean()), float(series.std(ddof=0))

    def _apply(self, series: pd.Series, first: float, second: float) -> pd.Series:
        scale = _safe_scale(second, self._eps)
        margin = self._n_sigmas * scale
        return series.clip(first - margin, first + margin)


class Fillna(Processor):
    """Fill missing values with a caller-supplied constant, without fitting.

    Specify fields when the frame also contains labels.
    """

    name = "fillna"
    requires_fit = False

    def __init__(
        self,
        fields: Sequence[str] | None = None,
        *,
        value: float = 0.0,
    ) -> None:
        self._requested = None if fields is None else tuple(fields)
        self._value = value

    @property
    def fields(self) -> tuple[str, ...]:
        return self._requested or ()

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fill the requested fields, leaving every other column untouched."""
        columns = _numeric_fields(frame, self._requested)
        out = frame.copy()
        if not columns:
            return out
        out[columns] = out[columns].fillna(self._value)
        return out


class DropnaRows(Processor):
    """Drop missing rows in selected fields before row-count-sensitive fitting.

    Training and inference frames may shrink by different amounts.
    """

    name = "dropna_rows"
    requires_fit = False

    def __init__(self, fields: Sequence[str] | None = None) -> None:
        self._requested = None if fields is None else tuple(fields)

    @property
    def fields(self) -> tuple[str, ...]:
        return self._requested or ()

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the rows that are complete in every requested field."""
        columns = _numeric_fields(frame, self._requested)
        if not columns:
            return frame.copy()
        return frame.dropna(subset=columns).copy()
