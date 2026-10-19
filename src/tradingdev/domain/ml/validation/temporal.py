"""Apply fold geometry to shared dates and verify actual label intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tradingdev.domain.ml.validation.schemes import build_validation_splits

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt


@dataclass(frozen=True)
class TemporalFold:
    """Row positions of a fold, selected on dates rather than storage order."""

    train_rows: tuple[int, ...]
    test_rows: tuple[int, ...]
    purged_rows: int
    embargoed_rows: int


def _mask(
    positions: npt.NDArray[np.int64], windows: Sequence[tuple[int, int]]
) -> npt.NDArray[np.bool_]:
    mask = np.zeros(len(positions), dtype=bool)
    for start, end in windows:
        mask |= (positions >= start) & (positions < end)
    return mask


def temporal_folds(
    frame: pd.DataFrame,
    *,
    scheme: str,
    horizon: int,
    n_folds: int,
    embargo: int,
    min_train_dates: int,
) -> list[TemporalFold]:
    """Split shared dates and purge overlapping original label intervals.

    Intervals precede feature drops and calendar alignment. Reject carried
    observations; exclude a whole date if any symbol's label overlaps.
    """
    required = {"timestamp", "label_start", "label_end"}
    if min_train_dates < 1:
        raise ValueError("min_train_bars must be at least 1")
    if not required <= set(frame.columns):
        raise ValueError("dataset has no label intervals; rebuild it before validation")
    dates = pd.to_datetime(frame["timestamp"], utc=True)
    starts = pd.to_datetime(frame["label_start"], utc=True)
    ends = pd.to_datetime(frame["label_end"], utc=True)
    if dates.isna().any() or starts.isna().any() or ends.isna().any():
        raise ValueError("label timestamps must be finite")
    if (starts > dates).any() or (ends <= starts).any():
        raise ValueError("invalid label observation interval")
    if not starts.equals(dates):
        raise ValueError(
            "validation requires original observation dates; rebuild the panel "
            "with intersection or missing_policy=drop"
        )
    calendar = pd.DatetimeIndex(dates.unique()).sort_values()
    positions = np.asarray(
        calendar.get_indexer(pd.DatetimeIndex(dates)), dtype=np.int64
    )
    settings = {
        "n_bars": len(calendar),
        "horizon": horizon,
        "n_folds": n_folds,
        "min_train_bars": 1,
    }
    splits = build_validation_splits(scheme, embargo=embargo, **settings)
    unembargoed = build_validation_splits(scheme, embargo=0, **settings)
    folds: list[TemporalFold] = []
    for split, without_embargo in zip(splits, unembargoed, strict=True):
        test_mask = _mask(positions, [split.test])
        train_mask = _mask(positions, split.train)
        purged_mask = _mask(positions, without_embargo.train)
        eligible = positions < split.test[0]
        if scheme == "purged_kfold":
            eligible |= positions >= split.test[1]
        test_start = starts.loc[test_mask].min()
        test_end = ends.loc[test_mask].max()
        overlaps = (ends >= test_start) & (starts <= test_end)
        unsafe_dates = dates.loc[train_mask & overlaps].unique()
        unsafe_mask = np.asarray(dates.isin(unsafe_dates), dtype=bool) & train_mask
        train_mask &= ~unsafe_mask
        if len(np.unique(positions[train_mask])) < min_train_dates:
            raise ValueError(
                "not enough training bars (distinct dates) remain "
                "after label interval purge"
            )
        train_rows = tuple(int(row) for row in np.flatnonzero(train_mask))
        test_rows = tuple(int(row) for row in np.flatnonzero(test_mask))
        if not train_rows or not test_rows:
            raise ValueError("validation requires non-empty training and test rows")
        if overlaps.loc[train_mask].any():
            raise ValueError("validation label intervals overlap")
        folds.append(
            TemporalFold(
                train_rows=train_rows,
                test_rows=test_rows,
                purged_rows=int((eligible & ~purged_mask).sum() + unsafe_mask.sum()),
                embargoed_rows=int(
                    (purged_mask & ~_mask(positions, split.train)).sum()
                ),
            )
        )
    return folds
