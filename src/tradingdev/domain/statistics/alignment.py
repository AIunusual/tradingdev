"""Timestamp-preserving alignment of two saved observation series.

This module prepares paired inputs; it does not resample, impute, or invent
observations. Both series keep their original UTC coordinates. A value that
is missing (NaN/inf) stays at its coordinate and is classified as warm-up
when it precedes that series' first valid observation, or as an interior gap
otherwise. Only coordinates valid on both sides become a pair.

The first version supports exact UTC daily and hourly calendars. Any other
declared cadence is reported as unsupported; callers must not compress an
irregular timeline into a contiguous one before describing it as complete.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

_X: Literal["a"] = "a"
_Y: Literal["b"] = "b"
_SUPPORTED_CADENCES = frozenset({"1d", "1h"})
_FREQUENCIES = {"1d": "D", "1h": "h"}
_OFFSET = re.compile(r"(Z|[+-]\d{2}:\d{2})$")

type ExclusionReason = Literal[
    "not_in_a", "not_in_b", "warmup", "interior_gap", "empty_series"
]


@dataclass(frozen=True)
class ExcludedObservation:
    """One original coordinate that did not become a valid pair."""

    timestamp: str
    absent_from: Literal["a", "b"]
    reason: ExclusionReason


@dataclass(frozen=True)
class MaskedPairAlignment:
    """The shared valid index plus the audit trail of everything removed."""

    common: pd.DataFrame
    paired: pd.DataFrame
    valid_mask: tuple[bool, ...]
    original_lengths: dict[str, int]
    excluded: tuple[ExcludedObservation, ...]
    overlap: int
    valid_count: int
    warmup_excluded: dict[str, int]
    gap_excluded: dict[str, int]
    missing_timestamps: tuple[str, ...]
    cadence: str
    cadence_complete: bool
    calendar_policy: Literal["exact_utc_intersection_no_fill"]
    timezone: Literal["UTC"]


def _utc_index(values: object, label: str) -> pd.DatetimeIndex:
    if not isinstance(values, pd.Index):
        raise ValueError(f"series {label} requires a DatetimeIndex")
    if not isinstance(values, pd.DatetimeIndex):
        raise ValueError(f"series {label} requires datetime timestamps")
    if values.tz is None:
        raise ValueError(
            f"series {label} has timezone-naive timestamps; "
            "localize them to UTC before comparison"
        )
    return pd.DatetimeIndex(values).tz_convert("UTC")


def _numeric(values: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        raise ValueError(f"series {label} must contain real numbers")
    try:
        converted = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"series {label} must contain real numbers") from exc
    return converted.astype("float64")


def _reason_map(series: pd.Series) -> dict[pd.Timestamp, ExclusionReason]:
    """Classify each missing coordinate on one series' own timeline."""
    valid = np.array(series.notna().to_numpy(), dtype=bool, copy=True)
    valid &= np.isfinite(series.to_numpy())
    found = np.flatnonzero(valid)
    if found.size == 0:
        return dict.fromkeys(series.index, "empty_series")
    result: dict[pd.Timestamp, ExclusionReason] = {}
    first = int(found[0])
    for position, timestamp in enumerate(series.index):
        if not bool(valid[position]):
            result[timestamp] = "warmup" if position < first else "interior_gap"
    return result


def align_masked_pair(
    a: pd.Series,
    b: pd.Series,
    *,
    cadence: str = "1d",
) -> MaskedPairAlignment:
    """Align two series on their exact shared UTC timestamps.

    Raises ``ValueError`` for malformed input (non-series, duplicate
    timestamps, timezone mixing, non-numeric values). Missing values are not
    malformed: they are preserved and reported as exclusions.
    """
    if not isinstance(a, pd.Series) or not isinstance(b, pd.Series):
        raise ValueError("both inputs must be pandas Series")
    a_index = _utc_index(a.index, _X)
    b_index = _utc_index(b.index, _Y)
    for label, index in ((_X, a_index), (_Y, b_index)):
        if index.has_duplicates:
            raise ValueError(f"series {label} has duplicate timestamps")
    left = _numeric(a, _X)
    right = _numeric(b, _Y)
    left.index = a_index
    right.index = b_index

    common = a_index.intersection(b_index).sort_values()
    frame = pd.concat(
        [left.reindex(common).rename(_X), right.reindex(common).rename(_Y)],
        axis=1,
        join="inner",
    )
    left_reasons = _reason_map(left)
    right_reasons = _reason_map(right)
    excluded: list[ExcludedObservation] = []
    for timestamp in a_index.difference(b_index):
        excluded.append(ExcludedObservation(timestamp.isoformat(), _Y, "not_in_b"))
    for timestamp in b_index.difference(a_index):
        excluded.append(ExcludedObservation(timestamp.isoformat(), _X, "not_in_a"))
    for timestamp in common:
        reason = left_reasons.get(timestamp)
        if reason is not None:
            excluded.append(ExcludedObservation(timestamp.isoformat(), _X, reason))
        reason = right_reasons.get(timestamp)
        if reason is not None:
            excluded.append(ExcludedObservation(timestamp.isoformat(), _Y, reason))

    finite = np.isfinite(frame.to_numpy(dtype="float64"))
    mask = tuple(bool(value) for value in np.all(finite, axis=1))
    paired = frame.iloc[np.all(finite, axis=1)]

    missing: tuple[str, ...] = ()
    complete = False
    if cadence in _SUPPORTED_CADENCES and len(common) >= 2:
        expected = pd.date_range(
            common[0], common[-1], freq=_FREQUENCIES[cadence], tz="UTC"
        )
        missing = tuple(
            timestamp.isoformat() for timestamp in expected.difference(common)
        )
        complete = len(missing) == 0

    def count(reason: ExclusionReason, side: str) -> int:
        return sum(
            1 for item in excluded if item.absent_from == side and item.reason == reason
        )

    return MaskedPairAlignment(
        common=frame,
        paired=paired,
        valid_mask=mask,
        original_lengths={_X: int(len(left)), _Y: int(len(right))},
        excluded=tuple(excluded),
        overlap=int(len(common)),
        valid_count=int(len(paired)),
        warmup_excluded={_X: count("warmup", _X), _Y: count("warmup", _Y)},
        gap_excluded={
            _X: count("interior_gap", _X) + count("empty_series", _X),
            _Y: count("interior_gap", _Y) + count("empty_series", _Y),
        },
        missing_timestamps=missing,
        cadence=cadence if cadence in _SUPPORTED_CADENCES else "unsupported",
        cadence_complete=complete,
        calendar_policy="exact_utc_intersection_no_fill",
        timezone="UTC",
    )


def finite_series(values: pd.Series) -> bool:
    """True when a series carries only finite numeric observations."""
    return bool(np.all(np.isfinite(values.to_numpy(dtype="float64"))))


def is_finite_number(value: object) -> bool:
    """Shared guard for frozen numeric settings without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(float(value))
