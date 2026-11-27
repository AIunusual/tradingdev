"""Timestamp-preserving alignment contract for paired comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.statistics.alignment import align_masked_pair


def _series(
    days: tuple[int, ...], values: tuple[float | None, ...], *, tz: str = "UTC"
) -> pd.Series:
    index = pd.to_datetime(
        [f"2024-01-{day:02d}T00:00:00" for day in days], utc=True
    ).tz_convert(tz)
    return pd.Series(values, index=index, dtype="float64")


def test_partial_overlap_reports_original_lengths_and_exclusions() -> None:
    a = _series((1, 2, 3), (0.01, 0.02, 0.03))
    b = _series((2, 3, 4), (0.04, 0.05, 0.06))
    result = align_masked_pair(a, b)
    assert result.original_lengths == {"a": 3, "b": 3}
    assert result.overlap == 2
    assert result.valid_count == 2
    assert [item.reason for item in result.excluded] == ["not_in_b", "not_in_a"]
    assert list(result.paired.index) == list(a.index[1:])
    assert result.calendar_policy == "exact_utc_intersection_no_fill"


def test_no_overlap_has_an_independent_reason() -> None:
    a = _series((1, 2), (0.01, 0.02))
    b = _series((3, 4), (0.03, 0.04))
    result = align_masked_pair(a, b)
    assert result.overlap == 0
    assert result.valid_count == 0
    assert {item.reason for item in result.excluded} == {"not_in_b", "not_in_a"}


def test_timezone_mixing_and_naive_index_are_rejected() -> None:
    aware = _series((1, 2), (0.01, 0.02))
    naive = pd.Series(
        [0.01, 0.02],
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        dtype="float64",
    )
    with pytest.raises(ValueError, match="timezone-naive"):
        align_masked_pair(naive, aware)
    local = _series((1, 2), (0.01, 0.02), tz="Asia/Taipei")
    result = align_masked_pair(aware, local)
    assert result.overlap == 2


def test_warm_up_nan_keeps_coordinates_and_is_not_shifted() -> None:
    a = _series((1, 2, 3, 4), (None, None, 0.03, 0.04))
    b = _series((1, 2, 3, 4), (None, 0.02, 0.03, 0.04))
    result = align_masked_pair(a, b)
    assert result.overlap == 4
    assert result.valid_count == 2
    assert list(result.paired.index) == list(a.index[2:])
    assert result.warmup_excluded == {"a": 2, "b": 1}
    assert result.gap_excluded == {"a": 0, "b": 0}
    assert result.valid_mask == (False, False, True, True)


def test_unsorted_input_is_classified_by_timestamp_order() -> None:
    index = pd.to_datetime(
        [
            "2024-01-01T00:00:00",
            "2024-01-02T00:00:00",
            "2024-01-03T00:00:00",
            "2024-01-04T00:00:00",
        ],
        utc=True,
    )
    a = pd.Series([0.04, np.nan, 0.02, np.nan], index=index, dtype="float64")
    b = pd.Series([0.04, 0.03, 0.02, 0.01], index=index, dtype="float64")
    result = align_masked_pair(a.iloc[np.array([2, 0, 3, 1])], b)
    assert result.warmup_excluded == {"a": 0, "b": 0}
    assert result.gap_excluded["a"] == 2
    assert list(result.paired.index) == [index[0], index[2]]


def test_interior_gap_is_not_relabelled_as_warmup() -> None:
    a = _series((1, 2, 3, 4), (0.01, None, 0.03, 0.04))
    b = _series((1, 2, 3, 4), (0.01, 0.02, 0.03, 0.04))
    result = align_masked_pair(a, b)
    assert result.warmup_excluded == {"a": 0, "b": 0}
    assert result.gap_excluded == {"a": 1, "b": 0}
    assert any(
        item.reason == "interior_gap" and item.absent_from == "a"
        for item in result.excluded
    )


def test_duplicate_timestamps_are_rejected() -> None:
    a = _series((1, 2, 3), (0.01, 0.02, 0.03))
    duplicated = pd.concat([a.iloc[:1], a])
    with pytest.raises(ValueError, match="duplicate timestamps"):
        align_masked_pair(duplicated, a)


def test_missing_calendar_period_is_reported_not_compressed() -> None:
    a = _series((1, 2, 4), (0.01, 0.02, 0.04))
    b = _series((1, 2, 4), (0.01, 0.02, 0.04))
    result = align_masked_pair(a, b)
    assert result.cadence_complete is False
    assert result.missing_timestamps == ("2024-01-03T00:00:00+00:00",)
    assert result.valid_count == 3


def test_declared_daily_cadence_with_hourly_steps_is_incomplete() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="12h", tz="UTC")
    series = pd.Series([0.01, 0.02, 0.03, 0.04], index=index, dtype="float64")
    result = align_masked_pair(series, series, cadence="1d")
    assert result.valid_count == 4
    assert result.cadence_complete is False


def test_unsupported_cadence_refuses_inference_qualification() -> None:
    a = _series((1, 2, 3), (0.01, 0.02, 0.03))
    result = align_masked_pair(a, a, cadence="3d")
    assert result.cadence == "unsupported"
    assert result.cadence_complete is False


def test_non_finite_values_stay_excluded_without_imputation() -> None:
    a = _series((1, 2, 3), (0.01, np.nan, 0.03))
    b = _series((1, 2, 3), (0.01, 0.02, 0.03))
    result = align_masked_pair(a, b)
    assert result.valid_count == 2
    assert np.isnan(result.common.loc[a.index[1], "a"])
    assert result.paired.loc[a.index[2], "a"] == 0.03
