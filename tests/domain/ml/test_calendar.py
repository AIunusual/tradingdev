"""Calendar and panel alignment over explicit timestamps, without market data or IO."""

from __future__ import annotations

import pandas as pd
import pytest

from tradingdev.domain.ml.calendar import (
    UNAVAILABLE,
    CalendarMode,
    MissingPolicy,
    TradingCalendar,
    align_frames,
    align_index,
    build_calendar,
)

_JAN = pd.date_range("2024-01-01", periods=5, freq="D")
# A late lister: it misses 2024-01-01 and 2024-01-02 entirely.
_LATE = pd.DatetimeIndex(list(_JAN[2:]))


def _frame(index: pd.DatetimeIndex, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": values}, index=index)


def test_intersection_keeps_only_dates_every_symbol_traded() -> None:
    calendar = build_calendar({"A": _JAN, "B": _LATE})
    assert calendar.mode is CalendarMode.INTERSECTION
    assert tuple(calendar.symbols) == ("A", "B")
    assert list(calendar.dates) == list(_JAN[2:])
    assert calendar.n_dates == 3
    assert calendar.start == _JAN[2]
    assert calendar.end == _JAN[4]


def test_union_keeps_every_date_any_symbol_traded() -> None:
    extra = pd.DatetimeIndex([pd.Timestamp("2024-01-06")])
    calendar = build_calendar({"A": _JAN, "B": _LATE.append(extra)}, mode="union")
    assert calendar.mode is CalendarMode.UNION
    assert list(calendar.dates) == list(_JAN) + [pd.Timestamp("2024-01-06")]


def test_symbols_are_sorted_for_stable_output() -> None:
    calendar = build_calendar({"zeta": _JAN, "alpha": _JAN})
    assert tuple(calendar.symbols) == ("alpha", "zeta")


def test_aware_indices_are_pinned_to_utc() -> None:
    aware = pd.DatetimeIndex(list(_JAN)).tz_localize("UTC")
    shifted = pd.DatetimeIndex(list(_JAN)).tz_localize("Asia/Tokyo").tz_convert("UTC")
    calendar = build_calendar({"utc": aware, "tokyo": shifted}, mode="union")
    assert calendar.dates[0].tz is not None
    assert list(calendar.dates) == list(aware.append(shifted).unique().sort_values())


def test_mixed_timezone_awareness_is_rejected() -> None:
    """Naive and aware timestamps are not comparable; guessing would shift bars."""
    aware = pd.DatetimeIndex(list(_JAN)).tz_localize("UTC")
    with pytest.raises(ValueError, match="uniformly timezone-aware"):
        build_calendar({"aware": aware, "naive": _JAN})


def test_no_symbols_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one symbol"):
        build_calendar({})


def test_duplicate_timestamps_are_rejected() -> None:
    duplicated = pd.DatetimeIndex([_JAN[0], _JAN[0], _JAN[1]])
    with pytest.raises(ValueError, match="duplicates"):
        build_calendar({"A": duplicated})


def test_nat_timestamps_are_rejected() -> None:
    broken = pd.DatetimeIndex([_JAN[0], pd.NaT])
    with pytest.raises(ValueError, match="NaT"):
        build_calendar({"A": broken})


def test_empty_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="no timestamps"):
        build_calendar({"A": pd.DatetimeIndex([])})


def test_disjoint_symbols_have_no_calendar() -> None:
    other = pd.date_range("2025-01-01", periods=3, freq="D")
    with pytest.raises(ValueError, match="no date is traded by every symbol"):
        build_calendar({"A": _JAN, "B": other})


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_calendar({"A": _JAN}, mode="weekly")


def test_calendar_rejects_unsorted_dates() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        TradingCalendar(
            mode=CalendarMode.UNION,
            symbols=("A",),
            dates=(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-01")),
        )


def test_coverage_reports_missing_dates_and_fraction() -> None:
    calendar = build_calendar({"A": _JAN, "B": _LATE}, mode="union")
    report = calendar.coverage(_LATE, symbol="B")
    assert report.complete is False
    assert report.n_missing == 2
    assert report.n_present == 3
    assert report.n_calendar_dates == 5
    assert report.coverage == pytest.approx(0.6)
    assert list(report.missing) == list(_JAN[:2])


def test_coverage_of_a_complete_symbol_has_no_gaps() -> None:
    calendar = build_calendar({"A": _JAN})
    report = calendar.coverage(_JAN, symbol="A")
    assert report.complete is True
    assert report.coverage == pytest.approx(1.0)


def test_coverage_refuses_a_timezone_mismatch() -> None:
    calendar = build_calendar({"A": _JAN.tz_localize("UTC")})
    with pytest.raises(ValueError, match="timezone-aware"):
        calendar.coverage(_JAN, symbol="A")


def test_align_index_maps_exact_dates_to_their_own_row() -> None:
    calendar = build_calendar({"A": _JAN})
    alignment = align_index(_JAN, calendar, symbol="A")
    assert alignment.positions == (0, 1, 2, 3, 4)
    assert alignment.staleness == (0, 0, 0, 0, 0)
    assert alignment.n_dropped == 0
    assert alignment.n_forward_filled == 0


def test_drop_policy_leaves_untraded_dates_unsupplied() -> None:
    calendar = build_calendar({"A": _JAN, "B": _LATE}, mode="union")
    alignment = align_index(_LATE, calendar, symbol="B")
    assert alignment.positions == (UNAVAILABLE, UNAVAILABLE, 0, 1, 2)
    assert alignment.n_dropped == 2
    assert alignment.dropped_positions == (0, 1)
    assert alignment.available() == (False, False, True, True, True)


def test_forward_fill_carries_the_last_bar() -> None:
    calendar = build_calendar({"A": _JAN, "B": _LATE}, mode="union")
    alignment = align_index(
        _LATE, calendar, symbol="B", policy=MissingPolicy.FORWARD_FILL
    )
    assert alignment.positions == (UNAVAILABLE, UNAVAILABLE, 0, 1, 2)
    assert alignment.n_dropped == 2
    assert alignment.n_forward_filled == 0


def test_forward_fill_carries_across_an_interior_gap() -> None:
    gapped = pd.DatetimeIndex(list(_JAN[:2]) + list(_JAN[3:]))
    calendar = build_calendar({"A": _JAN})
    alignment = align_index(gapped, calendar, symbol="A", policy="forward_fill")
    assert alignment.positions == (0, 1, 1, 2, 3)
    assert alignment.staleness == (0, 0, 1, 0, 0)
    assert alignment.n_dropped == 0
    assert alignment.n_forward_filled == 1


def test_max_staleness_bounds_how_far_a_bar_travels() -> None:
    sparse = pd.DatetimeIndex([_JAN[0], _JAN[4]])
    calendar = build_calendar({"A": _JAN})
    bounded = align_index(
        sparse, calendar, symbol="A", policy="forward_fill", max_staleness=2
    )
    assert bounded.positions == (0, 0, 0, UNAVAILABLE, 1)
    assert bounded.staleness == (0, 1, 2, 3, 0)


def test_zero_max_staleness_forbids_carrying() -> None:
    sparse = pd.DatetimeIndex([_JAN[0], _JAN[4]])
    calendar = build_calendar({"A": _JAN})
    alignment = align_index(
        sparse, calendar, symbol="A", policy="forward_fill", max_staleness=0
    )
    assert alignment.positions == (0, UNAVAILABLE, UNAVAILABLE, UNAVAILABLE, 1)


def test_negative_max_staleness_is_rejected() -> None:
    calendar = build_calendar({"A": _JAN})
    with pytest.raises(ValueError, match="must not be negative"):
        align_index(_JAN, calendar, symbol="A", max_staleness=-1)


def test_error_policy_refuses_any_gap() -> None:
    calendar = build_calendar({"A": _JAN})
    with pytest.raises(ValueError, match="B: no bar on 2024-01-01"):
        align_index(_LATE, calendar, symbol="B", policy=MissingPolicy.ERROR)


def test_intersection_never_drops_a_date() -> None:
    """Intersection is the safe default: no symbol is asked to invent a row."""
    frames = {
        "A": _frame(_JAN, [1.0, 2.0, 3.0, 4.0, 5.0]),
        "B": _frame(_LATE, [3.0, 4.0, 5.0]),
    }
    alignment = align_frames(frames)
    assert list(alignment.calendar.dates) == list(_JAN[2:])
    assert alignment.dropped_dates == ()
    assert list(alignment.frames["A"]["close"]) == [3.0, 4.0, 5.0]
    assert list(alignment.frames["B"]["close"]) == [3.0, 4.0, 5.0]


def test_union_with_drop_removes_dates_a_symbol_cannot_supply() -> None:
    frames = {
        "A": _frame(_JAN, [1.0, 2.0, 3.0, 4.0, 5.0]),
        "B": _frame(_LATE, [3.0, 4.0, 5.0]),
    }
    alignment = align_frames(frames, mode="union")
    assert list(alignment.calendar.dates) == list(_JAN[2:])
    assert alignment.dropped_dates == tuple(_JAN[:2])
    assert list(alignment.frames["A"]["close"]) == [3.0, 4.0, 5.0]


def test_align_frames_reports_coverage_against_the_requested_calendar() -> None:
    frames = {"A": _frame(_JAN, [1.0] * 5), "B": _frame(_LATE, [1.0] * 3)}
    alignment = align_frames(frames, mode="union")
    (_, late) = alignment.coverage
    assert late.symbol == "B"
    assert late.n_calendar_dates == 5
    assert late.n_missing == 2


def test_align_frames_forward_fills_and_counts_the_carry() -> None:
    gapped = pd.DatetimeIndex(list(_JAN[:2]) + list(_JAN[3:]))
    frames = {
        "A": _frame(_JAN, [1.0, 2.0, 3.0, 4.0, 5.0]),
        "B": _frame(gapped, [10.0, 20.0, 40.0, 50.0]),
    }
    alignment = align_frames(frames, mode="union", policy="forward_fill")
    # The panel keeps every date: B's 2024-01-03 row is carried from 01-02.
    assert alignment.calendar.n_dates == 5
    assert list(alignment.frames["B"]["close"]) == [10.0, 20.0, 20.0, 40.0, 50.0]
    assert alignment.forward_filled == {"A": 0, "B": 1}
    assert alignment.dropped_dates == ()


def test_align_frames_keeps_the_calendar_as_the_shared_index() -> None:
    frames = {"A": _frame(_JAN, [1.0] * 5), "B": _frame(_LATE, [1.0] * 3)}
    alignment = align_frames(frames)
    for frame in alignment.frames.values():
        assert list(frame.index) == list(alignment.calendar.dates)


def test_align_frames_fails_when_no_date_survives() -> None:
    sparse = pd.DatetimeIndex([_JAN[0]])
    frames = {"A": _frame(sparse, [1.0]), "B": _frame(_LATE, [1.0, 2.0, 3.0])}
    with pytest.raises(ValueError, match="no date can be supplied by every symbol"):
        align_frames(frames, mode="union")


def test_align_frames_refuses_unsorted_frames() -> None:
    """Positions are computed on the sorted order, so rows would be mis-dated."""
    unsorted_index = pd.DatetimeIndex(list(reversed(_JAN)))
    frames = {
        "A": _frame(_JAN, [1.0, 2.0, 3.0, 4.0, 5.0]),
        "B": _frame(unsorted_index, [5.0, 4.0, 3.0, 2.0, 1.0]),
    }
    with pytest.raises(ValueError, match="B: frame timestamps must be sorted"):
        align_frames(frames)


def test_align_frames_requires_at_least_one_frame() -> None:
    with pytest.raises(ValueError, match="at least one symbol frame"):
        align_frames({})
