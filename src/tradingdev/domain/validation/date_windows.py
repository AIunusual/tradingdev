"""UTC calendar-day boundaries shared by walk-forward execution and display."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta


class WalkForwardSplitError(ValueError):
    """Raised when explicit walk-forward dates do not describe usable folds."""


def utc_calendar_day(value: datetime) -> date:
    """Interpret naive values as UTC and convert offsets before taking the day."""
    if value.tzinfo is None:
        return value.date()
    return value.astimezone(UTC).date()


def inclusive_day_window(begin: datetime, end: datetime) -> tuple[datetime, datetime]:
    """Half-open UTC range covering every calendar day from ``begin`` to ``end``.

    Offset-aware values are converted to UTC before discarding their time.
    The upper boundary is midnight after the inclusive end day.
    """
    lower = datetime.combine(utc_calendar_day(begin), time.min, tzinfo=UTC)
    upper = datetime.combine(utc_calendar_day(end), time.min, tzinfo=UTC)
    if upper < lower:
        msg = f"Window end {end} precedes its start {begin}"
        raise WalkForwardSplitError(msg)
    try:
        exclusive_upper = upper + timedelta(days=1)
    except OverflowError as exc:
        msg = f"Window end {end} has no representable following UTC day"
        raise WalkForwardSplitError(msg) from exc
    return lower, exclusive_upper
