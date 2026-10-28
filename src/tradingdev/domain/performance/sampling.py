"""Declared bar frequencies: their duration and their daily-sampling fitness."""

from __future__ import annotations

import re

DAILY_EQUITY_METRICS = frozenset(
    {
        "annual_return",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "annual_volatility",
        "daily_max_drawdown",
        *(
            f"{period}_pnl_{stat}"
            for period in ("daily", "monthly")
            for stat in ("mean", "std", "min", "max", "median")
        ),
    }
)

_DAY_NS = 86_400_000_000_000
_FIXED_UNIT_NS = {
    "ns": 1,
    "us": 1_000,
    "ms": 1_000_000,
    "s": 1_000_000_000,
    "sec": 1_000_000_000,
    "m": 60_000_000_000,
    "min": 60_000_000_000,
    "T": 60_000_000_000,
    "h": 3_600_000_000_000,
    "H": 3_600_000_000_000,
    "d": _DAY_NS,
    "D": _DAY_NS,
}
_WEEK_UNITS = frozenset({"w", "wk", "W"})
_COARSE_UNITS = frozenset({"w", "wk", "W", "mo", "M", "MS", "ME"})


def _parse_frequency(frequency: str) -> tuple[int, str] | None:
    """Split ``"4h"`` into ``(4, "h")``; ``None`` when the shape is unusable."""
    match = re.fullmatch(r"([0-9]+)?([A-Za-z]+)", frequency.strip())
    if match is None:
        return None
    try:
        count = int(match.group(1) or "1")
    except ValueError:
        return None
    if count <= 0:
        return None
    return count, match.group(2)


def bar_duration_ns(frequency: str) -> int | None:
    """Length of one bar in nanoseconds, or ``None`` when it is ambiguous.

    Months and years have no fixed length, so they are refused rather than
    approximated; a caller that annualises on them would be inventing a
    calendar. Unit case is significant: ``m`` is minutes and ``M`` is months.
    """
    parsed = _parse_frequency(frequency)
    if parsed is None:
        return None
    count, unit = parsed
    if unit in _WEEK_UNITS:
        return count * 7 * _DAY_NS
    unit_ns = _FIXED_UNIT_NS.get(unit)
    return None if unit_ns is None else count * unit_ns


def daily_observation_unavailable_reason(frequency: str) -> str | None:
    """Reject bars coarser than a day without inferring bars from timestamp gaps.

    The multiplier must be a positive integer. Unit case is significant: ``m``
    is minutes and ``M`` is months. Unknown formats cannot establish daily
    observations. Calendar gaps in an explicitly daily series remain allowed.
    """
    parsed = _parse_frequency(frequency)
    if parsed is None:
        return "unknown_bar_frequency"
    count, unit = parsed
    if unit in _COARSE_UNITS:
        return "unsupported_daily_sampling"
    unit_ns = _FIXED_UNIT_NS.get(unit)
    if unit_ns is None:
        return "unknown_bar_frequency"
    if count * unit_ns > _DAY_NS:
        return "unsupported_daily_sampling"
    return None
