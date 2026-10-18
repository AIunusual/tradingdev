"""Multi-symbol trading calendars and panel alignment.

A single-instrument dataset has one obvious index: the bars the venue reported.
A *panel* of several instruments does not, because real markets do not agree
on when they traded:

* crypto trades continuously, equities do not;
* a symbol that listed late has no bars before its first trade;
* halts, maintenance windows and provider gaps punch holes in an index.

Joining such frames naively is where panel datasets go wrong. Reindexing on
the union fills the holes with ``NaN`` and leaves the model to guess; dropping
them silently reweights the sample towards whichever symbols traded most; and
an unbounded forward fill lets a week-old price pose as today's observation,
which is survivorship bias with a friendly face.

This module makes the choice explicit and auditable. :func:`build_calendar`
decides *which dates* the panel has, :func:`align_index` decides *where each
symbol's row comes from* on those dates, and both are pure index arithmetic:
no market data, no I/O, and no hidden state, so the whole policy can be unit
tested and reported back to the caller as :class:`SymbolCoverage`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

UNAVAILABLE = -1
"""Sentinel position for a calendar date a symbol cannot supply."""


class CalendarMode(StrEnum):
    """How the panel calendar is derived from the symbols' own indices."""

    INTERSECTION = "intersection"
    """Keep only dates every symbol traded; no row is ever invented."""

    UNION = "union"
    """Keep every date any symbol traded; holes need a missing-value policy."""


class MissingPolicy(StrEnum):
    """What to do when a symbol has no bar on a calendar date."""

    DROP = "drop"
    """Leave the date unsupplied; the date is removed from the panel."""

    FORWARD_FILL = "forward_fill"
    """Carry the last bar forward, bounded by ``max_staleness``."""

    ERROR = "error"
    """Refuse the panel; a gap is a data problem, not a modelling choice."""


def _normalize(index: object, *, label: str) -> pd.DatetimeIndex:
    """Coerce one symbol's timestamps into a sorted, unique, tz-aware-free index.

    Duplicates and ``NaT`` are rejected rather than silently collapsed: both
    mean the upstream bar source is broken, and every downstream position
    computed from a duplicated timestamp would be ambiguous.
    """
    try:
        # The caller may hand over a list, an Index or a Series; the shape is
        # validated at runtime, so the element type is only asserted here.
        converted = pd.DatetimeIndex(pd.Index(cast("Sequence[Any]", index)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: timestamps must be datetimes") from error
    if converted.hasnans:
        raise ValueError(f"{label}: timestamps contain NaT")
    if converted.empty:
        raise ValueError(f"{label}: no timestamps supplied")
    if converted.has_duplicates:
        raise ValueError(f"{label}: timestamps contain duplicates")
    return converted.sort_values()


def _to_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Put an index in UTC, treating naive timestamps as already UTC."""
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")


def _match_timezone(
    index: pd.DatetimeIndex, reference: pd.DatetimeIndex, *, label: str
) -> pd.DatetimeIndex:
    """Align a symbol's index to the calendar's tz-awareness.

    Naive and aware timestamps are not comparable, so mixing them is refused
    instead of silently producing an index where nothing ever matches.
    """
    reference_tz = reference.tz
    if (index.tz is None) != (reference_tz is None):
        expected = "timezone-aware" if reference_tz is not None else "naive"
        raise ValueError(f"{label}: timestamps must be {expected} to match calendar")
    if reference_tz is not None:
        return index.tz_convert(reference_tz)
    return index


@dataclass(frozen=True)
class SymbolCoverage:
    """How much of a calendar one symbol can actually supply."""

    symbol: str
    n_present: int
    n_calendar_dates: int
    missing: tuple[pd.Timestamp, ...]

    @property
    def n_missing(self) -> int:
        """Number of calendar dates without a bar for this symbol."""
        return len(self.missing)

    @property
    def coverage(self) -> float:
        """Fraction of calendar dates this symbol can supply."""
        if self.n_calendar_dates == 0:
            return 0.0
        return self.n_present / self.n_calendar_dates

    @property
    def complete(self) -> bool:
        """Whether the symbol has a bar on every calendar date."""
        return self.n_missing == 0


@dataclass(frozen=True)
class TradingCalendar:
    """The ordered dates a panel shares, plus how each symbol covers them."""

    mode: CalendarMode
    symbols: tuple[str, ...]
    dates: tuple[pd.Timestamp, ...]

    def __post_init__(self) -> None:
        """Reject an empty calendar, repeated symbols and unsorted dates."""
        if not self.symbols:
            raise ValueError("a calendar needs at least one symbol")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("calendar symbols must be unique")
        if not self.dates:
            raise ValueError("a calendar needs at least one date")
        for earlier, later in zip(self.dates, self.dates[1:], strict=False):
            if later <= earlier:
                raise ValueError("calendar dates must be strictly increasing")

    @property
    def n_dates(self) -> int:
        """Number of dates in the calendar."""
        return len(self.dates)

    @property
    def start(self) -> pd.Timestamp:
        """First calendar date."""
        return self.dates[0]

    @property
    def end(self) -> pd.Timestamp:
        """Last calendar date."""
        return self.dates[-1]

    def index(self) -> pd.DatetimeIndex:
        """The calendar as a :class:`pandas.DatetimeIndex`."""
        return pd.DatetimeIndex(list(self.dates))

    def coverage(self, index: Iterable[object], *, symbol: str) -> SymbolCoverage:
        """Report which calendar dates one symbol can supply.

        Args:
            index: The symbol's own timestamps.
            symbol: Name recorded in the report.

        Raises:
            ValueError: On an unusable index or a timezone mismatch.
        """
        source = _match_timezone(
            _normalize(index, label=symbol), self.index(), label=symbol
        )
        present = set(source)
        missing = tuple(date for date in self.dates if date not in present)
        return SymbolCoverage(
            symbol=symbol,
            n_present=self.n_dates - len(missing),
            n_calendar_dates=self.n_dates,
            missing=missing,
        )


@dataclass(frozen=True)
class CalendarAlignment:
    """Where each calendar date's row comes from for one symbol.

    ``positions[i]`` is the positional index of the source bar supplying
    calendar date ``i``, or :data:`UNAVAILABLE` when the policy leaves it
    unsupplied. ``staleness[i]`` counts how many calendar dates that bar was
    carried across: ``0`` means the symbol traded on that very date, ``-1``
    means there is no usable bar at all.
    """

    symbol: str
    positions: tuple[int, ...]
    staleness: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject mismatched position and staleness lengths."""
        if len(self.positions) != len(self.staleness):
            raise ValueError("positions and staleness must have equal length")

    @property
    def n_dropped(self) -> int:
        """Calendar dates this symbol cannot supply under the chosen policy."""
        return sum(1 for position in self.positions if position == UNAVAILABLE)

    @property
    def n_forward_filled(self) -> int:
        """Calendar dates filled from an earlier bar of the same symbol."""
        return sum(
            1
            for position, age in zip(self.positions, self.staleness, strict=True)
            if position != UNAVAILABLE and age > 0
        )

    @property
    def dropped_positions(self) -> tuple[int, ...]:
        """Calendar positions this symbol cannot supply."""
        return tuple(
            index
            for index, position in enumerate(self.positions)
            if position == UNAVAILABLE
        )

    def available(self) -> tuple[bool, ...]:
        """One flag per calendar date: does this symbol supply it?"""
        return tuple(position != UNAVAILABLE for position in self.positions)


def build_calendar(
    indices: Mapping[str, Iterable[object]],
    *,
    mode: CalendarMode | str = CalendarMode.INTERSECTION,
) -> TradingCalendar:
    """Derive the panel calendar from each symbol's own timestamps.

    Args:
        indices: One timestamp sequence per symbol.
        mode: :attr:`CalendarMode.INTERSECTION` keeps the dates every symbol
            traded, :attr:`CalendarMode.UNION` keeps every date any symbol
            traded.

    Returns:
        A sorted, unique :class:`TradingCalendar`.

    Raises:
        ValueError: On no symbols, an unusable index, a mixed-timezone input,
            an unknown mode, or an empty result.
    """
    if not indices:
        raise ValueError("at least one symbol is required")
    resolved = CalendarMode(mode)
    symbols = tuple(sorted(indices))
    normalized = {
        symbol: _normalize(indices[symbol], label=symbol) for symbol in symbols
    }
    # Naive and aware timestamps are not comparable, so a mix would silently
    # intersect to nothing (or to nonsense). It is almost always a loader bug,
    # and the fix belongs in the loader, not in a guess made here.
    awareness = [index.tz is not None for index in normalized.values()]
    if any(awareness) and not all(awareness):
        raise ValueError(
            "timestamps must be uniformly timezone-aware or uniformly naive"
        )
    if all(awareness):
        normalized = {symbol: _to_utc(index) for symbol, index in normalized.items()}

    if resolved is CalendarMode.INTERSECTION:
        common = set(normalized[symbols[0]])
        for symbol in symbols[1:]:
            common &= set(normalized[symbol])
        if not common:
            raise ValueError("no date is traded by every symbol")
        dates = pd.DatetimeIndex(sorted(common))
    else:
        union: set[pd.Timestamp] = set()
        for index in normalized.values():
            union |= set(index)
        dates = pd.DatetimeIndex(sorted(union))

    return TradingCalendar(mode=resolved, symbols=symbols, dates=tuple(dates))


def align_index(
    index: Iterable[object],
    calendar: TradingCalendar,
    *,
    symbol: str,
    policy: MissingPolicy | str = MissingPolicy.DROP,
    max_staleness: int | None = None,
) -> CalendarAlignment:
    """Map each calendar date to a row of one symbol's own index.

    Args:
        index: The symbol's own timestamps.
        calendar: The panel calendar to align onto.
        symbol: Name recorded in the result and in error messages.
        policy: How to treat calendar dates the symbol did not trade on.
        max_staleness: With :attr:`MissingPolicy.FORWARD_FILL`, the largest
            number of calendar dates one bar may be carried across. ``None``
            allows unbounded carry; ``0`` forbids carry (equivalent to
            :attr:`MissingPolicy.DROP`).

    Returns:
        Positions and staleness, one entry per calendar date.

    Raises:
        ValueError: On an unusable index, a timezone mismatch, a negative
            ``max_staleness``, or a gap under :attr:`MissingPolicy.ERROR`.
    """
    resolved = MissingPolicy(policy)
    if max_staleness is not None and max_staleness < 0:
        raise ValueError("max_staleness must not be negative")
    calendar_index = calendar.index()
    source = _match_timezone(
        _normalize(index, label=symbol), calendar_index, label=symbol
    )

    positions: list[int] = []
    staleness: list[int] = []
    for position, date in enumerate(calendar.dates):
        candidate = int(source.searchsorted(date, side="right")) - 1
        if candidate >= 0 and source[candidate] == date:
            positions.append(candidate)
            staleness.append(0)
            continue
        if resolved is MissingPolicy.ERROR:
            raise ValueError(f"{symbol}: no bar on {date.isoformat()}")
        if resolved is MissingPolicy.DROP or candidate < 0:
            # candidate < 0 means the symbol had not listed yet: there is no
            # earlier bar to carry, so the date stays unsupplied.
            positions.append(UNAVAILABLE)
            staleness.append(UNAVAILABLE)
            continue
        carried = source[candidate]
        anchor = int(calendar_index.searchsorted(carried, side="right")) - 1
        age = position - max(anchor, 0)
        if max_staleness is not None and age > max_staleness:
            positions.append(UNAVAILABLE)
        else:
            positions.append(candidate)
        staleness.append(age)

    return CalendarAlignment(
        symbol=symbol,
        positions=tuple(positions),
        staleness=tuple(staleness),
    )


@dataclass(frozen=True)
class PanelAlignment:
    """Several symbol frames reindexed onto one calendar.

    ``coverage`` is measured against the *requested* calendar (before the
    unusable dates were removed), so it explains what ``dropped_dates``
    removed instead of trivially reporting full coverage.
    """

    calendar: TradingCalendar
    frames: dict[str, pd.DataFrame]
    dropped_dates: tuple[pd.Timestamp, ...]
    coverage: tuple[SymbolCoverage, ...]
    forward_filled: dict[str, int]

    @property
    def n_dates(self) -> int:
        """Number of dates in the aligned calendar."""
        return self.calendar.n_dates


def align_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    mode: CalendarMode | str = CalendarMode.INTERSECTION,
    policy: MissingPolicy | str = MissingPolicy.DROP,
    max_staleness: int | None = None,
) -> PanelAlignment:
    """Reindex per-symbol frames onto one shared calendar.

    Each frame is indexed by its own timestamps; the returned frames carry the
    aligned calendar as their index, so a carried row keeps the *older* bar's
    column values while its index reports the calendar date it stands for.

    A date enters the panel only when every symbol can supply it, so the panel
    never contains a half-populated row. Dates no symbol combination can fill
    are reported through ``dropped_dates``.

    Args:
        frames: One timestamp-indexed frame per symbol, in ascending date order.
        mode: Calendar derivation, see :func:`build_calendar`.
        policy: Missing-date treatment, see :func:`align_index`.
        max_staleness: Carry bound for :attr:`MissingPolicy.FORWARD_FILL`.

    Returns:
        The aligned frames and the audit trail behind them.

    Raises:
        ValueError: On no frames, unusable or unsorted indices, or an empty
            result. An unsorted index is refused rather than sorted here,
            because positions are computed against the sorted order and every
            row would otherwise be attached to the wrong date.
    """
    if not frames:
        raise ValueError("at least one symbol frame is required")
    for symbol, frame in frames.items():
        if not frame.index.is_monotonic_increasing:
            raise ValueError(f"{symbol}: frame timestamps must be sorted")
    requested = build_calendar(
        {symbol: frame.index for symbol, frame in frames.items()}, mode=mode
    )
    alignments = {
        symbol: align_index(
            frame.index,
            requested,
            symbol=symbol,
            policy=policy,
            max_staleness=max_staleness,
        )
        for symbol, frame in frames.items()
    }
    keep = [
        all(alignment.positions[i] != UNAVAILABLE for alignment in alignments.values())
        for i in range(requested.n_dates)
    ]
    if not any(keep):
        raise ValueError(
            "no date can be supplied by every symbol under "
            f"{MissingPolicy(policy).value}"
        )
    kept_dates = tuple(
        date for date, retained in zip(requested.dates, keep, strict=True) if retained
    )
    final = TradingCalendar(
        mode=requested.mode, symbols=requested.symbols, dates=kept_dates
    )
    aligned: dict[str, pd.DataFrame] = {}
    for symbol in final.symbols:
        frame = frames[symbol]
        positions = [
            position
            for position, retained in zip(
                alignments[symbol].positions, keep, strict=True
            )
            if retained
        ]
        selected = frame.iloc[positions].reset_index(drop=True)
        selected.index = final.index()
        aligned[symbol] = selected
    return PanelAlignment(
        calendar=final,
        frames=aligned,
        dropped_dates=tuple(
            date
            for date, retained in zip(requested.dates, keep, strict=True)
            if not retained
        ),
        coverage=tuple(
            requested.coverage(frames[symbol].index, symbol=symbol)
            for symbol in requested.symbols
        ),
        forward_filled={
            symbol: alignments[symbol].n_forward_filled for symbol in requested.symbols
        },
    )
