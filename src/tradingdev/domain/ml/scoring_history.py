"""Time-identity rules for scoring a window that overlaps saved training rows.

Strategies keep the frame passed to ``fit()`` so that later scoring calls can
use it as warm-up context.  When the scored frame is that same frame -- which
is what ``fit(train)`` followed by ``generate_signals(train)`` does for training
metrics -- concatenating both copies produces a context whose timestamps go
forward, jump back to the start, and continue.  Features computed across that
seam mix bars from two different times.

Only the training rows that strictly precede the scored frame are history.  The
overlap is detected from timestamps, so both frames must carry a time identity:
a ``timestamp`` column or a ``DatetimeIndex``.  A frame without one cannot be
ordered against the other, and guessing would reintroduce the same seam.
"""

from __future__ import annotations

import pandas as pd


def align_scoring_history(
    history: pd.DataFrame,
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``history`` restricted to rows strictly before ``scored`` starts.

    Args:
        history: Training frame retained by ``fit()``.
        scored: Frame about to be scored.

    Returns:
        The clipped history and the scored frame, with time identities copied
        into UTC ``timestamp`` columns for concatenation. Inputs are unchanged.

    Raises:
        ValueError: If either frame has no time identity, or if its timestamps
            are missing, duplicated, or not increasing.
    """
    if not len(scored):
        return history.iloc[:0], scored

    scored_stamps = _time_identity(scored, "scoring frame")
    scored = _with_utc_time(scored, scored_stamps)
    if not len(history):
        return history, scored

    history_stamps = _time_identity(history, "training history")
    history = _with_utc_time(history, history_stamps)
    return history.loc[history_stamps < scored_stamps[0]], scored


def _time_identity(frame: pd.DataFrame, role: str) -> pd.DatetimeIndex:
    if "timestamp" in frame.columns:
        stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    elif isinstance(frame.index, pd.DatetimeIndex):
        stamps = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    else:
        msg = (
            f"{role} requires timestamps: add a timestamp column or a "
            "DatetimeIndex so overlapping training rows can be excluded"
        )
        raise ValueError(msg)
    if stamps.hasnans or not stamps.is_monotonic_increasing or not stamps.is_unique:
        msg = f"{role} timestamps must be non-missing, unique and increasing"
        raise ValueError(msg)
    return stamps


def _with_utc_time(
    frame: pd.DataFrame,
    stamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Keep time identity when consumers concatenate with ``ignore_index=True``."""
    return frame.assign(timestamp=stamps)
