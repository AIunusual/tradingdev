"""Leakage gate for walk-forward folds.

The walk-forward splitter cuts a price series into train/test windows that are
disjoint *in time*, which is enough when a strategy only ever looks backwards.
It is not enough once training targets are interval-based labels: the label of
the bar at ``t`` is decided by prices up to ``t + horizon``, so the last bars
of a training window carry answers computed from test-window prices.

This module is the gate for that case. It converts a fold's positional windows
into the auditable :class:`Split` objects used by the ML validation layer and
reports any overlap. Keeping it a pure function of ``(windows, horizon)`` is
the point: the gate can be locked down with small integer arrays, without a
DataFrame, a strategy, or a backtest engine.

The gate only *detects*; it never silently repairs a fold. Deciding how to
purge is the caller's choice, and a blocked run is preferable to a fold whose
out-of-sample score was quietly inflated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.domain.ml.validation import Split, leakage_audit

if TYPE_CHECKING:
    from collections.abc import Sequence

type FoldWindow = tuple[tuple[int, int], tuple[int, int]]


class WalkForwardLeakageError(RuntimeError):
    """Raised when walk-forward folds leak label information into training."""

    def __init__(self, violations: Sequence[str]) -> None:
        self.violations: list[str] = list(violations)
        detail = "; ".join(self.violations)
        super().__init__(f"walk-forward folds leak label information: {detail}")


def fold_splits(
    windows: Sequence[FoldWindow], *, name_prefix: str = "fold"
) -> list[Split]:
    """Convert positional fold windows into auditable :class:`Split` objects.

    Degenerate windows (empty train or test) carry no leakage signal and are
    skipped rather than crashing the audit.
    """
    splits: list[Split] = []
    for index, (train, test) in enumerate(windows):
        if train[0] >= train[1] or test[0] >= test[1]:
            continue
        splits.append(Split(name=f"{name_prefix}/{index}", train=(train,), test=test))
    return splits


def audit_fold_windows(windows: Sequence[FoldWindow], *, horizon: int) -> list[str]:
    """Report label leakage in the fold windows.

    Args:
        windows: One ``(train, test)`` pair of half-open positional ranges
            per fold.
        horizon: Bars a label looks ahead. ``0`` or negative disables the
            audit, because a plain price-series walk-forward has no
            forward-looking label to leak.

    Returns:
        Human-readable violations; empty means no *label* leakage was found.
        Embargo width is deliberately not audited, since it guards against
        serial correlation rather than a provable overlap.
    """
    if horizon < 1:
        return []
    return leakage_audit(fold_splits(windows), horizon=horizon)


def enforce_no_leakage(windows: Sequence[FoldWindow], *, horizon: int) -> None:
    """Raise :class:`WalkForwardLeakageError` when the folds leak."""
    violations = audit_fold_windows(windows, horizon=horizon)
    if violations:
        raise WalkForwardLeakageError(violations)
