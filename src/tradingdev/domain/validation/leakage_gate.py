"""Detect strategy fold label overlap without repairing split boundaries.

Convert positional windows to ML Split objects and audit intervals extending
through ``t + horizon``; disjoint decision dates alone do not prevent leakage.
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
    """Convert half-open positional windows to Split objects, skipping empty windows."""
    splits: list[Split] = []
    for index, (train, test) in enumerate(windows):
        if train[0] >= train[1] or test[0] >= test[1]:
            continue
        splits.append(Split(name=f"{name_prefix}/{index}", train=(train,), test=test))
    return splits


def audit_fold_windows(windows: Sequence[FoldWindow], *, horizon: int) -> list[str]:
    """Report label overlap in half-open (train, test) positional windows.

    Non-positive horizon disables the audit. Empty results mean no detected
    label overlap, not an assessment of embargo or serial correlation.
    """
    if horizon < 1:
        return []
    return leakage_audit(fold_splits(windows), horizon=horizon)


def enforce_no_leakage(windows: Sequence[FoldWindow], *, horizon: int) -> None:
    """Raise :class:`WalkForwardLeakageError` when the folds leak."""
    violations = audit_fold_windows(windows, horizon=horizon)
    if violations:
        raise WalkForwardLeakageError(violations)
