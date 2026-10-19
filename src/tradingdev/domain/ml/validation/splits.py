"""Leak-free train/test splits for interval-based labels.

Financial labels are not points in time: the label of the bar at index ``t``
is decided by prices up to index ``t + horizon``. A naive time-series split
therefore hands the model training samples whose answer is computed from
prices inside the test window.

Two gaps keep a split honest:

* **Purge** removes training bars whose label intervals overlap test label
  intervals, on either side of the test window. Its width is determined by
  ``horizon``.
* **Embargo** removes an extra buffer on both sides of the test window. Its
  width is a conservative choice, not a derivable quantity: financial
  features are serially correlated, so purge alone does not remove every
  path by which test information reaches the model.

Both are applied here as pure index arithmetic, so the result can be checked
by :func:`leakage_audit` without touching any market data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class Split:
    """One partition of a bar index range into training and test windows.

    ``train`` holds one or more half-open ``[start, end)`` ranges because a
    K-fold split leaves training bars on both sides of the test window.
    """

    name: str
    train: tuple[tuple[int, int], ...]
    test: tuple[int, int]
    purged_bars: int = 0
    embargoed_bars: int = 0

    def __post_init__(self) -> None:
        """Reject empty windows and any train/test overlap."""
        if self.test[0] >= self.test[1]:
            raise ValueError("test window must be non-empty")
        for start, end in self.train:
            if start >= end:
                raise ValueError("train window must be non-empty")
            if start < self.test[1] and self.test[0] < end:
                raise ValueError("train window overlaps the test window")
        if self.purged_bars < 0 or self.embargoed_bars < 0:
            raise ValueError("removed bar counts must not be negative")

    @property
    def train_bars(self) -> int:
        """Total number of bars retained for training."""
        return sum(end - start for start, end in self.train)

    @property
    def test_bars(self) -> int:
        """Number of bars in the test window."""
        return self.test[1] - self.test[0]


def _blocks(n_bars: int, count: int) -> list[tuple[int, int]]:
    """Split ``range(n_bars)`` into contiguous, near-equal blocks."""
    if n_bars < 1:
        raise ValueError("n_bars must be at least 1")
    if count < 1:
        raise ValueError("block count must be at least 1")
    if count > n_bars:
        raise ValueError("block count must not exceed n_bars")
    size, remainder = divmod(n_bars, count)
    blocks: list[tuple[int, int]] = []
    start = 0
    for index in range(count):
        end = start + size + (1 if index < remainder else 0)
        blocks.append((start, end))
        start = end
    return blocks


def _apply_gap(
    train: tuple[int, int],
    test: tuple[int, int],
    horizon: int,
    embargo: int,
) -> tuple[tuple[int, int] | None, int, int]:
    """Carve the purge and embargo gaps out of one training range."""
    start, end = train
    test_start, test_end = test
    if end <= test_start:
        # Labels of these bars run forward and may land inside the test window.
        purge_from = test_start - horizon
        cutoff = purge_from - embargo
        purged = max(0, min(end, test_start) - max(start, purge_from))
        embargoed = max(0, min(end, purge_from) - max(start, cutoff))
        remaining_end = min(end, cutoff)
        remaining = (start, remaining_end) if remaining_end > start else None
        return remaining, purged, embargoed
    if start >= test_end:
        # The last test label extends beyond its decision bar into this range.
        purge_until = test_end + horizon
        cutoff = purge_until + embargo
        purged = max(0, min(end, purge_until) - start)
        embargoed = max(0, min(end, cutoff) - max(start, purge_until))
        remaining_start = max(start, cutoff)
        remaining = (remaining_start, end) if end > remaining_start else None
        return remaining, purged, embargoed
    raise ValueError("train range overlaps the test window")


def _build_split(
    name: str,
    ranges: Sequence[tuple[int, int]],
    test: tuple[int, int],
    horizon: int,
    embargo: int,
    min_train_bars: int,
) -> Split:
    """Assemble one split, refusing folds that leave too little training data."""
    remaining: list[tuple[int, int]] = []
    purged_total = 0
    embargoed_total = 0
    for train_range in ranges:
        kept, purged, embargoed = _apply_gap(train_range, test, horizon, embargo)
        if kept is not None:
            remaining.append(kept)
        purged_total += purged
        embargoed_total += embargoed
    split = Split(
        name=name,
        train=tuple(remaining),
        test=test,
        purged_bars=purged_total,
        embargoed_bars=embargoed_total,
    )
    if split.train_bars < min_train_bars:
        raise ValueError(
            f"{name}: only {split.train_bars} training bars remain after "
            f"purging {purged_total} and embargoing {embargoed_total} bars; "
            f"min_train_bars is {min_train_bars}"
        )
    return split


def _validate(
    n_bars: int,
    horizon: int,
    folds: int,
    embargo: int,
    min_train_bars: int,
) -> None:
    if n_bars < 1:
        raise ValueError("n_bars must be at least 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if horizon >= n_bars:
        raise ValueError("horizon must be shorter than the bar sequence")
    if folds < 1:
        raise ValueError("folds must be at least 1")
    if embargo < 0:
        raise ValueError("embargo must not be negative")
    if min_train_bars < 1:
        raise ValueError("min_train_bars must be at least 1")


def purged_walk_forward_splits(
    n_bars: int,
    *,
    horizon: int,
    n_splits: int = 5,
    embargo: int = 0,
    min_train_bars: int = 1,
) -> list[Split]:
    """Anchored walk-forward folds, each purged and embargoed.

    The series is cut into ``n_splits + 1`` blocks; block 0 is always
    training-only and every later block becomes one test window trained on
    all preceding blocks.

    Args:
        n_bars: Total number of bars available.
        horizon: Label horizon in bars; sets the purge width.
        n_splits: Number of test folds to emit.
        embargo: Extra bars removed on each side of every test window.
        min_train_bars: Smallest training window accepted per fold.

    Returns:
        One :class:`Split` per fold, ordered by test window position.
    """
    _validate(n_bars, horizon, n_splits, embargo, min_train_bars)
    blocks = _blocks(n_bars, n_splits + 1)
    splits: list[Split] = []
    for index in range(1, len(blocks)):
        test = blocks[index]
        splits.append(
            _build_split(
                f"fold/{index - 1}",
                [(0, test[0])],
                test,
                horizon,
                embargo,
                min_train_bars,
            )
        )
    return splits


def purged_kfold_splits(
    n_bars: int,
    *,
    horizon: int,
    n_folds: int = 5,
    embargo: int = 0,
    min_train_bars: int = 1,
) -> list[Split]:
    """Contiguous K-fold splits, each purged and embargoed.

    Every bar appears in exactly one test window. Training bars on both
    sides of the window are kept, except for the label-overlap purge and
    additional embargo buffer on either side. This is a retrospective
    validation scheme: later training windows are not available in a live
    walk-forward simulation, even when their labels do not overlap the test.
    """
    _validate(n_bars, horizon, n_folds, embargo, min_train_bars)
    blocks = _blocks(n_bars, n_folds)
    splits: list[Split] = []
    for index, test in enumerate(blocks):
        ranges: list[tuple[int, int]] = []
        if test[0] > 0:
            ranges.append((0, test[0]))
        if test[1] < n_bars:
            ranges.append((test[1], n_bars))
        splits.append(
            _build_split(
                f"fold/{index}",
                ranges,
                test,
                horizon,
                embargo,
                min_train_bars,
            )
        )
    return splits


def leakage_audit(splits: Sequence[Split], *, horizon: int) -> list[str]:
    """Report overlapping training and test label intervals on either side.

    An empty list means every split is free of *label* leakage. Embargo
    widths cannot be audited this way because they guard against serial
    correlation rather than a provable overlap.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    violations: list[str] = []
    for split in splits:
        test_start, test_end = split.test
        for start, end in split.train:
            if end <= test_start and end > test_start - horizon:
                low = max(start, test_start - horizon)
                high = min(end, test_start)
                violations.append(
                    f"{split.name}: training bars [{low}, {high}) have labels "
                    f"reaching test start {test_start}"
                )
            elif start >= test_end and start < test_end + horizon:
                high = min(end, test_end + horizon)
                violations.append(
                    f"{split.name}: training bars [{start}, {high}) overlap "
                    f"test labels ending at {test_end - 1 + horizon}"
                )
    return violations
