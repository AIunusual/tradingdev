"""Index splits for labels spanning ``[t, t + horizon]``.

Purge excludes training labels overlapping test labels on either side.
Embargo adds a caller-chosen buffer for serial correlation; it does not prove
independence. ``leakage_audit`` checks label overlap, not embargo adequacy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class Split:
    """Train/test partition using half-open positional ranges.

    Training may contain ranges on both sides of a K-fold test window.
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
    """Return anchored purged folds in test-window order.

    Divide into ``n_splits + 1`` blocks: the first is training-only, each later
    block tests against preceding blocks. Remove horizon plus embargo before
    testing and require ``min_train_bars`` retained training rows.
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
    """Test each contiguous block once, training on both sides after exclusions.

    Remove horizon and embargo buffers around each test window. This is
    retrospective validation: later training data was unavailable at test time.
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
    """Report overlapping train/test label intervals on either side.

    An empty result establishes only no label overlap; it does not audit
    serial correlation or embargo adequacy.
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
