"""Purged and embargoed splits keep training labels out of the test window."""

from __future__ import annotations

import pytest

from tradingdev.domain.ml.validation import (
    Split,
    leakage_audit,
    purged_kfold_splits,
    purged_walk_forward_splits,
)

_WALK_CASES = [
    (40, 3, 3, 0),
    (40, 3, 3, 2),
    (100, 10, 4, 5),
    (469, 1, 3, 0),
    (469, 20, 3, 10),
]

_KFOLD_CASES = [
    (40, 3, 4, 0),
    (100, 10, 5, 3),
    (469, 5, 5, 5),
]


def _train_indices(split: Split) -> list[int]:
    return [index for start, end in split.train for index in range(start, end)]


@pytest.mark.parametrize(("n_bars", "horizon", "n_splits", "embargo"), _WALK_CASES)
def test_walk_forward_reports_no_label_leak(
    n_bars: int, horizon: int, n_splits: int, embargo: int
) -> None:
    splits = purged_walk_forward_splits(
        n_bars, horizon=horizon, n_splits=n_splits, embargo=embargo
    )
    assert len(splits) == n_splits
    assert leakage_audit(splits, horizon=horizon) == []


@pytest.mark.parametrize(("n_bars", "horizon", "n_folds", "embargo"), _KFOLD_CASES)
def test_kfold_reports_no_label_leak(
    n_bars: int, horizon: int, n_folds: int, embargo: int
) -> None:
    splits = purged_kfold_splits(
        n_bars, horizon=horizon, n_folds=n_folds, embargo=embargo
    )
    assert len(splits) == n_folds
    assert leakage_audit(splits, horizon=horizon) == []


@pytest.mark.parametrize(("n_bars", "horizon", "n_splits", "embargo"), _WALK_CASES)
def test_walk_forward_keeps_the_full_gap(
    n_bars: int, horizon: int, n_splits: int, embargo: int
) -> None:
    """Training must stop at least horizon + embargo bars before the test."""
    for split in purged_walk_forward_splits(
        n_bars, horizon=horizon, n_splits=n_splits, embargo=embargo
    ):
        assert split.train[-1][1] <= split.test[0] - horizon - embargo


@pytest.mark.parametrize(("n_bars", "horizon", "n_splits", "embargo"), _WALK_CASES)
def test_purge_width_is_exactly_the_horizon(
    n_bars: int, horizon: int, n_splits: int, embargo: int
) -> None:
    for split in purged_walk_forward_splits(
        n_bars, horizon=horizon, n_splits=n_splits, embargo=embargo
    ):
        assert split.purged_bars == horizon
        assert split.embargoed_bars == embargo


def test_embargo_removes_exactly_the_requested_extra_bars() -> None:
    baseline = purged_walk_forward_splits(200, horizon=5, n_splits=3, embargo=0)
    widened = purged_walk_forward_splits(200, horizon=5, n_splits=3, embargo=7)
    for before, after in zip(baseline, widened, strict=True):
        assert before.train_bars - after.train_bars == 7
        assert after.embargoed_bars == 7
        assert after.test == before.test


def test_kfold_test_windows_partition_every_bar() -> None:
    n_bars = 101
    splits = purged_kfold_splits(n_bars, horizon=4, n_folds=5)
    covered = sorted(
        index for split in splits for index in range(split.test[0], split.test[1])
    )
    assert covered == list(range(n_bars))


def test_walk_forward_test_windows_are_ordered_and_disjoint() -> None:
    splits = purged_walk_forward_splits(200, horizon=5, n_splits=4)
    windows = [split.test for split in splits]
    assert windows == sorted(windows)
    for earlier, later in zip(windows, windows[1:], strict=False):
        assert earlier[1] <= later[0]


def test_audit_flags_a_naive_unpurged_split() -> None:
    naive = [
        Split(name="fold/0", train=((0, 10),), test=(10, 20)),
        Split(name="fold/1", train=((0, 20),), test=(20, 30)),
    ]
    violations = leakage_audit(naive, horizon=3)
    assert len(violations) == 2
    assert "fold/0" in violations[0] and "[7, 10)" in violations[0]
    assert "fold/1" in violations[1] and "[17, 20)" in violations[1]


def test_audit_accepts_a_manually_purged_split() -> None:
    clean = [Split(name="fold/0", train=((0, 7),), test=(10, 20))]
    assert leakage_audit(clean, horizon=3) == []


def test_audit_flags_test_labels_reaching_right_hand_training() -> None:
    """Test labels extend beyond the last test decision bar."""
    split = [Split(name="fold/0", train=((20, 30),), test=(10, 20))]
    violations = leakage_audit(split, horizon=5)
    assert len(violations) == 1
    assert "[20, 25)" in violations[0]
    assert "ending at 24" in violations[0]


def test_audit_accepts_right_hand_training_beyond_test_label_end() -> None:
    split = [Split(name="fold/0", train=((25, 30),), test=(10, 20))]
    assert leakage_audit(split, horizon=5) == []


def test_kfold_purges_both_sides_before_adding_embargo() -> None:
    splits = purged_kfold_splits(120, horizon=5, n_folds=3, embargo=2)
    middle = splits[1]
    assert middle.test == (40, 80)
    assert middle.train == ((0, 33), (87, 120))
    assert middle.purged_bars == 10
    assert middle.embargoed_bars == 4


@pytest.mark.parametrize(("n_bars", "horizon", "n_folds", "embargo"), _KFOLD_CASES)
def test_kfold_training_and_test_labels_have_no_shared_price_bar(
    n_bars: int, horizon: int, n_folds: int, embargo: int
) -> None:
    for split in purged_kfold_splits(
        n_bars, horizon=horizon, n_folds=n_folds, embargo=embargo
    ):
        test_label_bars = set(range(split.test[0], split.test[1] + horizon))
        for train_bar in _train_indices(split):
            assert test_label_bars.isdisjoint(range(train_bar, train_bar + horizon + 1))


def test_splits_are_deterministic() -> None:
    first = purged_walk_forward_splits(200, horizon=5, n_splits=4, embargo=2)
    second = purged_walk_forward_splits(200, horizon=5, n_splits=4, embargo=2)
    assert first == second


def test_split_rejects_overlapping_or_empty_windows() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        Split(name="bad", train=((0, 15),), test=(10, 20))
    with pytest.raises(ValueError, match="non-empty"):
        Split(name="bad", train=((0, 10),), test=(10, 10))
    with pytest.raises(ValueError, match="non-empty"):
        Split(name="bad", train=((5, 5),), test=(10, 20))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_bars": 0, "horizon": 1},
        {"n_bars": 100, "horizon": 0},
        {"n_bars": 100, "horizon": 100},
        {"n_bars": 100, "horizon": 5, "n_splits": 0},
        {"n_bars": 100, "horizon": 5, "embargo": -1},
        {"n_bars": 100, "horizon": 5, "min_train_bars": 0},
    ],
)
def test_walk_forward_rejects_impossible_arguments(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        purged_walk_forward_splits(**kwargs)


def test_walk_forward_rejects_folds_without_training_bars() -> None:
    """A horizon longer than the first block must fail loudly, not shrink."""
    with pytest.raises(ValueError, match="training bars"):
        purged_walk_forward_splits(30, horizon=6, n_splits=5)


def test_min_train_bars_is_enforced() -> None:
    splits = purged_walk_forward_splits(200, horizon=5, n_splits=3, min_train_bars=40)
    assert all(split.train_bars >= 40 for split in splits)
    with pytest.raises(ValueError, match="training bars"):
        purged_walk_forward_splits(200, horizon=5, n_splits=3, min_train_bars=400)


def test_purge_cost_grows_with_the_label_horizon() -> None:
    short = purged_walk_forward_splits(469, horizon=1, n_splits=3)
    long = purged_walk_forward_splits(469, horizon=20, n_splits=3)
    assert [split.purged_bars for split in short] == [1, 1, 1]
    assert [split.purged_bars for split in long] == [20, 20, 20]
    assert long[0].train_bars < short[0].train_bars


def test_train_and_test_never_share_a_bar() -> None:
    for splits in (
        purged_walk_forward_splits(120, horizon=7, n_splits=4, embargo=3),
        purged_kfold_splits(120, horizon=7, n_folds=4, embargo=3),
    ):
        for split in splits:
            test = set(range(split.test[0], split.test[1]))
            assert test.isdisjoint(_train_indices(split))
