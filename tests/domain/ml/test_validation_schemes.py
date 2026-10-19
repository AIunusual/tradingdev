"""The validation registry shared by get_contract and train_model must map
advertised names and parameters to buildable splits.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from tradingdev.domain.ml.validation import (
    Split,
    audit_validation_splits,
    build_validation_splits,
    get_validation_scheme,
    validation_scheme_names,
)
from tradingdev.domain.ml.validation.schemes import VALIDATION_SCHEMES

if TYPE_CHECKING:
    from collections.abc import Sequence

_EXPECTED_NAMES = ("purged_walk_forward", "purged_kfold")


def _test_indices(split: Split) -> list[int]:
    return list(range(split.test[0], split.test[1]))


class TestRegistry:
    def test_names_are_stable_and_complete(self) -> None:
        """Order is part of the contract: the menu must not reshuffle."""
        assert validation_scheme_names() == _EXPECTED_NAMES

    def test_every_name_resolves(self) -> None:
        for name in validation_scheme_names():
            assert get_validation_scheme(name).name == name

    def test_the_dictionary_is_keyed_by_the_scheme_name(self) -> None:
        assert set(VALIDATION_SCHEMES) == set(validation_scheme_names())
        for key, scheme in VALIDATION_SCHEMES.items():
            assert scheme.name == key

    def test_every_scheme_is_described_for_the_contract(self) -> None:
        for name in validation_scheme_names():
            scheme = get_validation_scheme(name)
            assert scheme.description.strip()
            # The model needs a selection reason beyond the name.
            assert scheme.description != name

    def test_unknown_name_is_rejected_and_names_the_alternatives(self) -> None:
        with pytest.raises(ValueError, match="unknown validation scheme: rolling"):
            get_validation_scheme("rolling")
        with pytest.raises(ValueError) as caught:
            get_validation_scheme("rolling")
        message = str(caught.value)
        for name in _EXPECTED_NAMES:
            assert name in message

    def test_case_is_not_folded(self) -> None:
        with pytest.raises(ValueError, match="unknown validation scheme"):
            get_validation_scheme("PURGED_KFOLD")


class TestFoldParameterMapping:
    """The registry maps differing builder keywords."""

    @pytest.mark.parametrize("name", _EXPECTED_NAMES)
    def test_the_declared_keyword_is_the_one_the_builder_takes(self, name: str) -> None:
        scheme = get_validation_scheme(name)
        parameters = inspect.signature(scheme.build).parameters
        assert scheme.fold_parameter in parameters, (
            f"{name} declares {scheme.fold_parameter!r} but its builder takes "
            f"{sorted(parameters)}"
        )

    @pytest.mark.parametrize("n_folds", [1, 2, 3, 4])
    def test_walk_forward_honours_the_fold_count(self, n_folds: int) -> None:
        splits = build_validation_splits(
            "purged_walk_forward", n_bars=240, horizon=5, n_folds=n_folds, embargo=2
        )
        assert len(splits) == n_folds

    @pytest.mark.parametrize("n_folds", [2, 3, 4])
    def test_kfold_honours_the_fold_count(self, n_folds: int) -> None:
        splits = build_validation_splits(
            "purged_kfold", n_bars=240, horizon=5, n_folds=n_folds, embargo=2
        )
        assert len(splits) == n_folds

    def test_kfold_with_one_fold_leaves_nothing_to_train_on(self) -> None:
        """One fold tests every bar, so there is no training window left."""
        with pytest.raises(ValueError, match="training bars"):
            build_validation_splits("purged_kfold", n_bars=240, horizon=5, n_folds=1)


class TestBuilding:
    @pytest.mark.parametrize("name", _EXPECTED_NAMES)
    def test_every_advertised_scheme_builds_clean_splits(self, name: str) -> None:
        splits = build_validation_splits(
            name, n_bars=200, horizon=5, n_folds=4, embargo=2
        )
        assert splits
        assert all(isinstance(split, Split) for split in splits)
        assert audit_validation_splits(splits, horizon=5) == ()

    @pytest.mark.parametrize("name", _EXPECTED_NAMES)
    @pytest.mark.parametrize(
        ("n_bars", "horizon", "n_folds", "embargo"),
        [(40, 3, 3, 0), (200, 5, 4, 2), (469, 10, 3, 7), (1000, 1, 5, 0)],
    )
    def test_no_scheme_leaks_a_label_into_its_own_test_window(
        self, name: str, n_bars: int, horizon: int, n_folds: int, embargo: int
    ) -> None:
        splits = build_validation_splits(
            name,
            n_bars=n_bars,
            horizon=horizon,
            n_folds=n_folds,
            embargo=embargo,
        )
        assert audit_validation_splits(splits, horizon=horizon) == ()

    def test_walk_forward_never_trains_after_the_test_window(self) -> None:
        for split in build_validation_splits(
            "purged_walk_forward", n_bars=200, horizon=5, n_folds=4, embargo=2
        ):
            for _, end in split.train:
                assert end <= split.test[0]

    def test_kfold_trains_on_both_sides_of_the_test_window(self) -> None:
        splits = build_validation_splits(
            "purged_kfold", n_bars=200, horizon=5, n_folds=4, embargo=2
        )
        assert len(splits[1].train) == 2
        assert splits[1].train[0][1] <= splits[1].test[0]
        assert splits[1].train[1][0] >= splits[1].test[1]

    def test_kfold_test_windows_cover_every_bar_exactly_once(self) -> None:
        n_bars = 240
        covered = [
            index
            for split in build_validation_splits(
                "purged_kfold", n_bars=n_bars, horizon=5, n_folds=4
            )
            for index in _test_indices(split)
        ]
        assert sorted(covered) == list(range(n_bars))

    def test_walk_forward_leaves_the_first_block_untested(self) -> None:
        splits = build_validation_splits(
            "purged_walk_forward", n_bars=200, horizon=5, n_folds=4
        )
        first_block_end = splits[0].test[0]
        tested = {index for split in splits for index in _test_indices(split)}
        assert not tested & set(range(first_block_end))

    def test_the_two_schemes_do_not_produce_the_same_geometry(self) -> None:
        walk = build_validation_splits(
            "purged_walk_forward", n_bars=200, horizon=5, n_folds=4
        )
        kfold = build_validation_splits(
            "purged_kfold", n_bars=200, horizon=5, n_folds=4
        )
        assert [split.test for split in walk] != [split.test for split in kfold]

    def test_walk_forward_purges_the_horizon_on_every_fold(self) -> None:
        splits = build_validation_splits(
            "purged_walk_forward", n_bars=200, horizon=7, n_folds=4, embargo=3
        )
        assert [split.purged_bars for split in splits] == [7, 7, 7, 7]
        assert [split.embargoed_bars for split in splits] == [3, 3, 3, 3]

    def test_kfold_purges_after_the_first_test_window(self) -> None:
        """Fold 0 still has test labels extending into right-hand training."""
        splits = build_validation_splits(
            "purged_kfold", n_bars=200, horizon=7, n_folds=4, embargo=3
        )
        assert splits[0].purged_bars == 7
        assert [split.purged_bars for split in splits[1:]] == [14, 14, 7]

    def test_embargo_is_charged_on_top_of_the_purge(self) -> None:
        baseline = build_validation_splits(
            "purged_walk_forward", n_bars=200, horizon=5, n_folds=3, embargo=0
        )
        widened = build_validation_splits(
            "purged_walk_forward", n_bars=200, horizon=5, n_folds=3, embargo=6
        )
        for before, after in zip(baseline, widened, strict=True):
            assert before.purged_bars == after.purged_bars == 5
            assert before.train_bars - after.train_bars == 6
            assert after.test == before.test

    def test_the_horizon_alone_decides_the_purge_width(self) -> None:
        short = build_validation_splits(
            "purged_walk_forward", n_bars=469, horizon=1, n_folds=3
        )
        long = build_validation_splits(
            "purged_walk_forward", n_bars=469, horizon=20, n_folds=3
        )
        assert sum(split.purged_bars for split in short) == 3
        assert sum(split.purged_bars for split in long) == 60
        assert long[0].train_bars < short[0].train_bars

    def test_unknown_scheme_is_rejected_before_any_geometry(self) -> None:
        with pytest.raises(ValueError, match="unknown validation scheme"):
            build_validation_splits("rolling", n_bars=200, horizon=5)

    @pytest.mark.parametrize("name", _EXPECTED_NAMES)
    def test_more_folds_than_bars_is_rejected(self, name: str) -> None:
        with pytest.raises(ValueError):
            build_validation_splits(name, n_bars=10, horizon=5, n_folds=20)

    @pytest.mark.parametrize("name", _EXPECTED_NAMES)
    def test_a_horizon_as_long_as_the_series_is_rejected(self, name: str) -> None:
        with pytest.raises(ValueError, match="horizon"):
            build_validation_splits(name, n_bars=50, horizon=50, n_folds=3)

    def test_min_train_bars_is_enforced_per_fold(self) -> None:
        splits = build_validation_splits(
            "purged_walk_forward",
            n_bars=400,
            horizon=5,
            n_folds=3,
            min_train_bars=80,
        )
        assert all(split.train_bars >= 80 for split in splits)
        with pytest.raises(ValueError, match="training bars"):
            build_validation_splits(
                "purged_walk_forward",
                n_bars=400,
                horizon=5,
                n_folds=3,
                min_train_bars=10_000,
            )

    def test_building_is_deterministic(self) -> None:
        kwargs = {"n_bars": 240, "horizon": 5, "n_folds": 4, "embargo": 2}
        assert build_validation_splits("purged_kfold", **kwargs) == (
            build_validation_splits("purged_kfold", **kwargs)
        )


class TestAudit:
    def test_clean_splits_audit_empty(self) -> None:
        splits = build_validation_splits(
            "purged_walk_forward", n_bars=200, horizon=5, n_folds=3, embargo=2
        )
        assert audit_validation_splits(splits, horizon=5) == ()

    def test_a_naive_split_is_caught(self) -> None:
        naive: Sequence[Split] = (
            Split(name="fold/0", train=((0, 10),), test=(10, 20)),
            Split(name="fold/1", train=((0, 20),), test=(20, 30)),
        )
        violations = audit_validation_splits(naive, horizon=3)
        assert len(violations) == 2
        assert "fold/0" in violations[0]
        assert "fold/1" in violations[1]

    def test_the_horizon_used_is_the_one_given(self) -> None:
        """A one-bar gap is sufficient for horizon 1 but leaks labels at horizon 3."""
        tight: Sequence[Split] = (Split(name="fold/0", train=((0, 9),), test=(10, 20)),)
        assert audit_validation_splits(tight, horizon=1) == ()
        assert len(audit_validation_splits(tight, horizon=3)) == 1
