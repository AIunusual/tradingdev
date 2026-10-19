"""The feature registry enumerates named sets with a stable column contract."""

from __future__ import annotations

import pandas as pd
import pytest

from tradingdev.domain.ml.features.registry import (
    FeatureSet,
    feature_set_names,
    get_feature_set,
    list_feature_sets,
    register,
)

_EXPECTED_NAMES = {"returns", "price_ratios", "volume", "technical", "comprehensive"}


@pytest.fixture
def empty_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate registration so register() can be exercised without side effects."""
    monkeypatch.setattr("tradingdev.domain.ml.features.registry._REGISTRY", {})


def test_every_registered_set_emits_its_contract_columns(
    large_ohlcv_df: pd.DataFrame,
) -> None:
    for feature_set in list_feature_sets():
        result = feature_set.build(large_ohlcv_df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(feature_set.columns)
        # Row-preserving: no silent dropna, so a model can keep bar alignment.
        assert result.shape[0] == large_ohlcv_df.shape[0]
        assert result.index.equals(large_ohlcv_df.index)


def test_feature_set_names_are_unique_and_sorted() -> None:
    names = feature_set_names()
    assert len(names) == len(set(names))
    assert names == tuple(sorted(names))


def test_registry_exposes_the_expected_builtin_sets() -> None:
    assert _EXPECTED_NAMES.issubset(set(feature_set_names()))


def test_get_feature_set_returns_the_registered_entry() -> None:
    entry = get_feature_set("returns")
    assert isinstance(entry, FeatureSet)
    assert entry.name == "returns"
    assert entry.n_features == 8


def test_unknown_feature_set_raises_key_error() -> None:
    with pytest.raises(KeyError, match="unknown feature set"):
        get_feature_set("does_not_exist")


def test_comprehensive_is_the_union_of_the_other_sets(
    large_ohlcv_df: pd.DataFrame,
) -> None:
    comprehensive = get_feature_set("comprehensive")
    others = [
        get_feature_set(name).columns
        for name in ("returns", "price_ratios", "volume", "technical")
    ]
    union: tuple[str, ...] = tuple(col for group in others for col in group)
    assert comprehensive.columns == union

    result = comprehensive.build(large_ohlcv_df)
    assert list(result.columns) == list(union)


def test_builder_does_not_mutate_its_input(
    large_ohlcv_df: pd.DataFrame,
) -> None:
    before = large_ohlcv_df.copy()
    get_feature_set("comprehensive").build(large_ohlcv_df)
    pd.testing.assert_frame_equal(large_ohlcv_df, before)


def test_build_requires_ohlcv_columns() -> None:
    bare = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError):
        get_feature_set("returns").build(bare)


def test_descriptions_and_tags_are_populated() -> None:
    for feature_set in list_feature_sets():
        assert feature_set.description.strip()
        assert all(isinstance(tag, str) for tag in feature_set.tags)


def test_register_rejects_duplicate_names(empty_registry: None) -> None:
    register(
        FeatureSet(name="dup", description="d", columns=("a",), build=lambda df: df)
    )
    with pytest.raises(ValueError, match="already registered"):
        register(
            FeatureSet(name="dup", description="d", columns=("a",), build=lambda df: df)
        )


def test_register_makes_the_set_retrievable(empty_registry: None) -> None:
    register(
        FeatureSet(
            name="toy",
            description="a toy set",
            columns=("x",),
            build=lambda df: df[["x"]],
        )
    )
    assert "toy" in feature_set_names()
    assert get_feature_set("toy").n_features == 1
