"""What a feature cache store may hand back, and what it must refuse.

Every test here is a rejection. That is deliberate: the store's only failure
mode that matters is serving a matrix that is not the one that was asked for,
and the only way to be sure of that is to check, on the way out, that the files
still say what they said when they were written.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.feature_cache import FeatureCacheIdentity
from tradingdev.domain.ml.registry import FeatureCacheStore

if TYPE_CHECKING:
    from pathlib import Path

_COLUMNS = ("feat_a", "feat_b")
_ROWS = 12


def _frame(seed: float = 0.0) -> pd.DataFrame:
    index = np.arange(_ROWS, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=_ROWS, freq="h", tz="UTC"),
            "close": 100.0 + index + seed,
            "feat_a": np.sin(index / 3.0) + seed,
            "feat_b": np.cos(index / 3.0) - seed,
        }
    )


def _identity(**overrides: object) -> FeatureCacheIdentity:
    values: dict[str, object] = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "data_fingerprint": "bars-1",
        "feature_set": "returns",
        "code_fingerprint": "code-v1",
    }
    values.update(overrides)
    return FeatureCacheIdentity(**values)  # type: ignore[arg-type]


@pytest.fixture
def store(tmp_path: Path) -> FeatureCacheStore:
    return FeatureCacheStore(tmp_path / "feature_cache")


class TestRoundTrip:
    def test_an_empty_store_misses(self, store: FeatureCacheStore) -> None:
        assert store.load(_identity(), feature_columns=_COLUMNS) is None

    def test_a_saved_matrix_comes_back_unchanged(
        self, store: FeatureCacheStore
    ) -> None:
        frame = _frame()
        store.save(_identity(), frame, feature_columns=_COLUMNS)
        loaded = store.load(_identity(), feature_columns=_COLUMNS)
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, frame)

    def test_saving_twice_keeps_one_entry(self, store: FeatureCacheStore) -> None:
        store.save(_identity(), _frame(), feature_columns=_COLUMNS)
        store.save(_identity(), _frame(), feature_columns=_COLUMNS)
        assert store.keys() == (_identity().key,)

    def test_the_sidecar_records_what_it_was_computed_from(
        self, store: FeatureCacheStore
    ) -> None:
        identity = _identity()
        store.save(identity, _frame(), feature_columns=_COLUMNS)
        record = json.loads(store.meta_path_for(identity.key).read_text("utf-8"))
        assert record["symbol"] == "BTC/USDT"
        assert record["data_fingerprint"] == "bars-1"
        assert record["feature_set"] == "returns"
        assert record["code_fingerprint"] == "code-v1"
        assert record["feature_columns"] == list(_COLUMNS)
        assert record["rows"] == _ROWS

    def test_two_computations_live_side_by_side(self, store: FeatureCacheStore) -> None:
        store.save(_identity(), _frame(), feature_columns=_COLUMNS)
        other = _identity(symbol="ETH/USDT", data_fingerprint="bars-2")
        store.save(other, _frame(seed=5.0), feature_columns=_COLUMNS)
        assert len(store.keys()) == 2
        loaded = store.load(other, feature_columns=_COLUMNS)
        assert loaded is not None
        assert loaded["close"].iloc[0] == pytest.approx(105.0)


class TestRejections:
    @pytest.mark.parametrize(
        "field",
        ["symbol", "timeframe", "data_fingerprint", "feature_set", "code_fingerprint"],
    )
    def test_a_different_identity_misses(
        self, store: FeatureCacheStore, field: str
    ) -> None:
        store.save(_identity(), _frame(), feature_columns=_COLUMNS)
        assert (
            store.load(_identity(**{field: "other"}), feature_columns=_COLUMNS) is None
        )

    def test_a_different_declared_contract_misses(
        self, store: FeatureCacheStore
    ) -> None:
        """Same set name, different columns: the matrix is not this one."""
        store.save(_identity(), _frame(), feature_columns=_COLUMNS)
        assert store.load(_identity(), feature_columns=("feat_a",)) is None

    def test_a_tampered_matrix_is_discarded(self, store: FeatureCacheStore) -> None:
        identity = _identity()
        store.save(identity, _frame(), feature_columns=_COLUMNS)
        edited = _frame(seed=1.0)
        edited.to_parquet(store.path_for(identity.key), index=False)
        assert store.load(identity, feature_columns=_COLUMNS) is None

    def test_a_truncated_matrix_is_discarded(self, store: FeatureCacheStore) -> None:
        identity = _identity()
        store.save(identity, _frame(), feature_columns=_COLUMNS)
        path = store.path_for(identity.key)
        path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 2])
        assert store.load(identity, feature_columns=_COLUMNS) is None

    def test_a_matrix_with_a_rewritten_column_is_discarded(
        self, store: FeatureCacheStore
    ) -> None:
        identity = _identity()
        frame = _frame()
        store.save(identity, frame, feature_columns=_COLUMNS)
        swapped = frame.copy()
        swapped["feat_a"] = frame["feat_b"]
        swapped.to_parquet(store.path_for(identity.key), index=False)
        assert store.load(identity, feature_columns=_COLUMNS) is None

    def test_a_matrix_with_fewer_rows_is_discarded(
        self, store: FeatureCacheStore
    ) -> None:
        identity = _identity()
        store.save(identity, _frame(), feature_columns=_COLUMNS)
        short = _frame().iloc[: _ROWS - 1]
        short.to_parquet(store.path_for(identity.key), index=False)
        assert store.load(identity, feature_columns=_COLUMNS) is None

    def test_a_corrupt_sidecar_is_a_miss(self, store: FeatureCacheStore) -> None:
        identity = _identity()
        store.save(identity, _frame(), feature_columns=_COLUMNS)
        store.meta_path_for(identity.key).write_text("{not json", "utf-8")
        assert store.load(identity, feature_columns=_COLUMNS) is None

    def test_an_incomplete_sidecar_is_a_miss(self, store: FeatureCacheStore) -> None:
        """Without a content hash the entry cannot be verified, so it is unusable."""
        identity = _identity()
        store.save(identity, _frame(), feature_columns=_COLUMNS)
        record = json.loads(store.meta_path_for(identity.key).read_text("utf-8"))
        del record["content_hash"]
        store.meta_path_for(identity.key).write_text(json.dumps(record), "utf-8")
        assert store.load(identity, feature_columns=_COLUMNS) is None

    def test_a_sidecar_without_its_matrix_is_a_miss(
        self, store: FeatureCacheStore
    ) -> None:
        identity = _identity()
        store.save(identity, _frame(), feature_columns=_COLUMNS)
        store.path_for(identity.key).unlink()
        assert store.load(identity, feature_columns=_COLUMNS) is None

    def test_a_matrix_without_its_sidecar_is_a_miss(
        self, store: FeatureCacheStore
    ) -> None:
        identity = _identity()
        frame = _frame()
        frame.to_parquet(store.path_for(identity.key), index=False)
        assert store.load(identity, feature_columns=_COLUMNS) is None
        assert store.keys() == ()

    def test_a_rejection_leaves_the_entry_recomputable(
        self, store: FeatureCacheStore
    ) -> None:
        """A miss must be a miss, not a broken store: saving again must work."""
        identity = _identity()
        store.save(identity, _frame(), feature_columns=_COLUMNS)
        store.meta_path_for(identity.key).write_text("garbage", "utf-8")
        assert store.load(identity, feature_columns=_COLUMNS) is None
        store.save(identity, _frame(seed=2.0), feature_columns=_COLUMNS)
        loaded = store.load(identity, feature_columns=_COLUMNS)
        assert loaded is not None
        assert loaded["close"].iloc[0] == pytest.approx(102.0)
