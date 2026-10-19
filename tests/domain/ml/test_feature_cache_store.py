"""Feature cache reads verify identity, declared columns, content and shape."""

from __future__ import annotations

import errno
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pyarrow import parquet as pq

from tradingdev.domain.ml.feature_cache import FeatureCacheIdentity
from tradingdev.domain.ml.registry import FeatureCacheStore

if TYPE_CHECKING:
    import io
    from typing import Any

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
        ("field", "value"),
        [
            ("rows", "invalid"),
            ("rows", str(_ROWS)),
            ("rows", float(_ROWS)),
            ("rows", None),
            ("rows", True),
            ("rows", -1),
            ("columns", None),
            ("columns", "timestamp,close,feat_a,feat_b"),
            ("columns", {"timestamp": 0, "close": 1, "feat_a": 2, "feat_b": 3}),
            ("columns", ["timestamp", "close", "feat_a", None]),
            ("columns", ["timestamp", "close", "feat_a", "feat_a"]),
            ("feature_columns", None),
            ("feature_columns", 2),
            ("feature_columns", {"feat_a": 0, "feat_b": 1}),
            ("feature_columns", ["feat_a", "unknown"]),
            ("content_hash", None),
            ("content_hash", "g" * 64),
            ("symbol", ["BTC/USDT"]),
            ("used_at", "later"),
            ("used_at", True),
            ("used_at", float("nan")),
            ("used_at", float("inf")),
            ("used_at", 10**400),
        ],
    )
    def test_malformed_sidecar_values_are_recomputable_misses(
        self, store: FeatureCacheStore, field: str, value: object
    ) -> None:
        identity = _identity()
        store.save(identity, _frame(), feature_columns=_COLUMNS)
        sidecar = store.meta_path_for(identity.key)
        record = json.loads(sidecar.read_text("utf-8"))
        record[field] = value
        sidecar.write_text(json.dumps(record), encoding="utf-8")
        assert store.load(identity, feature_columns=_COLUMNS) is None
        store.save(identity, _frame(), feature_columns=_COLUMNS)
        rebuilt = store.load(identity, feature_columns=_COLUMNS)
        assert rebuilt is not None
        pd.testing.assert_frame_equal(rebuilt, _frame())

    @pytest.mark.parametrize("record", [None, [], "invalid", 12])
    def test_a_sidecar_must_be_a_json_object(
        self, store: FeatureCacheStore, record: object
    ) -> None:
        identity = _identity()
        store.save(identity, _frame(), feature_columns=_COLUMNS)
        store.meta_path_for(identity.key).write_text(json.dumps(record), "utf-8")
        assert store.load(identity, feature_columns=_COLUMNS) is None

    def test_invalid_parquet_bytes_are_a_miss_even_when_the_hash_matches(
        self, store: FeatureCacheStore
    ) -> None:
        identity = _identity()
        store.save(identity, _frame(), feature_columns=_COLUMNS)
        invalid = b"not a parquet matrix"
        store.path_for(identity.key).write_bytes(invalid)
        meta = store.meta_path_for(identity.key)
        record = json.loads(meta.read_text("utf-8"))
        record["content_hash"] = hashlib.sha256(invalid).hexdigest()
        meta.write_text(json.dumps(record), "utf-8")
        assert store.load(identity, feature_columns=_COLUMNS) is None

    @pytest.mark.parametrize(
        ("columns", "error_type"),
        [
            (None, TypeError),
            ([None], AttributeError),
            pytest.param("missing", KeyError, id="missing-columns"),
        ],
    )
    def test_invalid_pandas_metadata_is_a_recomputable_miss(
        self,
        store: FeatureCacheStore,
        columns: object,
        error_type: type[Exception],
    ) -> None:
        identity = _identity()
        frame = _frame()
        matrix = store.save(identity, frame, feature_columns=_COLUMNS)
        table = pq.read_table(matrix)
        metadata = dict(table.schema.metadata)
        pandas_metadata = json.loads(metadata[b"pandas"])
        if columns == "missing":
            del pandas_metadata["columns"]
        else:
            pandas_metadata["columns"] = columns
        metadata[b"pandas"] = json.dumps(pandas_metadata).encode("utf-8")
        pq.write_table(table.replace_schema_metadata(metadata), matrix)
        sidecar = store.meta_path_for(identity.key)
        record = json.loads(sidecar.read_text("utf-8"))
        record["content_hash"] = hashlib.sha256(matrix.read_bytes()).hexdigest()
        sidecar.write_text(json.dumps(record), "utf-8")

        with pytest.raises(error_type):
            pd.read_parquet(matrix)
        assert store.load(identity, feature_columns=_COLUMNS) is None
        store.save(identity, frame, feature_columns=_COLUMNS)
        rebuilt = store.load(identity, feature_columns=_COLUMNS)
        assert rebuilt is not None
        pd.testing.assert_frame_equal(rebuilt, frame)

    @pytest.mark.parametrize("method", ["read_text", "read_bytes"])
    @pytest.mark.parametrize("error_number", [errno.EACCES, errno.EIO])
    def test_file_access_failures_are_not_hidden_as_cache_misses(
        self,
        store: FeatureCacheStore,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        error_number: int,
    ) -> None:
        identity = _identity()
        store.save(identity, _frame(), feature_columns=_COLUMNS)
        target = (
            store.meta_path_for(identity.key)
            if method == "read_text"
            else store.path_for(identity.key)
        )
        original = getattr(Path, method)
        failure = OSError(error_number, "cache access failed")

        def refuse(path: Path, *args: Any, **kwargs: Any) -> Any:
            if path == target:
                raise failure
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, method, refuse)
        with pytest.raises(OSError) as caught:
            store.load(identity, feature_columns=_COLUMNS)
        assert caught.value is failure

    def test_decoding_uses_the_bytes_whose_hash_was_verified(
        self, store: FeatureCacheStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        identity = _identity()
        original_frame = _frame()
        store.save(identity, original_frame, feature_columns=_COLUMNS)
        original_read = pd.read_parquet

        def replace_before_decode(source: io.BytesIO) -> pd.DataFrame:
            _frame(seed=20).to_parquet(store.path_for(identity.key), index=False)
            return original_read(source)

        monkeypatch.setattr(pd, "read_parquet", replace_before_decode)
        loaded = store.load(identity, feature_columns=_COLUMNS)
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, original_frame)
        assert store.load(identity, feature_columns=_COLUMNS) is None

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
        """The feature-set name alone cannot verify its columns."""
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
        identity = _identity()
        store.save(identity, _frame(), feature_columns=_COLUMNS)
        store.meta_path_for(identity.key).write_text("garbage", "utf-8")
        assert store.load(identity, feature_columns=_COLUMNS) is None
        store.save(identity, _frame(seed=2.0), feature_columns=_COLUMNS)
        loaded = store.load(identity, feature_columns=_COLUMNS)
        assert loaded is not None
        assert loaded["close"].iloc[0] == pytest.approx(102.0)
