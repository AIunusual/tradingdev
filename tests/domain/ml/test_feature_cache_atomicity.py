"""Cache operations serialize publication, recency updates and eviction."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from threading import Event
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from filelock import FileLock, Timeout

from tradingdev.domain.ml.feature_cache import FeatureCacheIdentity
from tradingdev.domain.ml.registry import FeatureCacheStore
from tradingdev.domain.ml.registry import feature_cache_store as cache_module

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any


def _identity(version: str = "one") -> FeatureCacheIdentity:
    return FeatureCacheIdentity("BTC/USDT", "1h", version, "returns", "test-code")


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"close": [100.0, 101.0, 102.0], "feature": [0.0, 0.1, 0.2]})


def _published_bytes(store: FeatureCacheStore) -> dict[Path, bytes]:
    return {
        path: path.read_bytes()
        for pattern in ("*.parquet", "*.json")
        for path in store.root.glob(pattern)
    }


def test_successful_publications_use_unique_temporary_files_and_remove_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FeatureCacheStore(tmp_path)
    sources: list[Path] = []
    original_replace = os.replace

    def replace(source: Path, target: Path) -> None:
        assert source.parent == target.parent == tmp_path
        assert source.suffix == ".tmp"
        sources.append(source)
        original_replace(source, target)

    monkeypatch.setattr(os, "replace", replace)
    store.save(_identity(), _frame(), feature_columns=("feature",))
    store.save(_identity(), _frame(), feature_columns=("feature",))
    assert len(sources) == len(set(sources)) == 4
    assert list(tmp_path.glob("*.tmp")) == []
    assert len(store.keys()) == 1


@pytest.mark.parametrize(
    "operation",
    ["parquet-write", "hash", "parquet-replace", "sidecar-write", "sidecar-replace"],
)
def test_failed_save_removes_unique_staging_and_releases_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    store = FeatureCacheStore(tmp_path)
    store.save(_identity(), _frame(), feature_columns=("feature",))
    before = _published_bytes(store)
    replaced = os.replace

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("injected cache publication failure")

    def fail_write(frame: pd.DataFrame, path: Path, **kwargs: Any) -> None:
        path.write_bytes(b"partial parquet content")
        fail()

    def replace(source: Path, target: Path) -> None:
        if target.suffix == (".parquet" if operation == "parquet-replace" else ".json"):
            fail()
        replaced(source, target)

    with monkeypatch.context() as patch:
        if operation == "parquet-write":
            patch.setattr(pd.DataFrame, "to_parquet", fail_write)
        elif operation == "hash":
            patch.setattr(cache_module, "sha256_file", fail)
        elif operation == "sidecar-write":
            # Fail after creating the sidecar's staging file, before replace.
            @contextmanager
            def temporary(**kwargs: Any) -> Iterator[Any]:
                with NamedTemporaryFile(**kwargs) as stream:
                    if kwargs.get("suffix") == ".json.tmp":
                        patch.setattr(stream, "write", fail)
                    yield stream

            patch.setattr(cache_module, "NamedTemporaryFile", temporary)
        else:
            patch.setattr(os, "replace", replace)
        with pytest.raises(OSError, match="injected cache publication failure"):
            store.save(_identity("two"), _frame(), feature_columns=("feature",))

    assert {path: path.read_bytes() for path in before} == before
    assert list(tmp_path.glob("*.tmp")) == []
    retried = FeatureCacheStore(tmp_path, lock_timeout=0)
    retried.save(_identity("two"), _frame(), feature_columns=("feature",))
    loaded = retried.load(_identity("two"), feature_columns=("feature",))
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, _frame())
    assert set(retried.keys()) == {_identity().key, _identity("two").key}
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("operation", ["load", "save", "evict", "stats", "keys"])
def test_lock_timeout_does_not_read_publish_or_sweep_cache_entries(
    tmp_path: Path, operation: str
) -> None:
    store = FeatureCacheStore(tmp_path, lock_timeout=0)
    store.save(_identity(), _frame(), feature_columns=("feature",))
    before = _published_bytes(store)
    lock_path = tmp_path / ".feature-cache.lock"
    original_inode = lock_path.stat().st_ino
    with FileLock(lock_path), pytest.raises(Timeout):
        if operation == "save":
            store.save(_identity("two"), _frame(), feature_columns=("feature",))
        elif operation == "load":
            store.load(_identity(), feature_columns=("feature",))
        else:
            getattr(store, operation)()
    assert _published_bytes(store) == before
    assert list(tmp_path.glob("*.tmp")) == []
    store.save(_identity("two"), _frame(), feature_columns=("feature",))
    assert len(store.keys()) == 2
    assert lock_path.stat().st_ino == original_inode
    assert store.stats().total_bytes == sum(
        len(blob) for blob in _published_bytes(store).values()
    )


def test_eviction_waits_for_read_and_uses_the_updated_recency(tmp_path: Path) -> None:
    old, newer = _identity(), _identity("two")
    FeatureCacheStore(tmp_path, clock=lambda: 1).save(
        old, _frame(), feature_columns=("feature",)
    )
    FeatureCacheStore(tmp_path, clock=lambda: 2).save(
        newer, _frame(), feature_columns=("feature",)
    )
    touching, release = Event(), Event()
    evicting, evicted = Event(), Event()

    class PausedReader(FeatureCacheStore):
        def _touch(self, key: str, record: dict[str, Any]) -> None:
            touching.set()
            if not release.wait(10):
                raise TimeoutError("test did not release cache recency update")
            super()._touch(key, record)

    reader = PausedReader(tmp_path, clock=lambda: 3)
    bounded = FeatureCacheStore(tmp_path, max_entries=1)

    def evict() -> tuple[str, ...]:
        evicting.set()
        report = bounded.evict()
        evicted.set()
        return report.removed

    with ThreadPoolExecutor(max_workers=2) as pool:
        reading = pool.submit(reader.load, old, feature_columns=("feature",))
        try:
            assert touching.wait(10)
            eviction = pool.submit(evict)
            assert evicting.wait(10)
            assert not evicted.wait(0.2)
            release.set()
            loaded = reading.result(timeout=10)
            assert loaded is not None
            pd.testing.assert_frame_equal(loaded, _frame())
            assert eviction.result(timeout=10) == (newer.key,)
        finally:
            release.set()
    assert bounded.keys() == (old.key,)
    assert bounded.stats().over_limit is False
    assert list(tmp_path.glob("*.tmp")) == []


def test_failed_recency_update_keeps_a_readable_entry_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = FeatureCacheStore(tmp_path)
    store.save(_identity(), _frame(), feature_columns=("feature",))
    before = _published_bytes(store)

    def fail(*args: object, **kwargs: object) -> None:
        raise PermissionError("cannot update recency")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail)
        loaded = store.load(_identity(), feature_columns=("feature",))
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, _frame())
    assert _published_bytes(store) == before
    assert "cannot update recency" in caplog.text
    assert list(tmp_path.glob("*.tmp")) == []
    assert (
        FeatureCacheStore(tmp_path, lock_timeout=0).load(
            _identity(), feature_columns=("feature",)
        )
        is not None
    )
