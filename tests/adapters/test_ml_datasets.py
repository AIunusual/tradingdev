"""Immutable ML dataset publication preserves complete, verifiable evidence."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
from filelock import FileLock, Timeout

from tradingdev.adapters.storage import ml_datasets
from tradingdev.adapters.storage.filesystem import write_json
from tradingdev.adapters.storage.ml_datasets import MLDatasetStore

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess
    from multiprocessing.synchronize import Event


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "feature": [0.1, 0.3, -0.2, 0.4],
            "forward_return": [0.01, -0.02, 0.03, -0.01],
            "target": [1, -1, 1, -1],
        }
    )


def _metadata() -> dict[str, Any]:
    return {"symbol": "BTC/USDT", "horizon": 1, "feature_columns": ["feature"]}


def _published_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for directory in root.glob("ds_*")
        for path in directory.iterdir()
        if path.is_file()
    }


def _stop_workers(workers: list[BaseProcess]) -> None:
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
        worker.join(5)
        if worker.is_alive():
            worker.kill()
            worker.join(5)
        assert not worker.is_alive(), "dataset test process cleanup failed"
        worker.close()


def _save_in_process(root: Path, ready: Event, staged: Event, release: Event) -> None:
    def pause_after_manifest(path: Path, payload: dict[str, Any]) -> None:
        write_json(path, payload)
        staged.set()
        if not release.wait(15):
            raise TimeoutError("test did not release the dataset publisher")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ml_datasets, "write_json", pause_after_manifest)
        ready.set()
        MLDatasetStore(root).save(_frame(), _metadata())


def _hold_publish_lock(root: Path, ready: Event, release: Event) -> None:
    with FileLock(root / ".publish.lock", timeout=5):
        ready.set()
        if not release.wait(15):
            raise TimeoutError("test did not release the dataset lock")


def test_save_round_trip_is_idempotent_without_replacing_published_files(
    tmp_path: Path,
) -> None:
    store = MLDatasetStore(tmp_path)
    frame = _frame()
    metadata = _metadata()
    path, manifest = store.save(frame, metadata)
    before = _published_bytes(tmp_path)
    stat = path.stat()

    repeated_path, repeated_manifest = store.save(frame.copy(), dict(metadata))
    loaded, loaded_manifest = store.load(manifest["dataset_id"])

    assert repeated_path == path
    assert repeated_manifest == loaded_manifest == manifest
    assert metadata == _metadata()
    assert path.stat().st_ino == stat.st_ino
    assert path.stat().st_mtime_ns == stat.st_mtime_ns
    assert _published_bytes(tmp_path) == before
    pd.testing.assert_frame_equal(loaded, frame)
    assert list(tmp_path.glob(".publish-*")) == []


@pytest.mark.parametrize("change", ["content", "metadata"])
def test_changed_evidence_gets_a_new_id_and_preserves_the_old_dataset(
    tmp_path: Path, change: str
) -> None:
    store = MLDatasetStore(tmp_path)
    _, original = store.save(_frame(), _metadata())
    before = _published_bytes(tmp_path)
    frame, metadata = _frame(), _metadata()
    if change == "content":
        frame.loc[0, "feature"] = 2.0
    else:
        metadata["horizon"] = 2
    _, changed = store.save(frame, metadata)

    assert changed["dataset_id"] != original["dataset_id"]
    assert all(
        _published_bytes(tmp_path)[key] == value for key, value in before.items()
    )
    old_frame, old_manifest = store.load(original["dataset_id"])
    assert old_manifest == original
    pd.testing.assert_frame_equal(old_frame, _frame())


@pytest.mark.parametrize("tamper", ["content", "metadata"])
def test_load_and_repeated_save_reject_tampering_without_repairing_evidence(
    tmp_path: Path, tamper: str
) -> None:
    store = MLDatasetStore(tmp_path)
    path, manifest = store.save(_frame(), _metadata())
    if tamper == "content":
        path.write_bytes(path.read_bytes() + b"tampered")
        message = "content hash mismatch"
    else:
        metadata_path = path.parent / "manifest.json"
        changed = json.loads(metadata_path.read_text("utf-8"))
        changed["horizon"] = 99
        metadata_path.write_text(json.dumps(changed), encoding="utf-8")
        message = "metadata identity mismatch"
    tampered = _published_bytes(tmp_path)

    with pytest.raises(ValueError, match=message):
        store.load(manifest["dataset_id"])
    with pytest.raises(ValueError, match=message):
        store.save(_frame(), _metadata())
    assert _published_bytes(tmp_path) == tampered
    assert list(tmp_path.glob(".publish-*")) == []


def test_concurrent_processes_publish_the_same_complete_dataset_once(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    release = context.Event()
    ready = [context.Event(), context.Event()]
    staged = [context.Event(), context.Event()]
    workers = [
        context.Process(
            target=_save_in_process,
            args=(tmp_path, ready[index], staged[index], release),
        )
        for index in range(2)
    ]
    started: list[BaseProcess] = []
    try:
        workers[0].start()
        started.append(workers[0])
        assert ready[0].wait(15)
        assert staged[0].wait(15)
        workers[1].start()
        started.append(workers[1])
        assert ready[1].wait(15)
        assert not staged[1].wait(0.3)
        assert list(tmp_path.glob("ds_*")) == []
        release.set()
        for worker in workers:
            worker.join(15)
            assert worker.exitcode == 0
        directories = list(tmp_path.glob("ds_*"))
        assert len(directories) == 1
        loaded, manifest = MLDatasetStore(tmp_path).load(directories[0].name)
        pd.testing.assert_frame_equal(loaded, _frame())
        assert manifest["symbol"] == "BTC/USDT"
        assert not staged[1].is_set()
        assert list(tmp_path.glob(".publish-*")) == []
    finally:
        release.set()
        _stop_workers(started)


@pytest.mark.parametrize("operation", ["bytes", "manifest", "rename"])
def test_failed_publication_removes_staging_and_preserves_previous_datasets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    store = MLDatasetStore(tmp_path)
    store.save(_frame(), _metadata())
    before = _published_bytes(tmp_path)
    changed = {**_metadata(), "horizon": 2}

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("injected dataset publication failure")

    with monkeypatch.context() as patch:
        if operation == "manifest":
            patch.setattr(ml_datasets, "write_json", fail)
        else:
            patch.setattr(
                Path, "write_bytes" if operation == "bytes" else "rename", fail
            )
        with pytest.raises(OSError, match="injected dataset publication failure"):
            store.save(_frame(), changed)
    assert _published_bytes(tmp_path) == before
    assert list(tmp_path.glob(".publish-*")) == []
    # A fresh store can acquire the released lock and publish successfully.
    _, manifest = MLDatasetStore(tmp_path, lock_timeout=0).save(_frame(), changed)
    assert store.load(manifest["dataset_id"])[1]["horizon"] == 2


def test_lock_timeout_creates_no_staging_and_leaves_existing_data_intact(
    tmp_path: Path,
) -> None:
    store = MLDatasetStore(tmp_path, lock_timeout=0.1)
    store.save(_frame(), _metadata())
    before = _published_bytes(tmp_path)
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    worker = context.Process(target=_hold_publish_lock, args=(tmp_path, ready, release))
    started: list[BaseProcess] = []
    changed = {**_metadata(), "horizon": 2}
    try:
        worker.start()
        started.append(worker)
        assert ready.wait(15)
        with pytest.raises(Timeout):
            store.save(_frame(), changed)
        assert _published_bytes(tmp_path) == before
        assert list(tmp_path.glob(".publish-*")) == []
        release.set()
        worker.join(15)
        assert worker.exitcode == 0
    finally:
        release.set()
        _stop_workers(started)
    _, manifest = store.save(_frame(), changed)
    assert store.load(manifest["dataset_id"])[1]["horizon"] == 2
