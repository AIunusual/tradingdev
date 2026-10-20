"""Model publication keeps readers on complete, content-bound artifacts."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from tempfile import NamedTemporaryFile
from threading import Event
from typing import TYPE_CHECKING, Any

import pytest
from filelock import FileLock, Timeout

from tradingdev.domain.ml.registry import (
    ModelArtifact,
    ModelArtifactStore,
    build_model_artifact,
    content_sha256,
)
from tradingdev.domain.ml.registry import artifact_store as store_module

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _artifact(blob: bytes = b"model weights") -> ModelArtifact:
    return build_model_artifact(
        model_name="direction",
        framework="test",
        kind="classifier",
        parameters={"seed": 42},
        feature_names=("returns",),
        content_hash=content_sha256(blob),
        n_train_samples=100,
    )


def _published(root: Path) -> dict[Path, bytes]:
    return {
        path: path.read_bytes()
        for path in root.iterdir()
        if path.suffix in {".bin", ".json"}
    }


def test_repeat_save_keeps_the_blob_and_atomically_publishes_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ModelArtifactStore(tmp_path)
    artifact, blob = _artifact(), b"model weights"
    sources: list[Path] = []
    original_replace = os.replace

    def publish(source: Path, target: Path) -> None:
        assert source.parent == target.parent == tmp_path
        assert source.suffix == ".tmp"
        sources.append(source)
        original_replace(source, target)

    monkeypatch.setattr(os, "replace", publish)
    store.save(model_bytes=blob, artifact=artifact)
    before = store.path_for(artifact.model_id).stat()
    updated = replace(artifact, n_train_samples=120)
    store.save(model_bytes=blob, artifact=updated)
    after = store.path_for(artifact.model_id).stat()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)
    assert store.load_with_bytes(artifact.model_id) == (updated, blob)
    assert len(sources) == len(set(sources)) == 3
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("published", [False, True], ids=["first", "repeated"])
def test_concurrent_writers_publish_complete_artifacts_without_blocking_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, published: bool
) -> None:
    artifact, blob = _artifact(), b"model weights"
    first, second = ModelArtifactStore(tmp_path), ModelArtifactStore(tmp_path)
    if published:
        first.save(model_bytes=blob, artifact=artifact)
    paused, release, second_started, second_published = (
        Event(),
        Event(),
        Event(),
        Event(),
    )
    original_replace = os.replace
    publications = 0

    def publish(source: Path, target: Path) -> None:
        nonlocal publications
        if target == tmp_path / f"{artifact.model_id}.json":
            publications += 1
            if publications == 1:
                paused.set()
                if not release.wait(10):
                    raise TimeoutError("test did not release model publication")
            else:
                second_published.set()
        original_replace(source, target)

    def save_second() -> ModelArtifact:
        second_started.set()
        return second.save(model_bytes=blob, artifact=artifact)

    monkeypatch.setattr(os, "replace", publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writing = pool.submit(first.save, model_bytes=blob, artifact=artifact)
        try:
            assert paused.wait(10)
            if published:
                assert first.load_with_bytes(artifact.model_id) == (artifact, blob)
                assert first.list() == [artifact]
            else:
                assert not first.exists(artifact.model_id)
                assert first.list() == []
                with pytest.raises(FileNotFoundError, match="no model artifact"):
                    first.load(artifact.model_id)
            waiting = pool.submit(save_second)
            assert second_started.wait(10)
            assert not second_published.wait(0.2)
            # Another identity must remain independent of this writer's lock.
            other_blob = b"another model"
            other = _artifact(other_blob)
            second.save(model_bytes=other_blob, artifact=other)
            assert second.load_with_bytes(other.model_id) == (other, other_blob)
            release.set()
            assert writing.result(timeout=10) == artifact
            assert waiting.result(timeout=10) == artifact
        finally:
            release.set()
    assert first.load_with_bytes(artifact.model_id) == (artifact, blob)
    assert second_published.is_set()
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("published", [False, True], ids=["first", "repeated"])
@pytest.mark.parametrize("failure", [OSError, TimeoutError])
@pytest.mark.parametrize("operation", ["write", "fsync", "replace"])
def test_failed_publication_cleans_staging_and_preserves_published_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published: bool,
    failure: type[Exception],
    operation: str,
) -> None:
    artifact, blob = _artifact(), b"model weights"
    store = ModelArtifactStore(tmp_path)
    if published:
        store.save(model_bytes=blob, artifact=artifact)
    before = _published(tmp_path)
    updated = replace(artifact, n_train_samples=120)
    original_replace, original_fsync = os.replace, os.fsync
    syncs = 0

    def fail(*args: object, **kwargs: object) -> None:
        raise failure("injected model publication failure")

    def publish(source: Path, target: Path) -> None:
        if target.suffix == ".json":
            fail()
        original_replace(source, target)

    def sync(descriptor: int) -> None:
        nonlocal syncs
        syncs += 1
        if syncs == (1 if published else 2):
            fail()
        original_fsync(descriptor)

    @contextmanager
    def temporary(**kwargs: Any) -> Iterator[Any]:
        with NamedTemporaryFile(**kwargs) as stream:
            if kwargs.get("suffix") == ".json.tmp":
                monkeypatch.setattr(stream, "write", fail)
            yield stream

    with monkeypatch.context() as patch:
        if operation == "write":
            patch.setattr(store_module, "NamedTemporaryFile", temporary)
        elif operation == "fsync":
            patch.setattr(os, "fsync", sync)
        else:
            patch.setattr(os, "replace", publish)
        with pytest.raises(failure, match="injected model publication failure"):
            store.save(model_bytes=blob, artifact=updated)
    assert _published(tmp_path) == before
    assert list(tmp_path.glob("*.tmp")) == []
    if published:
        assert store.load_with_bytes(artifact.model_id) == (artifact, blob)
    else:
        assert not store.exists(artifact.model_id)
    # A fresh writer must acquire the released lock and complete publication.
    ModelArtifactStore(tmp_path, lock_timeout=0).save(
        model_bytes=blob, artifact=updated
    )
    assert store.load_with_bytes(artifact.model_id) == (updated, blob)


def test_lock_timeout_preserves_data_and_allows_reads(tmp_path: Path) -> None:
    artifact, blob = _artifact(), b"model weights"
    store = ModelArtifactStore(tmp_path, lock_timeout=0)
    store.save(model_bytes=blob, artifact=artifact)
    before = _published(tmp_path)
    lock_path = tmp_path / f".{artifact.model_id}.lock"
    inode = lock_path.stat().st_ino
    with FileLock(lock_path):
        with pytest.raises(Timeout):
            store.save(model_bytes=blob, artifact=artifact)
        assert store.load_with_bytes(artifact.model_id) == (artifact, blob)
    assert _published(tmp_path) == before
    assert list(tmp_path.glob("*.tmp")) == []
    store.save(model_bytes=blob, artifact=artifact)
    assert lock_path.stat().st_ino == inode


@pytest.mark.parametrize("has_metadata", [False, True])
def test_existing_conflicting_blob_is_never_overwritten(
    tmp_path: Path, has_metadata: bool
) -> None:
    artifact, blob = _artifact(), b"model weights"
    store = ModelArtifactStore(tmp_path)
    if has_metadata:
        store.save(model_bytes=blob, artifact=artifact)
    store.path_for(artifact.model_id).write_bytes(b"conflicting content")
    before = _published(tmp_path)
    with pytest.raises(ValueError, match="model blob tampered"):
        store.save(model_bytes=blob, artifact=artifact)
    assert _published(tmp_path) == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_unpublished_matching_blob_can_complete_a_retry(tmp_path: Path) -> None:
    artifact, blob = _artifact(), b"model weights"
    store = ModelArtifactStore(tmp_path)
    path = store.path_for(artifact.model_id)
    path.write_bytes(blob)
    inode = path.stat().st_ino
    store.save(model_bytes=blob, artifact=artifact)
    assert store.load_with_bytes(artifact.model_id) == (artifact, blob)
    assert path.stat().st_ino == inode


@pytest.mark.parametrize("failure", [OSError, TimeoutError])
@pytest.mark.parametrize("operation", ["write", "fsync", "replace", "after-replace"])
def test_failed_initial_blob_publication_leaves_no_record_or_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[Exception],
    operation: str,
) -> None:
    artifact, blob = _artifact(), b"model weights"
    store = ModelArtifactStore(tmp_path)
    original_replace = os.replace

    def fail(*args: object, **kwargs: object) -> None:
        raise failure("injected initial blob failure")

    def publish(source: Path, target: Path) -> None:
        if operation == "after-replace":
            original_replace(source, target)
        fail()

    with monkeypatch.context() as patch:

        @contextmanager
        def temporary(**kwargs: Any) -> Iterator[Any]:
            with NamedTemporaryFile(**kwargs) as stream:
                patch.setattr(stream, "write", fail)
                yield stream

        if operation == "write":
            patch.setattr(store_module, "NamedTemporaryFile", temporary)
        elif operation == "fsync":
            patch.setattr(os, "fsync", fail)
        else:
            patch.setattr(os, "replace", publish)
        with pytest.raises(failure, match="injected initial blob failure"):
            store.save(model_bytes=blob, artifact=artifact)
    assert _published(tmp_path) == {}
    assert list(tmp_path.glob("*.tmp")) == []
    ModelArtifactStore(tmp_path, lock_timeout=0).save(
        model_bytes=blob, artifact=artifact
    )
    assert store.load_with_bytes(artifact.model_id) == (artifact, blob)


def test_a_model_id_collision_cannot_replace_the_published_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, blob = _artifact(), b"model weights"
    store = ModelArtifactStore(tmp_path)
    store.save(model_bytes=blob, artifact=artifact)
    before = _published(tmp_path)
    # Force the truncated hash collision; neither identity may overwrite the
    # other even if its blob happens to be identical.
    monkeypatch.setattr(
        store_module, "compute_model_id", lambda **kwargs: artifact.model_id
    )
    with pytest.raises(ValueError, match="model identity collision"):
        store.save(
            model_bytes=blob, artifact=replace(artifact, feature_names=("other",))
        )
    assert _published(tmp_path) == before
    assert store.load_with_bytes(artifact.model_id) == (artifact, blob)
