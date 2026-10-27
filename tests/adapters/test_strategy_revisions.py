"""Storage guarantees for immutable generated strategy revisions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
import yaml
from filelock import FileLock

from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_text
from tradingdev.adapters.storage.strategy_revisions import (
    StrategyNamespaceOccupiedError,
    StrategyRevisionConflictError,
    StrategyRevisionError,
    StrategyRevisionIntegrityError,
    StrategyRevisionStore,
    UnsupportedStrategyRevisionError,
)
from tradingdev.domain.strategies.schemas import StrategyStatus, ValidationResult


def _config() -> dict[str, Any]:
    return {
        "strategy": {
            "id": "forged",
            "revision_id": "forged",
            "class_name": "FixtureStrategy",
            "source_path": "/outside/forged.py",
            "source_hash": "forged",
            "parameters": {"threshold": 3},
        },
        "backtest": {"symbol": "BTC/USDT"},
    }


def _store(tmp_path: Path) -> StrategyRevisionStore:
    return StrategyRevisionStore(WorkspacePaths(tmp_path / "workspace"))


def test_each_save_creates_an_independent_draft_without_mutating_input(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    config = _config()
    original = deepcopy(config)
    first = store.create("fixture", "original source", config, "First request")
    first.validation = ValidationResult(
        revision_id=first.revision_id, checked_at=first.updated_at, success=True
    )
    first.status = StrategyStatus.VALIDATED
    store.update(first, expected_status=StrategyStatus.DRAFT)
    first_files = {
        path.name: path.read_bytes()
        for path in Path(first.source_path).parent.iterdir()
    }

    second = store.create(
        "fixture", "replacement source", config, expected_revision_id=first.revision_id
    )

    assert config == original
    assert UUID(second.revision_id).version == 4
    assert second.revision_id != first.revision_id
    assert second.status == StrategyStatus.DRAFT
    assert second.validation is None and second.dry_run is None
    assert store.load("fixture") == second
    assert store.load("fixture", first.revision_id) == first
    assert {
        path.name: path.read_bytes()
        for path in Path(first.source_path).parent.iterdir()
    } == first_files
    parsed = yaml.safe_load(Path(second.config_path).read_text())
    assert parsed["strategy"]["id"] == "fixture"
    assert parsed["strategy"]["revision_id"] == second.revision_id
    assert parsed["strategy"]["source_path"] == second.source_path
    assert "source_hash" not in parsed["strategy"]
    assert parsed["strategy"]["parameters"] == {"threshold": 3}
    pointer = Path(second.source_path).parents[2] / "current.json"
    assert json.loads(pointer.read_text()) == {"revision_id": second.revision_id}
    assert store.list_current() == [second]


def test_identical_saves_still_create_distinct_revisions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.create("fixture", "same source", _config())
    second = store.create(
        "fixture", "same source", _config(), expected_revision_id=first.revision_id
    )
    assert first.revision_id != second.revision_id
    assert first.source_hash == second.source_hash
    assert store.load("fixture", first.revision_id) == first


@pytest.mark.parametrize("code", ["first\nsecond\n", "first\r\nsecond\r\n"])
def test_source_bytes_and_hash_are_preserved_across_newline_styles(
    tmp_path: Path, code: str
) -> None:
    store = _store(tmp_path)
    metadata = store.create("fixture", code, _config())
    assert Path(metadata.source_path).read_bytes() == code.encode("utf-8")
    assert metadata.source_hash == sha256_text(code)
    assert store.load("fixture") == metadata


def test_updating_old_revision_does_not_move_current(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.create("fixture", "one", _config())
    second = store.create(
        "fixture", "two", _config(), expected_revision_id=first.revision_id
    )
    first.status = StrategyStatus.REJECTED
    store.update(first, expected_status=StrategyStatus.DRAFT)
    assert store.load("fixture") == second
    assert store.load("fixture", first.revision_id) == first


@pytest.mark.parametrize("strategy_id", ["../escape", "a/b", "A", "a\n", "", "."])
def test_invalid_strategy_identifiers_are_rejected_before_writing(
    tmp_path: Path, strategy_id: str
) -> None:
    store = _store(tmp_path)
    with pytest.raises(StrategyRevisionError, match="snake_case"):
        store.create(strategy_id, "source", _config())
    with pytest.raises(StrategyRevisionError, match="snake_case"):
        store.load(strategy_id)
    assert not (tmp_path / "workspace").exists()


@pytest.mark.parametrize(
    "revision_id",
    ["../escape", "bad", str(uuid4()), uuid4().hex.upper(), "0" * 32],
)
def test_invalid_revision_ids_cannot_resolve_paths(
    tmp_path: Path, revision_id: str
) -> None:
    with pytest.raises(StrategyRevisionError, match="UUID4 hex"):
        _store(tmp_path).load("fixture", revision_id)
    assert not (tmp_path / "workspace").exists()


def test_unknown_revision_and_absent_strategy_are_not_found(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.load("fixture") is None
    assert store.load("fixture", uuid4().hex) is None
    assert store.list_current() == []


@pytest.mark.parametrize("filename", ["strategy.py", "config.yaml"])
def test_content_tampering_is_rejected(tmp_path: Path, filename: str) -> None:
    store = _store(tmp_path)
    metadata = store.create("fixture", "source", _config())
    (Path(metadata.source_path).parent / filename).write_text("tampered")
    with pytest.raises(StrategyRevisionIntegrityError, match="content mismatch"):
        store.load("fixture")
    with pytest.raises(StrategyRevisionIntegrityError, match="content mismatch"):
        store.update(metadata)


@pytest.mark.parametrize("field", ["source_path", "config_path", "revision_id"])
def test_metadata_path_or_identity_tampering_is_rejected(
    tmp_path: Path, field: str
) -> None:
    store = _store(tmp_path)
    metadata = store.create("fixture", "source", _config())
    path = Path(metadata.source_path).parent / "metadata.json"
    raw = json.loads(path.read_text())
    raw[field] = str(tmp_path / "outside")
    path.write_text(json.dumps(raw))
    with pytest.raises(StrategyRevisionIntegrityError, match="match"):
        store.load("fixture")


def test_symlinked_strategy_root_cannot_write_outside_workspace(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    workspace.ensure()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace.generated_strategies / "fixture").symlink_to(
        outside, target_is_directory=True
    )
    with pytest.raises(StrategyRevisionIntegrityError, match="workspace"):
        StrategyRevisionStore(workspace).create("fixture", "source", _config())
    assert list(outside.iterdir()) == []


def test_symlinked_revision_source_is_rejected_even_with_matching_bytes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    metadata = store.create("fixture", "source", _config())
    outside = tmp_path / "source.py"
    outside.write_text("source")
    source = Path(metadata.source_path)
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(StrategyRevisionIntegrityError, match="workspace"):
        store.load("fixture")


def test_update_rejects_reassigned_content_hashes_and_cross_revision_evidence(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    first = store.create("fixture", "one", _config())
    second = store.create(
        "fixture", "two", _config(), expected_revision_id=first.revision_id
    )
    altered = first.model_copy(update={"source_hash": sha256_text("altered")})
    with pytest.raises(StrategyRevisionIntegrityError, match="immutable"):
        store.update(altered)
    first.validation = ValidationResult(
        revision_id=second.revision_id, checked_at=first.updated_at, success=True
    )
    with pytest.raises(StrategyRevisionIntegrityError, match="Evidence"):
        store.update(first)
    reloaded = store.load("fixture", first.revision_id)
    assert reloaded is not None and reloaded.validation is None


def test_legacy_metadata_is_reported_without_migration(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    workspace.ensure()
    legacy = workspace.generated_strategies / "fixture.json"
    content = '{"strategy_id": "fixture", "status": "runnable"}'
    legacy.write_text(content)
    store = StrategyRevisionStore(workspace)
    with pytest.raises(UnsupportedStrategyRevisionError, match="save.*explicitly"):
        store.load("fixture")
    assert store.list_current() == []
    assert store.list_legacy_ids() == ["fixture"]
    assert legacy.read_text() == content
    assert not (workspace.generated_strategies / "fixture").exists()
    with pytest.raises(StrategyNamespaceOccupiedError, match="already exists"):
        store.create("fixture", "new source", _config())
    assert store.list_current() == []
    assert store.list_legacy_ids() == ["fixture"]
    assert legacy.read_text() == content


@pytest.mark.parametrize("location", ["source", "config", "config_directory"])
def test_legacy_recovery_rejects_symlinked_paths(tmp_path: Path, location: str) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    workspace.ensure()
    source = workspace.generated_strategies / "fixture.py"
    config = workspace.configs / "fixture.yaml"
    source.write_text("original source")
    config.write_text("original config")
    outside = tmp_path / "outside"
    outside.mkdir()
    if location == "config_directory":
        config.unlink()
        workspace.configs.rmdir()
        workspace.configs.symlink_to(outside, target_is_directory=True)
    else:
        path = source if location == "source" else config
        path.unlink()
        target = outside / "untrusted"
        target.write_text("must not be read")
        path.symlink_to(target)
    store = StrategyRevisionStore(workspace)
    with pytest.raises(StrategyRevisionIntegrityError, match="path leaves workspace"):
        store.read_legacy_source("fixture")


def test_revision_is_complete_before_current_pointer_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    first = store.create("fixture", "one", _config())
    replace = os.replace
    observed = []

    def observe_replace(source: Path, destination: Path) -> None:
        if destination.name == "current.json":
            assert store.load("fixture") == first
            revision_id = json.loads(source.read_text())["revision_id"]
            published = store.load("fixture", revision_id)
            assert published is not None
            observed.append(published)
        replace(source, destination)

    monkeypatch.setattr(os, "replace", observe_replace)
    second = store.create(
        "fixture", "two", _config(), expected_revision_id=first.revision_id
    )
    assert observed == [second]
    assert store.load("fixture") == second


def test_failed_revision_write_cleans_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    first = store.create("fixture", "one", _config())

    def fail_serialization(_payload: dict[str, Any]) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(store, "_serialize", fail_serialization)
    with pytest.raises(OSError, match="disk full"):
        store.create(
            "fixture", "two", _config(), expected_revision_id=first.revision_id
        )
    assert store.load("fixture") == first
    revisions = Path(first.source_path).parent.parent
    assert [path.name for path in revisions.iterdir()] == [first.revision_id]
    assert not list((tmp_path / "workspace").rglob(".pending-*"))


def test_failed_pointer_replace_preserves_current_and_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    first = store.create("fixture", "one", _config())

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        store.create(
            "fixture", "two", _config(), expected_revision_id=first.revision_id
        )
    assert store.load("fixture") == first
    revisions = list(Path(first.source_path).parent.parent.iterdir())
    assert len(revisions) == 2
    for path in revisions:
        assert store.load("fixture", path.name) is not None
    assert not list((tmp_path / "workspace").rglob(".pending-*"))


def test_failed_metadata_replace_preserves_previous_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    metadata = store.create("fixture", "source", _config())
    changed = metadata.model_copy(update={"status": StrategyStatus.REJECTED})

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        store.update(changed, expected_status=StrategyStatus.DRAFT)
    assert store.load("fixture") == metadata
    assert not list((tmp_path / "workspace").rglob(".pending-*"))


def test_metadata_lock_timeout_preserves_evidence_and_allows_later_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    metadata = store.create("fixture", "source", _config())
    changed = metadata.model_copy(update={"status": StrategyStatus.REJECTED})
    lock_path = Path(metadata.source_path).parent / ".metadata.lock"

    def immediate_lock(path: Path, *, timeout: float) -> FileLock:
        return FileLock(path, timeout=0)

    monkeypatch.setattr(
        "tradingdev.adapters.storage.strategy_revisions.FileLock", immediate_lock
    )
    with FileLock(lock_path):
        with pytest.raises(StrategyRevisionError, match="busy; retry"):
            store.update(changed, expected_status=StrategyStatus.DRAFT)
        assert store.load("fixture") == metadata
        assert not list((tmp_path / "workspace").rglob(".pending-*"))
    store.update(changed, expected_status=StrategyStatus.DRAFT)
    assert store.load("fixture") == changed


def test_stale_lifecycle_update_cannot_replace_completed_evidence(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    original = store.create("fixture", "source", _config())
    validated = original.model_copy(update={"status": StrategyStatus.VALIDATED})
    store.update(validated, expected_status=StrategyStatus.DRAFT)
    rejected = original.model_copy(update={"status": StrategyStatus.REJECTED})
    with pytest.raises(StrategyRevisionError, match="status changed"):
        store.update(rejected, expected_status=StrategyStatus.DRAFT)
    assert store.load("fixture") == validated


def test_same_status_update_preserves_newer_validation_evidence(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    metadata = store.create("fixture", "source", _config())
    metadata.status = StrategyStatus.VALIDATED
    metadata.validation = ValidationResult(
        revision_id=metadata.revision_id, checked_at="old", success=True
    )
    store.update(metadata, expected_status=StrategyStatus.DRAFT)
    before_dry_run = metadata.model_copy(deep=True)
    newer_validation = metadata.model_copy(deep=True)
    assert newer_validation.validation is not None
    newer_validation.validation.checked_at = "new"
    # Keep both status and updated_at identical to prove the full state is checked.
    store.update(newer_validation, expected_metadata=before_dry_run)
    metadata.status = StrategyStatus.RUNNABLE
    metadata.dry_run = ValidationResult(
        revision_id=metadata.revision_id, checked_at="finished", success=True
    )

    with pytest.raises(StrategyRevisionError, match="metadata changed"):
        store.update(
            metadata,
            expected_status=StrategyStatus.VALIDATED,
            expected_metadata=before_dry_run,
        )

    assert store.load("fixture") == newer_validation
    assert not list((tmp_path / "workspace").rglob(".pending-*"))


def test_compare_and_swap_serializes_across_processes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metadata = store.create("fixture", "source", _config())
    script = """
import sys
import time
from pathlib import Path
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.strategy_revisions import (
    StrategyRevisionStore, StrategyRevisionError,
)
from tradingdev.domain.strategies.schemas import StrategyStatus
store = StrategyRevisionStore(WorkspacePaths(Path(sys.argv[1])))
metadata = store.load('fixture')
original = store._atomic_json
def delayed_write(path, payload):
    time.sleep(0.2)
    original(path, payload)
store._atomic_json = delayed_write
ready = Path(sys.argv[1]) / ('ready-' + sys.argv[2])
ready.touch()
deadline = time.monotonic() + 10
while len(list(ready.parent.glob('ready-*'))) != 2:
    if time.monotonic() > deadline:
        raise TimeoutError('Both update processes must reach the barrier')
    time.sleep(0.01)
metadata.status = StrategyStatus.VALIDATED
try:
    store.update(metadata, expected_status=StrategyStatus.DRAFT)
except StrategyRevisionError:
    print('conflict', flush=True)
else:
    print('updated', flush=True)
"""
    processes: list[subprocess.Popen[str]] = []
    try:
        for index in range(2):
            process = subprocess.Popen(
                [sys.executable, "-c", script, str(tmp_path / "workspace"), str(index)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append(process)
        results = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=15)
            assert process.returncode == 0, stderr
            results.append(stdout.strip())
        assert sorted(results) == ["conflict", "updated"]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.communicate(timeout=5)
    loaded = store.load("fixture", metadata.revision_id)
    assert loaded is not None and loaded.status == StrategyStatus.VALIDATED


@pytest.mark.parametrize(
    "occupied_path",
    [
        "generated_strategies/fixture.json",
        "generated_strategies/fixture.py",
        "configs/fixture.yaml",
        "generated_strategies/fixture/revisions/unfinished/strategy.py",
    ],
)
def test_new_namespace_rejects_legacy_and_incomplete_publications(
    tmp_path: Path, occupied_path: str
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    occupied = workspace.root / occupied_path
    occupied.parent.mkdir(parents=True)
    occupied.write_text("retained evidence")
    store = StrategyRevisionStore(workspace)

    with pytest.raises(StrategyNamespaceOccupiedError, match="already exists"):
        store.create("fixture", "new source", _config())

    assert occupied.read_text() == "retained evidence"
    assert not (workspace.generated_strategies / "fixture" / "current.json").exists()


def test_existing_strategy_requires_an_explicit_current_revision(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    original = store.create("fixture", "original", _config())
    pointer = Path(original.source_path).parents[2] / "current.json"
    pointer_bytes = pointer.read_bytes()

    with pytest.raises(StrategyNamespaceOccupiedError):
        store.create("fixture", "unrequested replacement", _config())
    with pytest.raises(StrategyRevisionConflictError):
        store.create(
            "fixture", "stale replacement", _config(), expected_revision_id=uuid4().hex
        )

    assert pointer.read_bytes() == pointer_bytes
    assert store.load("fixture") == original
    assert store.revision_ids("fixture") == [original.revision_id]


def test_revision_of_missing_strategy_does_not_create_it(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(StrategyRevisionConflictError):
        store.create("fixture", "source", _config(), expected_revision_id=uuid4().hex)
    assert store.load("fixture") is None
    assert store.revision_ids("fixture") == []


def test_concurrent_revisions_cannot_both_replace_the_same_expected_revision(
    tmp_path: Path,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    workspace = WorkspacePaths(tmp_path / "workspace")
    original = StrategyRevisionStore(workspace).create("fixture", "old", _config())

    def revise(code: str) -> str:
        try:
            return (
                StrategyRevisionStore(workspace)
                .create(
                    "fixture",
                    code,
                    _config(),
                    expected_revision_id=original.revision_id,
                )
                .revision_id
            )
        except StrategyRevisionConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(revise, ["first writer", "second writer"]))

    assert outcomes.count("conflict") == 1
    winner = next(result for result in outcomes if result != "conflict")
    current = StrategyRevisionStore(workspace).load("fixture")
    assert current is not None and current.revision_id == winner
    assert set(StrategyRevisionStore(workspace).revision_ids("fixture")) == {
        original.revision_id,
        winner,
    }
