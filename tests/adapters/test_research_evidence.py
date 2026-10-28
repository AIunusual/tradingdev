"""Immutable publication and bounded verified reads of research evidence."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.research_evidence import ResearchEvidenceStore
from tradingdev.domain.research_evidence import (
    MAX_RESEARCH_EVIDENCE_BYTES,
    ResearchEvidenceDraft,
    ResearchEvidenceRecord,
)

if TYPE_CHECKING:
    from pathlib import Path


def _record(
    *, evidence_id: str = "a" * 32, action: str = "Buy"
) -> ResearchEvidenceRecord:
    draft = ResearchEvidenceDraft.model_validate(
        {
            "presentation": {
                "title": "動能研究",
                "summary": "依價格變化測試進場假設。",
                "parameter_descriptions": {},
            },
            "rules": [
                {
                    "rule_id": "entry",
                    "condition": "Price rises",
                    "action": action,
                    "formula": "close[t] > close[t-1]",
                }
            ],
            "limits": ["Historical results do not establish future returns."],
        }
    )
    return ResearchEvidenceRecord.create(
        draft, evidence_id=evidence_id, created_at=datetime(2026, 1, 1, tzinfo=UTC)
    )


def test_complete_record_round_trips_and_identical_retry_preserves_bytes(
    tmp_path: Path,
) -> None:
    store = ResearchEvidenceStore(WorkspacePaths(tmp_path))
    record = _record()
    store.save(record)
    path = store.path(record.evidence_id)
    original = path.read_bytes()
    assert path == tmp_path / "research_evidence" / f"{record.evidence_id}.json"
    assert store.load(record.evidence_id) == record
    store.save(record)
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


def test_conflicting_record_cannot_replace_existing_content(tmp_path: Path) -> None:
    store = ResearchEvidenceStore(WorkspacePaths(tmp_path))
    record = _record()
    store.save(record)
    path = store.path(record.evidence_id)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="different content"):
        store.save(_record(action="Sell"))
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


def test_concurrent_publications_keep_exactly_one_complete_record(
    tmp_path: Path,
) -> None:
    store = ResearchEvidenceStore(WorkspacePaths(tmp_path))
    candidates = [_record(), _record(action="Sell")]

    def save(record: ResearchEvidenceRecord) -> str | None:
        try:
            store.save(record)
        except ValueError:
            return None
        return record.content_hash

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, candidates))
    successful = [result for result in results if result is not None]
    assert len(successful) == 1
    assert store.load(candidates[0].evidence_id).content_hash == successful[0]
    assert len(list((tmp_path / "research_evidence").iterdir())) == 1


@pytest.mark.parametrize(
    "field", ["content", "schema_version", "hash", "extra", "bytes"]
)
def test_tampered_content_schema_hash_or_bytes_are_rejected(
    tmp_path: Path, field: str
) -> None:
    store = ResearchEvidenceStore(WorkspacePaths(tmp_path))
    record = _record()
    store.save(record)
    path = store.path(record.evidence_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if field == "content":
        payload["draft"]["rules"][0]["action"] = "Sell"
    elif field == "schema_version":
        payload["schema_version"] = 999
    elif field == "hash":
        payload["content_hash"] = "0" * 64
    elif field == "extra":
        payload["unexpected"] = True
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        store.load(record.evidence_id)
    with pytest.raises(ValueError):
        store.save(record)


def test_valid_record_under_another_identity_is_rejected(tmp_path: Path) -> None:
    store = ResearchEvidenceStore(WorkspacePaths(tmp_path))
    record = _record()
    store.save(record)
    store.path("b" * 32).write_bytes(store.path(record.evidence_id).read_bytes())
    with pytest.raises(ValueError, match="identity mismatch"):
        store.load("b" * 32)


@pytest.mark.parametrize("content", [b"{}", b"{broken", b"\xff", b"x" * 524289])
def test_invalid_or_oversized_files_are_rejected(
    tmp_path: Path, content: bytes
) -> None:
    store = ResearchEvidenceStore(WorkspacePaths(tmp_path))
    path = store.path("a" * 32)
    path.parent.mkdir()
    path.write_bytes(content)
    with pytest.raises(ValueError):
        store.load("a" * 32)


def test_missing_record_does_not_create_workspace(tmp_path: Path) -> None:
    root = tmp_path / "absent"
    store = ResearchEvidenceStore(WorkspacePaths(root))
    with pytest.raises(FileNotFoundError):
        store.load("a" * 32)
    assert not root.exists()


@pytest.mark.parametrize("operation", ["fsync", "link"])
def test_failed_write_cleans_temporary_files_and_preserves_other_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    store = ResearchEvidenceStore(WorkspacePaths(tmp_path))
    original = _record()
    store.save(original)

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("publication interrupted")

    monkeypatch.setattr(os, operation, fail)
    with pytest.raises(OSError, match="interrupted"):
        store.save(_record(evidence_id="b" * 32))
    assert store.load(original.evidence_id) == original
    assert list((tmp_path / "research_evidence").iterdir()) == [
        store.path(original.evidence_id)
    ]


@pytest.mark.parametrize("evidence_id", ["../escape", "/absolute", "", "a/b", "A" * 32])
def test_invalid_id_cannot_be_used_as_a_path(tmp_path: Path, evidence_id: str) -> None:
    store = ResearchEvidenceStore(WorkspacePaths(tmp_path))
    with pytest.raises(ValueError, match="Invalid research evidence ID"):
        store.load(evidence_id)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("target", ["directory", "record", "workspace"])
def test_symlinked_paths_are_rejected_without_modifying_targets(
    tmp_path: Path, target: str
) -> None:
    root = tmp_path / "workspace"
    workspace = WorkspacePaths(root)
    store = ResearchEvidenceStore(workspace)
    outside = tmp_path / "outside"
    outside.mkdir()
    if target == "workspace":
        root.symlink_to(outside, target_is_directory=True)
    else:
        root.mkdir()
        if target == "directory":
            (root / "research_evidence").symlink_to(outside, target_is_directory=True)
        else:
            (root / "research_evidence").mkdir()
            (root / "research_evidence" / f"{'a' * 32}.json").symlink_to(
                outside / "absent.json"
            )
    with pytest.raises(ValueError, match="path leaves workspace"):
        store.save(_record())
    with pytest.raises(ValueError, match="path leaves workspace"):
        store.load("a" * 32)
    assert list(outside.iterdir()) == []


def test_named_pipe_is_rejected_without_blocking(tmp_path: Path) -> None:
    store = ResearchEvidenceStore(WorkspacePaths(tmp_path))
    path = store.path("a" * 32)
    path.parent.mkdir()
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular file"):
        store.load("a" * 32)


def test_save_rechecks_record_size_and_integrity(tmp_path: Path) -> None:
    record = _record()
    invalid = record.model_copy(update={"content_hash": "0" * 64})
    store = ResearchEvidenceStore(WorkspacePaths(tmp_path))
    with pytest.raises(ValueError, match="hash"):
        store.save(invalid)
    oversized_draft = record.draft.model_copy(
        update={"limits": ("x" * MAX_RESEARCH_EVIDENCE_BYTES,)}
    )
    oversized = record.model_copy(update={"draft": oversized_draft})
    with pytest.raises(ValueError):
        store.save(oversized)
    assert list(tmp_path.iterdir()) == []
