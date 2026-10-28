"""Saving research declarations is local, bounded and returns a compact identity."""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING

import httpx
import pytest

from tradingdev.adapters.execution.process_runner import ProcessRunner
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.contracts.common import ErrorResponse
from tradingdev.app.contracts.research_evidence import ResearchEvidenceSaved
from tradingdev.app.data_service import DataService
from tradingdev.app.preflight_service import PreflightService
from tradingdev.app.research_evidence_service import ResearchEvidenceService
from tradingdev.domain.research_evidence import ResearchEvidenceDraft
from tradingdev.domain.strategies.loader import StrategyLoader

if TYPE_CHECKING:
    from pathlib import Path


def _draft() -> ResearchEvidenceDraft:
    return ResearchEvidenceDraft.model_validate(
        {
            "presentation": {
                "title": "Momentum study",
                "summary": "Inspect momentum entry rules.",
                "parameter_descriptions": {
                    "/lookback": {"label": "Lookback", "description": "History length"}
                },
            },
            "rules": [
                {
                    "rule_id": "entry",
                    "condition": "Price rises",
                    "action": "Buy",
                    "formula": "close[t] > close[t-1]",
                    "source_ids": ["momentum"],
                }
            ],
            "limits": [
                "Research hypothesis; external source contents are not checked."
            ],
            "sources": [
                {
                    "source_id": "momentum",
                    "title": "Host-reported source",
                    "url": "https://example.invalid/research",
                    "accessed_at": "2026-01-01T00:00:00Z",
                    "retrieval_kind": "host_web",
                    "support": "The host reports this page discusses momentum.",
                }
            ],
            "parameter_provenance": [
                {
                    "target_ids": ["base"],
                    "parameter_paths": ["/lookback"],
                    "provenance": "model_proposal",
                    "reason": "An exploratory model choice, not a source claim.",
                }
            ],
        }
    )


def test_save_returns_only_compact_identity_and_preserves_complete_draft(
    tmp_path: Path,
) -> None:
    service = ResearchEvidenceService(WorkspacePaths(tmp_path))
    draft = _draft()
    saved = service.save(draft)
    assert isinstance(saved, ResearchEvidenceSaved)
    assert set(saved.model_dump()) == {
        "success",
        "evidence_id",
        "content_hash",
        "source_count",
        "parameter_declaration_count",
    }
    assert saved.source_count == saved.parameter_declaration_count == 1
    assert len(saved.model_dump_json().encode()) < 300
    record = service.load(saved.evidence_id)
    assert record.draft == draft
    assert record.content_hash == saved.content_hash
    assert record.draft is not draft
    assert list(tmp_path.iterdir()) == [tmp_path / "research_evidence"]


def test_each_save_creates_new_identity_and_retains_earlier_records(
    tmp_path: Path,
) -> None:
    service = ResearchEvidenceService(WorkspacePaths(tmp_path))
    first, second = service.save(_draft()), service.save(_draft())
    assert isinstance(first, ResearchEvidenceSaved)
    assert isinstance(second, ResearchEvidenceSaved)
    assert first.evidence_id != second.evidence_id
    assert (
        service.load(first.evidence_id).draft == service.load(second.evidence_id).draft
    )
    assert len(list((tmp_path / "research_evidence").iterdir())) == 2


def test_source_declarations_are_saved_without_network_data_or_strategy_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Saving research must not fetch sources or execute research")

    for owner, name in (
        (socket, "create_connection"),
        (httpx.Client, "request"),
        (DataService, "load"),
        (DataService, "ensure"),
        (StrategyLoader, "create_from_config"),
        (StrategyLoader, "create_from_execution"),
        (PreflightService, "prepare"),
        (ProcessRunner, "spawn_module"),
        (ProcessRunner, "run_module"),
    ):
        monkeypatch.setattr(owner, name, unexpected)
    service = ResearchEvidenceService(WorkspacePaths(tmp_path))
    saved = service.save(_draft())
    assert isinstance(saved, ResearchEvidenceSaved)
    source = service.load(saved.evidence_id).draft.sources[0]
    assert source.url == "https://example.invalid/research"
    assert source.retrieval_kind == "host_web"
    assert "verified" not in source.model_dump()


def test_write_failure_returns_typed_error_without_claiming_saved_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ResearchEvidenceService(WorkspacePaths(tmp_path))

    def fail(record: object) -> None:
        raise OSError("storage is unavailable")

    monkeypatch.setattr(service.store, "save", fail)
    result = service.save(_draft())
    assert isinstance(result, ErrorResponse)
    assert result.code == "research_evidence_save_failed"
    assert result.error == "storage is unavailable"
    assert "evidence_id" not in result.model_dump()
    assert list(tmp_path.iterdir()) == []


def test_unsafe_invalid_model_copy_is_rejected_before_writing(tmp_path: Path) -> None:
    service = ResearchEvidenceService(WorkspacePaths(tmp_path))
    invalid = _draft().model_copy(update={"limits": ()})
    result = service.save(invalid)
    assert isinstance(result, ErrorResponse)
    assert result.code == "research_evidence_save_failed"
    assert list(tmp_path.iterdir()) == []


def test_internal_load_propagates_missing_or_corrupt_record(tmp_path: Path) -> None:
    service = ResearchEvidenceService(WorkspacePaths(tmp_path))
    with pytest.raises(FileNotFoundError):
        service.load("a" * 32)
    saved = service.save(_draft())
    assert isinstance(saved, ResearchEvidenceSaved)
    path = service.store.path(saved.evidence_id)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        service.load(saved.evidence_id)
