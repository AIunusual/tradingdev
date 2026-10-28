"""Require actual saved explanations before attributing a prepared plan to them."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import TypeAdapter, ValidationError

from tradingdev.domain.research_evidence import EvidenceId, ResearchEvidenceDraft

if TYPE_CHECKING:
    from tests.e2e.llm_client import ToolCall


def assert_saved_research(
    calls: list[ToolCall], prepare_index: int
) -> ResearchEvidenceDraft:
    """Inputs are normalized copies; original model calls remain diagnostic evidence."""
    prepared = calls[prepare_index]
    try:
        evidence_id = TypeAdapter(EvidenceId).validate_python(
            prepared.arguments.get("evidence_id")
        )
    except ValidationError as exc:
        raise AssertionError("Preparation needs a valid evidence_id") from exc
    saves = [
        call
        for call in calls[:prepare_index]
        if call.name == "save_research_evidence"
        and isinstance(call.result, dict)
        and call.result.get("success") is True
        and call.result.get("evidence_id") == evidence_id
    ]
    assert saves, "Missing successful save_research_evidence before preparation"
    try:
        draft = ResearchEvidenceDraft.model_validate(saves[-1].arguments.get("draft"))
    except ValidationError as exc:
        raise AssertionError("Saved research draft is invalid") from exc
    # These scenarios supply no URLs and the live host harness disables web tools.
    assert not draft.sources, "No-web fixtures cannot claim retrieved references"
    return draft
