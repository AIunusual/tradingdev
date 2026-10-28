"""Approval identities require the full saved research and simulator policy."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from tests.domain.test_confirmation_presentation import _manifest, _receipt
from tests.research_evidence_fixtures import research_evidence_draft

from tradingdev.domain.approved_research import ApprovedResearch
from tradingdev.domain.execution_plan import ExecutionPlan
from tradingdev.domain.presentation.confirmation import build_confirmation_document
from tradingdev.domain.presentation.research_confirmation import (
    enrich_research_confirmation,
)
from tradingdev.domain.research_evidence import ResearchEvidenceRecord


def _payload() -> dict[str, Any]:
    manifest = _manifest()
    receipt = _receipt(manifest)
    record = ResearchEvidenceRecord.create(research_evidence_draft(manifest))
    research = ApprovedResearch.create(record, manifest)
    document = enrich_research_confirmation(
        build_confirmation_document(
            manifest, record.draft.presentation, receipt, plan_id="a" * 32
        ),
        research.research,
    )
    payload = {
        "schema_version": 3,
        "plan_id": "a" * 32,
        "manifest": manifest.model_dump(mode="json"),
        "original_config_path": "/test/config.yaml",
        "preflight": receipt.model_dump(mode="json"),
        "research_evidence": research.model_dump(mode="json"),
        "document": document.model_dump(mode="json"),
        "created_at": "2026-10-28T01:00:00Z",
        "expires_at": "2026-10-28T02:00:00Z",
    }
    return payload


def test_complete_plan_round_trips_with_approved_research_in_hash() -> None:
    payload = _payload()
    plan = ExecutionPlan.model_validate(
        {**payload, "plan_hash": ExecutionPlan.digest(payload)}
    )
    assert ExecutionPlan.model_validate_json(plan.model_dump_json()) == plan
    plan.verify()
    original_hash = plan.plan_hash
    payload["research_evidence"]["research"]["record"]["evidence_id"] = "b" * 32
    assert ExecutionPlan.digest(payload) != original_hash
    with pytest.raises(ValidationError):
        ExecutionPlan.model_validate(
            {**payload, "plan_hash": ExecutionPlan.digest(payload)}
        )


@pytest.mark.parametrize("version", [2, True, "3", 3.0, 4])
def test_old_or_noninteger_plan_versions_cannot_gain_new_approval(version: Any) -> None:
    payload = _payload()
    payload["schema_version"] = version
    with pytest.raises(ValidationError):
        ExecutionPlan.model_validate(
            {**payload, "plan_hash": ExecutionPlan.digest(payload)}
        )


def test_rehashed_plan_without_research_snapshot_is_rejected() -> None:
    payload = _payload()
    payload.pop("research_evidence")
    with pytest.raises(ValidationError, match="research_evidence"):
        ExecutionPlan.model_validate(
            {**payload, "plan_hash": ExecutionPlan.digest(payload)}
        )


def test_valid_research_for_a_different_target_cannot_be_spliced_into_plan() -> None:
    payload = _payload()
    other = _manifest(parameters={"period": 99})
    record = ResearchEvidenceRecord.create(research_evidence_draft(other))
    payload["research_evidence"] = ApprovedResearch.create(record, other).model_dump(
        mode="json"
    )
    with pytest.raises(ValidationError, match="another execution"):
        ExecutionPlan.model_validate(
            {**payload, "plan_hash": ExecutionPlan.digest(payload)}
        )
