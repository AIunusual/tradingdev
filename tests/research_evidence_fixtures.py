"""Synthetic saved explanations for exact manifest-bound plan tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.integration.execution_fixtures import confirmation_presentation
from tradingdev.app.contracts.research_evidence import ResearchEvidenceSaved
from tradingdev.domain.presentation.confirmation import ConfirmationPresentation
from tradingdev.domain.research_evidence import (
    ResearchEvidenceDraft,
    enumerate_parameter_bindings,
)

if TYPE_CHECKING:
    from tradingdev.app.research_evidence_service import ResearchEvidenceService
    from tradingdev.domain.execution_target import ExecutionTarget


def research_evidence_draft(
    target: ExecutionTarget, presentation: ConfirmationPresentation | None = None
) -> ResearchEvidenceDraft:
    """Cover synthetic manifest values explicitly without inventing web research."""
    targets = enumerate_parameter_bindings(target)
    if presentation is None:
        # Presentation text is fixture-authored; parameter paths come from the manifest.
        descriptions = confirmation_presentation({})
        descriptions["parameter_descriptions"] = {
            entry.parameter_path: {
                "label": entry.parameter_path,
                "description": "控制本次合成測試策略的設定。",
                "unit": "策略單位",
            }
            for binding in targets
            for entry in binding.parameters
        }
        presentation = ConfirmationPresentation.model_validate(descriptions)
    return ResearchEvidenceDraft.model_validate(
        {
            "presentation": presentation.model_dump(mode="json"),
            "rules": [
                {
                    "rule_id": "fixture_rule",
                    "condition": "策略條件成立時",
                    "action": "依策略提供的方向進場",
                    "formula": "signal = strategy_rule(parameters)",
                    "variables": [
                        {"symbol": "signal", "meaning": "策略輸出方向", "unit": "方向"}
                    ],
                }
            ],
            "limits": ["測試用模型提案；未宣稱有外部來源或績效證據。"],
            "parameter_provenance": [
                {
                    "target_ids": [binding.target_id],
                    "parameter_paths": [
                        entry.parameter_path for entry in binding.parameters
                    ],
                    "provenance": "model_proposal",
                    "reason": "本測試明確固定的策略參數。",
                }
                for binding in targets
                if binding.parameters
            ],
        }
    )


def save_research_evidence(
    service: ResearchEvidenceService,
    target: ExecutionTarget,
    presentation: ConfirmationPresentation | None = None,
) -> str:
    saved = service.save(research_evidence_draft(target, presentation))
    assert isinstance(saved, ResearchEvidenceSaved), saved
    return saved.evidence_id
