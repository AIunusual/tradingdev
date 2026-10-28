"""Show the saved client evidence without turning it into execution authority."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tradingdev.domain.presentation.confirmation import (
    ConfirmationDocument,
    ConfirmationField,
    ConfirmationSection,
)

if TYPE_CHECKING:
    from tradingdev.domain.research_evidence import BoundResearchEvidence

_PROVENANCE_LABELS = {
    "user": "使用者提供",
    "project_default": "專案預設",
    "reference": "參考來源",
    "model_proposal": "模型提案（未經外部來源查證）",
}


def enrich_research_confirmation(
    document: ConfirmationDocument, evidence: BoundResearchEvidence
) -> ConfirmationDocument:
    """Append complete saved formulas, parameter origins and reference disclosures.

    Values and formulas remain inert text; all escaping belongs to the renderer.
    This projection never fetches a reference or resolves current execution policy.
    """
    evidence.verify()
    if document.identity.manifest_hash != evidence.manifest_hash:
        raise ValueError("Research confirmation belongs to another execution")
    draft = evidence.record.draft
    origins = {parameter.provenance for parameter in evidence.parameters}
    strategy_fields = [
        ConfirmationField(
            label="參數依據",
            value_text="、".join(
                label
                for origin, label in _PROVENANCE_LABELS.items()
                if origin in origins
            )
            or "沒有策略建構參數",
            description="分類依保存的客戶端聲明；不代表後端已核實來源或適用性。",
        )
    ]
    for index, rule in enumerate(draft.rules, 1):
        prefix = f"研究規則 {index}"
        strategy_fields.extend(
            (
                _field(prefix + "／條件", rule.condition),
                _field(prefix + "／動作", rule.action),
                _field(prefix + "／公式或精確條件", rule.formula),
            )
        )
        strategy_fields.extend(
            _field(
                prefix + f"／變數 {variable.symbol}",
                variable.meaning,
                f"單位：{variable.unit}",
            )
            for variable in rule.variables
        )
        strategy_fields.append(
            _field(
                prefix + "／公式來源", "、".join(rule.source_ids) or "未引用外部來源"
            )
        )
    parameter_fields = [
        _field(
            f"{parameter.target_id}／"
            + draft.presentation.parameter_descriptions[parameter.parameter_path].label
            + "／參數來源",
            _PROVENANCE_LABELS[parameter.provenance],
            "；".join(
                [
                    f"有效值：{_json(parameter.value)}",
                    f"選擇理由：{parameter.reason}",
                    "來源值聲明："
                    + (
                        "、".join(
                            f"{reference.source_id}／{reference.claim_id}"
                            for reference in parameter.source_claims
                        )
                        or "無"
                    ),
                ]
            ),
        )
        for parameter in evidence.parameters
    ]
    source_fields: list[ConfirmationField] = []
    for source in draft.sources:
        prefix = f"參考來源 {source.source_id}"
        source_fields.extend(
            (
                _field(prefix + "／標題", source.title),
                _field(prefix + "／網址", source.url),
                _field(prefix + "／存取時間", source.accessed_at.isoformat()),
                _field(
                    prefix + "／取得方式",
                    "客戶端回報網路查閱"
                    if source.retrieval_kind == "host_web"
                    else "使用者提供來源",
                ),
                _field(prefix + "／支持內容", source.support),
            )
        )
        source_fields.extend(
            _field(
                prefix + f"／參數聲明 {claim.claim_id}",
                _json(claim.value),
                claim.parameter,
            )
            for claim in source.claims
        )
        if not source.claims:
            source_fields.append(
                _field(
                    prefix + "／參數聲明", "未提供數值建議；只能作為規則或公式參考。"
                )
            )
    additions = {
        "strategy": strategy_fields,
        "parameters": parameter_fields,
        "limitations": source_fields,
    }
    sections: list[ConfirmationSection] = []
    for section in document.sections:
        paragraphs = section.paragraphs
        if section.id == "strategy":
            paragraphs += (
                "研究條件、動作、公式與變數解釋由客戶端提供；後端不執行公式，"
                "也不證明策略程式或 ML 模型符合這些敘述。",
            )
        elif section.id == "limitations":
            paragraphs += (
                *draft.limits,
                "來源記錄是客戶端回報，後端不證明已瀏覽網址或來源內容正確。"
                "參數值比對只核對聲明與有效值一致，來源語意與適用性仍須由客戶端核對。",
            )
            if not draft.sources:
                paragraphs += ("本次研究未提供外部參考來源。",)
        sections.append(
            section.model_copy(
                update={
                    "fields": section.fields + tuple(additions.get(section.id, ())),
                    "paragraphs": paragraphs,
                }
            )
        )
    identity = document.identity.model_copy(
        update={
            "details": document.identity.details
            + (
                ConfirmationField(
                    label="研究證據識別碼", value_text=evidence.record.evidence_id
                ),
                ConfirmationField(
                    label="研究證據內容摘要", value_text=evidence.record.content_hash
                ),
                ConfirmationField(
                    label="研究證據綁定摘要", value_text=evidence.content_hash
                ),
            )
        }
    )
    return ConfirmationDocument.model_validate(
        document.model_copy(
            update={"sections": tuple(sections), "identity": identity}
        ).model_dump(mode="python")
    )


def _field(label: str, value: str, description: str = "") -> ConfirmationField:
    return ConfirmationField(
        label=label,
        value_text=value,
        description=description,
        explanation_source="model",
    )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
