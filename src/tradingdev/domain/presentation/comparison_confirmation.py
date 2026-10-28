"""One review document covering every member of a fixed comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.domain.presentation.confirmation import (
    ConfirmationDocument,
    ConfirmationField,
    ConfirmationIdentity,
    ConfirmationPresentation,
    ConfirmationPresentationError,
    ConfirmationSection,
    build_confirmation_document,
    parameter_paths,
)

if TYPE_CHECKING:
    from tradingdev.domain.comparison import ComparisonManifest
    from tradingdev.domain.preflight import ComparisonPreflightReceipt


def build_comparison_confirmation(
    manifest: ComparisonManifest,
    presentation: ConfirmationPresentation,
    preflight: ComparisonPreflightReceipt,
    *,
    plan_id: str | None = None,
) -> ConfirmationDocument:
    """Reuse shared settings only after verifying all candidate bindings."""
    manifest.verify()
    preflight.verify_manifest(manifest)
    paths = sorted(
        {
            path
            for candidate in manifest.candidates
            for path in parameter_paths(candidate.manifest)
        }
    )
    descriptions = presentation.parameter_descriptions
    if set(descriptions) != set(paths) or len(
        {description.label for description in descriptions.values()}
    ) != len(descriptions):
        raise ConfirmationPresentationError(
            "Parameter descriptions must exactly and unambiguously cover every "
            "candidate constructor",
            required_parameter_paths=paths,
        )
    documents = [
        build_confirmation_document(
            candidate.manifest,
            presentation.model_copy(
                update={
                    "parameter_descriptions": {
                        path: descriptions[path]
                        for path in parameter_paths(candidate.manifest)
                    }
                }
            ),
            checked.receipt,
            plan_id=plan_id,
            comparison_candidate=True,
        )
        for candidate, checked in zip(
            manifest.candidates, preflight.candidates, strict=True
        )
    ]
    sections: list[ConfirmationSection] = []
    for section in documents[0].sections:
        if section.id in {"parameters", "preflight"}:
            fields: list[ConfirmationField] = []
            for candidate, document in zip(manifest.candidates, documents, strict=True):
                candidate_section = next(
                    item for item in document.sections if item.id == section.id
                )
                if section.id == "parameters":
                    fields.append(
                        ConfirmationField(
                            label=candidate.label,
                            value_text=candidate.rationale,
                            description="這組參數的選擇理由，由提案者提供。",
                            explanation_source="model",
                        )
                    )
                fields.extend(
                    field.model_copy(
                        update={"label": f"{candidate.label}／{field.label}"}
                    )
                    for field in candidate_section.fields
                )
            sections.append(section.model_copy(update={"fields": tuple(fields)}))
        elif section.id == "strategy":
            sections.append(
                section.model_copy(
                    update={
                        "fields": tuple(
                            field.model_copy(
                                update={
                                    "value_text": (
                                        f"固定參數比較（{len(manifest.candidates)}"
                                        " 組獨立回測）"
                                    )
                                }
                            )
                            if field.label == "執行方式"
                            else field
                            for field in section.fields
                        )
                    }
                )
            )
        elif section.id == "optimization":
            sections.append(
                section.model_copy(
                    update={
                        "fields": (),
                        "paragraphs": (
                            "只執行下列已固定的參數組合，不進行自動搜尋或新增候選。",
                            "每組使用同一份固定行情，獨立計算資金與交易；不合併為一條投資組合績效。",
                            "報告保留每組結果；失敗、取消或未執行時列出原因，不當作零報酬。",
                        ),
                    }
                )
            )
        else:
            sections.append(section)
    return ConfirmationDocument(
        title=presentation.title,
        summary=presentation.summary,
        sections=tuple(sections),
        identity=ConfirmationIdentity(
            plan_id=plan_id,
            manifest_hash=manifest.manifest_hash,
            details=documents[0].identity.details
            + tuple(
                ConfirmationField(
                    label=f"{candidate.label}／執行內容摘要",
                    value_text=candidate.manifest.manifest_hash,
                )
                for candidate in manifest.candidates
            ),
        ),
    )
