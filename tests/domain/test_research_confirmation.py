"""Research explanations preserve complete inert text in saved confirmation output."""

from __future__ import annotations

from html import unescape

import pytest
from tests.domain.test_comparison_confirmation import comparison_fixture
from tests.domain.test_confirmation_presentation import _manifest, _receipt
from tests.domain.test_research_evidence import _reference_draft
from tests.research_evidence_fixtures import research_evidence_draft

from tradingdev.adapters.presentation.confirmation import (
    render_confirmation_html,
    render_confirmation_summary,
    render_confirmation_text,
)
from tradingdev.domain.presentation.comparison_confirmation import (
    build_comparison_confirmation,
)
from tradingdev.domain.presentation.confirmation import build_confirmation_document
from tradingdev.domain.presentation.research_confirmation import (
    enrich_research_confirmation,
)
from tradingdev.domain.research_evidence import (
    ResearchEvidenceDraft,
    ResearchEvidenceRecord,
    bind_research_evidence,
)


def test_complete_research_is_visible_without_claiming_external_verification() -> None:
    manifest = _manifest()
    draft = _reference_draft()
    bound = bind_research_evidence(ResearchEvidenceRecord.create(draft), manifest)
    original = build_confirmation_document(
        manifest, draft.presentation, _receipt(manifest)
    )
    document = enrich_research_confirmation(original, bound)
    html = unescape(render_confirmation_html(document))
    text = render_confirmation_text(document)
    for rendered in (html, text):
        for expected in (
            draft.rules[0].condition,
            draft.rules[0].action,
            draft.rules[0].formula,
            draft.rules[0].variables[0].meaning,
            draft.rules[0].variables[0].unit,
            draft.sources[0].url,
            draft.sources[0].title,
            draft.sources[0].support,
            draft.sources[0].accessed_at.isoformat(),
            draft.parameter_provenance[0].reason,
            "manual／period",
            "客戶端回報網路查閱",
            "後端不證明已瀏覽網址",
            "來源語意與適用性仍須由客戶端核對",
            "有效值：20",
        ):
            assert expected in rendered
    assert document.identity.manifest_hash == original.identity.manifest_hash
    assert bound.record.evidence_id in html
    assert original != document
    assert all(
        "公式或精確條件" not in field.label
        for section in original.sections
        for field in section.fields
    )


def test_formula_only_sources_remain_distinct_from_model_parameter_proposals() -> None:
    manifest = _manifest()
    payload = _reference_draft().model_dump(mode="json")
    payload["sources"][0]["claims"] = []
    payload["sources"][0]["retrieval_kind"] = "user_supplied"
    payload["parameter_provenance"][0].update(
        provenance="model_proposal", source_claims=[]
    )
    draft = ResearchEvidenceDraft.model_validate(payload)
    bound = bind_research_evidence(ResearchEvidenceRecord.create(draft), manifest)
    document = enrich_research_confirmation(
        build_confirmation_document(manifest, draft.presentation, _receipt(manifest)),
        bound,
    )
    text = render_confirmation_text(document)
    assert "模型提案" in text
    assert "參數依據：模型提案（未經外部來源查證）" in render_confirmation_summary(
        document, mode="signal"
    )
    assert "使用者提供來源" in text
    assert "未提供數值建議；只能作為規則或公式參考" in text
    assert "來源值聲明：無" in text


def test_formula_source_and_claim_markup_are_escaped_as_inert_text() -> None:
    manifest = _manifest()
    payload = _reference_draft().model_dump(mode="json")
    attack = '</dd><script>alert("source")</script><img src=x>'
    payload["rules"][0].update(condition=attack, action=attack, formula=attack)
    payload["rules"][0]["variables"][0].update(
        symbol=attack, meaning=attack, unit=attack
    )
    payload["sources"][0].update(title=attack, support=attack)
    payload["sources"][0]["claims"][0]["parameter"] = attack
    payload["parameter_provenance"][0]["reason"] = attack
    payload["limits"] = [attack]
    draft = ResearchEvidenceDraft.model_validate(payload)
    bound = bind_research_evidence(ResearchEvidenceRecord.create(draft), manifest)
    document = enrich_research_confirmation(
        build_confirmation_document(manifest, draft.presentation, _receipt(manifest)),
        bound,
    )
    html = render_confirmation_html(document)
    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html
    assert attack in render_confirmation_text(document)


def test_every_comparison_candidate_has_its_own_parameter_origin() -> None:
    manifest, receipt = comparison_fixture()
    draft = research_evidence_draft(manifest)
    bound = bind_research_evidence(ResearchEvidenceRecord.create(draft), manifest)
    document = enrich_research_confirmation(
        build_comparison_confirmation(manifest, draft.presentation, receipt), bound
    )
    text = render_confirmation_text(document)
    for index, value in enumerate((10, 20, 30), 1):
        assert f"candidate_{index}／/period／參數來源" in text
        assert f"有效值：{value}" in text
    assert "本次研究未提供外部參考來源" in text


def test_enrichment_rejects_another_manifest() -> None:
    manifest = _manifest()
    draft = _reference_draft()
    bound = bind_research_evidence(ResearchEvidenceRecord.create(draft), manifest)
    other = _manifest(parameters={"period": 99})
    with pytest.raises(ValueError, match="another execution"):
        enrich_research_confirmation(
            build_confirmation_document(other, draft.presentation, _receipt(other)),
            bound,
        )
