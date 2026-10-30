"""Reports carry fixed research evidence through complete and failed comparisons."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from tests.app.test_comparison_report import ComparisonFixture, _fixture, _payload
from tests.app.test_submitted_research import approve

from tradingdev.adapters.storage.filesystem import sha256_file
from tradingdev.adapters.storage.submitted_research import SubmittedResearchStore

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.domain.approved_research import ApprovedResearch


def _record(fixture: ComparisonFixture) -> ApprovedResearch:
    evidence = approve(fixture.manifest)
    job = fixture.store.get_job("comparison")
    assert job is not None
    job["research_evidence_hash"] = evidence.content_hash
    fixture.store.upsert_job(job)
    snapshots = SubmittedResearchStore(fixture.workspace)
    snapshots.save("comparison", evidence, fixture.manifest)
    for run in fixture.store.list_runs():
        path = snapshots.publish_run(
            "comparison", run["run_id"], evidence, fixture.manifest
        )
        fixture.store.create_artifact(
            artifact_id=f"{run['run_id']}:research_evidence",
            run_id=run["run_id"],
            artifact_type="research_evidence",
            path=path,
            sha256=sha256_file(path),
            metadata={
                "job_id": "comparison",
                "research_evidence_hash": evidence.content_hash,
            },
        )
    return evidence


def test_comparison_deduplicates_research_and_keeps_exact_scope_sources(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, ("succeeded", "failed", "succeeded"), status="done")
    evidence = _record(fixture)
    response = fixture.service.generate_report(
        comparison_job_id="comparison", sections=["research", "settings"]
    )
    html, payload = _payload(response)
    assert list(payload["research_documents"]) == [evidence.content_hash]
    document = payload["research_documents"][evidence.content_hash]
    assert document["draft"] == evidence.research.record.draft.model_dump(mode="json")
    assert "target" not in document
    assert len(document["parameters"]) == 3
    assert "close &gt; SMA(close, period)" in html
    assert "模型提案（未經來源查證）" in html
    assert "Window 2" in html
    for index, run in zip((1, 3), payload["runs"], strict=True):
        assert run["research_evidence_hash"] == evidence.content_hash
        scope = run["scopes"][0]
        assert scope["parameter_sources"][0]["target_id"] == f"candidate_{index}"
        assert scope["parameter_sources"][0]["value"] == index
        assert scope["execution_assumptions"]["policy_version"] == "execution-v2"
        assert any(
            source["artifact_id"].endswith(":research_evidence")
            for source in run["provenance"]["artifacts"]
        )
    manifest = json.loads(
        (
            fixture.workspace.root / "reports" / response["report_id"] / "manifest.json"
        ).read_text()
    )
    assert manifest["research_documents"][evidence.content_hash]["record_hash"] == (
        evidence.research.record.content_hash
    )


@pytest.mark.parametrize("omit", [False, True])
def test_all_failed_comparison_preserves_research_even_when_section_is_omitted(
    tmp_path: Path, omit: bool
) -> None:
    fixture = _fixture(tmp_path, ("not_run", "not_run"), snapshot=False)
    evidence = _record(fixture)
    result = fixture.service.generate_report(
        comparison_job_id="comparison", sections=[] if omit else ["research"]
    )
    html, payload = _payload(result)
    assert "沒有單組交易明細入口" in result["navigation_instructions"]
    assert result["navigation_instructions"] in html
    assert payload["navigation_instructions"] == result["navigation_instructions"]
    assert payload["runs"] == []
    assert evidence.content_hash in payload["research_documents"]
    assert payload["fixed_comparison"]["research_evidence_status"] == "recorded"
    assert ("data-section='research'" in html) is not omit
    if not omit:
        assert "close &gt; SMA(close, period)" in html
        assert "Window 1" in html and "Window 2" in html


def test_detached_run_report_uses_registered_copy_and_rejects_lost_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, ("succeeded",))
    evidence = _record(fixture)
    snapshots = SubmittedResearchStore(fixture.workspace)
    snapshots.path("comparison").unlink()
    html, payload = _payload(
        fixture.service.generate_report(
            run_ids=["comparison_candidate_1"], sections=["research"]
        )
    )
    assert evidence.content_hash in payload["research_documents"]
    assert "close &gt; SMA(close, period)" in html
    snapshots.path("comparison_candidate_1", run=True).unlink()
    result = fixture.service.generate_report(
        run_ids=["comparison_candidate_1"], sections=[]
    )
    assert result["success"] is False
    assert result["code"] == "research_evidence_invalid"


def test_legacy_report_does_not_invent_research(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, ("succeeded",))
    html, payload = _payload(
        fixture.service.generate_report(
            run_ids=["comparison_candidate_1"], sections=["research"]
        )
    )
    assert payload["research_documents"] == {}
    assert "未保存交易規則" in html


@pytest.mark.parametrize("comparison", [False, True])
def test_research_without_overview_keeps_visible_links_to_each_success(
    tmp_path: Path, comparison: bool
) -> None:
    fixture = _fixture(tmp_path, ("succeeded", "failed", "succeeded"))
    _record(fixture)
    selection = (
        {"comparison_job_id": "comparison"}
        if comparison
        else {"run_ids": ["comparison_candidate_1", "comparison_candidate_3"]}
    )
    html, _payload_data = _payload(
        fixture.service.generate_report(**selection, sections=["research", "settings"])
    )
    main = html.split("<div id='overview-view'>", 1)[1].split(
        "<div id='result-view' hidden>", 1
    )[0]
    assert "data-section='research'" in main
    for index in (0, 1):
        assert f"href='#result-{index}-0'" in main
        assert f"id='result-{index}-0'" in html
    assert "href='#result-2-0'" not in main
