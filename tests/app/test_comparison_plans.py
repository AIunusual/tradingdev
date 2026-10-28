"""Whole-set approval binds all candidates; process launch is substituted here."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from tests.app.test_comparison_preflight import _comparison
from tests.mcp.execution_fixtures import RecordingRunner
from tests.preflight_fixtures import make_preflight_fixture
from tests.research_evidence_fixtures import save_research_evidence

from tradingdev.adapters.execution.preflight_worker import execute_preflight
from tradingdev.adapters.storage.comparisons import ComparisonStore
from tradingdev.app.contracts.plans import PlanRejected, PlanResponse, PlanStarted
from tradingdev.app.execution_plan_service import ExecutionPlanService
from tradingdev.app.preflight_service import PreflightResult, PreflightService
from tradingdev.app.prepared_execution import PreparedComparison
from tradingdev.domain.comparison import ComparisonManifest
from tradingdev.domain.preflight import ComparisonPreflightReceipt, PreflightRequest

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths


class InlineComparisonPreflight(PreflightService):
    """Keep real program checks but substitute the supervisor in state tests."""

    def __init__(self, workspace: WorkspacePaths, directory: Path) -> None:
        self.workspace = workspace
        self.directory = directory

    def prepare(self, request: PreflightRequest) -> PreflightResult:
        payload = execute_preflight(
            request, source_workspace=self.workspace, directory=self.directory
        )
        assert isinstance(payload.manifest, ComparisonManifest)
        assert isinstance(payload.receipt, ComparisonPreflightReceipt)
        return PreflightResult(
            PreparedComparison(
                payload.manifest,
                Path(payload.original_config_path),
                {
                    row.candidate_id: tuple(row.receipt.synthetic_fixture_rows)
                    for row in payload.receipt.candidates
                },
            ),
            payload.receipt,
        )


def context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ExecutionPlanService, RecordingRunner, PlanResponse]:
    workspace, single = make_preflight_fixture(tmp_path, monkeypatch)
    runner = RecordingRunner()
    service = ExecutionPlanService(
        workspace,
        process_runner=runner,
        preflight=InlineComparisonPreflight(workspace, tmp_path / "checks"),
    )
    request = _comparison(single)
    captured = service.preflight.prepare(request)
    evidence_id = save_research_evidence(service.evidence, captured.prepared.manifest)
    result = service.prepare(request, evidence_id)
    assert isinstance(result, PlanResponse), result
    assert not runner.calls
    assert service.jobs.get_job(result.plan_id) is None
    assert "2 組獨立回測" in result.confirmation_summary
    html = Path(result.html_path).read_text()
    assert "Threshold 0" in html and "Threshold 1" in html
    return service, runner, result


def test_parallel_approval_launches_one_comparison_with_every_reviewed_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, runner, result = context(tmp_path, monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(service.decide, result.plan_id, decision="approve")
            for _ in range(2)
        ]
        replies = [future.result(timeout=15) for future in futures]
    assert all(isinstance(reply, PlanStarted) for reply in replies)
    assert len(runner.calls) == 1
    assert runner.calls[0][0] == "tradingdev.mcp.workers.comparison"
    manifest = service.jobs.load_manifest(result.plan_id)
    assert isinstance(manifest, ComparisonManifest)
    assert manifest.manifest_hash == result.manifest_hash
    assert len(manifest.candidates) == 2
    ledger = ComparisonStore(service.workspace).load_ledger(
        result.plan_id, expected_hash=manifest.manifest_hash
    )
    assert all(row.status == "pending" for row in ledger.candidates)


@pytest.mark.parametrize("decision", ["decline", "revise"])
def test_whole_comparison_rejection_launches_no_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
) -> None:
    service, runner, result = context(tmp_path, monkeypatch)
    assert decision in {"decline", "revise"}
    reply = service.decide(
        result.plan_id, decision="decline" if decision == "decline" else "revise"
    )
    assert isinstance(reply, PlanRejected)
    assert not runner.calls
    assert service.jobs.get_job(result.plan_id) is None


def test_change_to_shared_source_after_review_blocks_all_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, runner, result = context(tmp_path, monkeypatch)
    plan, _ = service.plans.get(result.plan_id)
    assert isinstance(plan.manifest, ComparisonManifest)
    config = plan.manifest.common_config()
    Path(config["strategy"]["source_path"]).write_text("# changed after review\n")
    reply = service.decide(result.plan_id, decision="approve")
    assert isinstance(reply, PlanRejected)
    assert not runner.calls
    assert service.jobs.get_job(result.plan_id) is None
