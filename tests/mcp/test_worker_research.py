"""Queued jobs recheck disclosure policy before real data or strategy execution."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

import pytest
from tests.app.test_comparison_lifecycle import _comparison
from tests.app.test_submitted_research import approve
from tests.domain.test_confirmation_presentation import _manifest

from tradingdev.adapters.execution.process_runner import WorkerHandle
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.submitted_research import SubmittedResearchStore
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.job_store import JobStore
from tradingdev.mcp.workers import backtest, comparison, optimization

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.domain.execution_target import ExecutionTarget


@pytest.mark.parametrize("kind", ["backtest", "optimization", "comparison"])
@pytest.mark.parametrize("change", ["missing", "policy", "tamper", "none", "valid"])
def test_worker_checks_research_before_data_or_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: Literal["backtest", "optimization", "comparison"],
    change: str,
) -> None:
    reached_execution = []
    target: ExecutionTarget
    if kind == "comparison":
        fixture = _comparison(tmp_path)
        jobs, target = fixture.jobs, fixture.bundle
        job_id = "comparison"
        monkeypatch.setattr(
            comparison, "ComparisonService", lambda **_kwargs: fixture.service
        )

        def stop_comparison(*_args: object) -> None:
            reached_execution.append("data")
            raise RuntimeError("reached authorized execution")

        monkeypatch.setattr(fixture.backtest, "load_execution_dataset", stop_comparison)
    else:
        target = _manifest(kind)
        jobs = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
        job_id = "job"
        jobs.create_job(job_id=job_id, manifest=target)

        def stop_execution(*_args: object, **_kwargs: object) -> None:
            reached_execution.append("strategy")
            raise RuntimeError("reached authorized execution")

        monkeypatch.setattr(
            BacktestService,
            "run_manifest" if kind == "backtest" else "prepare_strategy",
            stop_execution,
        )
    monkeypatch.setenv("TRADINGDEV_WORKSPACE", str(jobs.workspace.root))
    monkeypatch.setenv(
        "TRADINGDEV_WORKER_IDENTITY",
        json.dumps(WorkerHandle(4321, 100.0, "a" * 32).job_fields()),
    )
    if change != "none":
        evidence = approve(target)
        jobs.update_job(job_id, research_evidence_hash=evidence.content_hash)
        snapshots = SubmittedResearchStore(jobs.workspace)
        if change != "missing":
            path = snapshots.save(job_id, evidence, target)
            if change == "tamper":
                path.write_text(path.read_text() + " ")
        if change == "policy":
            monkeypatch.setattr(
                "tradingdev.domain.approved_research.resolve_execution_assumptions",
                lambda manifest: evidence.assumptions_for(
                    manifest.manifest_hash
                ).model_copy(update={"policy_version": "changed"}),
            )
    if kind == "comparison":
        comparison._run_comparison(job_id)
    elif kind == "optimization":
        optimization._run_optimization(job_id)
    else:
        backtest._run_backtest(job_id)
    assert bool(reached_execution) == (change in {"none", "valid"})
    job = jobs.get_job(job_id)
    assert job is not None and job["status"] == "failed"
    assert not job["data_downloaded"]
    if kind == "comparison":
        assert [row.status for row in fixture.ledger().candidates] == ["not_run"] * 3
