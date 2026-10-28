"""Accepted research survives submission, publication failures and detached reads."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from tests.app.test_comparison_report import _fixture
from tests.app.test_execution_submission import _Runner
from tests.app.test_run_publisher import _manifest, _pipeline
from tests.domain.test_comparison_confirmation import comparison_fixture
from tests.domain.test_confirmation_presentation import _manifest as simple_manifest
from tests.domain.test_research_evidence import evidence_draft

from tradingdev.adapters.storage.comparisons import ComparisonStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.adapters.storage.research_evidence import ResearchEvidenceStore
from tradingdev.adapters.storage.submitted_research import SubmittedResearchStore
from tradingdev.app.execution_submission import ExecutionSubmissionService
from tradingdev.app.job_store import JobStore
from tradingdev.app.prepared_execution import PreparedComparison, PreparedExecution
from tradingdev.app.report_comparison import load_report_comparison
from tradingdev.app.run_publisher import RunPublicationBinding, RunPublisher
from tradingdev.app.trade_history_service import HistoryReadError, TradeHistoryService
from tradingdev.domain.approved_research import ApprovedResearch
from tradingdev.domain.comparison import ComparisonManifest
from tradingdev.domain.execution import ManifestError
from tradingdev.domain.research_evidence import ResearchEvidenceRecord

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.domain.execution_target import ExecutionTarget


def approve(target: ExecutionTarget) -> ApprovedResearch:
    return ApprovedResearch.create(
        ResearchEvidenceRecord.create(evidence_draft(target)), target
    )


def _single(
    tmp_path: Path,
) -> tuple[JobStore, RunPublisher, RunPublicationBinding, ApprovedResearch]:
    source = tmp_path / "strategy.py"
    source.write_text("# immutable source\n")
    manifest = _manifest(source, 1)
    evidence = approve(manifest)
    jobs = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    ExecutionSubmissionService(jobs, _Runner()).submit(
        PreparedExecution(manifest, source, research_evidence=evidence), job_id="job"
    )
    jobs.update_job("job", dataset_id="test-dataset")
    return (
        jobs,
        RunPublisher(jobs.workspace, jobs.store),
        RunPublicationBinding("job", "job", manifest),
        evidence,
    )


def test_saved_snapshot_and_run_need_no_original_source_or_confirmation(
    tmp_path: Path,
) -> None:
    jobs, publisher, binding, evidence = _single(tmp_path)
    assert binding.expected_manifest is not None
    records = ResearchEvidenceStore(jobs.workspace)
    records.save(evidence.research.record)
    confirmation = jobs.workspace.root / "confirmation.html"
    confirmation.write_text("reviewed")
    publisher.publish(
        binding, {"total_return": 0.1}, pipeline=_pipeline(binding.expected_manifest)
    )
    files = {
        path: path.read_bytes()
        for path in (jobs.workspace.runs / "job").iterdir()
        if path.is_file()
    }
    publisher.publish(
        binding, {"total_return": 0.1}, pipeline=_pipeline(binding.expected_manifest)
    )
    assert files == {
        path: path.read_bytes()
        for path in (jobs.workspace.runs / "job").iterdir()
        if path.is_file()
    }
    records.path(evidence.research.record.evidence_id).unlink()
    confirmation.unlink()
    (tmp_path / "strategy.py").unlink()
    SubmittedResearchStore(jobs.workspace).path("job").unlink()
    loaded = TradeHistoryService(workspace=jobs.workspace, store=jobs.store).load_scope(
        "job"
    )
    assert loaded.research_evidence == evidence
    assert loaded.execution_assumptions == evidence.execution_assumptions[0]
    assert loaded.parameter_sources == evidence.research.parameters


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "tamper",
        "artifact_missing",
        "declared_missing",
        "declared_hash",
        "parent_hash",
        "artifact_identity",
    ],
)
def test_declared_run_evidence_cannot_downgrade_to_legacy(
    tmp_path: Path, change: str
) -> None:
    jobs, publisher, binding, evidence = _single(tmp_path)
    assert binding.expected_manifest is not None
    publisher.publish(
        binding, {"total_return": 0.1}, pipeline=_pipeline(binding.expected_manifest)
    )
    path = SubmittedResearchStore(jobs.workspace).path("job", run=True)
    if change == "missing":
        path.unlink()
    elif change == "tamper":
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "artifact_missing":
        with jobs.store.connect() as connection:
            connection.execute(
                "DELETE FROM artifacts WHERE artifact_id = ?",
                ("job:research_evidence",),
            )
    elif change in {"declared_missing", "declared_hash", "parent_hash"}:
        job = jobs.get_job("job")
        assert job is not None
        if change == "declared_missing":
            job.pop("research_evidence_hash")
        else:
            job[
                "research_evidence_hash"
                if change == "declared_hash"
                else "manifest_hash"
            ] = "0" * 64
        jobs.store.upsert_job(job)
    else:
        artifact = jobs.get_artifact("job:research_evidence")
        assert artifact is not None
        jobs.store.create_artifact(
            artifact_id=artifact["artifact_id"],
            run_id="job",
            artifact_type="research_evidence",
            path=path,
            sha256=sha256_file(path),
            metadata={
                "job_id": "other",
                "research_evidence_hash": evidence.content_hash,
            },
        )
    with pytest.raises(HistoryReadError, match="research|Research") as error:
        TradeHistoryService(workspace=jobs.workspace, store=jobs.store).load_run("job")
    assert error.value.code == "research_evidence_invalid"


def test_submission_rejects_changed_policy_before_creating_a_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = simple_manifest()
    evidence = approve(manifest)
    original = evidence.execution_assumptions[0]
    monkeypatch.setattr(
        "tradingdev.domain.approved_research.resolve_execution_assumptions",
        lambda _manifest: original.model_copy(update={"policy_version": "next"}),
    )
    evidence.verify(manifest)  # Historical meaning remains readable.
    jobs = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    runner = _Runner()
    with pytest.raises(ValueError, match="policy changed"):
        ExecutionSubmissionService(jobs, runner).submit(
            PreparedExecution(manifest, tmp_path / "config", research_evidence=evidence)
        )
    assert not jobs.list_all_jobs() and not runner.calls


@pytest.mark.parametrize("failure", ["save", "spawn"])
@pytest.mark.parametrize("comparison", [False, True])
def test_submission_failure_is_terminal_and_retains_complete_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, comparison: bool
) -> None:
    target = comparison_fixture()[0] if comparison else simple_manifest()
    evidence = approve(target)
    jobs = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    runner = _Runner()

    def fail(*_args: object, **_kwargs: object) -> Any:
        raise OSError("cannot publish or launch")

    monkeypatch.setattr(
        SubmittedResearchStore if failure == "save" else runner,
        "save" if failure == "save" else "spawn_module",
        fail,
    )
    prepared = (
        PreparedComparison(target, tmp_path / "config", {}, evidence)
        if isinstance(target, ComparisonManifest)
        else PreparedExecution(target, tmp_path / "config", research_evidence=evidence)
    )
    with pytest.raises(OSError):
        ExecutionSubmissionService(jobs, runner).submit(prepared, job_id="failed")
    job = jobs.get_job("failed")
    assert job is not None and job["status"] == "failed"
    assert job["research_evidence_hash"] == evidence.content_hash
    if failure == "spawn":
        assert (
            SubmittedResearchStore(jobs.workspace).load_for_job(job, target) == evidence
        )
    if comparison:
        ledger = ComparisonStore(jobs.workspace).load_ledger(
            "failed", expected_hash=target.manifest_hash
        )
        assert [row.status for row in ledger.candidates] == ["not_run"] * 3


def test_publication_failure_rolls_back_run_copy_and_retry_restores_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs, publisher, binding, evidence = _single(tmp_path)
    assert binding.expected_manifest is not None
    original = PerformanceStore.publish

    def fail(self: PerformanceStore, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        raise OSError("after performance")

    monkeypatch.setattr(PerformanceStore, "publish", fail)
    with pytest.raises(OSError, match="after performance"):
        publisher.publish(
            binding,
            {"total_return": 0.1},
            pipeline=_pipeline(binding.expected_manifest),
        )
    snapshots = SubmittedResearchStore(jobs.workspace)
    assert not snapshots.path("job", run=True).exists()
    assert snapshots.path("job").exists()
    assert jobs.get_run("job") is None and not jobs.list_artifacts()
    monkeypatch.setattr(PerformanceStore, "publish", original)
    publisher.publish(
        binding, {"total_return": 0.1}, pipeline=_pipeline(binding.expected_manifest)
    )
    assert (
        TradeHistoryService(workspace=jobs.workspace, store=jobs.store)
        .load_run("job")
        .research_evidence
        == evidence
    )


@pytest.mark.parametrize("change", ["missing", "tamper", "wrong_hash", "wrong_binding"])
def test_publisher_rejects_invalid_submitted_evidence_before_writing_result(
    tmp_path: Path, change: str
) -> None:
    jobs, publisher, binding, _evidence = _single(tmp_path)
    assert binding.expected_manifest is not None
    path = SubmittedResearchStore(jobs.workspace).path("job")
    if change == "missing":
        path.unlink()
    elif change == "tamper":
        path.write_text(path.read_text() + " ")
    elif change == "wrong_hash":
        jobs.update_job("job", research_evidence_hash="0" * 64)
    else:
        binding = replace(binding, run_id="another")
    assert binding.expected_manifest is not None
    with pytest.raises(ValueError):
        publisher.publish(
            binding,
            {"total_return": 0.1},
            pipeline=_pipeline(binding.expected_manifest),
        )
    assert not jobs.list_runs() and not jobs.list_artifacts()


@pytest.mark.parametrize(
    "statuses",
    [
        ("not_run", "not_run", "not_run"),
        ("failed", "failed", "failed"),
        ("succeeded", "failed", "not_run"),
    ],
)
def test_comparison_research_survives_zero_or_partial_success(
    tmp_path: Path, statuses: Any
) -> None:
    fixture = _fixture(tmp_path, statuses, snapshot=statuses[0] != "not_run")
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
    loaded = load_report_comparison(
        fixture.workspace,
        fixture.store,
        TradeHistoryService(workspace=fixture.workspace, store=fixture.store),
        "comparison",
    )
    assert loaded.research_evidence == evidence
    assert len(loaded.research_evidence.research.targets) == 3
    assert loaded.evidence["research_evidence_status"] == "recorded"
    snapshots.path("comparison").unlink()
    with pytest.raises(ManifestError):
        load_report_comparison(
            fixture.workspace,
            fixture.store,
            TradeHistoryService(workspace=fixture.workspace, store=fixture.store),
            "comparison",
        )


def test_snapshot_store_enforces_immutable_identity_bytes_and_safe_paths(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    workspace.ensure()
    snapshots = SubmittedResearchStore(workspace)
    target = simple_manifest()
    evidence = approve(target)
    path = snapshots.save("job", evidence, target)
    assert snapshots.save("job", evidence, target) == path
    with pytest.raises(ManifestError, match="different content"):
        snapshots.save("job", approve(target), target)
    for identifier in ("../outside", "/absolute", "bad/name", ""):
        with pytest.raises(ManifestError):
            snapshots.path(identifier)
    path.unlink()
    path.symlink_to(tmp_path / "outside")
    with pytest.raises(ManifestError, match="leaves workspace"):
        snapshots.load("job", expected_hash=evidence.content_hash, target=target)


@pytest.mark.parametrize(
    "change", ["schema", "id", "run_id", "target", "evidence", "noncanonical"]
)
def test_submitted_snapshot_rejects_changed_envelope(
    tmp_path: Path, change: str
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    snapshots = SubmittedResearchStore(workspace)
    target = simple_manifest()
    evidence = approve(target)
    path = snapshots.save("job", evidence, target)
    payload = json.loads(path.read_text())
    if change == "schema":
        payload["schema_version"] = True
    elif change == "id":
        payload["job_id"] = "another"
    elif change == "run_id":
        payload["run_id"] = "job"
    elif change == "target":
        payload["target"] = simple_manifest(parameters={"period": 99}).model_dump(
            mode="json"
        )
    elif change == "evidence":
        payload["research_evidence"]["execution_assumptions"][0]["policy_version"] = (
            "forged"
        )
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        + ("\n " if change == "noncanonical" else "\n")
    )
    with pytest.raises(ValueError):
        snapshots.load("job", expected_hash=evidence.content_hash, target=target)


@pytest.mark.parametrize("corrupt", [False, True])
def test_recovery_requires_registered_research_copy(
    tmp_path: Path, corrupt: bool
) -> None:
    from tests.app.test_comparison_lifecycle import _comparison, _publish_first

    fixture = _comparison(tmp_path)
    evidence = approve(fixture.bundle)
    fixture.jobs.update_job("comparison", research_evidence_hash=evidence.content_hash)
    snapshots = SubmittedResearchStore(fixture.jobs.workspace)
    snapshots.save("comparison", evidence, fixture.bundle)
    run_id = _publish_first(fixture)
    marker = fixture.jobs.workspace.runs / run_id / ".result-publication"
    marker.mkdir()
    binding = RunPublicationBinding(
        "comparison", run_id, fixture.bundle.candidates[0].manifest
    )
    publisher = RunPublisher(fixture.jobs.workspace, fixture.jobs.store)
    if corrupt:
        snapshots.path(run_id, run=True).unlink()
        with pytest.raises(ValueError):
            publisher.verify_published(binding)
        assert marker.exists()
    else:
        publisher.verify_published(binding)
        assert not marker.exists()


def test_legacy_comparison_and_direct_run_explicitly_have_no_research(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    history = TradeHistoryService(workspace=fixture.workspace, store=fixture.store)
    result = load_report_comparison(
        fixture.workspace, fixture.store, history, "comparison"
    )
    assert result.evidence["research_evidence_status"] == "not_recorded"
    loaded = history.load_scope("comparison_candidate_1")
    assert loaded.research_evidence is None
    assert loaded.parameter_sources == () and loaded.execution_assumptions is None
