"""Fixed comparison execution retains every outcome and recovers published runs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
import yaml
from tests.app.test_backtest_service import (
    _DataServiceStub,
    _raw_config,
    _service,
    _SignalStrategy,
    _StrategyLoaderStub,
)
from tests.app.test_execution_dataset import _candidate

from tradingdev.adapters.execution.process_runner import ProcessRunner, WorkerHandle
from tradingdev.adapters.storage.comparisons import ComparisonStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.comparison_service import ComparisonService
from tradingdev.app.contracts.jobs import JobStatus
from tradingdev.app.execution_submission import ExecutionSubmissionService
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.prepared_execution import PreparedComparison
from tradingdev.app.run_publisher import RunPublicationBinding, RunPublisher
from tradingdev.domain.comparison import ComparisonCandidate, ComparisonManifest
from tradingdev.domain.execution import ExecutionManifest, ManifestError

if TYPE_CHECKING:
    import pandas as pd

    from tradingdev.app.backtest_service import BacktestService
    from tradingdev.app.execution_dataset import ExecutionDataset
    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
    from tradingdev.domain.backtest.schemas import ParallelConfig
    from tradingdev.domain.comparison_results import ComparisonLedger
    from tradingdev.domain.strategies.execution import StrategyExecution


@dataclass(frozen=True)
class _Comparison:
    jobs: JobStore
    store: ComparisonStore
    bundle: ComparisonManifest
    backtest: BacktestService
    data: _DataServiceStub
    loader: _StrategyLoaderStub
    service: ComparisonService

    def ledger(self) -> ComparisonLedger:
        return self.store.load_ledger(
            "comparison", expected_hash=self.bundle.manifest_hash
        )


def _comparison(tmp_path: Path, *, count: int = 3) -> _Comparison:
    backtest, data, loader, _strategy = _service(tmp_path)
    data.dataset.frame["execution_data"] = True
    original = backtest.prepare_execution(_raw_config(), kind="backtest")
    bundle = ComparisonManifest.create(
        [
            ComparisonCandidate(
                candidate_id=f"candidate_{number}",
                label=f"Setting {number}",
                rationale=f"Fixed hypothesis {number}, recorded before execution.",
                manifest=_candidate(original, {"number": number}),
            )
            for number in range(1, count + 1)
        ]
    )
    jobs = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    jobs.create_job(job_id="comparison", manifest=bundle)
    return _Comparison(
        jobs,
        ComparisonStore(jobs.workspace),
        bundle,
        backtest,
        data,
        loader,
        ComparisonService(job_store=jobs, backtest_service=backtest),
    )


def _statuses(fixture: _Comparison) -> list[str]:
    return [row.status for row in fixture.ledger().candidates]


def _publish_first(fixture: _Comparison) -> str:
    """Simulate a completed publication immediately before its ledger transition."""
    candidate = fixture.bundle.candidates[0]
    snapshot = fixture.backtest.load_execution_dataset(candidate.manifest)
    fixture.store.publish_snapshot(
        "comparison",
        expected_hash=fixture.bundle.manifest_hash,
        snapshot_bytes=snapshot.payload,
        data_binding_json=snapshot.data_binding_json,
        source_dataset_id=snapshot.source_dataset_id,
    )
    fixture.jobs.update_job(
        "comparison",
        status="running_backtest",
        data_downloaded=True,
        dataset_id=snapshot.dataset_id,
        source_dataset_id=snapshot.source_dataset_id,
    )
    ledger = fixture.store.transition(
        "comparison",
        expected_hash=fixture.bundle.manifest_hash,
        candidate_id=candidate.candidate_id,
        status="running",
    )
    row = ledger.candidates[0]
    run = fixture.backtest.run_manifest(candidate.manifest, loaded_dataset=snapshot)
    RunPublisher(fixture.jobs.workspace, fixture.jobs.store).publish(
        RunPublicationBinding("comparison", row.run_id, candidate.manifest),
        run.metrics,
        pipeline=run.pipeline,
    )
    return row.run_id


def test_eight_candidates_share_one_snapshot_and_keep_independent_execution_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _comparison(tmp_path, count=8)
    frames: list[pd.DataFrame] = []
    strategies: list[_SignalStrategy] = []
    engines: list[BaseBacktestEngine] = []

    class Candidate(_SignalStrategy):
        def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
            if "execution_data" in df:
                frames.append(df)
                assert df.at[0, "close"] == 100.0
            return super().generate_signals(df)

    def create(
        _config: dict[str, Any],
        _execution: StrategyExecution,
        engine: BaseBacktestEngine | None,
        _parallel: ParallelConfig | None = None,
    ) -> Candidate:
        strategy = Candidate()
        if engine is not None:
            strategies.append(strategy)
            engines.append(engine)
        return strategy

    monkeypatch.setattr(fixture.loader, "create_from_execution", create)

    ledger = fixture.service.execute("comparison")

    assert ledger.is_complete
    assert _statuses(fixture) == ["succeeded"] * 8
    assert len(fixture.data.loads) == 1
    assert len(frames) == len(strategies) == len(engines) == 8
    assert len({id(frame) for frame in frames}) == 8
    assert len({id(strategy) for strategy in strategies}) == 8
    assert len({id(engine) for engine in engines}) == 8
    assert fixture.data.dataset.frame.at[0, "close"] == 100.0
    receipt, snapshot = fixture.store.load_snapshot(
        "comparison", expected_hash=fixture.bundle.manifest_hash
    )
    dataset_id = f"sha256:{hashlib.sha256(snapshot).hexdigest()}"
    job = fixture.jobs.get_job("comparison")
    assert job is not None
    assert job["status"] == "done"
    assert job["manifest_hash"] == fixture.bundle.manifest_hash
    assert job["dataset_id"] == dataset_id
    assert job["source_dataset_id"] == receipt.source_dataset_id == "dataset-fixture"
    assert fixture.jobs.get_run("comparison") is None
    projection = yaml.safe_load(Path(job["config_path"]).read_text())
    assert "parameters" not in projection["strategy"]
    for row, candidate in zip(
        ledger.candidates, fixture.bundle.candidates, strict=True
    ):
        assert row.run_id == f"comparison_{candidate.candidate_id}"
        run = fixture.jobs.get_run(row.run_id)
        assert run is not None
        assert run["job_id"] == "comparison"
        assert run["manifest_hash"] == candidate.manifest.manifest_hash
        assert run["dataset_id"] == dataset_id


@pytest.mark.parametrize("fail_all", [False, True])
@pytest.mark.parametrize("raise_after_mutation", [False, True])
def test_strategy_failure_preserves_all_outcomes_and_continues_later_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_all: bool,
    raise_after_mutation: bool,
) -> None:
    fixture = _comparison(tmp_path)
    attempted: list[int] = []

    class Candidate(_SignalStrategy):
        def __init__(self, number: int) -> None:
            super().__init__()
            self.number = number

        def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
            if "execution_data" in df:
                attempted.append(self.number)
                assert df.at[0, "close"] == 100.0
                if fail_all or self.number == 2:
                    df.loc[:, "close"] = -1.0
                    if raise_after_mutation:
                        raise RuntimeError(f"setting {self.number} failed")
            return super().generate_signals(df)

    monkeypatch.setattr(
        fixture.loader,
        "create_from_execution",
        lambda _config, execution, *_args: Candidate(
            execution.constructor_kwargs["number"]
        ),
    )

    ledger = fixture.service.execute("comparison")

    assert attempted == [1, 2, 3]
    assert len(fixture.data.loads) == 1
    assert _statuses(fixture) == (
        ["failed"] * 3 if fail_all else ["succeeded", "failed", "succeeded"]
    )
    assert ledger.is_complete
    for row in ledger.candidates:
        if row.status == "failed":
            reason = "failed" if raise_after_mutation else "input_mutated"
            assert row.reason is not None and reason in row.reason
            assert fixture.jobs.get_run(row.run_id) is None
    job = fixture.jobs.get_job("comparison")
    assert job is not None and job["status"] == "done"
    response = JobService(job_store=fixture.jobs).get_job_status("comparison")
    assert response["status"] == "done"
    public = JobStatus.model_validate(response)
    assert public.comparison is not None
    assert public.comparison.counts["failed"] == (3 if fail_all else 1)
    assert public.comparison.outcome == (
        "no_success" if fail_all else "partial_success"
    )
    assert response.get("run_id") is None
    assert response.get("metrics") is None
    assert len(response["comparison"]["candidates"]) == 3


def test_changed_source_stops_comparison_before_shared_data_load(
    tmp_path: Path,
) -> None:
    fixture = _comparison(tmp_path)
    (tmp_path / "fixture.py").write_text("# changed after approval\n")

    ledger = fixture.service.execute("comparison")

    assert ledger.is_complete
    assert _statuses(fixture) == ["not_run"] * 3
    assert fixture.data.loads == []
    assert fixture.jobs.list_runs() == []
    job = fixture.jobs.get_job("comparison")
    assert job is not None and job["status"] == "failed"


def test_shared_dataset_binding_mismatch_stops_all_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _comparison(tmp_path)
    original_load = fixture.backtest.load_execution_dataset

    def wrong_binding(manifest: ExecutionManifest) -> ExecutionDataset:
        snapshot = original_load(manifest)
        return replace(snapshot, data_binding_json='{"different":"request"}')

    monkeypatch.setattr(fixture.backtest, "load_execution_dataset", wrong_binding)

    ledger = fixture.service.execute("comparison")

    assert ledger.is_complete
    assert _statuses(fixture) == ["not_run"] * 3
    assert len(fixture.data.loads) == 1
    assert fixture.jobs.list_runs() == []
    job = fixture.jobs.get_job("comparison")
    assert job is not None and job["status"] == "failed"


def test_terminal_comparison_cannot_be_replayed_or_reload_data(tmp_path: Path) -> None:
    fixture = _comparison(tmp_path)
    fixture.service.execute("comparison")
    original = fixture.ledger()

    with pytest.raises(ManifestError):
        fixture.service.execute("comparison")

    assert fixture.ledger() == original
    assert len(fixture.data.loads) == 1
    assert len(fixture.jobs.list_runs()) == 3


def test_queued_cancellation_preserves_complete_not_run_ledger(tmp_path: Path) -> None:
    fixture = _comparison(tmp_path)
    service = JobService(job_store=fixture.jobs)

    response = service.cancel_job("comparison")

    assert response["success"] is True
    assert response["process_terminated"] is False
    assert _statuses(fixture) == ["not_run"] * 3
    status = service.get_job_status("comparison")
    assert status["status"] == "cancelled"
    assert len(status["comparison"]["candidates"]) == 3
    assert fixture.data.loads == []


def test_cancellation_stops_worker_before_recovering_published_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _comparison(tmp_path)
    run_id = _publish_first(fixture)
    fixture.store.transition(
        "comparison",
        expected_hash=fixture.bundle.manifest_hash,
        candidate_id="candidate_2",
        status="running",
    )
    handle = WorkerHandle(12345, 100.0, "b" * 32)
    fixture.jobs.update_job("comparison", **handle.job_fields())
    stopped: list[WorkerHandle] = []

    def stop(_root: Path, selected: WorkerHandle) -> bool:
        assert _statuses(fixture) == ["running", "running", "pending"]
        stopped.append(selected)
        return True

    monkeypatch.setattr("tradingdev.app.job_service.request_worker_stop", stop)

    response = JobService(job_store=fixture.jobs).cancel_job("comparison")

    assert response["success"] is True
    assert stopped == [handle]
    assert _statuses(fixture) == ["succeeded", "cancelled", "not_run"]
    assert fixture.jobs.get_run(run_id) is not None


def test_failed_worker_cleanup_does_not_finalize_comparison_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _comparison(tmp_path)
    _publish_first(fixture)
    fixture.jobs.update_job(
        "comparison", **WorkerHandle(12345, 100.0, "b" * 32).job_fields()
    )

    def fail_stop(_root: Path, _selected: WorkerHandle) -> bool:
        raise RuntimeError("cleanup not confirmed")

    monkeypatch.setattr("tradingdev.app.job_service.request_worker_stop", fail_stop)

    response = JobService(job_store=fixture.jobs).cancel_job("comparison")

    assert response["success"] is False
    assert response["code"] == "worker_cleanup_failed"
    assert _statuses(fixture) == ["running", "pending", "pending"]


def test_dead_worker_recovers_completed_publication_before_finalizing_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _comparison(tmp_path)
    run_id = _publish_first(fixture)
    handle = WorkerHandle(12345, 100.0, "b" * 32)
    fixture.jobs.update_job("comparison", **handle.job_fields())
    service = JobService(job_store=fixture.jobs)
    monkeypatch.setattr(service, "_is_process_alive", lambda _job: False)
    stopped: list[WorkerHandle] = []

    def stop(_root: Path, selected: WorkerHandle) -> bool:
        assert _statuses(fixture) == ["running", "pending", "pending"]
        stopped.append(selected)
        return False

    monkeypatch.setattr("tradingdev.app.job_service.request_worker_stop", stop)

    response = service.get_job_status("comparison")

    assert response["status"] == "failed"
    assert stopped == [handle]
    assert _statuses(fixture) == ["succeeded", "not_run", "not_run"]
    assert fixture.jobs.get_run(run_id) is not None


@pytest.mark.parametrize("damage", ["artifact", "parent", "manifest"])
def test_recovery_does_not_promote_corrupt_or_foreign_published_results(
    tmp_path: Path, damage: str
) -> None:
    fixture = _comparison(tmp_path)
    run_id = _publish_first(fixture)
    if damage == "artifact":
        artifact = fixture.jobs.get_artifact(f"{run_id}:strategy_source")
        assert artifact is not None
        Path(artifact["path"]).write_text("# replaced source snapshot\n")
    else:
        column = "job_id" if damage == "parent" else "manifest_hash"
        value = "another_parent" if damage == "parent" else "0" * 64
        with fixture.jobs.store.connect() as connection:
            connection.execute(
                f"update runs set {column} = ? where run_id = ?", (value, run_id)
            )

    ledger = fixture.service.reconcile(
        "comparison", reason="Worker stopped before ledger completion."
    )

    assert ledger.is_complete
    assert _statuses(fixture) == ["failed", "not_run", "not_run"]


def test_status_does_not_reload_large_shared_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _comparison(tmp_path)
    fixture.service.execute("comparison")

    def no_snapshot_read(*_args: object, **_kwargs: object) -> object:
        pytest.fail("Status polling must not read the complete market snapshot")

    monkeypatch.setattr(ComparisonStore, "load_snapshot", no_snapshot_read)

    response = JobService(job_store=fixture.jobs).get_job_status("comparison")

    assert response["status"] == "done"
    candidates = cast("list[dict[str, Any]]", response["comparison"]["candidates"])
    assert [row["status"] for row in candidates] == ["succeeded"] * 3


@pytest.mark.parametrize("mismatch", ["pending", "run_id", "manifest", "parent"])
def test_child_publication_requires_running_membership_and_exact_child_binding(
    tmp_path: Path, mismatch: str
) -> None:
    fixture = _comparison(tmp_path)
    _publish_first(fixture)
    candidate = fixture.bundle.candidates[1]
    manifest = candidate.manifest
    job_id = "comparison"
    run_id = "comparison_candidate_2"
    if mismatch != "pending":
        fixture.store.transition(
            job_id,
            expected_hash=fixture.bundle.manifest_hash,
            candidate_id="candidate_2",
            status="running",
        )
    if mismatch == "run_id":
        run_id = "unreserved_child"
    elif mismatch == "manifest":
        manifest = _candidate(manifest, {"number": 99})
    elif mismatch == "parent":
        job_id = "other_parent"
        fixture.jobs.create_job(job_id=job_id, manifest=fixture.bundle)
        fixture.jobs.update_job(job_id, dataset_id="sha256:" + "a" * 64)
    run = fixture.backtest.run_manifest(manifest)

    with pytest.raises(ManifestError):
        RunPublisher(fixture.jobs.workspace, fixture.jobs.store).publish(
            RunPublicationBinding(job_id, run_id, manifest),
            run.metrics,
            pipeline=run.pipeline,
        )

    assert fixture.jobs.get_run(run_id) is None
    assert len(fixture.jobs.list_runs()) == 1


@pytest.mark.parametrize("startup_failure", [False, True])
def test_submission_persists_all_candidates_before_one_worker_launch(
    tmp_path: Path, startup_failure: bool
) -> None:
    fixture = _comparison(tmp_path)
    jobs = JobStore(workspace=WorkspacePaths(tmp_path / "submission-workspace"))
    comparisons = ComparisonStore(jobs.workspace)
    launches: list[tuple[str, tuple[str, ...]]] = []

    class Runner(ProcessRunner):
        def __init__(self) -> None:
            pass

        def spawn_module(self, module: str, *args: str) -> WorkerHandle:
            launches.append((module, args))
            ledger = comparisons.load_ledger(
                "approved_comparison", expected_hash=fixture.bundle.manifest_hash
            )
            assert [row.status for row in ledger.candidates] == ["pending"] * 3
            assert jobs.load_manifest("approved_comparison") == fixture.bundle
            if startup_failure:
                raise RuntimeError("comparison startup failed")
            return WorkerHandle(4321, 100.0, "a" * 32)

    prepared = PreparedComparison(
        manifest=fixture.bundle,
        original_config_path=tmp_path / "original.yaml",
        candidate_fixture_rows={
            candidate.candidate_id: (80, 300) for candidate in fixture.bundle.candidates
        },
    )
    submission = ExecutionSubmissionService(jobs, Runner())
    if startup_failure:
        with pytest.raises(RuntimeError, match="comparison startup failed"):
            submission.submit(prepared, job_id="approved_comparison")
        failed = jobs.get_job("approved_comparison")
        assert failed is not None and failed["status"] == "failed"
        ledger = comparisons.load_ledger(
            "approved_comparison", expected_hash=fixture.bundle.manifest_hash
        )
        assert [row.status for row in ledger.candidates] == ["not_run"] * 3
    else:
        result = submission.submit(prepared, job_id="approved_comparison")
        assert result["job_id"] == "approved_comparison"
        assert result["manifest_hash"] == fixture.bundle.manifest_hash
    original = jobs.get_job("approved_comparison")

    with pytest.raises(ValueError, match="already exists"):
        submission.submit(prepared, job_id="approved_comparison")

    assert jobs.get_job("approved_comparison") == original
    assert launches == [("tradingdev.mcp.workers.comparison", ("approved_comparison",))]
    assert fixture.data.loads == []


@pytest.mark.parametrize("write", ["job", "legacy_summary", "foreign_run"])
def test_pending_comparison_reserves_child_ids_before_any_results(
    tmp_path: Path, write: str
) -> None:
    fixture = _comparison(tmp_path)
    child = fixture.bundle.candidates[0].manifest
    run_id = "comparison_candidate_1"
    with pytest.raises(ManifestError, match="reserved"):
        if write == "job":
            fixture.jobs.create_job(job_id=run_id, manifest=child)
        elif write == "legacy_summary":
            fixture.jobs.save_result(run_id, {"total_return": 99.0})
        else:
            fixture.jobs.create_job(job_id="foreign", manifest=child)
            run = fixture.backtest.run_manifest(child)
            RunPublisher(fixture.jobs.workspace, fixture.jobs.store).publish(
                RunPublicationBinding("foreign", run_id, child),
                run.metrics,
                pipeline=run.pipeline,
            )
    assert fixture.jobs.get_job(run_id) is None
    assert fixture.jobs.get_run(run_id) is None
    assert _statuses(fixture) == ["pending"] * 3


def test_recovery_accepts_complete_publication_with_leftover_marker(
    tmp_path: Path,
) -> None:
    fixture = _comparison(tmp_path)
    run_id = _publish_first(fixture)
    marker = fixture.jobs.workspace.runs / run_id / ".result-publication"
    marker.mkdir()

    ledger = fixture.service.reconcile("comparison", reason="Worker exited.")

    assert ledger.is_complete
    assert _statuses(fixture) == ["succeeded", "not_run", "not_run"]
    assert not marker.exists()


def test_wrong_worker_cannot_claim_queued_comparison(tmp_path: Path) -> None:
    fixture = _comparison(tmp_path)
    original = WorkerHandle(12345, 100.0, "b" * 32)
    fixture.jobs.update_job("comparison", **original.job_fields())
    with pytest.raises(ManifestError, match="different worker"):
        fixture.service.execute(
            "comparison", worker_handle=WorkerHandle(99999, 200.0, "c" * 32)
        )
    job = fixture.jobs.get_job("comparison")
    assert job is not None and job["status"] == "queued"
    assert WorkerHandle.from_job(job) == original
    assert fixture.data.loads == []
    assert _statuses(fixture) == ["pending"] * 3


def test_status_preserves_completion_between_liveness_check_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _comparison(tmp_path)
    fixture.jobs.update_job(
        "comparison",
        **WorkerHandle(12345, 100.0, "b" * 32).job_fields(),
    )
    service = JobService(job_store=fixture.jobs)
    monkeypatch.setattr(service, "_is_process_alive", lambda _job: False)

    calls: list[WorkerHandle] = []

    def stopped(_root: Path, _handle: WorkerHandle) -> bool:
        calls.append(_handle)
        fixture.service.execute("comparison", worker_handle=_handle)
        return False

    monkeypatch.setattr("tradingdev.app.job_service.request_worker_stop", stopped)

    response = service.get_job_status("comparison")

    assert response["status"] == "done"
    assert calls == [WorkerHandle(12345, 100.0, "b" * 32)]
    assert _statuses(fixture) == ["succeeded"] * 3
