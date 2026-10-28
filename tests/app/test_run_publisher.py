"""Distinct run identities retain their parent job and immutable publication."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from filelock import FileLock, Timeout

from tradingdev.adapters.execution.process_runner import ProcessRunner, WorkerHandle
from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file
from tradingdev.adapters.storage.performance import (
    PerformanceArtifactError,
    PerformanceStore,
)
from tradingdev.app.execution_submission import (
    ExecutionSubmissionService,
)
from tradingdev.app.job_store import JobStore
from tradingdev.app.prepared_execution import PreparedExecution
from tradingdev.app.run_publisher import RunPublicationBinding, RunPublisher
from tradingdev.app.strategy_service import StrategyNotExecutableError
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.result import BacktestResult
from tradingdev.domain.execution import ExecutionManifest, ManifestError
from tradingdev.domain.performance.artifacts import (
    PerformanceArtifacts,
    bundles_from_pipeline,
)
from tradingdev.domain.strategies.execution import StrategyExecution

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any


def _manifest(source: Path, window: int) -> ExecutionManifest:
    return ExecutionManifest.create(
        kind="backtest",
        config={
            "strategy": {
                "id": "fixture",
                "source_path": str(source),
                "source_hash": sha256_file(source),
                "parameters": {"window": window},
            },
            "backtest": {
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
                "init_cash": 100.0,
            },
        },
        strategy_execution=StrategyExecution(
            kind="generated", constructor_kwargs={"window": window}
        ),
    )


def _pipeline(manifest: ExecutionManifest) -> PipelineResult:
    return PipelineResult(
        mode="simple",
        config_snapshot=manifest.config_copy(),
        execution_manifest=manifest,
        backtest_result=BacktestResult(
            metrics={"total_return": 0.1},
            equity_curve=np.array([100.0, 110.0]),
            returns=np.array([0.0, 0.1]),
            timestamps=pd.date_range("2024-01-01", periods=2, tz="UTC").to_numpy(),
            trades=[],
            init_cash=100.0,
            mode="signal",
        ),
    )


def _submitted(
    tmp_path: Path,
) -> tuple[JobStore, RunPublisher, RunPublicationBinding, Path]:
    source = tmp_path / "strategy.py"
    source.write_text("# pinned source\n")
    parent = _manifest(source, 1)
    jobs = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    jobs.create_job(job_id="parent", manifest=parent)
    jobs.update_job("parent", dataset_id="dataset-fixture")
    # This fixture supplies a trusted binding independently of the result. It
    # exercises publication only, not comparison submission or a worker.
    binding = RunPublicationBinding(
        job_id="parent", run_id="candidate", expected_manifest=_manifest(source, 2)
    )
    return jobs, RunPublisher(jobs.workspace, jobs.store), binding, source


def test_multiple_run_publications_keep_job_and_artifact_identity_separate(
    tmp_path: Path,
) -> None:
    jobs, publisher, first, source = _submitted(tmp_path)
    original_job = jobs.get_job("parent")
    parent_manifest = (jobs.workspace.runs / "parent" / "manifest.json").read_bytes()
    second = replace(
        first, run_id="candidate_two", expected_manifest=_manifest(source, 3)
    )

    for binding in (first, second):
        assert binding.expected_manifest is not None
        pipeline = _pipeline(binding.expected_manifest)
        path = publisher.publish(binding, {"total_return": 0.1}, pipeline=pipeline)
        assert path == jobs.workspace.runs / binding.run_id / "result.json"
        run = jobs.get_run(binding.run_id)
        assert run is not None
        assert run["job_id"] == "parent"
        assert run["dataset_id"] == "dataset-fixture"
        assert run["manifest_hash"] == binding.expected_manifest.manifest_hash
        artifacts = jobs.list_artifacts(binding.run_id)
        assert {artifact["artifact_type"] for artifact in artifacts} == {
            "result_json",
            "execution_manifest",
            "config_snapshot",
            "strategy_source",
            "dataset_fingerprint",
            "pipeline_result",
            "performance_json",
            "observations_json",
        }
        for artifact in artifacts:
            assert artifact["artifact_id"].startswith(f"{binding.run_id}:")
            assert artifact["run_id"] == binding.run_id
            assert artifact["path"].startswith(str(path.parent) + "/")
            if artifact["artifact_type"] not in {
                "performance_json",
                "observations_json",
            }:
                assert artifact["metadata"]["job_id"] == "parent"
        assert (
            ExecutionManifestStore(jobs.workspace).load(
                binding.run_id, expected_hash=binding.expected_manifest.manifest_hash
            )
            == binding.expected_manifest
        )
        original = {item: item.read_bytes() for item in path.parent.iterdir()}
        original_run = jobs.get_run(binding.run_id)
        assert (
            publisher.publish(binding, {"total_return": 0.1}, pipeline=pipeline) == path
        )
        assert {item: item.read_bytes() for item in path.parent.iterdir()} == original
        assert jobs.get_run(binding.run_id) == original_run

    assert jobs.get_job("parent") == original_job
    assert jobs.get_run("parent") is None
    assert (
        jobs.workspace.runs / "parent" / "manifest.json"
    ).read_bytes() == parent_manifest
    assert not list(jobs.workspace.runs.rglob(".result-publication"))


@pytest.mark.parametrize(
    "mismatch",
    ["manifest", "config", "performance_run", "parent_strategy", "parent_revision"],
)
def test_rejects_results_outside_explicit_publication_identity(
    tmp_path: Path, mismatch: str
) -> None:
    jobs, publisher, binding, source = _submitted(tmp_path)
    assert binding.expected_manifest is not None
    pipeline = _pipeline(binding.expected_manifest)
    if mismatch == "manifest":
        pipeline = _pipeline(_manifest(source, 3))
    elif mismatch == "config":
        pipeline.config_snapshot["strategy"]["parameters"]["window"] = 99
    elif mismatch.startswith("parent_"):
        job = jobs.get_job("parent")
        assert job is not None
        job["strategy_name" if mismatch == "parent_strategy" else "revision_id"] = (
            "other"
        )
        jobs.store.upsert_job(job)

    with pytest.raises(ManifestError):
        if mismatch == "performance_run":
            details = bundles_from_pipeline(
                "wrong_run", pipeline, {"total_return": 0.1}
            )
            publisher.publish(
                binding,
                {"total_return": 0.1},
                performance=details,
                execution_manifest=binding.expected_manifest,
            )
        else:
            publisher.publish(binding, {"total_return": 0.1}, pipeline=pipeline)
    assert jobs.list_runs() == []
    assert jobs.list_artifacts() == []
    assert {path.name for path in (jobs.workspace.runs / binding.run_id).iterdir()} == {
        ".submission.lock"
    }


def test_rejects_changed_source_before_creating_child_artifacts(tmp_path: Path) -> None:
    jobs, publisher, binding, source = _submitted(tmp_path)
    assert binding.expected_manifest is not None
    pipeline = _pipeline(binding.expected_manifest)
    source.write_text("# changed source\n")
    with pytest.raises(StrategyNotExecutableError, match="source hash changed"):
        publisher.publish(binding, {"total_return": 0.1}, pipeline=pipeline)
    assert jobs.list_runs() == []
    assert {path.name for path in (jobs.workspace.runs / binding.run_id).iterdir()} == {
        ".submission.lock"
    }


@pytest.mark.parametrize("failure", [OSError("disk failure"), TimeoutError("timeout")])
def test_failed_child_publication_removes_new_manifest_and_retains_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    jobs, publisher, binding, _ = _submitted(tmp_path)
    assert binding.expected_manifest is not None
    before = {
        path: path.read_bytes()
        for path in jobs.workspace.runs.rglob("*")
        if path.is_file() and path.name != ".submission.lock"
    }
    original_publish = PerformanceStore.publish

    def fail_after_publication(
        self: PerformanceStore, artifacts: PerformanceArtifacts
    ) -> None:
        original_publish(self, artifacts)
        raise failure

    monkeypatch.setattr(PerformanceStore, "publish", fail_after_publication)
    with pytest.raises(type(failure), match=str(failure)):
        publisher.publish(
            binding,
            {"total_return": 0.1},
            pipeline=_pipeline(binding.expected_manifest),
        )
    assert jobs.list_runs() == []
    assert jobs.list_artifacts() == []
    assert {
        path: path.read_bytes()
        for path in jobs.workspace.runs.rglob("*")
        if path.is_file() and path.name != ".submission.lock"
    } == before
    assert {path.name for path in (jobs.workspace.runs / binding.run_id).iterdir()} == {
        ".submission.lock"
    }
    assert not list(jobs.workspace.runs.rglob(".result-publication"))


def test_existing_run_cannot_be_reassigned_to_another_parent(tmp_path: Path) -> None:
    jobs, publisher, binding, _ = _submitted(tmp_path)
    assert binding.expected_manifest is not None
    pipeline = _pipeline(binding.expected_manifest)
    publisher.publish(binding, {"total_return": 0.1}, pipeline=pipeline)
    original = jobs.get_run(binding.run_id)
    jobs.create_job(job_id="other_parent", manifest=jobs.load_manifest("parent"))
    jobs.update_job("other_parent", dataset_id="dataset-fixture")
    with pytest.raises(PerformanceArtifactError, match="different identity"):
        publisher.publish(
            replace(binding, job_id="other_parent"),
            {"total_return": 0.1},
            pipeline=pipeline,
        )
    assert jobs.get_run(binding.run_id) == original


def test_existing_child_manifest_is_never_replaced(tmp_path: Path) -> None:
    jobs, publisher, binding, source = _submitted(tmp_path)
    assert binding.expected_manifest is not None
    path = ExecutionManifestStore(jobs.workspace).publish(
        binding.run_id, _manifest(source, 9)
    )
    original = path.read_bytes()
    with pytest.raises(ManifestError, match="expected hash"):
        publisher.publish(
            binding,
            {"total_return": 0.1},
            pipeline=_pipeline(binding.expected_manifest),
        )
    assert path.read_bytes() == original
    assert jobs.list_runs() == []
    assert jobs.list_artifacts() == []
    assert {item.name for item in path.parent.iterdir()} == {
        "manifest.json",
        ".submission.lock",
    }


def test_child_run_id_cannot_overwrite_an_existing_job(tmp_path: Path) -> None:
    jobs, publisher, binding, _ = _submitted(tmp_path)
    assert binding.expected_manifest is not None
    jobs.create_job(job_id=binding.run_id, manifest=binding.expected_manifest)
    with pytest.raises(ManifestError, match="belongs to an existing job"):
        publisher.publish(
            binding,
            {"total_return": 0.1},
            pipeline=_pipeline(binding.expected_manifest),
        )
    assert jobs.list_runs() == []
    assert jobs.list_artifacts() == []


def test_legacy_summary_stays_readable_and_cannot_replace_an_existing_result(
    tmp_path: Path,
) -> None:
    jobs = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    jobs.create_job(job_id="legacy", strategy_name="fixture")
    path = jobs.save_result("legacy", {"total_return": 0.1})
    original = jobs.get_run("legacy")
    assert jobs.save_result("legacy", {"total_return": 0.1}) == path
    with pytest.raises(PerformanceArtifactError, match="different identity or results"):
        jobs.save_result("legacy", {"total_return": 0.2})
    assert jobs.get_run("legacy") == original
    assert jobs.load_result(str(path)) == {"total_return": 0.1}


def test_publication_requires_existing_job_and_explicit_child_manifest(
    tmp_path: Path,
) -> None:
    jobs = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    publisher = RunPublisher(jobs.workspace, jobs.store)
    binding = RunPublicationBinding(
        job_id="legacy", run_id="child", expected_manifest=None
    )
    with pytest.raises(ValueError, match="existing job"):
        publisher.publish(binding, {})
    jobs.create_job(job_id="legacy", strategy_name="fixture")
    with pytest.raises(ManifestError, match="not bound"):
        publisher.publish(binding, {})
    assert jobs.list_runs() == []
    assert jobs.list_artifacts() == []
    assert {path.name for path in (jobs.workspace.runs / "child").iterdir()} == {
        ".submission.lock"
    }


def test_unsubmitted_summary_remains_unregistered_and_cannot_attach_a_manifest(
    tmp_path: Path,
) -> None:
    jobs = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    path = jobs.save_result("historical", {"total_return": 0.1})
    assert jobs.load_result(str(path)) == {"total_return": 0.1}
    assert jobs.list_all_jobs() == []
    assert jobs.list_runs() == []
    assert jobs.list_artifacts() == []
    source = tmp_path / "strategy.py"
    source.write_text("# fixture\n")
    with pytest.raises(ManifestError, match="not bound to a submitted job"):
        jobs.save_result(
            "historical", {"total_return": 0.2}, execution_manifest=_manifest(source, 1)
        )
    assert jobs.load_result(str(path)) == {"total_return": 0.1}


class _Runner(ProcessRunner):
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def spawn_module(self, module: str, *args: str) -> WorkerHandle:
        self.calls.append((module, args))
        return WorkerHandle(4321, 100.0, "a" * 32)


@pytest.mark.parametrize("submission", [False, True])
def test_child_run_reserves_its_id_against_later_job_creation(
    tmp_path: Path, submission: bool
) -> None:
    jobs, publisher, binding, source = _submitted(tmp_path)
    assert binding.expected_manifest is not None
    publisher.publish(
        binding, {"total_return": 0.1}, pipeline=_pipeline(binding.expected_manifest)
    )
    original_run = jobs.get_run(binding.run_id)
    original_artifacts = jobs.list_artifacts(binding.run_id)
    original_files = {
        path: path.read_bytes()
        for path in (jobs.workspace.runs / binding.run_id).iterdir()
    }
    runner = _Runner()
    with pytest.raises(ValueError, match="Job ID belongs to an existing run"):
        if submission:
            ExecutionSubmissionService(jobs, runner).submit(
                PreparedExecution(binding.expected_manifest, source),
                job_id=binding.run_id,
            )
        else:
            jobs.create_job(job_id=binding.run_id, manifest=binding.expected_manifest)
    assert jobs.get_job(binding.run_id) is None
    assert jobs.get_run(binding.run_id) == original_run
    assert jobs.list_artifacts(binding.run_id) == original_artifacts
    assert {
        path: path.read_bytes()
        for path in (jobs.workspace.runs / binding.run_id).iterdir()
    } == original_files
    assert runner.calls == []


@pytest.mark.parametrize("winner", ["child", "job"])
@pytest.mark.parametrize("submission", [False, True])
def test_concurrent_child_and_job_creation_share_one_id_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    winner: str,
    submission: bool,
) -> None:
    jobs, publisher, binding, source = _submitted(tmp_path)
    assert binding.expected_manifest is not None
    prepared = PreparedExecution(binding.expected_manifest, source)
    pipeline = _pipeline(binding.expected_manifest)
    other_jobs = JobStore(workspace=jobs.workspace)
    other_publisher = RunPublisher(other_jobs.workspace, other_jobs.store)
    runner = _Runner()
    first_checked = Event()
    release_first = Event()
    second_blocked = Event()
    lock_path = jobs.workspace.runs / binding.run_id / ".submission.lock"

    def create_job(store: JobStore) -> object:
        if submission:
            return ExecutionSubmissionService(store, runner).submit(
                prepared, job_id=binding.run_id
            )
        return store.create_job(job_id=binding.run_id, manifest=prepared.manifest)

    persistence: Callable[..., None] = (
        jobs.store.create_run if winner == "child" else jobs.store.upsert_job
    )

    def delayed_persistence(*args: Any, **kwargs: Any) -> None:
        first_checked.set()
        assert release_first.wait(5), "First publisher was not released"
        persistence(*args, **kwargs)

    monkeypatch.setattr(
        jobs.store,
        "create_run" if winner == "child" else "upsert_job",
        delayed_persistence,
    )

    def first_publish() -> object:
        if winner == "child":
            return publisher.publish(binding, {"total_return": 0.1}, pipeline=pipeline)
        return create_job(jobs)

    def second_publish() -> object:
        # The first caller has checked the shared ID but has not inserted its
        # row. Verify mutual exclusion in that window with an independent lock.
        with pytest.raises(Timeout), FileLock(lock_path, timeout=0):
            pytest.fail("Shared identity lock was not held before persistence")
        second_blocked.set()
        if winner == "child":
            return create_job(other_jobs)
        return other_publisher.publish(
            binding, {"total_return": 0.1}, pipeline=pipeline
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(first_publish)
        try:
            assert first_checked.wait(5), "First caller did not reach persistence"
            second = executor.submit(second_publish)
            assert second_blocked.wait(5), "Second caller did not observe the lock"
        finally:
            release_first.set()
        first.result(timeout=5)
        with pytest.raises(ValueError, match="belongs to an existing"):
            second.result(timeout=5)

    assert (jobs.get_run(binding.run_id) is not None) == (winner == "child")
    assert (jobs.get_job(binding.run_id) is not None) == (winner == "job")
    assert len(runner.calls) == int(submission and winner == "job")
    assert lock_path.is_file()
    assert not list(jobs.workspace.runs.rglob(".result-publication"))


def test_legacy_extra_payload_cannot_change_the_reserved_job_id(tmp_path: Path) -> None:
    jobs, publisher, binding, _ = _submitted(tmp_path)
    assert binding.expected_manifest is not None
    publisher.publish(
        binding, {"total_return": 0.1}, pipeline=_pipeline(binding.expected_manifest)
    )
    original = jobs.get_run(binding.run_id)
    with pytest.raises(ManifestError, match="cannot replace execution identity"):
        jobs.create_job(job_id="unused", extra_payload={"job_id": binding.run_id})
    assert jobs.get_job("unused") is None
    assert jobs.get_job(binding.run_id) is None
    assert jobs.get_run(binding.run_id) == original


def test_legacy_summary_cannot_overwrite_a_registered_child_result(
    tmp_path: Path,
) -> None:
    jobs, publisher, binding, _ = _submitted(tmp_path)
    assert binding.expected_manifest is not None
    path = publisher.publish(
        binding, {"total_return": 0.1}, pipeline=_pipeline(binding.expected_manifest)
    )
    original_run = jobs.get_run(binding.run_id)
    original_artifacts = jobs.list_artifacts(binding.run_id)
    original_files = {item: item.read_bytes() for item in path.parent.iterdir()}

    with pytest.raises(ValueError, match="Result ID belongs to an existing run"):
        jobs.save_result(binding.run_id, {"total_return": 99})

    assert jobs.get_run(binding.run_id) == original_run
    assert jobs.list_artifacts(binding.run_id) == original_artifacts
    assert {item: item.read_bytes() for item in path.parent.iterdir()} == original_files
    assert jobs.load_result(str(path)) == {"total_return": 0.1}
    assert jobs.get_job(binding.run_id) is None


@pytest.mark.parametrize("winner", ["child", "legacy"])
def test_concurrent_legacy_summary_and_child_publication_preserve_registered_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, winner: str
) -> None:
    jobs, publisher, binding, _ = _submitted(tmp_path)
    assert binding.expected_manifest is not None
    pipeline = _pipeline(binding.expected_manifest)
    other_jobs = JobStore(workspace=jobs.workspace)
    other_publisher = RunPublisher(other_jobs.workspace, other_jobs.store)
    first_checked = Event()
    release_first = Event()
    second_blocked = Event()
    lock_path = jobs.workspace.runs / binding.run_id / ".submission.lock"

    def wait_before_write() -> None:
        first_checked.set()
        assert release_first.wait(5), "First writer was not released"

    if winner == "child":
        create_run = jobs.store.create_run

        def delayed_create_run(**kwargs: Any) -> None:
            wait_before_write()
            create_run(**kwargs)

        monkeypatch.setattr(jobs.store, "create_run", delayed_create_run)
    else:
        get_run = jobs.get_run

        def delayed_missing_run(run_id: str) -> dict[str, Any] | None:
            result = get_run(run_id)
            assert result is None
            wait_before_write()
            return result

        monkeypatch.setattr(jobs, "get_run", delayed_missing_run)

    def first_write() -> Path:
        if winner == "child":
            return publisher.publish(binding, {"total_return": 0.1}, pipeline=pipeline)
        return jobs.save_result(binding.run_id, {"total_return": 99})

    def second_write() -> Path:
        # Both the no-job/no-run classification and the subsequent file write
        # must be protected, even while the competing run row is still absent.
        with pytest.raises(Timeout), FileLock(lock_path, timeout=0):
            pytest.fail("Identity lock was not held across the no-job write")
        second_blocked.set()
        if winner == "child":
            return other_jobs.save_result(binding.run_id, {"total_return": 99})
        return other_publisher.publish(
            binding, {"total_return": 0.1}, pipeline=pipeline
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(first_write)
        try:
            assert first_checked.wait(5), "First writer did not reach the protected gap"
            second = executor.submit(second_write)
            assert second_blocked.wait(5), "Second writer did not observe the lock"
        finally:
            release_first.set()
        result_path = first.result(timeout=5)
        if winner == "child":
            with pytest.raises(
                ValueError, match="Result ID belongs to an existing run"
            ):
                second.result(timeout=5)
        else:
            assert second.result(timeout=5) == result_path

    run = other_jobs.get_run(binding.run_id)
    assert run is not None and run["metrics"] == {"total_return": 0.1}
    assert other_jobs.load_result(str(result_path)) == run["metrics"]
    artifact = other_jobs.get_artifact(f"{binding.run_id}:result_json")
    assert artifact is not None and artifact["sha256"] == sha256_file(result_path)
    assert other_jobs.get_job(binding.run_id) is None
    assert not list(jobs.workspace.runs.rglob(".result-publication"))
