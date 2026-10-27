"""Preparation and submission keep one captured specification across the boundary."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import TYPE_CHECKING

import pytest
from filelock import FileLock, Timeout

from tradingdev.adapters.execution.process_runner import ProcessRunner, WorkerHandle
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.execution_submission import (
    ExecutionSubmissionService,
)
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.prepared_execution import PreparedExecution
from tradingdev.domain.execution import ManifestError

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


class _Runner(ProcessRunner):
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def spawn_module(self, module: str, *args: str) -> WorkerHandle:
        self.calls.append((module, args))
        return WorkerHandle(4321, 100.0, "a" * 32)


def _prepare(
    root: Path, *, walk_forward: bool = False
) -> tuple[PreparedExecution, JobStore, _Runner]:
    store = JobStore(workspace=WorkspacePaths(root / "workspace"))
    runner = _Runner()
    service = JobService(job_store=store, process_runner=runner)
    prepare = service.prepare_walk_forward if walk_forward else service.prepare_backtest
    prepared = prepare(
        strategy_id="kd_crossover",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        # Bundled explicit validation includes the whole final UTC calendar day.
        end_date=("2025-12-31T23:59:59.999999" if walk_forward else "2025-12-31"),
    )
    assert isinstance(prepared, PreparedExecution)
    return prepared, store, runner


@pytest.mark.parametrize("walk_forward", [False, True])
def test_preparation_creates_no_job_or_worker_and_submission_does_not_reread_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, walk_forward: bool
) -> None:
    prepared, store, runner = _prepare(tmp_path, walk_forward=walk_forward)
    expected = prepared.manifest.model_dump(mode="json")
    assert store.list_all_jobs() == []
    assert store.list_runs() == []
    assert list(store.workspace.runs.iterdir()) == []
    assert runner.calls == []

    def unexpected_config_read(*_args: object) -> None:
        pytest.fail("Submission must not rebuild an already prepared execution")

    monkeypatch.setattr(
        "tradingdev.app.job_service.load_config", unexpected_config_read
    )
    monkeypatch.setattr(
        "tradingdev.app.backtest_service.BacktestService.prepare_execution",
        unexpected_config_read,
    )
    response = ExecutionSubmissionService(store, runner).submit(
        prepared, job_id="confirmed_plan"
    )

    assert response == {
        "job_id": "confirmed_plan",
        "revision_id": None,
        "manifest_hash": prepared.manifest.manifest_hash,
    }
    assert store.load_manifest("confirmed_plan").model_dump(mode="json") == expected
    job = store.get_job("confirmed_plan")
    assert job is not None
    assert job["job_type"] == ("walk_forward" if walk_forward else "backtest")
    assert job["original_config_path"] == str(prepared.original_config_path)
    assert job["symbol"] == "BTC/USDT"
    assert job["timeframe"] == "1h"
    assert runner.calls == [("tradingdev.mcp.workers.backtest", ("confirmed_plan",))]


def test_submission_rejects_changed_prepared_settings_before_creating_job(
    tmp_path: Path,
) -> None:
    prepared, store, runner = _prepare(tmp_path)
    backtest = prepared.manifest.config["backtest"]
    assert isinstance(backtest, dict)
    backtest["fees"] = 0.9

    with pytest.raises(ManifestError, match="hash does not match"):
        ExecutionSubmissionService(store, runner).submit(prepared)

    assert store.list_all_jobs() == []
    assert list(store.workspace.runs.iterdir()) == []
    assert runner.calls == []


def test_submission_rejects_existing_job_without_replacing_or_restarting_it(
    tmp_path: Path,
) -> None:
    prepared, store, runner = _prepare(tmp_path)
    service = ExecutionSubmissionService(store, runner)
    service.submit(prepared, job_id="confirmed_plan")
    original = store.get_job("confirmed_plan")

    with pytest.raises(ValueError, match="Job already exists"):
        service.submit(prepared, job_id="confirmed_plan")

    assert store.get_job("confirmed_plan") == original
    assert len(runner.calls) == 1


def test_concurrent_submission_serializes_check_creation_and_worker_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, store, runner = _prepare(tmp_path)
    other_store = JobStore(workspace=store.workspace)
    first_checked = Event()
    release_first = Event()
    second_blocked = Event()
    create_job = store.create_job
    lock_path = store.workspace.runs / "confirmed_plan" / ".submission.lock"

    def delayed_create(**kwargs: Any) -> dict[str, Any]:
        first_checked.set()
        assert release_first.wait(5), "First submitter was not released"
        return create_job(**kwargs)

    def second_submit() -> dict[str, Any]:
        # The first caller has checked absence but not created the record yet.
        # A nonblocking contender proves that this gap is already protected.
        with pytest.raises(Timeout), FileLock(lock_path, timeout=0):
            pytest.fail("Submission lock was not held before job creation")
        second_blocked.set()
        return ExecutionSubmissionService(other_store, runner).submit(
            prepared, job_id="confirmed_plan"
        )

    monkeypatch.setattr(store, "create_job", delayed_create)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            ExecutionSubmissionService(store, runner).submit,
            prepared,
            job_id="confirmed_plan",
        )
        try:
            assert first_checked.wait(5), "First submitter did not reach creation"
            second = executor.submit(second_submit)
            assert second_blocked.wait(5), "Second submitter did not observe the lock"
        finally:
            release_first.set()
        assert first.result(timeout=5)["job_id"] == "confirmed_plan"
        with pytest.raises(ValueError, match="Job already exists"):
            second.result(timeout=5)

    assert len(store.list_all_jobs()) == 1
    assert len(runner.calls) == 1
    assert lock_path.is_file()


def test_spawn_failure_releases_submission_lock_and_cannot_restart_failed_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, store, runner = _prepare(tmp_path)
    calls: list[str] = []

    def failed_spawn(module: str, *args: str) -> WorkerHandle:
        calls.append(module)
        raise RuntimeError("startup failed")

    monkeypatch.setattr(runner, "spawn_module", failed_spawn)
    submission = ExecutionSubmissionService(store, runner)
    with pytest.raises(RuntimeError, match="startup failed"):
        submission.submit(prepared, job_id="failed_plan")
    lock_path = store.workspace.runs / "failed_plan" / ".submission.lock"
    with FileLock(lock_path, timeout=0):
        failed = store.get_job("failed_plan")
        assert failed is not None and failed["status"] == "failed"
    with pytest.raises(ValueError, match="Job already exists"):
        submission.submit(prepared, job_id="failed_plan")
    assert len(calls) == 1
    assert store.get_job("failed_plan") == failed
    assert lock_path.is_file()
