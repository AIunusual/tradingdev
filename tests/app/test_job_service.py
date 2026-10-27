"""Job service tests."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import TYPE_CHECKING

import psutil
import pytest

from tradingdev.adapters.execution.process_runner import ProcessRunner, WorkerHandle
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.data_service import DataService
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore

if TYPE_CHECKING:
    from pathlib import Path


class _FakeRunner(ProcessRunner):
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def spawn_module(self, module: str, *args: str) -> WorkerHandle:
        self.calls.append((module, args))
        return WorkerHandle(4321, 100.0, "a" * 32)


def test_start_walk_forward_uses_bundled_walkforward_config(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    job_store = JobStore(workspace=workspace, store=SQLiteStore(workspace))
    runner = _FakeRunner()
    service = JobService(
        data_service=DataService(workspace),
        job_store=job_store,
        process_runner=runner,
        project_root=tmp_path,
    )

    response = service.start_walk_forward(
        strategy_id="kd_crossover",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        # Keep the bundled validation window inside this explicit research period.
        end_date="2025-12-31T23:59:59.999999",
    )

    assert response["job_id"]
    job = job_store.get_job(str(response["job_id"]))
    assert job is not None
    assert job["job_type"] == "walk_forward"
    assert job["worker_control_id"] == "a" * 32
    assert job["original_config_path"].endswith(
        "bundled/kd_strategy/walkforward_config.yaml"
    )
    assert runner.calls == [
        (
            "tradingdev.mcp.workers.backtest",
            (response["job_id"],),
        )
    ]
    manifest = job_store.load_manifest(response["job_id"])
    assert manifest.kind == "walk_forward"
    assert manifest.manifest_hash == response["manifest_hash"]
    assert manifest.manifest_hash == job["manifest_hash"]


@pytest.mark.parametrize("walk_forward", [False, True])
@pytest.mark.parametrize(
    "failure_type", [OSError, RuntimeError, KeyboardInterrupt, asyncio.CancelledError]
)
def test_start_worker_failure_is_persisted_before_reraising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    walk_forward: bool,
    failure_type: type[BaseException],
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = JobStore(workspace=workspace)
    runner = _FakeRunner()
    failure = failure_type("worker identity unavailable")

    def fail_spawn(module: str, *args: str) -> WorkerHandle:
        raise failure

    monkeypatch.setattr(runner, "spawn_module", fail_spawn)
    service = JobService(
        data_service=DataService(workspace),
        job_store=store,
        process_runner=runner,
        project_root=tmp_path,
    )
    start = service.start_walk_forward if walk_forward else service.start_backtest

    with pytest.raises(failure_type) as caught:
        start(
            strategy_id="kd_crossover",
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01",
            # Reach the worker failure with a valid complete-day validation window.
            end_date=("2025-12-31T23:59:59.999999" if walk_forward else "2025-12-31"),
        )

    assert caught.value is failure
    jobs = store.list_all_jobs()
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "failed"
    assert job["job_type"] == ("walk_forward" if walk_forward else "backtest")
    assert job["error"] == (
        f"Worker failed to start: {failure_type.__name__}: worker identity unavailable"
    )
    assert datetime.fromisoformat(job["ended_at"]) >= datetime.fromisoformat(
        job["created_at"]
    )
    assert job["pid"] is None
    assert job["process_create_time"] is None
    response = service.get_job_status(job["job_id"])
    assert response["status"] == "failed"
    assert response["error"] == job["error"]


@pytest.mark.parametrize("stopped", [True, False])
def test_cancel_job_uses_control_identity_before_marking_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stopped: bool,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    job_store = JobStore(workspace=workspace, store=SQLiteStore(workspace))
    job_store.create_job(
        job_id="job_cancel",
        strategy_name="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
        config_path="fixture.yaml",
    )
    job_store.update_job(
        "job_cancel",
        status="running_backtest",
        **WorkerHandle(12345, 100.0, "b" * 32).job_fields(),
    )

    requests: list[tuple[Path, WorkerHandle]] = []

    def stop(root: Path, handle: WorkerHandle) -> bool:
        requests.append((root, handle))
        return stopped

    monkeypatch.setattr("tradingdev.app.job_service.request_worker_stop", stop)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    service = JobService(job_store=job_store)

    response = service.cancel_job("job_cancel")

    assert response == {
        "success": True,
        "job_id": "job_cancel",
        "status": "cancelled",
        "process_terminated": stopped,
    }
    assert requests == [(workspace.root, WorkerHandle(12345, 100.0, "b" * 32))]
    assert not signals
    cancelled = job_store.get_job("job_cancel")
    assert cancelled is not None
    assert cancelled["status"] == "cancelled"
    assert cancelled["ended_at"]
    assert cancelled["error"] == "Cancelled by user."


def test_cancel_job_rejects_terminal_job(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    job_store = JobStore(workspace=workspace, store=SQLiteStore(workspace))
    job_store.create_job(
        job_id="job_done",
        strategy_name="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
        config_path="fixture.yaml",
    )
    job_store.update_job("job_done", status="done")

    response = JobService(job_store=job_store).cancel_job("job_done")

    assert response["success"] is False
    assert response["status"] == "done"


def test_get_job_status_returns_run_id_for_completed_optimization(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    job_store = JobStore(workspace=workspace, store=SQLiteStore(workspace))
    job_store.create_job(
        job_id="job_optimization",
        strategy_name="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
        config_path="fixture.yaml",
    )
    job_store.save_result(
        "job_optimization",
        {
            "best_params": {"window": 10},
            "train_metrics": {"max_drawdown": 0.1},
            "test_metrics": {"max_drawdown": 0.08},
            "optimization_metric": "max_drawdown",
            "direction": "minimize",
            "total_combinations": 3,
        },
    )
    job_store.update_job(
        "job_optimization",
        status="done",
        job_type="optimization",
    )

    response = JobService(job_store=job_store).get_job_status("job_optimization")

    assert response["status"] == "done"
    assert response["job_type"] == "optimization"
    assert response["run_id"] == "job_optimization"
    assert response["best_params"] == {"window": 10}
    assert response["direction"] == "minimize"


@pytest.mark.parametrize("created_at", [None, 100.0])
def test_cancel_rejects_worker_without_control_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    created_at: float | None,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = JobStore(workspace=workspace)
    store.create_job(job_id="old_worker")
    store.update_job(
        "old_worker",
        status="running_backtest",
        pid=12345,
        process_create_time=created_at,
    )

    class ReplacementProcess:
        def create_time(self) -> float:
            return 200.0

    monkeypatch.setattr(psutil, "Process", lambda _pid: ReplacementProcess())
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    result = JobService(job_store=store).cancel_job("old_worker")

    assert result["success"] is False
    assert "cannot confirm process cleanup" in result["error"]
    assert not signals
    job = store.get_job("old_worker")
    assert job is not None
    assert job["status"] == "running_backtest"


@pytest.mark.parametrize("status", ["queued", "running_backtest", "optimizing"])
def test_job_status_detects_reused_pid_as_terminated_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = JobStore(workspace=workspace)
    store.create_job(job_id="old_worker")
    store.update_job("old_worker", status=status, pid=12345, process_create_time=100.0)

    class ReplacementProcess:
        def create_time(self) -> float:
            return 200.0

    monkeypatch.setattr(psutil, "Process", lambda _pid: ReplacementProcess())

    result = JobService(job_store=store).get_job_status("old_worker")

    assert result["status"] == "failed"
    assert result["error"] == "Worker process terminated unexpectedly."
    job = store.get_job("old_worker")
    assert job is not None
    assert job["status"] == "failed"
    assert job["error"] == "Worker process terminated unexpectedly."


def test_job_status_keeps_active_job_when_process_inspection_is_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = JobStore(workspace=workspace)
    store.create_job(job_id="worker")
    store.update_job(
        "worker", status="running_backtest", pid=12345, process_create_time=100.0
    )

    def denied(pid: int) -> psutil.Process:
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", denied)
    service = JobService(job_store=store)

    with pytest.raises(psutil.AccessDenied):
        service.get_job_status("worker")
    job = store.get_job("worker")
    assert job is not None
    assert job["status"] == "running_backtest"


@pytest.mark.parametrize(
    "failure_type", [OSError, ValueError, RuntimeError, TimeoutError]
)
def test_cancel_does_not_claim_success_when_control_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[Exception],
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = JobStore(workspace=workspace)
    store.create_job(job_id="worker")
    store.update_job(
        "worker",
        status="running_backtest",
        **WorkerHandle(12345, 100.0, "b" * 32).job_fields(),
    )

    def fail_stop(root: Path, handle: WorkerHandle) -> bool:
        raise failure_type("control unavailable")

    monkeypatch.setattr("tradingdev.app.job_service.request_worker_stop", fail_stop)

    response = JobService(job_store=store).cancel_job("worker")

    assert response["success"] is False
    assert "control unavailable" in response["error"]
    job = store.get_job("worker")
    assert job is not None
    assert job["status"] == "running_backtest"
    assert job["ended_at"] is None


def test_queued_job_without_worker_can_be_polled_and_cancelled(tmp_path: Path) -> None:
    store = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    store.create_job(job_id="queued")
    service = JobService(job_store=store)

    assert service.get_job_status("queued")["status"] == "queued"
    result = service.cancel_job("queued")

    assert result["success"] is True
    assert result["process_terminated"] is False
    assert service.get_job_status("queued")["status"] == "cancelled"


def test_running_job_without_any_worker_identity_cannot_claim_cancellation(
    tmp_path: Path,
) -> None:
    store = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    store.create_job(job_id="worker")
    store.update_job("worker", status="optimizing")

    response = JobService(job_store=store).cancel_job("worker")

    assert response["success"] is False
    assert "cannot confirm process cleanup" in response["error"]
