"""Job and worker boundaries preserve a submitted execution specification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tradingdev.adapters.execution.process_runner import ProcessRunner, WorkerHandle
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.backtest_service import BacktestRun, BacktestService
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.result import BacktestResult
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.mcp.workers import backtest


class _Runner(ProcessRunner):
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def spawn_module(self, module: str, *args: str) -> WorkerHandle:
        self.calls.append((module, args))
        return WorkerHandle(4321, 100.0, "a" * 32)


def _queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[JobStore, dict[str, Any], _Runner]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    monkeypatch.setenv("TRADINGDEV_WORKSPACE", str(workspace.root))
    monkeypatch.setenv(
        "TRADINGDEV_WORKER_IDENTITY",
        json.dumps(WorkerHandle(4321, 100.0, "a" * 32).job_fields()),
    )
    store = JobStore(workspace=workspace)
    runner = _Runner()
    result = JobService(job_store=store, process_runner=runner).start_backtest(
        strategy_id="kd_crossover",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
    )
    return store, result, runner


def test_worker_uses_manifest_when_config_projection_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, started, runner = _queued(tmp_path, monkeypatch)
    job_id = started["job_id"]
    manifest = store.load_manifest(job_id)
    assert runner.calls == [("tradingdev.mcp.workers.backtest", (job_id,))]
    job = store.get_job(job_id)
    assert job is not None
    Path(job["config_path"]).write_text("invalid: [yaml", encoding="utf-8")
    executed: list[ExecutionManifest] = []

    def run_manifest(
        _service: BacktestService, selected: ExecutionManifest
    ) -> BacktestRun:
        executed.append(selected)
        return BacktestRun(
            mode="simple",
            pipeline=PipelineResult(
                mode="simple",
                backtest_result=BacktestResult(
                    metrics={"total_return": 0.1},
                    equity_curve=np.array([10000.0, 11000.0]),
                    init_cash=10000.0,
                    returns=np.array([0.0, 0.1]),
                ),
                config_snapshot=selected.config_copy(),
                execution_manifest=selected,
            ),
            metrics={"total_return": 0.1},
            processed_path=tmp_path / "unused.parquet",
            dataset_id="fixture-dataset",
        )

    monkeypatch.setattr(BacktestService, "run_manifest", run_manifest)
    backtest._run_backtest(job_id)

    completed = store.get_job(job_id)
    assert completed is not None and completed["status"] == "done"
    assert executed == [manifest]
    run = store.get_run(job_id)
    assert run is not None and run["manifest_hash"] == started["manifest_hash"]


@pytest.mark.parametrize("recompute_hash", [False, True])
def test_worker_rejects_tampered_manifest_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recompute_hash: bool
) -> None:
    store, started, _runner = _queued(tmp_path, monkeypatch)
    job_id = started["job_id"]
    path = store.workspace.runs / job_id / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["backtest"]["fees"] = 0.9
    if recompute_hash:
        payload = ExecutionManifest.create(
            kind="backtest",
            config=payload["config"],
            strategy_execution=StrategyExecution.model_validate(
                payload["strategy_execution"]
            ),
        ).model_dump(mode="json")
    path.write_text(json.dumps(payload), encoding="utf-8")

    def unexpected_run(*_args: object) -> BacktestRun:
        pytest.fail("Worker must reject the manifest before executing a backtest")

    monkeypatch.setattr(BacktestService, "run_manifest", unexpected_run)
    backtest._run_backtest(job_id)

    job = store.get_job(job_id)
    assert job is not None and job["status"] == "failed"
    assert "manifest" in job["error"]
    assert store.get_run(job_id) is None
    assert not (store.workspace.runs / job_id / "result.json").exists()


def test_bundled_job_can_be_submitted_from_outside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    store, started, _runner = _queued(tmp_path, monkeypatch)

    manifest = store.load_manifest(started["job_id"])
    assert isinstance(manifest, ExecutionManifest)
    source = Path(manifest.config_copy()["strategy"]["source_path"])
    assert source.is_absolute() and source.is_file()
    assert manifest.manifest_hash == started["manifest_hash"]
