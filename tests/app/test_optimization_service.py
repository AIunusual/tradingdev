"""Optimization service tests."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import pytest
import yaml

from tradingdev.adapters.execution.process_runner import WorkerHandle
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.execution_submission import (
    ExecutionSubmissionService,
)
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.optimization_service import OptimizationService
from tradingdev.app.prepared_execution import PreparedExecution
from tradingdev.app.strategy_service import StrategyNotExecutableError
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.strategies.schemas import StrategySpec, StrategyStatus

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from tradingdev.adapters.execution.process_runner import ProcessRunner
    from tradingdev.app.strategy_service import StrategyService

_EXECUTABLE = {StrategyStatus.RUNNABLE, StrategyStatus.PROMOTED}


class _StrategyServiceStub:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata

    def resolve_executable(
        self, strategy_id: str, revision_id: str | None = None
    ) -> StrategySpec:
        status = StrategyStatus(str(self.metadata["status"]))
        if status not in _EXECUTABLE:
            msg = (
                "Strategy must be runnable or promoted before execution. "
                f"Current status: {status.value}"
            )
            raise StrategyNotExecutableError(msg)
        return StrategySpec(
            strategy_id=strategy_id,
            class_name="Fixture",
            kind="bundled",
            source_path=str(self.metadata["source_path"]),
            config_path=str(self.metadata["config_path"]),
            status=status,
        )


class _RunnerStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def spawn_module(self, module: str, *args: str) -> WorkerHandle:
        self.calls.append((module, args))
        return WorkerHandle(2468, 100.0, "a" * 32)


class _StrategyLoaderStub(StrategyLoader):
    def resolve_execution(self, strategy_cfg: dict[str, Any]) -> StrategyExecution:
        return StrategyExecution(
            kind="bundled",
            constructor_kwargs={"config": {"window": 10, "threshold": 0.1}},
        )

    def validate_parameter_grid(
        self,
        strategy_cfg: dict[str, Any],
        execution: StrategyExecution,
        combinations: Iterable[dict[str, Any]],
    ) -> None:
        for combination in combinations:
            invalid = set(combination) - {"window", "threshold"}
            if invalid:
                raise ValueError(f"Unknown optimization parameters: {sorted(invalid)}")


def _service(
    tmp_path: Path,
    *,
    metadata: dict[str, Any],
) -> tuple[OptimizationService, JobStore, _RunnerStub]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source_path = tmp_path / "strategy.py"
    source_path.write_text("class Fixture: pass\n", encoding="utf-8")
    metadata["source_path"] = str(source_path)
    job_store = JobStore(workspace=workspace, store=SQLiteStore(workspace))
    runner = _RunnerStub()
    service = OptimizationService(
        strategy_service=cast("StrategyService", _StrategyServiceStub(metadata)),
        job_store=job_store,
        process_runner=cast("ProcessRunner", runner),
        strategy_loader=_StrategyLoaderStub(workspace_root=workspace.root),
        project_root=tmp_path,
    )
    return service, job_store, runner


def test_prepared_optimization_keeps_search_after_original_config_changes(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text(
        "strategy:\n  id: fixture\nbacktest:\n  symbol: ETH/USDT\n"
        "  timeframe: 4h\n  start_date: '2024-01-01'\n"
        "  end_date: '2024-12-31'\n  init_cash: 10000\n  periods_per_year: 365\n",
        encoding="utf-8",
    )
    service, store, runner = _service(
        tmp_path,
        metadata={"status": "runnable", "config_path": str(config_path)},
    )
    ranges = {"window": [10, 20]}
    prepared = service.prepare_optimization(
        strategy_id="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        param_ranges=ranges,
        optimization_metric="sharpe_ratio",
        train_start="2024-01-01",
        train_end="2024-02-01",
        test_start="2024-02-02",
        test_end="2024-03-01",
    )
    assert isinstance(prepared, PreparedExecution)
    assert store.list_all_jobs() == []
    assert list(store.workspace.runs.iterdir()) == []
    assert runner.calls == []
    expected = prepared.manifest.model_dump(mode="json")
    ranges["window"].append(30)
    config_path.write_text("invalid: [yaml", encoding="utf-8")

    submitted = ExecutionSubmissionService(store, cast("ProcessRunner", runner)).submit(
        prepared, job_id="confirmed_search"
    )

    assert submitted["job_id"] == "confirmed_search"
    assert store.load_manifest("confirmed_search").model_dump(mode="json") == expected
    job = store.get_job("confirmed_search")
    assert job is not None
    assert job["total_combinations"] == 2
    assert job["original_config_path"] == str(config_path)
    assert runner.calls == [
        ("tradingdev.mcp.workers.optimization", ("confirmed_search",))
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "another_strategy", "strategy.id"),
        ("revision_id", "another_revision", "revision_id"),
        ("class_name", "AnotherStrategy", "class_name"),
        ("source_path", "/another/strategy.py", "source_path"),
    ],
)
def test_optimization_rejects_configuration_identity_changed_after_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text(
        "strategy:\n  id: fixture\nbacktest:\n  symbol: ETH/USDT\n"
        "  timeframe: 4h\n  start_date: '2024-01-01'\n"
        "  end_date: '2024-12-31'\n  init_cash: 10000\n  periods_per_year: 365\n",
        encoding="utf-8",
    )
    service, store, runner = _service(
        tmp_path,
        metadata={"status": "runnable", "config_path": str(config_path)},
    )
    original = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    original["strategy"][field] = value
    monkeypatch.setattr(
        "tradingdev.app.optimization_service.load_config", lambda _path: original
    )

    response = service.start_optimization(
        strategy_id="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        param_ranges={"window": [10]},
        optimization_metric="sharpe_ratio",
        train_start="2024-01-01",
        train_end="2024-02-01",
        test_start="2024-02-02",
        test_end="2024-03-01",
    )

    assert response["code"] == "invalid_optimization_request"
    assert message in response["message"]
    assert store.list_all_jobs() == []
    assert list(store.workspace.runs.iterdir()) == []
    assert runner.calls == []


def test_start_optimization_creates_job_and_spawns_worker(tmp_path: Path) -> None:
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text(
        "strategy:\n  id: fixture\nbacktest:\n  symbol: ETH/USDT\n"
        "  timeframe: 4h\n  start_date: '2024-01-01'\n"
        "  end_date: '2024-12-31'\n  init_cash: 10000\n  periods_per_year: 365\n",
        encoding="utf-8",
    )
    service, job_store, runner = _service(
        tmp_path,
        metadata={"status": "runnable", "config_path": str(config_path)},
    )

    response = service.start_optimization(
        strategy_id="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        param_ranges={"window": [10, 20], "threshold": [0.1, 0.2, 0.3]},
        optimization_metric="sharpe_ratio",
        train_start="2024-01-01",
        train_end="2024-02-01",
        test_start="2024-02-02",
        test_end="2024-03-01",
    )

    assert response["job_id"]
    assert response["total_combinations"] == 6
    assert response["direction"] == "maximize"
    assert runner.calls == [
        ("tradingdev.mcp.workers.optimization", (response["job_id"],))
    ]
    job = job_store.get_job(str(response["job_id"]))
    assert job is not None
    assert job["job_type"] == "optimization"
    assert job["pid"] == 2468
    assert job["process_create_time"] == 100.0
    assert job["worker_control_id"] == "a" * 32
    assert job["total_combinations"] == 6
    manifest = job_store.load_manifest(str(response["job_id"]))
    assert response["manifest_hash"] == manifest.manifest_hash
    assert job["manifest_hash"] == manifest.manifest_hash
    assert manifest.kind == "optimization"
    assert manifest.optimization is not None
    assert manifest.optimization.optimization_metric == "sharpe_ratio"
    assert manifest.optimization.param_ranges == {
        "window": [10, 20],
        "threshold": [0.1, 0.2, 0.3],
    }
    assert list(manifest.optimization.param_ranges) == ["threshold", "window"]
    assert "optimization_metric" not in job
    assert "param_ranges" not in job
    config = manifest.config_copy()
    assert config["backtest"]["symbol"] == "BTC/USDT"
    assert config["backtest"]["fees"] == 0.0006
    assert config["parallel"]["reserve_cores"] == 2
    assert config["backtest"]["end_date"] == "2024-03-01T23:59:59.999999"
    assert manifest.strategy_execution.constructor_kwargs == {
        "config": {"window": 10, "threshold": 0.1}
    }

    # A later source YAML edit cannot alter the accepted specification.
    config_path.write_text("strategy: changed\n", encoding="utf-8")
    reloaded = job_store.load_manifest(str(response["job_id"]))
    assert reloaded == manifest


def test_invalid_parallel_overhead_rejected_before_optimization_starts(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text(
        "strategy:\n  id: fixture\nbacktest:\n  symbol: BTC/USDT\n"
        "  timeframe: 1h\n  start_date: '2024-01-01'\n"
        "  end_date: '2024-12-31'\n  init_cash: 10000\n  periods_per_year: 365\n"
        "parallel:\n  overhead_multiplier: 0\n",
        encoding="utf-8",
    )
    service, store, runner = _service(
        tmp_path, metadata={"status": "runnable", "config_path": str(config_path)}
    )
    response = service.start_optimization(
        strategy_id="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        param_ranges={"window": [10]},
        optimization_metric="sharpe_ratio",
        train_start="2024-01-01",
        train_end="2024-02-01",
        test_start="2024-02-02",
        test_end="2024-03-01",
    )
    assert response["code"] == "invalid_optimization_request"
    assert "parallel.overhead_multiplier" in response["message"]
    assert store.list_all_jobs() == []
    assert list(store.workspace.runs.iterdir()) == []
    assert runner.calls == []


@pytest.mark.parametrize(
    ("metric", "mode", "annualization", "timeframe", "message"),
    [
        ("sharpe_ratio", "volume", 365, "1h", "not applicable"),
        ("max_drawdown", "volume", 365, "1h", "not applicable"),
        ("sortino_ratio", "signal", None, "1h", "requires periods_per_year"),
        ("n_days", "signal", 365, "1h", "not an optimization objective"),
        ("sharpe_ratio", "signal", 365, "1w", "unsupported_daily_sampling"),
        ("daily_max_drawdown", "signal", 365, "1wk", "unsupported_daily_sampling"),
        ("daily_pnl_mean", "volume", None, "3d", "unsupported_daily_sampling"),
        ("monthly_pnl_std", "volume", None, "1M", "unsupported_daily_sampling"),
        ("annual_return", "signal", 365, "unknown", "unknown_bar_frequency"),
    ],
)
def test_ineligible_objective_is_rejected_before_job_creation(
    tmp_path: Path,
    metric: str,
    mode: str,
    annualization: int | None,
    timeframe: str,
    message: str,
) -> None:
    config_path = tmp_path / "strategy.yaml"
    capital = "  position_size: 100\n" if mode == "volume" else "  init_cash: 10000\n"
    config_path.write_text(
        "strategy:\n  id: fixture\nbacktest:\n  symbol: BTC/USDT\n"
        "  timeframe: 1h\n  start_date: '2024-01-01'\n"
        "  end_date: '2024-12-31'\n"
        f"  mode: {mode}\n"
        + capital
        + (f"  periods_per_year: {annualization}\n" if annualization else ""),
        encoding="utf-8",
    )
    service, store, runner = _service(
        tmp_path,
        metadata={"status": "runnable", "config_path": str(config_path)},
    )
    response = service.start_optimization(
        strategy_id="fixture",
        symbol="BTC/USDT",
        timeframe=timeframe,
        param_ranges={"window": [10, 20]},
        optimization_metric=metric,
        train_start="2024-01-01",
        train_end="2024-02-01",
        test_start="2024-02-02",
        test_end="2024-03-01",
    )
    assert response["code"] == "invalid_optimization_request"
    assert message in response["message"]
    assert store.list_all_jobs() == []
    assert runner.calls == []


@pytest.mark.parametrize(
    "failure_type", [OSError, RuntimeError, KeyboardInterrupt, asyncio.CancelledError]
)
def test_optimization_spawn_failure_is_persisted_before_reraising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[BaseException],
) -> None:
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text(
        "strategy:\n  id: fixture\nbacktest:\n  symbol: ETH/USDT\n"
        "  timeframe: 4h\n  start_date: '2024-01-01'\n"
        "  end_date: '2024-12-31'\n  init_cash: 10000\n  periods_per_year: 365\n",
        encoding="utf-8",
    )
    service, store, runner = _service(
        tmp_path,
        metadata={"status": "runnable", "config_path": str(config_path)},
    )
    failure = failure_type("worker identity unavailable")

    def fail_spawn(module: str, *args: str) -> WorkerHandle:
        raise failure

    monkeypatch.setattr(runner, "spawn_module", fail_spawn)

    with pytest.raises(failure_type) as caught:
        service.start_optimization(
            strategy_id="fixture",
            symbol="BTC/USDT",
            timeframe="1h",
            param_ranges={"window": [10, 20]},
            optimization_metric="sharpe_ratio",
            train_start="2024-01-01",
            train_end="2024-02-01",
            test_start="2024-02-02",
            test_end="2024-03-01",
        )

    assert caught.value is failure
    jobs = store.list_all_jobs()
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "failed"
    assert job["job_type"] == "optimization"
    assert job["error"] == (
        f"Worker failed to start: {failure_type.__name__}: worker identity unavailable"
    )
    assert datetime.fromisoformat(job["ended_at"]) >= datetime.fromisoformat(
        job["created_at"]
    )
    assert job["pid"] is None
    assert job["process_create_time"] is None
    response = JobService(job_store=store).get_job_status(job["job_id"])
    assert response["status"] == "failed"
    assert response["error"] == job["error"]


def test_start_optimization_rejects_invalid_request_before_spawning(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text("strategy:\n  id: fixture\n", encoding="utf-8")
    service, job_store, runner = _service(
        tmp_path,
        metadata={"status": "runnable", "config_path": str(config_path)},
    )

    response = service.start_optimization(
        strategy_id="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        param_ranges={"window": []},
        optimization_metric="sortino_ratio",
        train_start="2024-01-01",
        train_end="2024-02-01",
        test_start="2024-02-01",
        test_end="2024-03-01",
    )

    assert response["job_id"] == ""
    assert response["total_combinations"] == 0
    assert response["code"] == "invalid_optimization_request"
    assert "window" in response["message"]
    assert job_store.list_all_jobs() == []
    assert runner.calls == []


def test_start_optimization_requires_runnable_strategy(tmp_path: Path) -> None:
    service, job_store, runner = _service(
        tmp_path,
        metadata={"status": "draft", "config_path": str(tmp_path / "missing.yaml")},
    )

    response = service.start_optimization(
        strategy_id="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        param_ranges={"window": [10]},
        optimization_metric="sharpe_ratio",
        train_start="2024-01-01",
        train_end="2024-02-01",
        test_start="2024-02-01",
        test_end="2024-03-01",
    )

    assert response["job_id"] == ""
    assert "must be runnable or promoted" in response["message"]
    assert job_store.list_all_jobs() == []
    assert runner.calls == []


def test_start_optimization_rejects_walk_forward_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text(
        "strategy:\n  id: fixture\nvalidation:\n  mode: automatic\n  n_splits: 2\n",
        encoding="utf-8",
    )
    service, job_store, runner = _service(
        tmp_path,
        metadata={"status": "runnable", "config_path": str(config_path)},
    )
    response = service.start_optimization(
        strategy_id="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        param_ranges={"window": [10]},
        optimization_metric="total_return",
        train_start="2024-01-01",
        train_end="2024-01-03",
        test_start="2024-01-04",
        test_end="2024-01-07",
    )

    assert response["job_id"] == ""
    assert response["code"] == "invalid_optimization_request"
    assert "must not contain validation settings" in response["message"]
    assert job_store.list_all_jobs() == []
    assert runner.calls == []


def test_start_optimization_rejects_invalid_strategy_parameters_before_job(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text(
        "strategy:\n  id: fixture\nbacktest:\n  symbol: ETH/USDT\n"
        "  timeframe: 4h\n  start_date: '2024-01-01'\n"
        "  end_date: '2024-12-31'\n  init_cash: 10000\n  periods_per_year: 365\n",
        encoding="utf-8",
    )
    service, job_store, runner = _service(
        tmp_path,
        metadata={"status": "runnable", "config_path": str(config_path)},
    )

    response = service.start_optimization(
        strategy_id="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        param_ranges={"windwo": [10]},
        optimization_metric="sharpe_ratio",
        train_start="2024-01-01",
        train_end="2024-02-01",
        test_start="2024-02-02",
        test_end="2024-03-01",
    )

    assert response["job_id"] == ""
    assert response["code"] == "invalid_optimization_request"
    assert "windwo" in response["message"]
    assert job_store.list_all_jobs() == []
    assert runner.calls == []
