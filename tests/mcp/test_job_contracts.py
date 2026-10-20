"""Job contract regressions at the MCPServer boundary."""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import pytest
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.execution_submission import PreparedExecution
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.optimization_service import OptimizationService
from tradingdev.mcp.tools import jobs

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def job_service(tmp_path: Path) -> Iterator[JobService]:
    with TemporaryDirectory(dir=tmp_path) as directory:
        workspace = WorkspacePaths(Path(directory))
        store = JobStore(workspace=workspace)
        yield JobService(job_store=store)


@pytest.mark.parametrize("corruption", ["extra_field", "unknown_state", "missing_code"])
def test_job_output_rejects_unknown_fields_and_invalid_state(
    job_service: JobService, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    server = MCPServer("contract-test")
    jobs.register(server, job_service)
    # A service regression must fail at the boundary, not silently publish an
    # arbitrary object or discard an internal field the client should not see.
    malformed = {
        "status": "not_found",
        "error": "missing",
        "code": "job_not_found",
    }
    if corruption == "extra_field":
        malformed["worker_control_id"] = "private"
    elif corruption == "unknown_state":
        malformed["status"] = "unexpected"
    else:
        del malformed["code"]
    monkeypatch.setattr(job_service, "get_job_status", lambda _: malformed)
    with pytest.raises(ToolError):
        asyncio.run(server.call_tool("get_job_status", {"job_id": "missing"}))


@pytest.mark.parametrize(
    "state",
    [
        "queued",
        "downloading_data",
        "running_backtest",
        "estimating",
        "optimizing",
        "testing_oos",
        "done",
        "failed",
        "cancelled",
        "estimation_timeout",
    ],
)
def test_job_lifecycle_payloads_satisfy_advertised_contract(
    job_service: JobService, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    # Exercise the real service formatting for every worker state without
    # starting a worker. Process liveness is the only substituted dependency.
    store = job_service._job_store
    store.create_job(
        job_id="job",
        strategy_name="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
        config_path="missing.yaml",
    )
    store.update_job(
        "job",
        status=state,
        job_type="optimization",
        total_combinations=2,
        completed=1,
        time_per_combo=0.5,
        estimated_total_seconds=1.0,
        n_parallel_workers=1,
        estimated_remaining_seconds=0.5,
        error="fixture failure" if state == "failed" else None,
    )
    if state == "done":
        result_path = store.save_result(
            "job",
            {
                "best_params": {"period": 3},
                "train_metrics": {"sharpe_ratio": 0.0},
                "test_metrics": {"sharpe_ratio": None},
                "optimization_metric": "sharpe_ratio",
                "direction": "maximize",
                "total_combinations": 2,
            },
        )
        store.update_job("job", result_path=str(result_path))
    monkeypatch.setattr(job_service, "_is_process_alive", lambda _: True)
    server = MCPServer("contract-test")
    jobs.register(server, job_service)
    result = asyncio.run(server.call_tool("get_job_status", {"job_id": "job"}))
    assert isinstance(result, CallToolResult)
    assert result.structured_content is not None
    payload: Any = result.structured_content["result"]
    assert payload["status"] == state
    assert payload["manifest_hash"] is None
    if state == "done":
        assert payload["direction"] == "maximize"
        assert payload["train_metrics"]["sharpe_ratio"] == 0.0
        assert payload["test_metrics"]["sharpe_ratio"] is None


def test_cancel_contract_does_not_claim_success_when_identity_missing(
    job_service: JobService,
) -> None:
    store = job_service._job_store
    store.create_job(
        job_id="job",
        strategy_name="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
        config_path="missing.yaml",
    )
    store.update_job("job", status="running_backtest", pid=12345)
    server = MCPServer("contract-test")
    jobs.register(server, job_service)
    result = asyncio.run(server.call_tool("cancel_job", {"job_id": "job"}))
    assert isinstance(result, CallToolResult)
    assert result.structured_content is not None
    payload = result.structured_content["result"]
    assert not payload["success"]
    assert payload["code"] == "worker_identity_unavailable"
    assert "process_terminated" not in payload
    persisted = store.get_job("job")
    assert persisted is not None and persisted["status"] == "running_backtest"


def test_old_confirmation_tool_is_no_longer_registered(job_service: JobService) -> None:
    server = MCPServer("retired-confirmation")
    jobs.register(server, job_service)
    with pytest.raises(ToolError, match="Unknown tool"):
        asyncio.run(server.call_tool("confirm_optimization", {"job_id": "legacy"}))


@pytest.mark.parametrize("metric", ["max_drawdown", "daily_pnl_mean"])
def test_optimization_preparation_preserves_catalog_objective_direction(
    job_service: JobService, metric: str
) -> None:
    store = job_service._job_store
    prepared = OptimizationService(job_store=store).prepare_optimization(
        strategy_id="kd_crossover",
        symbol="BTC/USDT",
        timeframe="1h",
        param_ranges={"k_period": [3, 5]},
        optimization_metric=metric,
        train_start="2024-01-01",
        train_end="2024-01-03",
        test_start="2024-01-04",
        test_end="2024-01-07",
    )
    assert isinstance(prepared, PreparedExecution), prepared
    expected = "minimize" if metric == "max_drawdown" else "maximize"
    manifest = prepared.manifest
    assert manifest.optimization is not None
    assert manifest.optimization.optimization_metric == metric
    assert manifest.optimization.direction == expected
    assert store.list_all_jobs() == []
