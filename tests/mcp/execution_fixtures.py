"""Real plan storage and MCP dispatch with an explicitly substituted sample worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tests.integration.execution_fixtures import execution_options

from tradingdev.adapters.execution.process_runner import ProcessRunner, WorkerHandle
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.execution_plan_service import ExecutionPlanService
from tradingdev.app.execution_submission import PreparedExecution
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.preflight_service import PreflightResult, PreflightService
from tradingdev.domain.preflight import (
    PreflightCheck,
    PreflightReceipt,
    PreflightRequest,
    PreflightWindow,
)
from tradingdev.mcp.strict_server import StrictMCPServer
from tradingdev.mcp.tools import execution

if TYPE_CHECKING:
    from pathlib import Path


class RecordingRunner(ProcessRunner):
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def spawn_module(self, module: str, *arguments: str) -> WorkerHandle:
        self.calls.append((module, arguments))
        return WorkerHandle(999999, 123.0, "a" * 32)


class FixedPreflight(PreflightService):
    """Protocol tests do not claim to execute a real sample engine."""

    def __init__(self, prepared: PreparedExecution) -> None:
        self.prepared = prepared
        self.requests: list[PreflightRequest] = []

    def prepare(self, request: PreflightRequest) -> PreflightResult:
        self.requests.append(request)
        checked_paths: list[PreflightCheck] = [
            "configuration",
            "signals",
            "engine",
            "serialization",
        ]
        if self.prepared.manifest.strategy_execution.kind == "generated":
            checked_paths.append("signal_contract")
        return PreflightResult(
            self.prepared,
            PreflightReceipt(
                manifest_hash=self.prepared.manifest.manifest_hash,
                elapsed_seconds=0.5,
                sample_bars_requested=request.sample_bars,
                sample_bars_used=128,
                minimum_history_bars=request.minimum_history_bars,
                data_source=self.prepared.manifest.config_copy()["data"][
                    "requirements"
                ]["market"]["source"],
                windows=[
                    PreflightWindow(
                        role="full",
                        start="2024-01-01",
                        end="2024-01-06",
                        rows=128,
                    )
                ],
                checked_paths=checked_paths,
                trade_count=1,
                execution_record_count=2,
                trading_path_exercised=True,
            ),
        )


@dataclass
class PlanContext:
    server: StrictMCPServer
    service: ExecutionPlanService
    runner: RecordingRunner
    preflight: FixedPreflight
    arguments: dict[str, Any]


def plan_context(root: Path) -> PlanContext:
    workspace = WorkspacePaths(root / "workspace")
    store = JobStore(workspace=workspace)
    arguments: dict[str, Any] = {
        "strategy_id": "kd_crossover",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-08",
    }
    prepared = JobService(job_store=store).prepare_backtest(**arguments)
    assert isinstance(prepared, PreparedExecution), prepared
    runner = RecordingRunner()
    preflight = FixedPreflight(prepared)
    service = ExecutionPlanService(
        workspace,
        job_store=store,
        process_runner=runner,
        preflight=preflight,
    )
    server = StrictMCPServer("plan-contracts")
    execution.register(server, service)
    return PlanContext(
        server,
        service,
        runner,
        preflight,
        {
            **arguments,
            **execution_options(
                prepared.manifest.strategy_execution.constructor_kwargs
            ),
        },
    )
