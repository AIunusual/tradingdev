"""Real plan storage and MCP dispatch with substituted offline validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tests.integration.execution_fixtures import research_arguments
from tests.research_evidence_fixtures import save_research_evidence

from tradingdev.adapters.execution.process_runner import ProcessRunner, WorkerHandle
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.execution_plan_service import ExecutionPlanService
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.preflight_service import PreflightResult, PreflightService
from tradingdev.app.prepared_execution import PreparedExecution
from tradingdev.domain.preflight import (
    PreflightCheck,
    PreflightReceipt,
    PreflightRequest,
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
    """Protocol tests substitute code checks and never claim historical execution."""

    def __init__(self, prepared: PreparedExecution) -> None:
        self.prepared = prepared
        self.requests: list[PreflightRequest] = []

    def prepare(self, request: PreflightRequest) -> PreflightResult:
        self.requests.append(request)
        checked_paths: list[PreflightCheck] = [
            "configuration",
            "execution_capabilities",
            "parameter_binding",
            "manifest_serialization",
        ]
        generated = self.prepared.manifest.strategy_execution.kind == "generated"
        if generated:
            checked_paths.append("signal_contract")
        return PreflightResult(
            self.prepared,
            PreflightReceipt(
                manifest_hash=self.prepared.manifest.manifest_hash,
                elapsed_seconds=0.5,
                checked_paths=checked_paths,
                synthetic_fixture_rows=[80, 240] if generated else [],
                signal_check=(
                    "synthetic_contract_passed" if generated else "bundled_not_executed"
                ),
                training_requirement="unknown" if generated else "optional",
                not_checked=[
                    "historical_data",
                    "backtest_engine",
                    "training",
                    "performance",
                ],
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
            **research_arguments(arguments),
            "evidence_id": save_research_evidence(service.evidence, prepared.manifest),
        },
    )
