"""TradingDev MCPServer server."""

from __future__ import annotations

import argparse
from pathlib import Path

from mcp.server.transport_security import TransportSecuritySettings

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import get_sqlite_store
from tradingdev.app.artifact_service import ArtifactService
from tradingdev.app.data_service import DataService
from tradingdev.app.execution_plan_service import ExecutionPlanService
from tradingdev.app.feature_request_service import FeatureRequestService
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.ml_service import MLService
from tradingdev.app.report_service import ReportService
from tradingdev.app.run_service import RunService
from tradingdev.app.strategy_cleanup_service import StrategyCleanupService
from tradingdev.app.strategy_service import StrategyService
from tradingdev.app.trade_history_service import TradeHistoryService
from tradingdev.mcp.prompts import SERVER_INSTRUCTIONS
from tradingdev.mcp.strict_server import StrictMCPServer
from tradingdev.mcp.tools import (
    artifacts,
    data,
    execution,
    feature_requests,
    history,
    jobs,
    ml,
    runs,
    strategy,
    strategy_cleanup,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
HTTP_TRANSPORT_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=False,
)


def create_server(workspace: WorkspacePaths | None = None) -> StrictMCPServer:
    """Compose an MCP server with one explicitly shared runtime workspace."""
    workspace = workspace or WorkspacePaths()
    store = get_sqlite_store(workspace)
    mcp = StrictMCPServer(
        name="tradingdev",
        instructions=SERVER_INSTRUCTIONS,
    )
    strategy_service = StrategyService(workspace, store=store)
    data_service = DataService(workspace)
    job_store = JobStore(workspace=workspace, store=store)
    job_service = JobService(
        strategy_service=strategy_service,
        data_service=data_service,
        job_store=job_store,
    )
    execution_service = ExecutionPlanService(workspace, job_store=job_store)
    run_service = RunService(workspace=workspace, store=store)
    artifact_service = ArtifactService(workspace=workspace, store=store)
    feature_request_service = FeatureRequestService(workspace=workspace, store=store)

    strategy.register(mcp, strategy_service, _PACKAGE_ROOT)
    strategy_cleanup.register(mcp, StrategyCleanupService(workspace, store=store))
    data.register(mcp, data_service)
    execution.register(mcp, execution_service)
    jobs.register(mcp, job_service)
    runs.register(mcp, run_service)
    history.register(
        mcp,
        TradeHistoryService(workspace=workspace, store=store),
        ReportService(workspace=workspace, store=store),
    )
    artifacts.register(mcp, artifact_service)
    feature_requests.register(mcp, feature_request_service)
    ml.register(mcp, MLService(workspace=workspace, data_service=data_service))
    return mcp


def main(argv: list[str] | None = None) -> None:
    """Run the TradingDev MCP server."""
    parser = argparse.ArgumentParser(description="TradingDev MCP Server")
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Runtime workspace. Defaults to TRADINGDEV_WORKSPACE or ./workspace.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Run as HTTP server. Default: stdio.",
    )
    parser.add_argument(
        "--transport",
        choices=["sse", "streamable-http"],
        default="streamable-http",
        help="HTTP transport when --web is set.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP port when --web is set.",
    )
    args = parser.parse_args(argv)
    mcp = create_server(WorkspacePaths(args.workspace))

    if args.web and args.transport == "streamable-http":
        mcp.run(
            transport="streamable-http",
            port=args.port,
            # Preserve SSE streaming for clients using the legacy protocol.
            json_response=False,
            transport_security=HTTP_TRANSPORT_SECURITY,
        )
    elif args.web:
        mcp.run(
            transport="sse",
            port=args.port,
            transport_security=HTTP_TRANSPORT_SECURITY,
        )
    else:
        mcp.run()


if __name__ == "__main__":
    main()
