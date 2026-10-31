"""TradingDev MCPServer server."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

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
from tradingdev.app.research_note_service import ResearchNoteService
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
    research_notes,
    runs,
    strategy,
    strategy_cleanup,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]

if TYPE_CHECKING:
    from collections.abc import Sequence


def http_transport_security(
    port: int = 8000,
    *,
    allowed_hosts: Sequence[str] = (),
    allowed_origins: Sequence[str] = (),
) -> TransportSecuritySettings:
    """Allow local HTTP names and explicitly configured exact proxy values."""
    if not 1 <= port <= 65535:
        raise ValueError("HTTP port must be between 1 and 65535")
    for host in allowed_hosts:
        parsed = urlsplit(f"//{host}")
        if (
            not parsed.hostname
            or parsed.netloc != host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
            or "*" in host
            or "\\" in host
            or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in host)
        ):
            raise ValueError(f"Allowed Host must be an exact hostname[:port]: {host!r}")
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError(f"Invalid allowed Host port: {host!r}")
    for origin in allowed_origins:
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or f"{parsed.scheme}://{parsed.netloc}" != origin
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
            or "*" in origin
            or "\\" in origin
            or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in origin)
        ):
            raise ValueError(
                f"Allowed Origin must be an exact http(s)://host[:port]: {origin!r}"
            )
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError(f"Invalid allowed Origin port: {origin!r}")
    hosts = ["127.0.0.1", "localhost", "[::1]"]
    hosts += [f"{host}:{port}" for host in hosts.copy()]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(dict.fromkeys([*hosts, *allowed_hosts])),
        allowed_origins=list(
            dict.fromkeys(
                [
                    *(f"http://{host}" for host in hosts),
                    *allowed_origins,
                ]
            )
        ),
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
    research_notes.register(mcp, ResearchNoteService(workspace, store))
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
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        help="Additional exact HTTP Host (hostname[:port]); repeat as needed.",
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="Additional exact Origin (http(s)://host[:port]); repeat as needed.",
    )
    args = parser.parse_args(argv)
    if not args.web and (args.allow_host or args.allow_origin):
        parser.error("--allow-host and --allow-origin require --web")
    try:
        security = (
            http_transport_security(
                args.port,
                allowed_hosts=args.allow_host,
                allowed_origins=args.allow_origin,
            )
            if args.web
            else None
        )
    except ValueError as error:
        parser.error(str(error))
    mcp = create_server(WorkspacePaths(args.workspace))

    if args.web and args.transport == "streamable-http":
        mcp.run(
            transport="streamable-http",
            host="127.0.0.1",
            port=args.port,
            # Preserve SSE streaming for clients using the legacy protocol.
            json_response=False,
            transport_security=security,
        )
    elif args.web:
        mcp.run(
            transport="sse",
            host="127.0.0.1",
            port=args.port,
            transport_security=security,
        )
    else:
        mcp.run()


if __name__ == "__main__":
    main()
