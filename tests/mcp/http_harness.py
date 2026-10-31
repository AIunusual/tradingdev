"""Real localhost HTTP transport with substituted sample and worker execution."""

from __future__ import annotations

import asyncio
import shutil
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Literal

import uvicorn
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from sse_starlette import sse
from tests.integration.mcp_harness import MCPClient
from tests.mcp.execution_fixtures import PlanContext, plan_context

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.mcp import server as server_module

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import pytest


@dataclass
class HTTPPlanServer:
    root: Path
    context: PlanContext
    server: uvicorn.Server
    task: asyncio.Task[None]
    listener: socket.socket
    address: tuple[str, int]
    shutdown_watchers: set[asyncio.Task[None]]
    transport: Literal["sse", "streamable-http"] = "streamable-http"

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[MCPClient]:
        host, port = self.address
        connection = (
            sse_client(f"http://{host}:{port}/sse")
            if self.transport == "sse"
            else streamable_http_client(f"http://{host}:{port}/mcp")
        )
        async with (
            connection as (read, write),
            ClientSession(
                read,
                write,
                read_timeout_seconds=5.0,
            ) as session,
        ):
            await session.initialize()
            tools = (await session.list_tools()).tools
            schemas = {
                tool.name: tool.output_schema
                for tool in tools
                if tool.output_schema is not None
            }
            yield MCPClient(session, output_schemas=schemas)


@asynccontextmanager
async def http_plan_server(
    monkeypatch: pytest.MonkeyPatch,
    *,
    transport: Literal["sse", "streamable-http"] = "streamable-http",
    allowed_hosts: tuple[str, ...] = (),
    allowed_origins: tuple[str, ...] = (),
) -> AsyncIterator[HTTPPlanServer]:
    root = Path(mkdtemp(prefix="tradingdev-http-test-")).resolve()
    task: asyncio.Task[None] | None = None
    shutdown_watchers: set[asyncio.Task[None]] = set()
    original_watcher = sse._shutdown_watcher

    async def tracked_shutdown_watcher() -> None:
        watcher = asyncio.current_task()
        assert watcher is not None
        shutdown_watchers.add(watcher)
        await original_watcher()

    stopped = True
    try:
        context = plan_context(root)
        # Exercise production composition, including its transport settings. The
        # fixture substitutes only sample execution and spawning a real worker.
        with monkeypatch.context() as patch:
            patch.setattr(
                server_module,
                "ExecutionPlanService",
                lambda *args, **kwargs: context.service,
            )
            mcp = server_module.create_server(WorkspacePaths(root / "workspace"))
        # sse-starlette keeps process-wide shutdown state and starts a detached
        # watcher. Scope both to this in-process server, including teardown, so
        # a previous server cannot immediately terminate the next SSE response.
        with monkeypatch.context() as patch, socket.socket() as listener:
            patch.setattr(sse.AppStatus, "should_exit", False)
            patch.setattr(sse, "_shutdown_watcher", tracked_shutdown_watcher)
            listener.bind(("127.0.0.1", 0))
            address = listener.getsockname()
            security = server_module.http_transport_security(
                address[1],
                allowed_hosts=allowed_hosts,
                allowed_origins=allowed_origins,
            )
            app = (
                mcp.sse_app(transport_security=security)
                if transport == "sse"
                else mcp.streamable_http_app(
                    json_response=False,
                    transport_security=security,
                )
            )
            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    log_level="error",
                    lifespan="on",
                    timeout_graceful_shutdown=2,
                )
            )
            task = asyncio.create_task(server.serve(sockets=[listener]))
            stopped = False
            try:
                async with asyncio.timeout(5):
                    while not server.started:
                        if task.done():
                            await task
                            raise RuntimeError("HTTP test server exited during startup")
                        await asyncio.sleep(0.01)
                yield HTTPPlanServer(
                    root,
                    context,
                    server,
                    task,
                    listener,
                    address,
                    shutdown_watchers,
                    transport,
                )
            finally:
                server.should_exit = True
                sse.AppStatus.should_exit = True
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
                if shutdown_watchers:
                    await asyncio.wait_for(
                        asyncio.gather(*shutdown_watchers), timeout=5
                    )
                assert not server.server_state.tasks
                assert not server.server_state.connections
                stopped = True
    finally:
        if not stopped:
            pending = [watcher for watcher in shutdown_watchers if not watcher.done()]
            if task is not None and not task.done():
                pending.append(task)
            for pending_task in pending:
                pending_task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise RuntimeError(
                f"HTTP test server cleanup failed; temporary directory retained: {root}"
            )
        shutil.rmtree(root)
        assert not root.exists()
