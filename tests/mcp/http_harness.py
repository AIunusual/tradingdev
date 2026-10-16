"""Real localhost HTTP transport with substituted sample and worker execution."""

from __future__ import annotations

import asyncio
import shutil
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING

import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
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

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[MCPClient]:
        host, port = self.address
        async with (
            streamable_http_client(f"http://{host}:{port}/mcp") as (read, write, _),
            ClientSession(
                read,
                write,
                read_timeout_seconds=timedelta(seconds=5),
            ) as session,
        ):
            await session.initialize()
            tools = (await session.list_tools()).tools
            schemas = {
                tool.name: tool.outputSchema
                for tool in tools
                if tool.outputSchema is not None
            }
            yield MCPClient(session, output_schemas=schemas)


@asynccontextmanager
async def http_plan_server(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[HTTPPlanServer]:
    root = Path(mkdtemp(prefix="tradingdev-http-test-")).resolve()
    task: asyncio.Task[None] | None = None
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
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            address = listener.getsockname()
            server = uvicorn.Server(
                uvicorn.Config(
                    mcp.streamable_http_app(),
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
                yield HTTPPlanServer(root, context, server, task, listener, address)
            finally:
                server.should_exit = True
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
                assert not server.server_state.tasks
                assert not server.server_state.connections
                stopped = True
    finally:
        if not stopped:
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            raise RuntimeError(
                f"HTTP test server cleanup failed; temporary directory retained: {root}"
            )
        shutil.rmtree(root)
        assert not root.exists()
