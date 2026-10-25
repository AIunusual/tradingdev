"""Real MCP subprocess fixtures with eager cleanup, including failed tests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys
from contextlib import asynccontextmanager, closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

import anyio
import psutil
from jsonschema import validate
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp_types import ElicitResult, PaginatedRequestParams

from tradingdev.adapters.execution.process_runner import (
    WorkerHandle,
    request_worker_stop,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator
    from typing import Literal

    import pandas as pd
    from mcp.client.session import ClientRequestContext, ElicitationFnT
    from mcp_types import ElicitRequestParams


@dataclass
class SimulatedUserApproval:
    """An explicit test user callback, never evidence of a real human UI action."""

    action: Literal["accept", "decline", "cancel"] = "accept"
    approved: bool = True
    requests: list[ElicitRequestParams] = field(default_factory=list)

    async def __call__(
        self,
        context: ClientRequestContext,
        params: ElicitRequestParams,
    ) -> ElicitResult:
        return await self.respond(params)

    async def respond(self, params: ElicitRequestParams) -> ElicitResult:
        """Record and answer a form from either supported test transport."""
        self.requests.append(params)
        return ElicitResult(
            action=self.action,
            content={"approved": self.approved} if self.action == "accept" else None,
        )


@dataclass
class MCPClient:
    session: ClientSession
    instructions: str = ""
    output_schemas: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def call(self, name: str, **arguments: Any) -> Any:
        result = await self.session.call_tool(name, arguments)
        assert not result.is_error, result.model_dump(mode="json", by_alias=True)
        assert result.structured_content is not None, result
        payload = result.structured_content
        assert name in self.output_schemas, f"No advertised outputSchema for {name}"
        validate(instance=payload, schema=self.output_schemas[name])
        return payload["result"] if set(payload) == {"result"} else payload

    async def read_all_pages(
        self, name: str, *, limit: int = 20, **arguments: Any
    ) -> list[dict[str, Any]]:
        """Explicitly consume every list page while the test workspace is stable."""
        items: list[dict[str, Any]] = []
        offset = 0
        total: int | None = None
        while True:
            page = await self.call(name, offset=offset, limit=limit, **arguments)
            assert set(page) == {"items", "offset", "limit", "total", "next_offset"}
            assert page["offset"] == offset and page["limit"] == limit
            if total is None:
                total = page["total"]
            assert page["total"] == total, "Test inventory changed during pagination"
            chunk = page["items"]
            assert isinstance(chunk, list) and len(chunk) <= limit
            items.extend(chunk)
            next_offset = page["next_offset"]
            if next_offset is None:
                break
            assert next_offset == offset + len(chunk) > offset
            offset = next_offset
        assert len(items) == total, "Pagination ended before returning every item"
        return items

    async def wait_for_job(self, job_id: str, timeout: float = 120) -> dict[str, Any]:
        with anyio.fail_after(timeout):
            while True:
                status = await self.call("get_job_status", job_id=job_id)
                if status["status"] in {"done", "failed", "cancelled"}:
                    return dict(status)
                await anyio.sleep(0.1)

    async def read_artifact_text(self, artifact_id: str) -> str:
        """Read every page of one unchanged artifact through the actual MCP API."""
        chunks: list[str] = []
        offset = 0
        digest: str | None = None
        total: int | None = None
        while True:
            result = await self.call(
                "get_artifact",
                artifact_id=artifact_id,
                include_content=True,
                offset=offset,
                expected_sha256=digest,
            )
            assert result["success"], result
            page = result["content_page"]
            assert page["offset"] == offset
            if digest is None:
                digest = page["content_sha256"]
                total = page["total_characters"]
            assert page["content_sha256"] == digest
            assert page["total_characters"] == total
            chunk = result["content"]
            assert isinstance(chunk, str)
            chunks.append(chunk)
            next_offset = page["next_offset"]
            if next_offset is None:
                break
            assert next_offset == offset + len(chunk) > offset
            offset = next_offset
        content = "".join(chunks)
        assert len(content) == total
        assert hashlib.sha256(content.encode("utf-8")).hexdigest() == digest
        return content


@dataclass
class MCPWorkspace:
    root: Path

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def data_root(self) -> Path:
        # Deliberately different from workspace/data to check propagation.
        return self.root / "market-data"

    def seed_market(self, frame: pd.DataFrame) -> Path:
        path = self.data_root / "processed" / "btcusdt_1h_2024.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return path

    def environment(self) -> dict[str, str]:
        """Confine caches and block market network access in the backend."""
        source_root = Path(__file__).resolve().parents[2] / "src"
        env = {
            "PYTHONPATH": str(source_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TRADINGDEV_WORKSPACE": str(self.workspace),
            "TRADINGDEV_DATA_ROOT": str(self.data_root),
            "TRADINGDEV_PROJECT_ROOT": str(self.root),
            "MYPY_CACHE_DIR": str(self.root / "mypy-cache"),
            "RUFF_CACHE_DIR": str(self.root / "ruff-cache"),
            "NUMBA_CACHE_DIR": str(self.root / "numba-cache"),
            "MPLCONFIGDIR": str(self.root / "matplotlib"),
            "XDG_CACHE_HOME": str(self.root / "cache"),
            "UV_OFFLINE": "1",
        }
        # Test-only network guard; the MCP server and workers remain real.
        guard = self.root / "python-guard"
        guard.mkdir(exist_ok=True)
        (guard / "sitecustomize.py").write_text(
            "import sys\n"
            "def deny_network(event, args):\n"
            "    if event in {'socket.connect', 'socket.getaddrinfo'}:\n"
            "        raise RuntimeError('Network disabled in MCP integration tests')\n"
            "sys.addaudithook(deny_network)\n",
            encoding="utf-8",
        )
        env["PYTHONPATH"] = os.pathsep.join((str(guard), str(source_root)))
        return env

    @asynccontextmanager
    async def connect(
        self, *, elicitation_callback: ElicitationFnT | None = None
    ) -> AsyncGenerator[MCPClient]:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "tradingdev.mcp.server"],
            cwd=str(self.root),
            env={**os.environ, **self.environment()},
        )
        log_path = self.root / "server.log"
        try:
            with log_path.open("a", encoding="utf-8") as errlog:
                async with (
                    stdio_client(params, errlog=errlog) as (read, write),
                    ClientSession(
                        read,
                        write,
                        read_timeout_seconds=100.0,
                        elicitation_callback=elicitation_callback,
                    ) as session,
                ):
                    initialized = await session.initialize()
                    assert initialized.server_info.name == "tradingdev"
                    assert "save_strategy" in (initialized.instructions or "")
                    try:
                        output_schemas: dict[str, dict[str, Any]] = {}
                        cursor = None
                        while True:
                            listed = await session.list_tools(
                                params=PaginatedRequestParams(cursor=cursor)
                            )
                            for tool in listed.tools:
                                assert tool.output_schema is not None, tool.name
                                output_schemas[tool.name] = tool.output_schema
                            cursor = listed.next_cursor
                            if cursor is None:
                                break
                        yield MCPClient(
                            session,
                            initialized.instructions or "",
                            output_schemas,
                        )
                    finally:
                        self.stop_workers()
        except BaseException:
            # Capture the failure log before successful workspace cleanup.
            print(log_path.read_text(encoding="utf-8"))
            raise

    def stop_workers(self) -> None:
        """Ask workspace supervisors to stop their workers before deleting files."""
        database = self.workspace / "tradingdev.sqlite"
        failures: list[Exception] = []
        handles: set[WorkerHandle] = set()
        if database.exists():
            try:
                with closing(sqlite3.connect(database)) as connection:
                    workers = connection.execute(
                        "SELECT pid, payload FROM jobs WHERE pid IS NOT NULL"
                    ).fetchall()
            except Exception as error:
                failures.append(error)
            else:
                for pid, payload in workers:
                    try:
                        record = json.loads(payload)
                        if not isinstance(record, dict):
                            raise AssertionError(f"Invalid worker record: PID {pid}")
                        handle = WorkerHandle.from_job({**record, "pid": pid})
                        if handle is None:
                            raise AssertionError(
                                f"Worker control identity missing or invalid: PID {pid}"
                            )
                        handles.add(handle)
                    except Exception as error:
                        failures.append(error)

        # A supervisor may have started before identity capture or DB update.
        # Inspect launch directories too: absence of start is not proof of exit.
        try:
            launches = list((self.workspace / ".workers").iterdir())
        except FileNotFoundError:
            launches = []
        except Exception as error:
            launches = []
            failures.append(error)
        for launch in launches:
            try:
                start = launch / "start.json"
                try:
                    record = json.loads(start.read_text(encoding="utf-8"))
                except FileNotFoundError as error:
                    raise AssertionError(
                        f"Unidentified worker launch; start record missing: {launch}"
                    ) from error
                handle = (
                    WorkerHandle.from_job(record) if isinstance(record, dict) else None
                )
                if handle is None or handle.control_id != start.parent.name:
                    raise AssertionError(f"Invalid worker control record: {start}")
                handles.add(handle)
            except Exception as error:
                failures.append(error)

        for handle in handles:
            try:
                request_worker_stop(self.workspace, handle)
            except Exception as error:
                failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ExceptionGroup("Test worker cleanup failed", failures)


def worker_is_alive(process: psutil.Process) -> bool:
    """Check the recorded process identity without treating zombies as active."""
    try:
        return bool(process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


@contextmanager
def temporary_mcp_workspace() -> Generator[MCPWorkspace]:
    # Also used outside pytest; stop detached workers before removing files.
    directory = Path(mkdtemp(prefix="tradingdev-mcp-test-")).resolve()
    workspace = MCPWorkspace(directory)
    try:
        yield workspace
    finally:
        try:
            workspace.stop_workers()
        except Exception as error:
            raise RuntimeError(
                f"Worker cleanup failed; temporary directory retained: {directory}"
            ) from error
        shutil.rmtree(directory)
        assert not directory.exists()
