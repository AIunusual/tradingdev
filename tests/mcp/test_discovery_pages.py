"""Bounded discovery remains complete and points to unabridged detail reads."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest
from jsonschema import Draft202012Validator
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.job_store import JobStore
from tradingdev.app.strategy_service import StrategyService
from tradingdev.mcp.server import create_server

if TYPE_CHECKING:
    from pathlib import Path

    from mcp.server import MCPServer

_TOOLS = (
    "list_strategies",
    "list_runs",
    "list_jobs",
    "list_available_data",
    "list_artifacts",
)
_LARGE = "unabridged detail " * 5000


async def _page(server: MCPServer, name: str, **arguments: object) -> dict[str, Any]:
    result = await server.call_tool(name, arguments)
    assert isinstance(result, CallToolResult) and result.structured_content is not None
    page = result.structured_content
    assert isinstance(page, dict)
    assert set(page) == {"items", "offset", "limit", "total", "next_offset"}
    schema = next(
        tool.output_schema for tool in await server.list_tools() if tool.name == name
    )
    assert schema is not None
    Draft202012Validator(schema).validate(page)
    return page


def _seed(
    workspace: WorkspacePaths, tool: str
) -> tuple[dict[str, object], str, list[str]]:
    store = SQLiteStore(workspace)
    strategies = StrategyService(workspace, store)
    jobs = JobStore(workspace=workspace, store=store)
    artifact_path = workspace.root / "detail.txt"
    artifact_path.write_text("original file", encoding="utf-8")
    for index in reversed(range(105)):
        identity = f"item_{index:03d}"
        created_at = f"2024-01-0{2 if index % 2 == 0 else 1}T00:00:00+00:00"
        if tool == "list_strategies":
            assert strategies.save_draft(
                identity,
                "class PageStrategy: pass\n",
                "strategy:\n  class_name: PageStrategy\n",
                request_summary=_LARGE,
            ).success
        elif tool == "list_runs":
            store.create_run(
                run_id=identity,
                job_id=identity,
                strategy_id="page_strategy",
                revision_id="revision",
                artifact_dir=workspace.root,
                metrics={"large": _LARGE},
            )
            with store.connect() as conn:
                conn.execute(
                    "update runs set created_at = ? where run_id = ?",
                    (created_at, identity),
                )
        elif tool == "list_jobs":
            jobs.create_job(
                job_id=identity,
                strategy_name="page_strategy",
                job_type="optimization",
                extra_payload={"metrics": {"large": _LARGE}},
            )
            jobs.update_job(identity, created_at=created_at, completed=3)
        elif tool == "list_artifacts":
            store.create_artifact(
                artifact_id=identity,
                run_id=None,
                artifact_type="page_fixture",
                path=artifact_path,
                metadata={"large": _LARGE},
            )
            with store.connect() as conn:
                conn.execute(
                    "update artifacts set created_at = ? where artifact_id = ?",
                    (created_at, identity),
                )
        else:
            (workspace.processed_data / f"market{index:03d}_1h_2024.parquet").touch()
    if tool == "list_strategies":
        return (
            {"kind": "generated"},
            "strategy_id",
            [f"item_{i:03d}" for i in range(105)],
        )
    if tool == "list_available_data":
        return {"timeframe": "1h"}, "symbol", [f"MARKET{i:03d}" for i in range(105)]
    key = {
        "list_runs": "run_id",
        "list_jobs": "job_id",
        "list_artifacts": "artifact_id",
    }[tool]
    filter_choices: dict[str, dict[str, object]] = {
        "list_runs": {"strategy_id": "page_strategy", "revision_id": "revision"},
        "list_jobs": {
            "status": "queued",
            "job_type": "optimization",
            "strategy_name": "page_strategy",
        },
        "list_artifacts": {"artifact_type": "page_fixture"},
    }
    return (
        filter_choices[tool],
        key,
        [f"item_{i:03d}" for parity in (0, 1) for i in range(105) if i % 2 == parity],
    )


@pytest.mark.parametrize("tool", _TOOLS)
def test_discovery_pages_preserve_all_matches_and_hide_large_details(
    tmp_path: Path, tool: str
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    workspace.ensure()
    filters, key, expected = _seed(workspace, tool)
    server = create_server(workspace)

    async def check() -> None:
        first = await _page(server, tool, **filters)
        assert first["offset"] == 0 and first["limit"] == 20
        assert first["total"] == 105 and first["next_offset"] == 20
        assert len(first["items"]) == 20
        assert "unabridged detail" not in json.dumps(first)
        received = list(first["items"])
        offset = first["next_offset"]
        while offset is not None:
            page = await _page(server, tool, **filters, offset=offset)
            assert page["total"] == 105
            received.extend(page["items"])
            offset = page["next_offset"]
        assert [item[key] for item in received] == expected
        assert len({item[key] for item in received}) == 105
        maximum = await _page(server, tool, **filters, limit=100)
        assert len(maximum["items"]) == 100 and maximum["next_offset"] == 100
        last = await _page(server, tool, **filters, offset=100, limit=100)
        assert len(last["items"]) == 5 and last["next_offset"] is None
        for offset in (105, 1000):
            beyond = await _page(server, tool, **filters, offset=offset)
            assert beyond["items"] == [] and beyond["total"] == 105
            assert beyond["next_offset"] is None

        if tool == "list_strategies":
            assert all(
                item["status"] == "draft" and item["revision_id"] for item in received
            )
            assert all("execution_capabilities" in item for item in received)
            assert "metadata" not in first["items"][0]
            detail = await server.call_tool("get_strategy", {"strategy_id": "item_000"})
            assert (
                isinstance(detail, CallToolResult)
                and detail.structured_content is not None
            )
            assert (
                detail.structured_content["result"]["metadata"]["request_summary"]
                == _LARGE
            )
            narrowed = await _page(server, tool, strategy_id="item_000")
            assert narrowed["total"] == 1
        elif tool == "list_artifacts":
            assert "metadata" not in first["items"][0]
            detail = await server.call_tool("get_artifact", {"artifact_id": "item_000"})
            assert (
                isinstance(detail, CallToolResult)
                and detail.structured_content is not None
            )
            assert (
                detail.structured_content["result"]["artifact"]["metadata"]["large"]
                == _LARGE
            )
            narrowed = await _page(server, tool, run_id="absent")
            assert narrowed["total"] == 0
        elif tool == "list_runs":
            assert "metrics" not in first["items"][0]
            detail = await server.call_tool("get_run", {"run_id": "item_000"})
            assert (
                isinstance(detail, CallToolResult)
                and detail.structured_content is not None
            )
            assert (
                detail.structured_content["result"]["run"]["metrics"]["large"] == _LARGE
            )
            narrowed = await _page(server, tool, revision_id="absent")
            assert narrowed["total"] == 0
        elif tool == "list_jobs":
            assert first["items"][0]["completed"] == 3
            assert "metrics" not in first["items"][0]
            narrowed = await _page(server, tool, status="done")
            assert narrowed["total"] == 0
        else:
            narrowed = await _page(server, tool, symbol="market/001", timeframe="1h")
            assert (
                narrowed["total"] == 1 and narrowed["items"][0]["symbol"] == "MARKET001"
            )

    asyncio.run(check())


@pytest.mark.parametrize("tool", _TOOLS)
def test_empty_discovery_pages_and_invalid_input_failures(
    tmp_path: Path, tool: str
) -> None:
    server = create_server(WorkspacePaths(tmp_path / "workspace"))

    async def check() -> None:
        filters = {"kind": "generated"} if tool == "list_strategies" else {}
        page = await _page(server, tool, **filters)
        assert page == {
            "items": [],
            "offset": 0,
            "limit": 20,
            "total": 0,
            "next_offset": None,
        }
        for arguments in (
            {"offset": -1},
            {"offset": True},
            {"offset": "0"},
            {"offset": 0.5},
            {"limit": 0},
            {"limit": 101},
            {"limit": False},
            {"limit": "20"},
            {"limit": 1.5},
        ):
            with pytest.raises(ToolError):
                await server.call_tool(tool, arguments)
        if tool == "list_strategies":
            with pytest.raises(ToolError):
                await server.call_tool(tool, {"kind": "unknown"})
        if tool == "list_jobs":
            with pytest.raises(ToolError):
                await server.call_tool(tool, {"status": "unknown"})

    asyncio.run(check())
