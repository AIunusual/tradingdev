"""MCP exploration entry points share one paired-comparison service."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, cast

import pytest
from jsonschema import Draft202012Validator
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from tests.app.test_paired_comparison import _publish_run

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.ml_service import MLService
from tradingdev.app.paired_comparison_service import PairedComparisonService
from tradingdev.app.run_service import RunService
from tradingdev.mcp.strict_server import StrictMCPServer
from tradingdev.mcp.tools import ml, runs

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(params=[MCPServer, StrictMCPServer], ids=["sdk", "strict"])
def server(
    tmp_path: Path, request: pytest.FixtureRequest
) -> tuple[MCPServer, WorkspacePaths]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    paired = PairedComparisonService(workspace=workspace, store=store)
    instance: MCPServer = request.param("paired-contract-test")
    runs.register(instance, RunService(workspace=workspace, store=store), paired)
    ml.register(
        instance,
        MLService(workspace=workspace),
        paired,
    )
    _publish_run(workspace, store, "run-a", returns=[0.01] * 30)
    _publish_run(workspace, store, "run-b", returns=[0.02] * 30)
    return instance, workspace


async def _call(
    server: MCPServer, name: str, arguments: dict[str, object]
) -> dict[str, Any]:
    result = await server.call_tool(name, arguments)
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        inner = structured.get("result")
        if isinstance(inner, dict):
            return cast("dict[str, Any]", inner)
        return cast("dict[str, Any]", structured)
    content = list(getattr(result, "content", ()))
    assert content
    text = getattr(content[0], "text", None)
    assert isinstance(text, str)
    return cast("dict[str, Any]", json.loads(text))


def test_tools_advertise_constrained_schemas(
    server: tuple[MCPServer, WorkspacePaths],
) -> None:
    instance, _ = server

    async def check() -> None:
        tools = {tool.name: tool for tool in await instance.list_tools()}
        for name in ("compare_saved_series", "compare_signal_series"):
            assert name in tools
            tool = tools[name]
            assert tool.output_schema is not None
            Draft202012Validator.check_schema(tool.input_schema)
            Draft202012Validator.check_schema(tool.output_schema)
            hints = tools[name].annotations
            assert hints is not None
            assert (
                hints.read_only_hint,
                hints.destructive_hint,
                hints.idempotent_hint,
                hints.open_world_hint,
            ) == (False, False, True, False)

    asyncio.run(check())


def test_compare_saved_series_through_real_dispatch(
    server: tuple[MCPServer, WorkspacePaths],
) -> None:
    instance, _ = server

    async def check() -> None:
        payload = await _call(
            instance,
            "compare_saved_series",
            {
                "a": {"kind": "run", "run_id": "run-a"},
                "b": {"kind": "run", "run_id": "run-b"},
                "mode": "descriptive",
            },
        )
        assert payload["success"] is True
        assert payload["status"] == "available"
        assert payload["result"]["estimate"] < 0
        assert payload["alignment"]["valid_observations"] == 30
        # An incomplete selector fails input validation before any service
        # call, so it is an operation error rather than a data-driven absence.
        with pytest.raises(ToolError, match="signal sources require"):
            await _call(
                instance,
                "compare_saved_series",
                {
                    "a": {"kind": "signal"},
                    "b": {"kind": "run", "run_id": "run-b"},
                },
            )

    asyncio.run(check())
