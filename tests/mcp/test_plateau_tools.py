"""Saved-grid analysis and continuation through both registered MCP servers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from jsonschema import Draft202012Validator
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from tests.app.test_parameter_plateau import _record
from tests.mcp.test_ml_leaderboard_tool import _dispatch

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.plateau_service import PlateauService
from tradingdev.mcp.strict_server import StrictMCPServer
from tradingdev.mcp.tools import plateau

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("server_type", [MCPServer, StrictMCPServer])
def test_registered_plateau_keeps_snapshot_and_schema(
    tmp_path: Path, server_type: type[MCPServer]
) -> None:
    workspace = WorkspacePaths(tmp_path)
    service = PlateauService(workspace=workspace, store=SQLiteStore(workspace))
    ids = _record(workspace, [0.5, None, 0.6])
    server = server_type("plateau-contract")
    plateau.register(server, service)
    args: dict[str, Any] = {
        "metric": "signal.ic_mean",
        "axes": {"/threshold": [1, 2, 3]},
        "entry_ids": ids,
        "limit": 1,
    }
    expected = service.analyze(**args)
    actual = _dispatch(server, "analyze_parameter_plateau", args)
    assert actual == expected
    _record(workspace, [8.0, 8.0, 8.0])
    continued = _dispatch(
        server,
        "get_parameter_plateau",
        {"analysis_id": actual["analysis_id"], "offset": 1, "limit": 1},
    )
    assert continued["source_hash"] == actual["source_hash"]
    assert continued["summary"] == actual["summary"]
    assert continued["cells"]["items"][0]["status"] == "undefined"
    assert continued["cells"]["next_offset"] == 2
    assert continued["artifact"] == actual["artifact"]
    refusal = _dispatch(
        server, "analyze_parameter_plateau", {**args, "entry_ids": ["missing"]}
    )
    assert refusal["success"] is False
    assert refusal["next_step"]

    async def schemas() -> dict[str, Any]:
        return {tool.name: tool.output_schema for tool in await server.list_tools()}

    for schema in asyncio.run(schemas()).values():
        Draft202012Validator.check_schema(schema)
        for result in (actual, continued, refusal):
            Draft202012Validator(schema).validate({"result": result})


@pytest.mark.parametrize(
    "bad",
    [
        {"axes": {"/threshold": [True]}},
        {"radius": 0},
        {"limit": 101},
        {"entry_ids": ["x"] * 1025},
    ],
)
def test_plateau_input_rejects_before_publication(
    tmp_path: Path, bad: dict[str, Any]
) -> None:
    workspace = WorkspacePaths(tmp_path)
    service = PlateauService(workspace=workspace, store=SQLiteStore(workspace))
    server = StrictMCPServer("plateau-input")
    plateau.register(server, service)

    async def rejected() -> None:
        with pytest.raises(ToolError):
            await server.call_tool(
                "analyze_parameter_plateau",
                {
                    "metric": "signal.ic_mean",
                    "axes": {"/threshold": [1]},
                    "entry_ids": [],
                    **bad,
                },
            )

    asyncio.run(rejected())
    assert not service.artifacts.root.exists()
