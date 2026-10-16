"""Actual MCP input validation must reject ignored argument typos before effects."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from jsonschema import Draft202012Validator, ValidationError
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import TextContent
from tests.integration.execution_fixtures import execution_options
from tests.mcp.execution_fixtures import plan_context

from tradingdev.adapters.execution.process_runner import ProcessRunner
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.job_store import JobStore
from tradingdev.mcp.server import create_server
from tradingdev.mcp.strict_server import StrictFastMCP

if TYPE_CHECKING:
    from pathlib import Path


_START_ARGUMENTS = {
    "strategy_id": "kd_crossover",
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "start_date": "2024-01-01",
    "end_date": "2024-01-03",
    **execution_options({}),
}


def test_revision_typo_is_protocol_error_before_job_or_worker_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    server = create_server(workspace)

    def unexpected_spawn(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Misspelled MCP arguments must not reach worker launch")

    monkeypatch.setattr(ProcessRunner, "spawn_module", unexpected_spawn)

    async def check() -> None:
        async with create_connected_server_and_client_session(server) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            arguments = _START_ARGUMENTS | {"revisions_id": "wrong-spelling"}
            with pytest.raises(ValidationError, match="revisions_id"):
                Draft202012Validator(tools["prepare_backtest"].inputSchema).validate(
                    arguments
                )
            result = await client.call_tool("prepare_backtest", arguments)
            assert result.isError
            text = " ".join(
                item.text for item in result.content if isinstance(item, TextContent)
            )
            assert "Unknown arguments for prepare_backtest: revisions_id" in text
            assert "revision_id" in text
            assert result.structuredContent is None

    asyncio.run(check())
    assert JobStore(workspace=workspace).list_all_jobs() == []
    assert not list(workspace.runs.iterdir())


def test_unknown_read_arguments_fail_in_direct_and_protocol_dispatch(
    tmp_path: Path,
) -> None:
    server = create_server(WorkspacePaths(tmp_path / "workspace"))
    with pytest.raises(ToolError, match="metrc_ids"):
        asyncio.run(
            server.call_tool(
                "get_run_metrics", {"run_id": "absent", "metrc_ids": ["total_pnl"]}
            )
        )

    async def check() -> None:
        async with create_connected_server_and_client_session(server) as client:
            for name, arguments in (
                ("get_run_metrics", {"run_id": "absent", "metrc_ids": ["total_pnl"]}),
                ("list_runs", {"unexpected": True}),
            ):
                result = await client.call_tool(name, arguments)
                assert result.isError
                assert any(
                    isinstance(item, TextContent) and "Unknown arguments" in item.text
                    for item in result.content
                )
            # Omitting optional arguments remains valid and reaches the service.
            result = await client.call_tool("get_run_metrics", {"run_id": "absent"})
            assert not result.isError
            assert result.structuredContent is not None
            assert result.structuredContent["result"]["code"] == "run_not_found"

    asyncio.run(check())


def test_revision_omission_prepares_bundled_strategy_without_launching(
    tmp_path: Path,
) -> None:
    context = plan_context(tmp_path)

    async def check() -> None:
        async with create_connected_server_and_client_session(context.server) as client:
            result = await client.call_tool("prepare_backtest", context.arguments)
            assert not result.isError
            assert result.structuredContent is not None
            prepared = result.structuredContent["result"]
            assert prepared["success"] and prepared["status"] == "ready"
            assert context.runner.calls == []
            assert context.service.jobs.list_all_jobs() == []
            started = await client.call_tool(
                "submit_execution_decision",
                {"plan_id": prepared["plan_id"], "decision": "approve"},
            )
            assert not started.isError
            assert started.structuredContent is not None
            assert started.structuredContent["result"]["success"] is True

    asyncio.run(check())
    assert len(context.runner.calls) == 1
    jobs = context.service.jobs.list_all_jobs()
    assert len(jobs) == 1 and jobs[0]["revision_id"] is None


def test_all_advertised_argument_objects_are_closed_and_sdk_schema_unchanged(
    tmp_path: Path,
) -> None:
    server = create_server(WorkspacePaths(tmp_path / "workspace"))

    async def check() -> None:
        tools = await server.list_tools()
        assert tools
        for tool in tools:
            schema = tool.inputSchema
            Draft202012Validator.check_schema(schema)
            assert schema["additionalProperties"] is False, tool.name
        backtest = next(tool for tool in tools if tool.name == "prepare_backtest")
        assert "revision_id" not in backtest.inputSchema["required"]
        optimization = next(
            tool for tool in tools if tool.name == "prepare_optimization"
        )
        assert isinstance(
            optimization.inputSchema["properties"]["param_ranges"][
                "additionalProperties"
            ],
            dict,
        )
        # list_tools is a public projection, not a mutation of SDK internals.
        sdk_tools = await FastMCP.list_tools(server)
        assert all("additionalProperties" not in tool.inputSchema for tool in sdk_tools)

    asyncio.run(check())


def test_dynamic_parameter_maps_and_late_tool_registration_remain_supported() -> None:
    server = StrictFastMCP("dynamic-input-test")

    @server.tool()
    def echo(parameters: dict[str, Any], optional: str | None = None) -> dict[str, Any]:
        return {"parameters": parameters, "optional": optional}

    async def check() -> None:
        await server.list_tools()

        @server.tool()
        def added_later(value: int) -> int:
            return value

        result = await server.call_tool(
            "echo", {"parameters": {"custom_strategy_parameter": 3}}
        )
        assert isinstance(result, tuple)
        assert result[1]["parameters"] == {"custom_strategy_parameter": 3}
        assert result[1]["optional"] is None
        result = await server.call_tool("added_later", {"value": 7})
        assert isinstance(result, tuple) and result[1]["result"] == 7
        with pytest.raises(ToolError, match="extra"):
            await server.call_tool("added_later", {"value": 7, "extra": True})
        with pytest.raises(ToolError, match="Unknown tool"):
            await server.call_tool("does_not_exist", {})

    asyncio.run(check())
