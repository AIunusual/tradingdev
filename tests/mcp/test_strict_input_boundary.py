"""Actual MCP input validation must reject ignored argument typos before effects."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from jsonschema import Draft202012Validator, ValidationError
from mcp import Client
from mcp.server import MCPServer
from mcp.server.mcpserver import Context  # noqa: TC002 — SDK resolves tool annotations.
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult, TextContent
from tests.mcp.execution_fixtures import plan_context

from tradingdev.adapters.execution.process_runner import ProcessRunner
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.job_store import JobStore
from tradingdev.mcp.server import create_server
from tradingdev.mcp.strict_server import StrictMCPServer

if TYPE_CHECKING:
    from pathlib import Path


_START_ARGUMENTS = {
    "strategy_id": "kd_crossover",
    "research": {
        "kind": "new",
        "strategy_outline": "測試 KD 指標交叉的買賣規則",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-03",
    },
    "evidence_id": "e" * 32,
}


@pytest.mark.parametrize("protocol_mode", ["legacy", "auto"])
def test_revision_typo_is_protocol_error_before_job_or_worker_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protocol_mode: str,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    server = create_server(workspace)

    def unexpected_spawn(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Misspelled MCP arguments must not reach worker launch")

    monkeypatch.setattr(ProcessRunner, "spawn_module", unexpected_spawn)

    async def check() -> None:
        async with Client(server, mode=protocol_mode) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            validator = Draft202012Validator(tools["prepare_backtest"].input_schema)
            validator.validate(_START_ARGUMENTS)
            arguments = _START_ARGUMENTS | {"revisions_id": "wrong-spelling"}
            with pytest.raises(ValidationError, match="revisions_id"):
                validator.validate(arguments)
            result = await client.call_tool("prepare_backtest", arguments)
            assert result.is_error
            text = " ".join(
                item.text for item in result.content if isinstance(item, TextContent)
            )
            assert "Unknown arguments for prepare_backtest: revisions_id" in text
            assert "revision_id" in text
            assert result.structured_content is None

    asyncio.run(check())
    assert JobStore(workspace=workspace).list_all_jobs() == []
    assert not list(workspace.runs.iterdir())


@pytest.mark.parametrize("protocol_mode", ["legacy", "auto"])
def test_unknown_read_arguments_fail_in_direct_and_protocol_dispatch(
    tmp_path: Path,
    protocol_mode: str,
) -> None:
    server = create_server(WorkspacePaths(tmp_path / "workspace"))
    with pytest.raises(ToolError, match="metrc_ids"):
        asyncio.run(
            server.call_tool(
                "get_run_metrics", {"run_id": "absent", "metrc_ids": ["total_pnl"]}
            )
        )

    async def check() -> None:
        async with Client(server, mode=protocol_mode) as client:
            for name, arguments in (
                ("get_run_metrics", {"run_id": "absent", "metrc_ids": ["total_pnl"]}),
                ("list_runs", {"unexpected": True}),
            ):
                result = await client.call_tool(name, arguments)
                assert result.is_error
                assert any(
                    isinstance(item, TextContent) and "Unknown arguments" in item.text
                    for item in result.content
                )
            # Omitting optional arguments remains valid and reaches the service.
            result = await client.call_tool("get_run_metrics", {"run_id": "absent"})
            assert not result.is_error
            assert result.structured_content is not None
            assert result.structured_content["result"]["code"] == "run_not_found"

    asyncio.run(check())


@pytest.mark.parametrize("protocol_mode", ["legacy", "auto"])
def test_revision_omission_prepares_bundled_strategy_without_launching(
    tmp_path: Path,
    protocol_mode: str,
) -> None:
    context = plan_context(tmp_path)

    async def check() -> None:
        async with Client(context.server, mode=protocol_mode) as client:
            result = await client.call_tool("prepare_backtest", context.arguments)
            assert not result.is_error
            assert result.structured_content is not None
            prepared = result.structured_content["result"]
            assert prepared["success"] and prepared["status"] == "ready"
            assert context.runner.calls == []
            assert context.service.jobs.list_all_jobs() == []
            started = await client.call_tool(
                "submit_execution_decision",
                {"plan_id": prepared["plan_id"], "decision": "approve"},
            )
            assert not started.is_error
            assert started.structured_content is not None
            assert started.structured_content["result"]["success"] is True

    asyncio.run(check())
    assert len(context.runner.calls) == 1
    jobs = context.service.jobs.list_all_jobs()
    assert len(jobs) == 1 and jobs[0]["revision_id"] is None


@pytest.mark.parametrize("protocol_mode", ["legacy", "auto"])
def test_strict_dispatch_forwards_request_context_and_dynamic_maps(
    protocol_mode: str,
) -> None:
    server = StrictMCPServer("context-forwarding")

    @server.tool()
    async def inspect_request(
        parameters: dict[str, Any], ctx: Context
    ) -> dict[str, Any]:
        return {
            "parameters": parameters,
            "method": ctx.request_context.method,
            "protocol_version": ctx.request_context.protocol_version,
        }

    async def check() -> None:
        async with Client(server, mode=protocol_mode) as client:
            result = await client.call_tool(
                "inspect_request", {"parameters": {"custom": {"nested": 3}}}
            )
            assert not result.is_error
            assert result.structured_content == {
                "parameters": {"custom": {"nested": 3}},
                "method": "tools/call",
                "protocol_version": client.protocol_version,
            }

    asyncio.run(check())


def test_all_advertised_argument_objects_are_closed_and_sdk_schema_unchanged(
    tmp_path: Path,
) -> None:
    server = create_server(WorkspacePaths(tmp_path / "workspace"))

    async def check() -> None:
        tools = await server.list_tools()
        assert tools
        for tool in tools:
            schema = tool.input_schema
            Draft202012Validator.check_schema(schema)
            assert schema["additionalProperties"] is False, tool.name
        backtest = next(tool for tool in tools if tool.name == "prepare_backtest")
        assert "revision_id" not in backtest.input_schema["required"]
        optimization = next(
            tool for tool in tools if tool.name == "prepare_optimization"
        )
        assert isinstance(
            optimization.input_schema["properties"]["param_ranges"][
                "additionalProperties"
            ],
            dict,
        )
        # list_tools is a public projection, not a mutation of SDK internals.
        sdk_tools = await MCPServer.list_tools(server)
        assert all(
            "additionalProperties" not in tool.input_schema for tool in sdk_tools
        )

    asyncio.run(check())


def test_dynamic_parameter_maps_and_late_tool_registration_remain_supported() -> None:
    server = StrictMCPServer("dynamic-input-test")

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
        assert isinstance(result, CallToolResult)
        assert result.structured_content is not None
        assert result.structured_content["parameters"] == {
            "custom_strategy_parameter": 3
        }
        assert result.structured_content["optional"] is None
        result = await server.call_tool("added_later", {"value": 7})
        assert isinstance(result, CallToolResult)
        assert (
            result.structured_content is not None
            and result.structured_content["result"] == 7
        )
        with pytest.raises(ToolError, match="extra"):
            await server.call_tool("added_later", {"value": 7, "extra": True})
        with pytest.raises(ToolError, match="Unknown tool"):
            await server.call_tool("does_not_exist", {})

    asyncio.run(check())
