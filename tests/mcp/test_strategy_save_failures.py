"""Regression coverage for generated saves rejected by the MCP boundary."""

from __future__ import annotations

import asyncio
from pathlib import Path

from jsonschema import Draft202012Validator
from mcp import Client
from mcp.server import MCPServer

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.strategy_service import StrategyService
from tradingdev.mcp.tools.strategy import register

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "tradingdev"


def test_reserved_strategy_id_returns_a_structured_rejection(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    server = MCPServer("strategy-save-failure-test")
    register(server, service, _PACKAGE_ROOT)

    async def check_rejected_save() -> None:
        # The SDK transport exercises a real MCP session and its error boundary,
        # while avoiding a process for this purely local contract regression.
        async with Client(server, mode="legacy") as client:
            listed = await client.call_tool("list_strategies", {})
            assert not listed.is_error
            assert listed.structured_content is not None
            bundled = next(
                item
                for item in listed.structured_content["result"]
                if item["kind"] == "bundled"
            )
            strategy_id = bundled["strategy_id"]
            before = await client.call_tool(
                "get_strategy", {"strategy_id": strategy_id}
            )
            assert not before.is_error
            assert before.structured_content is not None
            source = before.structured_content["result"]
            tools = await client.list_tools()
            schema = next(
                tool.output_schema
                for tool in tools.tools
                if tool.name == "save_strategy"
            )
            assert schema is not None
            Draft202012Validator.check_schema(schema)

            result = await client.call_tool(
                "save_strategy",
                {
                    "strategy_id": strategy_id,
                    "code": source["source_code"],
                    "yaml_config": source["yaml_config"],
                },
            )

            assert not result.is_error, result.model_dump(mode="json")
            assert result.structured_content is not None
            Draft202012Validator(schema).validate(result.structured_content)
            rejected = result.structured_content["result"]
            assert rejected["success"] is False
            assert rejected["code"] == "reserved_strategy_id"
            assert rejected["strategy_id"] == strategy_id
            assert rejected["status"] == "rejected"
            assert rejected["revision_id"] is None
            assert rejected["py_path"] == rejected["yaml_path"] == ""
            assert rejected["error"]

            after = await client.call_tool("get_strategy", {"strategy_id": strategy_id})
            assert not after.is_error
            assert after.structured_content == before.structured_content
            still_listed = await client.call_tool("list_strategies", {})
            assert not still_listed.is_error
            assert still_listed.structured_content == listed.structured_content
            assert not any(workspace.generated_strategies.iterdir())

    asyncio.run(check_rejected_save())
