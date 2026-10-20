"""Offline ML tool-scope discovery and dispatch checks for the local client."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, create_autospec

import httpx
import pytest
from mcp import ClientSession
from mcp_types import CallToolResult, ListToolsResult, Tool

from tests.e2e.llm_client import ToolCall, run_local_model
from tests.integration.mcp_harness import MCPClient


@pytest.fixture
def ml_client() -> tuple[MCPClient, AsyncMock]:
    session = create_autospec(ClientSession, instance=True)
    session.list_tools.return_value = ListToolsResult(
        tools=[
            Tool(name=name, input_schema={"type": "object", "properties": {}})
            for name in ("list_strategies", "get_ml_contract")
        ]
    )
    session.call_tool.return_value = CallToolResult(
        content=[], structured_content={"result": []}
    )
    return MCPClient(session, "Use MCP tools."), session.call_tool


def _reply(name: str | None = None) -> httpx.Response:
    message: dict[str, Any] = {"role": "assistant", "content": "Complete."}
    if name is not None:
        message.update(
            content=None,
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }
            ],
        )
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": message,
                    "finish_reason": "tool_calls" if name is not None else "stop",
                }
            ]
        },
    )


@pytest.mark.parametrize("outside_scope", [False, True])
def test_local_ml_tool_scope_controls_discovery_and_dispatch(
    outside_scope: bool, ml_client: tuple[MCPClient, AsyncMock]
) -> None:
    async def run() -> None:
        client, call_tool = ml_client
        requests = 0

        def respond(request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            body = json.loads(request.content)
            assert [tool["function"]["name"] for tool in body["tools"]] == [
                "get_ml_contract"
            ]
            if requests == 1:
                name = "list_strategies" if outside_scope else "get_ml_contract"
                return _reply(name)
            return _reply()

        async def invoke() -> list[ToolCall]:
            return await run_local_model(
                client,
                "Research ML",
                model="fixture",
                base_url="http://fixture/v1",
                timeout_seconds=5,
                tool_names=["get_ml_contract"],
                transport=httpx.MockTransport(respond),
            )

        if outside_scope:
            with pytest.raises(AssertionError, match="list_strategies"):
                await invoke()
            call_tool.assert_not_awaited()
        else:
            calls = await invoke()
            assert [call.name for call in calls] == ["get_ml_contract"]
            call_tool.assert_awaited_once_with("get_ml_contract", {})

    asyncio.run(run())
