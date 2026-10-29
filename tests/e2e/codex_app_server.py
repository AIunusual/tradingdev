"""A bounded Codex conversation over one app-server thread."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from jsonschema import validate
from mcp_types import ElicitRequestFormParams, ElicitRequestParams, ElicitResult

if TYPE_CHECKING:
    import asyncio

ElicitationCallback = Callable[[ElicitRequestParams], Awaitable[ElicitResult]]
TurnCallback = Callable[[list[dict[str, Any]]], Awaitable[str | None]]


class CodexAppServer:
    """Consume native events and reply only to explicitly handled MCP forms."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        tools: list[str],
        max_tool_calls: int,
        elicitation_callback: ElicitationCallback | None,
        observe_processes: Callable[[], None],
        on_completed_tool: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.tools = tools
        self.max_tool_calls = max_tool_calls
        self.elicitation_callback = elicitation_callback
        self.observe_processes = observe_processes
        self.on_completed_tool = on_completed_tool
        self.events: list[dict[str, Any]] = []
        self.seen_calls: set[str] = set()
        self.request_ids: set[str | int] = set()
        self.thread_id: str | None = None
        self.server_name = "tradingdev"
        self.turn_id: str | None = None
        self.turn_started = False
        self.completed = False
        self.sequence = 0

    async def send(self, message: dict[str, Any]) -> None:
        self.writer.write((json.dumps(message) + "\n").encode())
        await self.writer.drain()

    async def read(self) -> dict[str, Any]:
        line = await self.reader.readline()
        self.observe_processes()
        assert line, "Codex app-server closed before completing the request/turn"
        message = json.loads(line)
        assert isinstance(message, dict), "Codex event must be a JSON object"
        return message

    async def request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        pending_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.sequence += 1
        request_id = f"tradingdev-{self.sequence}"
        await self.send({"id": request_id, "method": method, "params": params})
        while True:
            message = await self.read()
            if "method" in message:
                if pending_events is None:
                    await self.dispatch(message)
                else:
                    pending_events.append(message)
                continue
            assert message.get("id") == request_id, "Unexpected Codex response ID"
            assert "error" not in message, f"Codex {method} failed: {message['error']}"
            result = message.get("result")
            assert isinstance(result, dict), f"Invalid Codex {method} response"
            return result

    def check_turn(self, params: dict[str, Any]) -> None:
        assert (
            self.turn_started and not self.completed and self.thread_id is not None
        ), "Unexpected Codex event outside the requested turn"
        assert params.get("threadId") == self.thread_id, "Unexpected Codex thread ID"
        if (turn_id := params.get("turnId")) is not None:
            assert isinstance(turn_id, str)
            assert turn_id == self.turn_id, "Unexpected Codex turn ID"

    async def dispatch(self, message: dict[str, Any]) -> None:
        self.events.append(message)
        method = message["method"]
        params = message.get("params", {})
        assert isinstance(params, dict), "Invalid Codex notification parameters"
        if "id" in message:
            await self.respond_to_form(message)
        elif method == "error":
            raise AssertionError(f"Codex turn error: {params}")
        elif method in {"item/started", "item/completed"}:
            self.check_turn(params)
            assert params.get("turnId") == self.turn_id, "Missing Codex item turn ID"
            item = params["item"]
            assert isinstance(item, dict)
            kind = item.get("type")
            assert kind in {
                "userMessage",
                "agentMessage",
                "reasoning",
                "plan",
                "contextCompaction",
                "mcpToolCall",
            }, f"Codex bypassed MCP with item type {kind!r}"
            if kind == "mcpToolCall":
                assert item.get("server") == self.server_name, item.get("server")
                assert item.get("tool") in self.tools, item.get("tool")
                self.seen_calls.add(str(item["id"]))
                assert len(self.seen_calls) <= self.max_tool_calls, (
                    f"Codex exceeded {self.max_tool_calls} MCP tool calls"
                )
                if method == "item/completed":
                    result = item.get("result")
                    returned_error = (
                        item.get("status") == "failed"
                        and item.get("error") is None
                        and isinstance(result, dict)
                        and bool(
                            result.get("content") or result.get("structuredContent")
                        )
                    )
                    assert item.get("status") == "completed" or returned_error, (
                        f"Codex MCP call failed without a tool response: {item}"
                    )
                    if self.on_completed_tool is not None:
                        self.on_completed_tool(deepcopy(message))
                    print(f"[codex] {item['tool']} completed", flush=True)
        elif method == "turn/completed":
            turn = params["turn"]
            self.check_turn({**params, "turnId": turn["id"]})
            assert turn.get("status") == "completed", f"Codex turn failed: {turn}"
            self.completed = True

    async def respond_to_form(self, message: dict[str, Any]) -> None:
        request_id = message["id"]
        assert isinstance(request_id, (str, int))
        assert request_id not in self.request_ids, "Duplicate Codex server request ID"
        self.request_ids.add(request_id)
        try:
            assert message["method"] == "mcpServer/elicitation/request", (
                f"Unsupported Codex server request: {message['method']}"
            )
            params = message["params"]
            self.check_turn(params)
            assert params.get("serverName") == self.server_name, (
                "Unexpected MCP elicitation server"
            )
            assert params.get("mode") == "form", (
                "Only MCP form elicitation is supported"
            )
            assert self.elicitation_callback is not None, (
                "MCP elicitation requires an explicit test user callback"
            )
            form = ElicitRequestFormParams.model_validate(
                {
                    "mode": "form",
                    "message": params["message"],
                    "requestedSchema": params["requestedSchema"],
                }
            )
            print(
                "[codex] MCP form received; calling the test user callback", flush=True
            )
            answer = await self.elicitation_callback(form)
            if answer.action == "accept":
                validate(answer.content, form.requested_schema)
            else:
                assert answer.content is None, "Declined/cancelled form has content"
            await self.send(
                {
                    "id": request_id,
                    "result": answer.model_dump(mode="json", by_alias=True),
                }
            )
        except Exception:
            await self.send(
                {
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": "Test client rejected the request",
                    },
                }
            )
            raise

    async def run(
        self,
        prompt: str,
        thread_params: dict[str, Any],
        *,
        mcp_config: dict[str, Any],
        expected_reasoning_effort: str | None = None,
        user_reply: TurnCallback | None = None,
        max_turns: int = 8,
    ) -> list[dict[str, Any]]:
        await self.request(
            "initialize",
            {
                "clientInfo": {"name": "tradingdev_tests", "version": "1.0.0"},
                "capabilities": {"experimentalApi": True},
            },
        )
        await self.send({"method": "initialized"})
        print("[codex] app-server initialized", flush=True)
        effective = await self.request("config/read", {"includeLayers": False})
        servers = effective["config"].get("mcp_servers", {})
        assert isinstance(servers, dict), "Invalid Codex MCP server configuration"
        # Config layers merge maps: an empty table does not erase user servers.
        # Pick an unused name so old transport/env fields cannot merge into ours.
        while self.server_name in servers:
            self.server_name += "_test"
        thread_params = {
            **thread_params,
            "config": {
                "mcp_servers": {
                    **{name: {"enabled": False} for name in servers},
                    self.server_name: mcp_config,
                }
            },
        }
        started = await self.request("thread/start", thread_params)
        if requested_model := thread_params.get("model"):
            assert started.get("model") == requested_model, (
                f"Codex selected {started.get('model')!r}, expected {requested_model!r}"
            )
        if expected_reasoning_effort is not None:
            assert started.get("reasoningEffort") == expected_reasoning_effort, (
                f"Codex reasoning effort {started.get('reasoningEffort')!r}, "
                f"expected {expected_reasoning_effort!r}"
            )
        self.thread_id = started["thread"]["id"]
        assert isinstance(self.thread_id, str) and self.thread_id
        print(
            f"[codex] thread started with model {started.get('model')}, "
            f"reasoning effort {started.get('reasoningEffort')}",
            flush=True,
        )
        turn_ids: set[str] = set()
        for turn_number in range(1, max_turns + 1):
            self.turn_started = True
            self.completed = False
            event_start = len(self.events)
            pending_events: list[dict[str, Any]] = []
            result = await self.request(
                "turn/start",
                {
                    "threadId": self.thread_id,
                    "input": [{"type": "text", "text": prompt, "text_elements": []}],
                },
                pending_events=pending_events,
            )
            self.turn_id = result["turn"]["id"]
            assert isinstance(self.turn_id, str) and self.turn_id
            assert self.turn_id not in turn_ids, "Codex reused a completed turn ID"
            turn_ids.add(self.turn_id)
            # Notifications may precede the ACK. Dispatch only after the ACK
            # establishes which turn this client actually started.
            for message in pending_events:
                await self.dispatch(message)
            while not self.completed:
                await self.dispatch(await self.read())
            if user_reply is None:
                return self.events
            reply = await user_reply(self.events[event_start:])
            if reply is None:
                return self.events
            assert reply.strip(), "Test user reply must not be empty"
            assert turn_number < max_turns, (
                f"Codex exceeded {max_turns} conversation turns"
            )
            prompt = reply
        raise AssertionError("Conversation requires at least one turn")
