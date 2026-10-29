"""Exercise the Codex client against a deterministic, in-memory JSON-RPC peer."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, cast

import pytest
from jsonschema import ValidationError
from mcp_types import ElicitResult

from tests.e2e.codex_app_server import CodexAppServer

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp_types import ElicitRequestParams

    from tests.e2e.codex_app_server import ElicitationCallback

_SCHEMA = {
    "type": "object",
    "properties": {"approved": {"type": "boolean", "title": "Run this plan?"}},
    "required": ["approved"],
}
_MESSAGE = "Approve the complete plan:\nstrategy: generated\ncapital: 10000\n期間：2025"
_MCP_CONFIG = {"command": "test-server", "args": [], "enabled": True}


def _form(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "approval-42",
        "method": "mcpServer/elicitation/request",
        "params": {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "serverName": "tradingdev",
            "mode": "form",
            "message": _MESSAGE,
            "requestedSchema": _SCHEMA,
            **overrides,
        },
    }


def _completed(status: str = "completed") -> dict[str, Any]:
    return {
        "method": "turn/completed",
        "params": {
            "threadId": "thread-1",
            "turn": {"id": "turn-1", "status": status},
        },
    }


def _tool(method: str = "item/completed", **overrides: Any) -> dict[str, Any]:
    return {
        "method": method,
        "params": {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "item": {
                "id": "call-1",
                "type": "mcpToolCall",
                "server": "tradingdev",
                "tool": "prepare_backtest",
                "arguments": {"strategy": "generated"},
                "status": "completed",
                "result": {"content": [], "structuredContent": {"success": True}},
                **overrides,
            },
        },
    }


class _Peer:
    """Answer client requests and record every JSON message sent to the peer."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        *,
        early_events: bool = False,
        rpc_failure: str | None = None,
        servers: dict[str, Any] | None = None,
        returned_model: str | None = "specified-model",
        returned_reasoning_effort: str | None = "xhigh",
        turns: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self.reader = asyncio.StreamReader()
        self.messages = messages
        self.turns = turns or [messages]
        self.turn_count = 0
        self.sent: list[dict[str, Any]] = []
        self.early_events = early_events
        self.rpc_failure = rpc_failure
        self.servers = servers or {}
        self.returned_model = returned_model
        self.returned_reasoning_effort = returned_reasoning_effort
        self.observations = 0

    def _feed(self, message: dict[str, Any]) -> None:
        self.reader.feed_data((json.dumps(message) + "\n").encode())

    def write(self, data: bytes) -> None:
        message = json.loads(data)
        self.sent.append(message)
        method = message.get("method")
        if self.rpc_failure is not None and method == self.rpc_failure:
            self._feed(
                {"id": message["id"], "error": {"code": -32000, "message": "bad"}}
            )
        elif method == "initialize":
            self._feed({"id": message["id"], "result": {"userAgent": "test-peer"}})
        elif method == "config/read":
            self._feed(
                {
                    "id": message["id"],
                    "result": {"config": {"mcp_servers": self.servers}},
                }
            )
        elif method == "thread/start":
            self._feed(
                {
                    "id": message["id"],
                    "result": {
                        "thread": {"id": "thread-1"},
                        "model": self.returned_model,
                        "reasoningEffort": self.returned_reasoning_effort,
                    },
                }
            )
        elif method == "turn/start":
            messages = self.turns[self.turn_count]
            self.turn_count += 1
            response = {
                "id": message["id"],
                "result": {"turn": {"id": f"turn-{self.turn_count}"}},
            }
            for event in (
                [*messages, response] if self.early_events else [response, *messages]
            ):
                self._feed(event)
            if self.turn_count == len(self.turns):
                self.reader.feed_eof()

    async def drain(self) -> None:
        pass

    def observe(self) -> None:
        self.observations += 1

    def client(
        self,
        callback: ElicitationCallback | None = None,
        *,
        max_tool_calls: int = 2,
        on_completed_tool: Callable[[dict[str, Any]], None] | None = None,
    ) -> CodexAppServer:
        return CodexAppServer(
            self.reader,
            cast("asyncio.StreamWriter", self),
            tools=["prepare_backtest"],
            max_tool_calls=max_tool_calls,
            elicitation_callback=callback,
            observe_processes=self.observe,
            on_completed_tool=on_completed_tool,
        )


@pytest.mark.parametrize("early_events", [False, True])
def test_handshake_and_turn_preserve_native_events(early_events: bool) -> None:
    async def run() -> None:
        events = [_tool(), _completed()]
        peer = _Peer(events, early_events=early_events)
        thread_params = {"ephemeral": True, "model": "specified-model"}
        assert (
            await peer.client().run(
                "Use MCP only", thread_params, mcp_config=_MCP_CONFIG
            )
            == events
        )
        assert [message["method"] for message in peer.sent] == [
            "initialize",
            "initialized",
            "config/read",
            "thread/start",
            "turn/start",
        ]
        assert peer.sent[0]["params"]["clientInfo"]["name"] == "tradingdev_tests"
        assert peer.sent[0]["params"]["capabilities"]["experimentalApi"] is True
        assert peer.sent[2]["params"] == {"includeLayers": False}
        assert peer.sent[3]["params"] == {
            **thread_params,
            "config": {"mcp_servers": {"tradingdev": _MCP_CONFIG}},
        }
        assert peer.sent[4]["params"] == {
            "threadId": "thread-1",
            "input": [{"type": "text", "text": "Use MCP only", "text_elements": []}],
        }
        assert peer.observations == 6

    asyncio.run(run())


def _conversation_turn(number: int) -> list[dict[str, Any]]:
    events = deepcopy([_tool(id=f"call-{number}"), _completed()])
    events[0]["params"]["turnId"] = f"turn-{number}"
    events[1]["params"]["turn"]["id"] = f"turn-{number}"
    return events


@pytest.mark.parametrize("early_events", [False, True])
def test_user_reply_starts_a_new_turn_on_the_same_codex_thread(
    early_events: bool,
) -> None:
    async def run() -> None:
        turns = [_conversation_turn(1), _conversation_turn(2)]
        peer = _Peer([], turns=turns, early_events=early_events)
        observed: list[list[dict[str, Any]]] = []

        async def user_reply(events: list[dict[str, Any]]) -> str | None:
            assert events[-1]["method"] == "turn/completed"
            observed.append(events)
            assert peer.turn_count == len(observed)
            return "我已閱讀，同意執行。" if len(observed) == 1 else None

        events = await peer.client().run(
            "先準備並等我確認", {}, mcp_config=_MCP_CONFIG, user_reply=user_reply
        )
        assert observed == turns
        assert events == turns[0] + turns[1]
        assert (
            sum(message.get("method") == "thread/start" for message in peer.sent) == 1
        )
        starts = [
            message for message in peer.sent if message.get("method") == "turn/start"
        ]
        assert len(starts) == 2
        assert {message["params"]["threadId"] for message in starts} == {"thread-1"}
        assert starts[1]["params"]["input"][0]["text"] == "我已閱讀，同意執行。"

    asyncio.run(run())


def test_codex_tool_budget_is_shared_across_conversation_turns() -> None:
    async def run() -> None:
        peer = _Peer([], turns=[_conversation_turn(1), _conversation_turn(2)])

        async def user_reply(events: list[dict[str, Any]]) -> str:
            return "同意執行"

        with pytest.raises(AssertionError, match="exceeded 1 MCP tool calls"):
            await peer.client(max_tool_calls=1).run(
                "prepare", {}, mcp_config=_MCP_CONFIG, user_reply=user_reply
            )

    asyncio.run(run())


def test_codex_user_observer_failure_does_not_start_another_turn() -> None:
    async def run() -> None:
        peer = _Peer([_completed()])

        async def user_reply(events: list[dict[str, Any]]) -> str:
            raise AssertionError("Plan was submitted before user consent")

        with pytest.raises(AssertionError, match="before user consent"):
            await peer.client().run(
                "prepare", {}, mcp_config=_MCP_CONFIG, user_reply=user_reply
            )
        assert peer.turn_count == 1

    asyncio.run(run())


def test_codex_conversation_turn_budget_is_enforced_before_next_turn() -> None:
    async def run() -> None:
        peer = _Peer([_completed()])

        async def user_reply(events: list[dict[str, Any]]) -> str:
            return "continue"

        with pytest.raises(AssertionError, match="exceeded 1 conversation turns"):
            await peer.client().run(
                "prepare",
                {},
                mcp_config=_MCP_CONFIG,
                user_reply=user_reply,
                max_turns=1,
            )
        assert peer.turn_count == 1

    asyncio.run(run())


@pytest.mark.parametrize("returned_model", ["gpt-5.6-luna", "another-model", None])
def test_explicit_model_must_match_thread_before_starting_turn(
    returned_model: str | None,
) -> None:
    async def run() -> None:
        peer = _Peer([_completed()], returned_model=returned_model)
        client = peer.client()
        if returned_model == "gpt-5.6-luna":
            assert await client.run(
                "prompt", {"model": "gpt-5.6-luna"}, mcp_config=_MCP_CONFIG
            ) == [_completed()]
            assert peer.sent[-1]["method"] == "turn/start"
        else:
            with pytest.raises(AssertionError, match="expected 'gpt-5.6-luna'"):
                await client.run(
                    "prompt", {"model": "gpt-5.6-luna"}, mcp_config=_MCP_CONFIG
                )
            assert peer.sent[-1]["method"] == "thread/start"
            assert not any(
                message.get("method") == "turn/start" for message in peer.sent
            )

    asyncio.run(run())


@pytest.mark.parametrize("returned_effort", ["xhigh", "ultra", "high", None])
def test_reasoning_effort_must_match_thread_before_starting_turn(
    returned_effort: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    async def run() -> None:
        peer = _Peer([_completed()], returned_reasoning_effort=returned_effort)
        if returned_effort == "xhigh":
            assert await peer.client().run(
                "prompt",
                {},
                mcp_config=_MCP_CONFIG,
                expected_reasoning_effort="xhigh",
            ) == [_completed()]
            assert "reasoning effort xhigh" in capsys.readouterr().out
        else:
            with pytest.raises(AssertionError, match="expected 'xhigh'"):
                await peer.client().run(
                    "prompt",
                    {},
                    mcp_config=_MCP_CONFIG,
                    expected_reasoning_effort="xhigh",
                )
            assert not any(
                message.get("method") == "turn/start" for message in peer.sent
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    ("existing_names", "test_name"),
    [
        (["other-server"], "tradingdev"),
        (["tradingdev", "other-server"], "tradingdev_test"),
        (["tradingdev", "tradingdev_test"], "tradingdev_test_test"),
    ],
)
def test_existing_servers_are_disabled_without_merging_their_config(
    existing_names: list[str], test_name: str
) -> None:
    async def run() -> None:
        async def callback(form: ElicitRequestParams) -> ElicitResult:
            return ElicitResult(action="accept", content={"approved": True})

        servers = {
            name: {
                "command": "user-command",
                "args": ["user-data"],
                "env": {"USER_SETTING": "must-not-merge"},
                "enabled": True,
            }
            for name in existing_names
        }
        events = [_form(serverName=test_name), _tool(server=test_name), _completed()]
        peer = _Peer(events, servers=servers)
        assert (
            await peer.client(callback).run("prompt", {}, mcp_config=_MCP_CONFIG)
            == events
        )
        configured = peer.sent[3]["params"]["config"]["mcp_servers"]
        assert configured == {
            **{name: {"enabled": False} for name in existing_names},
            test_name: _MCP_CONFIG,
        }
        assert servers[existing_names[0]]["enabled"] is True
        assert not any(
            message.get("method", "").startswith("config/")
            and message["method"] != "config/read"
            for message in peer.sent
        )

    asyncio.run(run())


@pytest.mark.parametrize("early_events", [False, True])
@pytest.mark.parametrize(
    ("action", "content"),
    [
        ("accept", {"approved": True}),
        ("accept", {"approved": False}),
        ("decline", None),
        ("cancel", None),
    ],
)
def test_form_preserves_callback_decision_and_request(
    early_events: bool,
    action: Literal["accept", "decline", "cancel"],
    content: dict[str, bool] | None,
) -> None:
    async def run() -> None:
        forms: list[ElicitRequestParams] = []

        async def callback(form: ElicitRequestParams) -> ElicitResult:
            forms.append(form)
            return ElicitResult(
                action=action, content=content, meta={"fixture": "decision"}
            )

        request = _form()
        peer = _Peer([request, _completed()], early_events=early_events)
        await peer.client(callback).run(
            "Confirm through the client", {}, mcp_config=_MCP_CONFIG
        )
        assert len(forms) == 1
        assert forms[0].message == _MESSAGE
        assert forms[0].model_dump(by_alias=True)["requestedSchema"] == _SCHEMA
        reply = peer.sent[-1]
        assert reply["id"] == request["id"]
        assert reply["result"]["action"] == action
        assert reply["result"]["content"] == content
        assert reply["result"]["_meta"] == {"fixture": "decision"}
        assert "meta" not in reply["result"]
        assert "error" not in reply

    asyncio.run(run())


@pytest.mark.parametrize("early_events", [False, True])
@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"threadId": "foreign-thread"}, "Unexpected Codex thread ID"),
        ({"turnId": "foreign-turn"}, "Unexpected Codex turn ID"),
        ({"turnId": ""}, "Unexpected Codex turn ID"),
        ({"serverName": "foreign-server"}, "Unexpected MCP elicitation server"),
        ({"mode": "url", "url": "https://example.invalid"}, "Only MCP form"),
    ],
)
def test_foreign_or_unsupported_forms_are_rejected_before_callback(
    overrides: dict[str, Any], error: str, early_events: bool
) -> None:
    async def run() -> None:
        async def callback(form: ElicitRequestParams) -> ElicitResult:
            pytest.fail(f"Unexpected callback: {form}")

        request = _form(**overrides)
        peer = _Peer([request], early_events=early_events)
        with pytest.raises(AssertionError, match=error):
            await peer.client(callback).run("prompt", {}, mcp_config=_MCP_CONFIG)
        assert peer.sent[-1]["id"] == request["id"]
        assert peer.sent[-1]["error"]["code"] == -32603

    asyncio.run(run())


def test_buffered_form_after_turn_completion_is_not_approved() -> None:
    async def run() -> None:
        async def callback(form: ElicitRequestParams) -> ElicitResult:
            pytest.fail(f"Unexpected callback after completion: {form}")

        peer = _Peer([_completed(), _form()], early_events=True)
        with pytest.raises(AssertionError, match="outside the requested turn"):
            await peer.client(callback).run("prompt", {}, mcp_config=_MCP_CONFIG)
        assert "error" in peer.sent[-1]

    asyncio.run(run())


def test_form_requires_explicit_callback() -> None:
    async def run() -> None:
        peer = _Peer([_form()])
        with pytest.raises(AssertionError, match="explicit test user callback"):
            await peer.client().run("prompt", {}, mcp_config=_MCP_CONFIG)
        assert "result" not in peer.sent[-1]
        assert peer.sent[-1]["error"]["code"] == -32603

    asyncio.run(run())


@pytest.mark.parametrize(
    "method",
    ["item/commandExecution/requestApproval", "item/fileChange/requestApproval"],
)
def test_other_server_requests_are_rejected(method: str) -> None:
    async def run() -> None:
        request = {"id": 314, "method": method, "params": {}}
        peer = _Peer([request])
        with pytest.raises(AssertionError, match="Unsupported Codex server request"):
            await peer.client().run("prompt", {}, mcp_config=_MCP_CONFIG)
        assert peer.sent[-1]["id"] == 314
        assert "error" in peer.sent[-1]

    asyncio.run(run())


@pytest.mark.parametrize("content", [{}, {"approved": "yes"}, {"approved": 1}])
def test_accepted_content_must_match_requested_schema(content: dict[str, Any]) -> None:
    async def run() -> None:
        async def callback(form: ElicitRequestParams) -> ElicitResult:
            return ElicitResult(action="accept", content=content)

        peer = _Peer([_form()])
        with pytest.raises(ValidationError):
            await peer.client(callback).run("prompt", {}, mcp_config=_MCP_CONFIG)
        assert "error" in peer.sent[-1]

    asyncio.run(run())


@pytest.mark.parametrize("action", ["decline", "cancel"])
def test_negative_decision_cannot_return_accepted_content(
    action: Literal["decline", "cancel"],
) -> None:
    async def run() -> None:
        async def callback(form: ElicitRequestParams) -> ElicitResult:
            return ElicitResult(action=action, content={"approved": True})

        peer = _Peer([_form()])
        with pytest.raises(AssertionError, match="Declined/cancelled form has content"):
            await peer.client(callback).run("prompt", {}, mcp_config=_MCP_CONFIG)
        assert "error" in peer.sent[-1]

    asyncio.run(run())


def test_callback_error_is_reported_without_approving_the_form() -> None:
    async def run() -> None:
        async def callback(form: ElicitRequestParams) -> ElicitResult:
            raise RuntimeError("Plan evidence did not match")

        peer = _Peer([_form()])
        with pytest.raises(RuntimeError, match="Plan evidence did not match"):
            await peer.client(callback).run("prompt", {}, mcp_config=_MCP_CONFIG)
        assert peer.sent[-1]["id"] == "approval-42"
        assert "error" in peer.sent[-1]
        assert "result" not in peer.sent[-1]

    asyncio.run(run())


@pytest.mark.parametrize(
    "method", ["initialize", "config/read", "thread/start", "turn/start"]
)
def test_rpc_error_is_not_a_completed_turn(method: str) -> None:
    async def run() -> None:
        peer = _Peer([], rpc_failure=method)
        with pytest.raises(AssertionError, match=f"Codex {method} failed"):
            await peer.client().run("prompt", {}, mcp_config=_MCP_CONFIG)

    asyncio.run(run())


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_unsuccessful_turn_is_rejected(status: str) -> None:
    async def run() -> None:
        peer = _Peer([_completed(status)])
        with pytest.raises(AssertionError, match="Codex turn failed"):
            await peer.client().run("prompt", {}, mcp_config=_MCP_CONFIG)

    asyncio.run(run())


def test_eof_without_completed_turn_is_rejected() -> None:
    async def run() -> None:
        with pytest.raises(AssertionError, match="closed before completing"):
            await _Peer([_tool()]).client().run("prompt", {}, mcp_config=_MCP_CONFIG)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"type": "commandExecution"}, "bypassed MCP"),
        ({"type": "fileChange"}, "bypassed MCP"),
        ({"type": "subAgentActivity"}, "bypassed MCP"),
        ({"server": "foreign-server"}, "foreign-server"),
        ({"tool": "unapproved_tool"}, "unapproved_tool"),
        ({"status": "failed", "result": None}, "MCP call failed"),
    ],
)
def test_tool_failures_and_bypass_are_rejected(
    overrides: dict[str, Any], error: str
) -> None:
    async def run() -> None:
        observed: list[dict[str, Any]] = []
        with pytest.raises(AssertionError, match=error):
            await (
                _Peer([_tool(**overrides)])
                .client(on_completed_tool=observed.append)
                .run("prompt", {}, mcp_config=_MCP_CONFIG)
            )
        assert observed == []

    asyncio.run(run())


@pytest.mark.parametrize("early_events", [False, True])
def test_completed_tool_observer_gets_one_isolated_event_before_turn_finishes(
    early_events: bool,
) -> None:
    async def run() -> None:
        completed = _tool()
        peer = _Peer(
            [_tool("item/started"), completed, _completed()], early_events=early_events
        )
        observed: list[dict[str, Any]] = []

        def observe(event: dict[str, Any]) -> None:
            observed.append(deepcopy(event))
            event["params"]["item"]["result"]["structuredContent"]["success"] = False

        client = peer.client(on_completed_tool=observe)
        events = await client.run("prompt", {}, mcp_config=_MCP_CONFIG)
        assert observed == [completed]
        assert events == [_tool("item/started"), completed, _completed()]

    asyncio.run(run())


def test_started_tool_without_completion_does_not_notify_observer() -> None:
    async def run() -> None:
        observed: list[dict[str, Any]] = []
        peer = _Peer([_tool("item/started")])
        with pytest.raises(AssertionError, match="closed before completing"):
            await peer.client(on_completed_tool=observed.append).run(
                "prompt", {}, mcp_config=_MCP_CONFIG
            )
        assert observed == []

    asyncio.run(run())


def test_tool_budget_counts_calls_instead_of_notifications() -> None:
    async def run() -> None:
        first = [_tool("item/started"), _tool()]
        peer = _Peer([*first, _completed()])
        assert await peer.client(max_tool_calls=1).run(
            "prompt", {}, mcp_config=_MCP_CONFIG
        ) == [
            *first,
            _completed(),
        ]
        peer = _Peer([*first, _tool("item/started", id="call-2")])
        with pytest.raises(AssertionError, match="exceeded 1 MCP tool calls"):
            await peer.client(max_tool_calls=1).run(
                "prompt", {}, mcp_config=_MCP_CONFIG
            )

    asyncio.run(run())


def test_application_failure_payload_is_preserved_for_workflow_assertions() -> None:
    async def run() -> None:
        result = {
            "content": [{"type": "text", "text": "confirmation declined"}],
            "structuredContent": {"success": False, "error": "confirmation_declined"},
        }
        events = (
            await _Peer([_tool(result=result), _completed()])
            .client()
            .run("prompt", {}, mcp_config=_MCP_CONFIG)
        )
        assert events[0]["params"]["item"]["result"] == result

    asyncio.run(run())


def test_deadline_cancels_waiting_approval_callback() -> None:
    async def run() -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def callback(form: ElicitRequestParams) -> ElicitResult:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            raise AssertionError("Unreachable")

        peer = _Peer([_form(), _completed()])
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await peer.client(callback).run("prompt", {}, mcp_config=_MCP_CONFIG)
        assert started.is_set()
        assert cancelled.is_set()
        assert not any(message.get("id") == "approval-42" for message in peer.sent)

    asyncio.run(run())
