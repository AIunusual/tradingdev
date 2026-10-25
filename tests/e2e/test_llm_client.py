"""Offline model-adapter tests, including a real stdio MCP round trip."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, create_autospec, patch

import httpx
import pytest
from mcp import ClientSession
from mcp_types import CallToolResult, ListToolsResult, TextContent, Tool

from tests.e2e.codex_harness import LIFECYCLE_TOOLS
from tests.e2e.llm_client import (
    ToolCall,
    codex_calls,
    compact_tool_result,
    run_local_model,
)
from tests.integration.mcp_harness import MCPClient, temporary_mcp_workspace

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn


_EMPTY_LIST_PAGE = {
    "items": [],
    "offset": 0,
    "limit": 20,
    "total": 0,
    "next_offset": None,
}


def _call(
    name: str = "list_strategies",
    arguments: str = "{}",
    call_id: str = "call-1",
) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _reply(
    calls: list[dict[str, Any]] | None = None,
    *,
    finish_reason: str | None = None,
) -> httpx.Response:
    message: dict[str, Any] = {"role": "assistant", "content": None}
    if calls is not None:
        message["tool_calls"] = calls
    else:
        message["content"] = "Complete."
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": message,
                    "finish_reason": finish_reason
                    or ("tool_calls" if calls else "stop"),
                }
            ]
        },
    )


def _mock_client(
    *, extra_tools: list[Tool] | None = None
) -> tuple[MCPClient, AsyncMock]:
    session = create_autospec(ClientSession, instance=True)
    session.list_tools.return_value = ListToolsResult(
        tools=[
            Tool(
                name="list_strategies",
                description="List strategies.",
                input_schema={"type": "object", "properties": {}},
            ),
            *(extra_tools or []),
        ]
    )
    session.call_tool.return_value = CallToolResult(
        content=[], structured_content=deepcopy(_EMPTY_LIST_PAGE)
    )
    return MCPClient(session, "Use MCP tools."), session.call_tool


@pytest.mark.parametrize("is_error", [False, True])
def test_local_model_receives_sdk_wire_aliases_and_compacted_structured_result(
    is_error: bool,
) -> None:
    client, invoke = _mock_client()
    structured = {"success": not is_error, "rows": 0}
    invoke.return_value = CallToolResult(
        content=[TextContent(type="text", text=json.dumps(structured))],
        structured_content=structured,
        is_error=is_error,
        meta={"fixture": "wire"},
    )
    requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return _reply([_call()])
        message = json.loads(request.content)["messages"][-1]
        assert message["role"] == "tool"
        assert json.loads(message["content"]) == {
            "structuredContent": structured,
            "isError": is_error,
            "resultType": "complete",
            "_meta": {"fixture": "wire"},
        }
        return _reply()

    calls = asyncio.run(
        run_local_model(
            client,
            "Inspect the tool result",
            model="fixture",
            base_url="http://fixture/v1",
            timeout_seconds=5,
            transport=httpx.MockTransport(respond),
        )
    )
    assert requests == 2
    assert calls == [ToolCall("list_strategies", {}, structured)]
    invoke.assert_awaited_once_with("list_strategies", {})


def test_local_user_reply_keeps_prior_messages_and_observes_only_completed_turns() -> (
    None
):
    async def run() -> None:
        client, call_tool = _mock_client()
        requests: list[dict[str, Any]] = []
        turns: list[ModelTurn] = []

        def respond(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) in {1, 3}:
                return _reply([_call(call_id=f"call-{len(requests)}")])
            return _reply()

        async def user_reply(turn: ModelTurn) -> str | None:
            turns.append(turn)
            assert turn.number == len(turns)
            assert turn.text == "Complete."
            assert len(turn.calls) == 1
            assert call_tool.await_count == len(turns)
            return "同意執行" if turn.number == 1 else None

        calls = await run_local_model(
            client,
            "先準備",
            model="fixture",
            base_url="http://fixture/v1",
            timeout_seconds=5,
            transport=httpx.MockTransport(respond),
            user_reply=user_reply,
        )
        assert len(calls) == len(turns) == 2
        assert requests[2]["messages"][:-2] == requests[1]["messages"]
        assert requests[2]["messages"][-2:] == [
            {"role": "assistant", "content": "Complete."},
            {"role": "user", "content": "同意執行"},
        ]
        assert (
            requests[3]["messages"][: len(requests[2]["messages"])]
            == requests[2]["messages"]
        )

    asyncio.run(run())


def test_local_tool_budget_remains_shared_after_user_reply() -> None:
    async def run() -> None:
        client, call_tool = _mock_client()
        count = 0

        def respond(request: httpx.Request) -> httpx.Response:
            nonlocal count
            count += 1
            return _reply() if count == 2 else _reply([_call(call_id=f"call-{count}")])

        async def user_reply(turn: ModelTurn) -> str:
            return "同意執行"

        with pytest.raises(AssertionError, match="exceeded 1 MCP tool calls"):
            await run_local_model(
                client,
                "prepare",
                model="fixture",
                base_url="http://fixture/v1",
                timeout_seconds=5,
                transport=httpx.MockTransport(respond),
                user_reply=user_reply,
                max_tool_calls=1,
            )
        assert call_tool.await_count == 1

    asyncio.run(run())


def test_local_user_observer_error_does_not_request_another_model_turn() -> None:
    async def run() -> None:
        client, call_tool = _mock_client()
        requests = 0

        def respond(request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            return _reply()

        async def user_reply(turn: ModelTurn) -> str:
            raise AssertionError("No reviewable plan")

        with pytest.raises(AssertionError, match="No reviewable plan"):
            await run_local_model(
                client,
                "prepare",
                model="fixture",
                base_url="http://fixture/v1",
                timeout_seconds=5,
                transport=httpx.MockTransport(respond),
                user_reply=user_reply,
            )
        assert requests == 1
        call_tool.assert_not_awaited()

    asyncio.run(run())


def test_local_model_receives_real_mcp_schemas_errors_and_repaired_result() -> None:
    async def exercise() -> None:
        with temporary_mcp_workspace() as workspace:
            async with workspace.connect() as client:
                advertised = {
                    tool.name: tool
                    for tool in (await client.session.list_tools()).tools
                }
                original_payloads = {
                    call_id: (await client.session.call_tool(name, {})).model_dump(
                        mode="json", by_alias=True
                    )
                    for call_id, name in (
                        ("list", "list_strategies"),
                        ("contract", "get_strategy_contract"),
                    )
                }
                requests: list[dict[str, Any]] = []
                repaired_id = ""

                def respond(request: httpx.Request) -> httpx.Response:
                    nonlocal repaired_id
                    assert (
                        str(request.url) == "http://localhost:11434/v1/chat/completions"
                    )
                    body = json.loads(request.content)
                    requests.append(body)
                    assert body["model"] == "offline-fixture-model"
                    assert "temperature" not in body
                    assert "reasoning_effort" not in body
                    assert body["messages"][0] == {
                        "role": "system",
                        "content": client.instructions,
                    }
                    schemas = {
                        item["function"]["name"]: item["function"]
                        for item in body["tools"]
                    }
                    assert set(schemas) == set(LIFECYCLE_TOOLS) & advertised.keys()
                    for name, schema in schemas.items():
                        assert schema["parameters"] == advertised[name].input_schema
                    if len(requests) == 1:
                        assert (
                            body["messages"][1]["content"] == "Find a valid strategy."
                        )
                        return _reply(
                            [
                                _call(call_id="list"),
                                _call(
                                    "get_strategy",
                                    '{"strategy_id": "missing_strategy"}',
                                    "missing",
                                ),
                                _call("get_strategy", "{}", "invalid-schema"),
                                _call("get_strategy_contract", "{}", "contract"),
                            ]
                        )
                    if len(requests) == 2:
                        responses = {
                            item["tool_call_id"]: json.loads(item["content"])
                            for item in body["messages"]
                            if item["role"] == "tool"
                        }
                        for call_id, original in original_payloads.items():
                            compact = responses[call_id]
                            assert compact == {
                                key: value
                                for key, value in original.items()
                                if key != "content"
                            }
                            assert len(json.dumps(compact)) < 0.7 * len(
                                json.dumps(original)
                            )
                        listed = responses["list"]["structuredContent"]["items"]
                        assert listed and all(
                            item["kind"] == "bundled" for item in listed
                        )
                        missing = responses["missing"]["structuredContent"]
                        assert set(missing) == {"result"}
                        semantic_error = missing["result"]
                        assert semantic_error["success"] is False
                        assert "missing_strategy" in semantic_error["error"]
                        assert responses["invalid-schema"]["isError"] is True
                        assert responses["invalid-schema"]["content"]
                        repaired_id = listed[0]["strategy_id"]
                        return _reply(
                            [
                                _call(
                                    "get_strategy",
                                    json.dumps({"strategy_id": repaired_id}),
                                    "repair",
                                )
                            ]
                        )
                    assert len(requests) == 3
                    repaired = json.loads(body["messages"][-1]["content"])
                    assert body["messages"][-1]["tool_call_id"] == "repair"
                    assert set(repaired["structuredContent"]) == {"result"}
                    repaired_payload = repaired["structuredContent"]["result"]
                    assert repaired_payload["success"] is True
                    assert repaired_payload["strategy_id"] == repaired_id
                    assert repaired_payload["source_code"]
                    return _reply()

                calls = await run_local_model(
                    client,
                    "Find a valid strategy.",
                    model="offline-fixture-model",
                    base_url="http://localhost:11434/v1/",
                    timeout_seconds=20,
                    transport=httpx.MockTransport(respond),
                )
                assert len(calls) == 5
                assert isinstance(calls[0].result["items"], list)
                assert calls[1].result["success"] is False
                assert calls[-1].arguments == {"strategy_id": repaired_id}
                assert calls[-1].result["success"] is True

    asyncio.run(exercise())


def _text_payload(value: Any) -> dict[str, str]:
    return {"type": "text", "text": json.dumps(value)}


@pytest.mark.parametrize(
    ("structured", "content"),
    [
        (
            {"success": True, "values": [1, "1", 0.1, None]},
            [_text_payload({"values": [1, "1", 0.1, None], "success": True})],
        ),
        (
            {"result": [{"id": "a"}, {"id": "b"}]},
            [_text_payload({"id": "a"}), _text_payload({"id": "b"})],
        ),
        ({"result": [{"id": "a"}]}, [_text_payload({"id": "a"})]),
        ({"result": [1, 2]}, [_text_payload([1, 2])]),
        ({"result": "message"}, [_text_payload("message")]),
        ({"result": False}, [_text_payload(False)]),
    ],
    ids=["whole-object", "sdk-list", "sdk-single-item", "list", "string", "boolean"],
)
def test_duplicate_content_is_removed_without_mutation_or_metadata_loss(
    structured: dict[str, Any], content: list[dict[str, Any]]
) -> None:
    payload = {
        "content": content,
        "structuredContent": structured,
        "isError": True,
        "meta": {"trace": "keep"},
        "unknown": [1, 2],
    }
    before = deepcopy(payload)
    compact = compact_tool_result(payload)
    assert compact == {key: value for key, value in before.items() if key != "content"}
    assert payload == before


@pytest.mark.parametrize(
    ("structured", "content"),
    [
        ({"success": True}, [_text_payload({"success": 1})]),
        ({"count": 1}, [_text_payload({"count": 1.0})]),
        (
            {"value": 9007199254740992.0},
            [{"type": "text", "text": '{"value": 9007199254740993.0}'}],
        ),
        ({"value": 0.0}, [{"type": "text", "text": '{"value": 1e-400}'}]),
        (
            {"success": True},
            [{"type": "text", "text": '{"success": false, "success": true}'}],
        ),
        (
            {"value": float("nan")},
            [{"type": "text", "text": '{"value": NaN}'}],
        ),
        (
            {"value": float("inf")},
            [{"type": "text", "text": '{"value": Infinity}'}],
        ),
        (
            {"value": float("inf")},
            [{"type": "text", "text": '{"value": 1e9999}'}],
        ),
        (
            {"success": True},
            [_text_payload({"success": True}), {"type": "text", "text": "Warning"}],
        ),
        (
            {"result": [{"id": "a"}, {"id": "b"}]},
            [_text_payload({"id": "b"}), _text_payload({"id": "a"})],
        ),
        (
            {"success": True},
            [
                {"type": "text", "text": '{"success":'},
                {"type": "text", "text": "true}"},
            ],
        ),
        (
            {"success": True},
            [_text_payload({"success": True}), {"type": "image", "data": "keep"}],
        ),
        (
            {"success": True},
            [
                {
                    **_text_payload({"success": True}),
                    "annotations": {"audience": ["user"]},
                }
            ],
        ),
        ({"success": True}, [{**_text_payload({"success": True}), "meta": {}}]),
        (
            {"success": True},
            [{**_text_payload({"success": True}), "_meta": {"trace": "keep"}}],
        ),
        (
            {"success": True},
            [{**_text_payload({"success": True}), "unknown": None}],
        ),
        (
            {"result": {"success": True}, "extra": "not a sole wrapper"},
            [_text_payload({"success": True})],
        ),
        ({"nested": {"success": True}}, [_text_payload({"success": True})]),
    ],
    ids=[
        "bool-v-int",
        "int-v-float",
        "float-rounding",
        "float-underflow",
        "duplicate-key",
        "nan",
        "infinity",
        "overflow",
        "additional-text",
        "different-order",
        "split-json",
        "image",
        "annotations",
        "metadata",
        "metadata-alias",
        "unknown-key",
        "not-sole-wrapper",
        "subtree",
    ],
)
def test_nonidentical_or_information_bearing_content_is_preserved(
    structured: dict[str, Any], content: list[dict[str, Any]]
) -> None:
    payload = {"content": content, "structuredContent": structured, "isError": True}
    assert compact_tool_result(payload) is payload


@pytest.mark.parametrize("temperature", [0.0, 0.7, 2.0])
def test_explicit_temperature_is_preserved_across_model_tool_rounds(
    temperature: float,
) -> None:
    client, invoke = _mock_client()
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert body["temperature"] == temperature
        return _reply([_call()] if len(requests) == 1 else None)

    calls = asyncio.run(
        run_local_model(
            client,
            "Use the selected sampling temperature",
            model="fixture",
            base_url="http://localhost/v1",
            timeout_seconds=1,
            temperature=temperature,
            transport=httpx.MockTransport(respond),
        )
    )
    assert len(requests) == 2
    assert calls == [ToolCall("list_strategies", {}, _EMPTY_LIST_PAGE)]
    invoke.assert_awaited_once_with("list_strategies", {})


def test_explicit_reasoning_effort_is_preserved_across_model_tool_rounds() -> None:
    client, invoke = _mock_client()
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert body["reasoning_effort"] == "low"
        return _reply([_call()] if len(requests) == 1 else None)

    calls = asyncio.run(
        run_local_model(
            client,
            "Use the selected reasoning effort",
            model="fixture",
            base_url="http://localhost/v1",
            timeout_seconds=1,
            reasoning_effort="low",
            transport=httpx.MockTransport(respond),
        )
    )
    assert len(requests) == 2
    assert calls == [ToolCall("list_strategies", {}, _EMPTY_LIST_PAGE)]
    invoke.assert_awaited_once_with("list_strategies", {})


def test_progress_reports_usage_and_results_without_source_or_reasoning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client, invoke = _mock_client()
    invoke.return_value = CallToolResult(
        content=[],
        structured_content={"success": True, "source_code": "PRIVATE_SOURCE"},
    )
    requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        payload = _reply([_call()] if requests == 1 else None).json()
        if requests == 1:
            payload["choices"][0]["message"]["reasoning"] = "PRIVATE_REASONING"
            payload["usage"] = {
                "prompt_tokens": 103,
                "completion_tokens": 17,
                "completion_tokens_details": {"reasoning_tokens": 7},
            }
        else:
            messages = json.loads(request.content)["messages"]
            assert messages[-2]["reasoning"] == "PRIVATE_REASONING"
            assert (
                json.loads(messages[-1]["content"])["structuredContent"]["source_code"]
                == "PRIVATE_SOURCE"
            )
            payload["choices"][0]["message"]["content"] = (
                "Task complete. " + "More detail. " * 100 + "UNPRINTED_TAIL"
            )
        return httpx.Response(200, json=payload)

    calls = asyncio.run(
        run_local_model(
            client,
            "Report progress",
            model="fixture",
            base_url="http://localhost/v1",
            timeout_seconds=1,
            transport=httpx.MockTransport(respond),
        )
    )
    output = capsys.readouterr().out
    assert calls[0].result["source_code"] == "PRIVATE_SOURCE"
    assert "request started" in output and "response received" in output
    assert "MCP tool list_strategies" in output and "success=True" in output
    assert "input_tokens=103" in output and "output_tokens=17" in output
    assert "reasoning_tokens=7" in output and "reasoning_chars=17" in output
    assert "model sampling seed=42" in output
    assert "Task complete." in output
    assert not any(
        secret in output
        for secret in ("PRIVATE_SOURCE", "PRIVATE_REASONING", "UNPRINTED_TAIL")
    )


def test_http_failure_keeps_status_error_and_bounded_response_diagnostic() -> None:
    client, invoke = _mock_client()
    with pytest.raises(httpx.HTTPStatusError) as failure:
        asyncio.run(
            run_local_model(
                client,
                "Unavailable model",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=1,
                transport=httpx.MockTransport(
                    lambda _request: httpx.Response(
                        400,
                        json={
                            "error": "Unsupported reasoning effort",
                            "detail": "x" * 2000 + "UNPRINTED_TAIL",
                        },
                    )
                ),
            )
        )
    error = failure.value
    assert error.response.status_code == 400
    assert str(error.request.url) == "http://localhost/v1/chat/completions"
    assert isinstance(error.__cause__, httpx.HTTPStatusError)
    assert error.__cause__.response is error.response
    assert "Unsupported reasoning effort" in str(error)
    assert "UNPRINTED_TAIL" not in str(error)
    invoke.assert_not_awaited()


_TOOL_XML_ERROR = (
    "XML syntax error on line 19: element <parameter> closed by </function>"
)


def _tool_serialization_failure() -> httpx.Response:
    # Ollama 0.35.0 /v1/chat/completions response from the live legacy scenario.
    return httpx.Response(
        500,
        json={"error": {"message": _TOOL_XML_ERROR, "type": "api_error"}},
    )


def test_provider_tool_error_receives_feedback_without_replaying_completed_calls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client, invoke = _mock_client(
        extra_tools=[
            Tool(
                name="get_job_status",
                description="Get job status.",
                input_schema={
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}},
                    "required": ["job_id"],
                },
            )
        ]
    )
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert str(request.url) == "http://localhost/v1/chat/completions"
        assert body["model"] == "fixture"
        assert body["seed"] == 42
        assert body["temperature"] == 0.7
        assert body["reasoning_effort"] == "none"
        if len(requests) == 1:
            return _reply([_call(call_id="completed")])
        if len(requests) == 2:
            return _tool_serialization_failure()
        if len(requests) == 3:
            before = requests[1]
            assert body["messages"][:-1] == before["messages"]
            assert {key: value for key, value in body.items() if key != "messages"} == {
                key: value for key, value in before.items() if key != "messages"
            }
            feedback = body["messages"][-1]
            assert feedback["role"] == "user"
            assert _TOOL_XML_ERROR in feedback["content"]
            assert (
                "No MCP tools from that response were executed" in feedback["content"]
            )
            assert "Do not repeat already completed tool calls" in feedback["content"]
            assert invoke.await_count == 1
            return _reply([_call("get_job_status", '{"job_id":"job-1"}', "next")])
        assert len(requests) == 4
        return _reply()

    calls = asyncio.run(
        run_local_model(
            client,
            "Continue after a provider formatting failure",
            model="fixture",
            base_url="http://localhost/v1",
            timeout_seconds=2,
            temperature=0.7,
            reasoning_effort="none",
            transport=httpx.MockTransport(respond),
        )
    )
    assert [(call.name, call.arguments) for call in calls] == [
        ("list_strategies", {}),
        ("get_job_status", {"job_id": "job-1"}),
    ]
    assert invoke.await_count == 2
    output = capsys.readouterr().out
    assert "HTTP 500 provider tool serialization error" in output
    assert "feedback repair 1/2" in output
    assert "provider_tool_repairs=1" in output
    assert _TOOL_XML_ERROR in output


def test_model_sampling_seed_is_preserved_across_all_tool_rounds() -> None:
    client, invoke = _mock_client()
    requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        assert json.loads(request.content)["seed"] == 42
        return _reply([_call()] if requests == 1 else None)

    asyncio.run(
        run_local_model(
            client,
            "Fix model sampling independently of the strategy execution seed",
            model="fixture",
            base_url="http://localhost/v1",
            timeout_seconds=1,
            transport=httpx.MockTransport(respond),
        )
    )
    assert requests == 2
    invoke.assert_awaited_once_with("list_strategies", {})


def test_provider_tool_repair_budget_applies_to_the_entire_conversation() -> None:
    client, invoke = _mock_client()
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) in {2, 4}:
            return _reply([_call(call_id=f"completed-{len(requests)}")])
        return _tool_serialization_failure()

    with pytest.raises(
        httpx.HTTPStatusError, match="repair budget exhausted"
    ) as failure:
        asyncio.run(
            run_local_model(
                client,
                "Keep encountering malformed provider output",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=1,
                transport=httpx.MockTransport(respond),
            )
        )
    assert len(requests) == 5
    assert invoke.await_count == 2
    assert failure.value.response.status_code == 500
    assert isinstance(failure.value.__cause__, httpx.HTTPStatusError)
    assert _TOOL_XML_ERROR in str(failure.value)
    feedback = [
        message
        for message in requests[-1]["messages"]
        if message["role"] == "user"
        and message["content"].startswith("Runtime feedback:")
    ]
    assert len(feedback) == 2


@pytest.mark.parametrize(
    ("status", "payload"),
    [
        (500, {"error": {"message": "Model runner crashed"}}),
        (500, {"error": {"message": "XML syntax error on line 19: unexpected EOF"}}),
        (500, {"error": {"message": _TOOL_XML_ERROR + " extra details"}}),
        (500, {"error": _TOOL_XML_ERROR}),
        (500, {"error": {"message": 123}}),
        (500, ["not an error object"]),
        (500, None),
        (401, {"error": {"message": _TOOL_XML_ERROR}}),
        (400, {"error": {"message": _TOOL_XML_ERROR}}),
        (400, {"error": {"message": "Unsupported parameter: seed"}}),
    ],
    ids=[
        "runner-crash",
        "other-xml-error",
        "nonexact-error",
        "unconfirmed-shape",
        "nonstring-message",
        "nonobject-body",
        "invalid-json",
        "authentication",
        "bad-request",
        "unsupported-seed",
    ],
)
def test_other_provider_failures_are_not_retried(status: int, payload: Any) -> None:
    client, invoke = _mock_client()
    requests = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if payload is None:
            return httpx.Response(status, text="Invalid JSON response")
        return httpx.Response(status, json=payload)

    with pytest.raises(httpx.HTTPStatusError) as failure:
        asyncio.run(
            run_local_model(
                client,
                "Provider unavailable",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=1,
                transport=httpx.MockTransport(respond),
            )
        )
    assert requests == 1
    assert failure.value.response.status_code == status
    invoke.assert_not_awaited()


@pytest.mark.parametrize("failure", ["deadline", "http-timeout"])
def test_provider_tool_feedback_does_not_restart_deadline_or_retry_timeout(
    failure: str,
) -> None:
    client, invoke = _mock_client()
    requests = 0
    transport_finished = False

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests, transport_finished
        requests += 1
        if requests == 1:
            return _tool_serialization_failure()
        try:
            if failure == "http-timeout":
                raise httpx.ReadTimeout("Model stalled after feedback", request=request)
            await asyncio.Event().wait()
            raise AssertionError("Unreachable")
        finally:
            transport_finished = True

    with pytest.raises(AssertionError, match="model round 2") as error:
        asyncio.run(
            run_local_model(
                client,
                "Feedback remains within the original deadline",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=0.02,
                transport=httpx.MockTransport(respond),
            )
        )
    assert requests == 2
    assert transport_finished
    assert isinstance(error.value.__cause__, (TimeoutError, httpx.ReadTimeout))
    invoke.assert_not_awaited()


@pytest.mark.parametrize(
    "invalid_batch", [False, True], ids=["tool-budget", "bad-batch"]
)
def test_provider_tool_feedback_preserves_dispatch_validation_and_tool_budget(
    invalid_batch: bool,
) -> None:
    client, invoke = _mock_client()
    requests = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return _reply([_call(call_id="completed")])
        if requests == 2:
            return _tool_serialization_failure()
        assert requests == 3
        next_call = _call(
            "shell" if invalid_batch else "list_strategies", call_id="bad"
        )
        return _reply([_call(call_id="not-dispatched"), next_call])

    with pytest.raises(AssertionError):
        asyncio.run(
            run_local_model(
                client,
                "Never dispatch an invalid batch after provider feedback",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=1,
                max_tool_calls=64 if invalid_batch else 2,
                transport=httpx.MockTransport(respond),
            )
        )
    assert requests == 3
    invoke.assert_awaited_once_with("list_strategies", {})


def test_model_timeout_includes_recent_tool_failure_and_preserves_cause() -> None:
    client, invoke = _mock_client()
    invoke.return_value = CallToolResult(
        content=[],
        structured_content={"success": False, "error": "Storage unavailable"},
    )
    requests = 0
    original_error: httpx.ReadTimeout | None = None

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests, original_error
        requests += 1
        if requests == 1:
            return _reply([_call()])
        original_error = httpx.ReadTimeout("Model stalled", request=request)
        raise original_error

    with pytest.raises(AssertionError, match="after 1 MCP calls") as failure:
        asyncio.run(
            run_local_model(
                client,
                "Retry after tool failure",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=1,
                transport=httpx.MockTransport(respond),
            )
        )
    assert failure.value.__cause__ is original_error
    diagnostic = str(failure.value)
    assert "model round 2" in diagnostic
    assert "list_strategies" in diagnostic and "success=False" in diagnostic
    assert "Storage unavailable" in diagnostic
    invoke.assert_awaited_once_with("list_strategies", {})


@pytest.mark.parametrize(
    "invalid_call",
    [
        _call("shell", '{"command": "touch should-not-exist"}', "bad"),
        _call(arguments="not-json", call_id="bad"),
        _call(arguments="[]", call_id="bad"),
        _call(call_id="valid"),
    ],
    ids=["unadvertised-shell", "malformed-json", "non-object", "duplicate-id"],
)
def test_invalid_batch_is_rejected_before_any_mcp_tool_runs(
    invalid_call: dict[str, Any],
) -> None:
    client, invoke = _mock_client()
    with pytest.raises((AssertionError, ValueError)):
        asyncio.run(
            run_local_model(
                client,
                "Bad batch",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=1,
                transport=httpx.MockTransport(
                    lambda _request: _reply([_call(call_id="valid"), invalid_call])
                ),
            )
        )
    invoke.assert_not_awaited()


def test_tool_budget_counts_prior_rounds_and_rejects_next_batch_atomically() -> None:
    client, invoke = _mock_client()
    requests = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return _reply(
            [_call()]
            if requests == 1
            else [_call(call_id="two"), _call(call_id="three")]
        )

    with pytest.raises(AssertionError, match="exceeded 2 MCP tool calls"):
        asyncio.run(
            run_local_model(
                client,
                "Keep calling",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=1,
                max_tool_calls=2,
                transport=httpx.MockTransport(respond),
            )
        )
    assert requests == 2
    invoke.assert_awaited_once_with("list_strategies", {})


def test_truncated_model_response_does_not_count_as_completion() -> None:
    client, invoke = _mock_client()
    with pytest.raises(AssertionError):
        asyncio.run(
            run_local_model(
                client,
                "Finish the task",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=1,
                transport=httpx.MockTransport(
                    lambda _request: _reply(finish_reason="length")
                ),
            )
        )
    invoke.assert_not_awaited()


@pytest.mark.parametrize("failure", ["deadline", "http-timeout"])
def test_model_timeout_is_bounded_and_reports_completed_tool_count(
    failure: str,
) -> None:
    client, invoke = _mock_client()
    transport_finished = False

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal transport_finished
        try:
            if failure == "http-timeout":
                raise httpx.ReadTimeout("No response", request=request)
            await asyncio.Event().wait()
            raise AssertionError("Unreachable")
        finally:
            transport_finished = True

    with pytest.raises(AssertionError, match="after 0 MCP calls") as error:
        asyncio.run(
            run_local_model(
                client,
                "Wait forever",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=0.02,
                transport=httpx.MockTransport(respond),
            )
        )
    assert transport_finished
    assert "model round 1" in str(error.value)
    assert isinstance(error.value.__cause__, (TimeoutError, httpx.ReadTimeout))
    invoke.assert_not_awaited()


def test_model_deadline_also_cancels_stalled_mcp_tool_discovery() -> None:
    client, invoke = _mock_client()
    discovery_finished = False

    async def stalled_discovery() -> ListToolsResult:
        nonlocal discovery_finished
        try:
            await asyncio.Event().wait()
            raise AssertionError("Unreachable")
        finally:
            discovery_finished = True

    def reject_model_request(_request: httpx.Request) -> httpx.Response:
        pytest.fail("The model cannot run before MCP tool discovery completes")

    async def exercise() -> None:
        # This outer bound prevents a regression from hanging the test suite.
        await asyncio.wait_for(
            run_local_model(
                client,
                "Discover tools",
                model="fixture",
                base_url="http://localhost/v1",
                timeout_seconds=0.02,
                transport=httpx.MockTransport(reject_model_request),
            ),
            timeout=1,
        )

    with (
        patch.object(
            client.session, "list_tools", new=AsyncMock(side_effect=stalled_discovery)
        ),
        pytest.raises(AssertionError, match="after 0 MCP calls") as error,
    ):
        asyncio.run(exercise())
    assert discovery_finished
    assert "MCP tool discovery" in str(error.value)
    invoke.assert_not_awaited()


def test_codex_events_normalize_arguments_and_both_mcp_result_encodings() -> None:
    completed: dict[str, Any] = {
        "method": "item/completed",
        "params": {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "item": {
                "type": "mcpToolCall",
                "status": "completed",
                "tool": "get_strategy",
                "arguments": '{"strategy_id": "example"}',
                "result": {"structuredContent": {"success": False, "error": "Repair"}},
            },
        },
    }
    wrapped = {
        "method": "item/completed",
        "params": {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "item": {
                "type": "mcpToolCall",
                "status": "completed",
                "tool": "list_strategies",
                "arguments": {},
                "result": {
                    "structuredContent": {
                        "result": {
                            "items": [{"strategy_id": "example"}],
                            "offset": 0,
                            "limit": 20,
                            "total": 1,
                            "next_offset": None,
                        }
                    }
                },
            },
        },
    }
    text_only = {
        "method": "item/completed",
        "params": {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "item": {
                "type": "mcpToolCall",
                "status": "completed",
                "tool": "get_job_status",
                "arguments": {"job_id": "job-1"},
                "result": {
                    "content": [
                        {"type": "text", "text": "Diagnostic before payload"},
                        {"type": "text", "text": '{"status": "done"}'},
                    ]
                },
            },
        },
    }
    events = [
        {"method": "item/started", "params": completed["params"]},
        {
            "method": "item/completed",
            "params": {"item": {"type": "agentMessage"}},
        },
        {
            "method": "item/completed",
            "params": {
                **completed["params"],
                "item": {**completed["params"]["item"], "status": "failed"},
            },
        },
        completed,
        wrapped,
        text_only,
    ]
    original = deepcopy(events)
    calls = codex_calls(events)
    assert calls == [
        ToolCall(
            "get_strategy",
            {"strategy_id": "example"},
            {"success": False, "error": "Repair"},
        ),
        ToolCall(
            "list_strategies",
            {},
            {
                "items": [{"strategy_id": "example"}],
                "offset": 0,
                "limit": 20,
                "total": 1,
                "next_offset": None,
            },
        ),
        ToolCall("get_job_status", {"job_id": "job-1"}, {"status": "done"}),
    ]
    assert events == original, "The adapter must retain raw app-server evidence"
