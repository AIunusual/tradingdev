"""Two model clients, one MCP conversation contract and one assertion format."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import httpx
from tooling.context_budget import RequestSizes, text_size

from tests.e2e.codex_harness import LIFECYCLE_TOOLS, successful_mcp_calls

if TYPE_CHECKING:
    from tests.e2e.conversation import UserReply
    from tests.integration.mcp_harness import MCPClient


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: Any


def tool_payload(result: dict[str, Any]) -> Any:
    """Read MCP wire results shared by the SDK and Codex app-server."""
    structured = result.get("structuredContent")
    if structured is None:
        for block in result.get("content", []):
            if block.get("type") == "text":
                try:
                    structured = json.loads(block["text"])
                except ValueError, KeyError:
                    continue
                break
    if isinstance(structured, dict) and set(structured) == {"result"}:
        return structured["result"]
    return structured


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def _exact_json_float(token: str) -> float:
    value = float(token)
    if Decimal(token) != Decimal(str(value)):
        raise ValueError("JSON number loses precision when parsed")
    return value


def _canonical_json(value: Any) -> str:
    # Values come from the MCP SDK's JSON serialization or strict json.loads.
    # JSON spelling distinguishes booleans from numbers (unlike Python ==).
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def compact_tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove only an entirely redundant plain-text MCP content sequence.

    Compare whole values, including an SDK result wrapper or list spread across
    text blocks. Preserve additional text, metadata and every other payload key.
    """
    structured = payload.get("structuredContent")
    content = payload.get("content")
    if not isinstance(structured, dict) or not isinstance(content, list) or not content:
        return payload
    decoded = []
    try:
        for block in content:
            if (
                not isinstance(block, dict)
                or block.get("type") != "text"
                or not isinstance(block.get("text"), str)
                or set(block) - {"type", "text", "annotations", "meta", "_meta"}
                # Pydantic emits these known optional fields even when absent.
                or any(
                    block.get(key) is not None
                    for key in ("annotations", "meta", "_meta")
                )
            ):
                return payload
            decoded.append(
                json.loads(
                    block["text"],
                    object_pairs_hook=_unique_json_object,
                    parse_constant=_reject_json_constant,
                    parse_float=_exact_json_float,
                )
            )
        candidates = [structured]
        if set(structured) == {"result"}:
            candidates.append(structured["result"])
        expected = {_canonical_json(candidate) for candidate in candidates}
        observed = [_canonical_json(decoded)]
        if len(decoded) == 1:
            observed.append(_canonical_json(decoded[0]))
        if not expected.intersection(observed):
            return payload
    except TypeError, ValueError, RecursionError:
        return payload
    return {key: value for key, value in payload.items() if key != "content"}


def codex_calls(events: list[dict[str, Any]]) -> list[ToolCall]:
    """Read completed app-server calls, retaining failed application responses."""
    calls = []
    for item in successful_mcp_calls(events):
        arguments = item.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        assert isinstance(arguments, dict), item
        calls.append(
            ToolCall(item["tool"], arguments, tool_payload(item.get("result") or {}))
        )
    return calls


def _progress(message: str) -> None:
    print(f"[local-llm] {message}", flush=True)


def _short_text(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[:limit] + " [truncated]"


def _tool_summary(call: ToolCall) -> str:
    """Keep useful status fields without logging source code or full MCP payloads."""
    result = call.result
    if isinstance(result, dict):
        fields = [
            f"{key}={_short_text(str(result[key]))}"
            for key in ("success", "status", "error", "plan_id", "job_id", "run_id")
            if key in result and isinstance(result[key], (str, bool, int, float))
        ]
        diagnostics = result.get("diagnostics")
        if isinstance(diagnostics, list):
            codes = [
                _short_text(str(item["code"]), 60)
                for item in diagnostics[:3]
                if isinstance(item, dict) and "code" in item
            ]
            if codes:
                fields.append(f"diagnostics={','.join(codes)}")
        summary = "; ".join(fields) or f"object with {len(result)} fields"
    elif isinstance(result, list):
        summary = f"{len(result)} items"
    else:
        summary = type(result).__name__
    return f"{call.name}: {summary}"


def _usage_summary(payload: dict[str, Any]) -> str:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return ""
    fields = [
        f"{label}={usage[key]}"
        for key, label in (
            ("prompt_tokens", "input_tokens"),
            ("completion_tokens", "output_tokens"),
        )
        if isinstance(usage.get(key), int)
    ]
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), int):
        fields.append(f"reasoning_tokens={details['reasoning_tokens']}")
    return " " + " ".join(fields) if fields else ""


_MAX_PROVIDER_TOOL_REPAIRS = 2
# Reproduce model sampling where the selected provider supports it. This is
# separate from the strategy's execution random_seed in the backtest manifest.
_MODEL_SAMPLING_SEED = 42
_PROVIDER_TOOL_XML_ERROR = re.compile(
    r"XML syntax error on line [1-9][0-9]*: "
    r"element <parameter> closed by </function>"
)


def _provider_tool_serialization_error(response: httpx.Response) -> str | None:
    """Recognize the confirmed Ollama tool parser failure, not generic HTTP 500s."""
    if response.status_code != 500:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if (
        not isinstance(message, str)
        or _PROVIDER_TOOL_XML_ERROR.fullmatch(message) is None
    ):
        return None
    return message


async def run_local_model(
    client: MCPClient,
    prompt: str,
    *,
    model: str,
    base_url: str,
    timeout_seconds: float,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    max_tool_calls: int = 64,
    transport: httpx.AsyncBaseTransport | None = None,
    user_reply: UserReply | None = None,
    max_turns: int = 8,
    tool_names: list[str] | None = None,
) -> list[ToolCall]:
    """Bridge local Chat Completions tool calls to real stdio MCP, with bounds.

    No shell or file tool is exposed. MCP errors are returned to the model so it
    can repair drafts. A confirmed provider tool serialization failure receives
    explicit model feedback at most twice per conversation, before any tool from
    that response runs. Other malformed responses and budget exhaustion fail.
    The caller owns the MCP session and its worker/file cleanup.
    """
    calls: list[ToolCall] = []
    recent_results: list[str] = []
    stage = "MCP tool discovery"
    round_number = 0
    provider_tool_repairs = 0
    request_sizes = RequestSizes()
    turn_number = 1
    turn_start = 0
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": client.instructions},
        {"role": "user", "content": prompt},
    ]
    try:
        async with (
            asyncio.timeout(timeout_seconds),
            httpx.AsyncClient(
                timeout=timeout_seconds, transport=transport, trust_env=False
            ) as http,
        ):
            _progress(f"{stage}: started")
            tools: list[dict[str, Any]] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                }
                for tool in (await client.session.list_tools()).tools
                if tool.name in (LIFECYCLE_TOOLS if tool_names is None else tool_names)
            ]
            allowed = {tool["function"]["name"] for tool in tools}
            _progress(f"{stage}: {len(allowed)} tools available")
            _progress(f"model sampling seed={_MODEL_SAMPLING_SEED}")
            while True:
                round_number += 1
                stage = f"model round {round_number}"
                request: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "stream": False,
                    "seed": _MODEL_SAMPLING_SEED,
                }
                if temperature is not None:
                    request["temperature"] = temperature
                if reasoning_effort is not None:
                    request["reasoning_effort"] = reasoning_effort
                _progress(f"{stage}: request started after {len(calls)} MCP calls")
                started = time.monotonic()
                outbound = http.build_request(
                    "POST",
                    base_url.rstrip("/") + "/chat/completions",
                    json=request,
                )
                _progress(f"{stage}: {request_sizes.observe(outbound.content)}")
                response = await http.send(outbound)
                elapsed = time.monotonic() - started
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    detail = _short_text(response.text, 1200)
                    serialization_error = _provider_tool_serialization_error(response)
                    if (
                        serialization_error is not None
                        and provider_tool_repairs < _MAX_PROVIDER_TOOL_REPAIRS
                    ):
                        provider_tool_repairs += 1
                        _progress(
                            f"{stage}: HTTP 500 provider tool serialization error "
                            f"after {elapsed:.2f}s; feedback repair "
                            f"{provider_tool_repairs}/{_MAX_PROVIDER_TOOL_REPAIRS}; "
                            f"no MCP calls dispatched from this response; "
                            f"error: {serialization_error}"
                        )
                        # No assistant message was received or dispatched. Keep
                        # every completed call and result; ask for a new response
                        # with feedback rather than replaying an identical request.
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Runtime feedback: the model provider rejected "
                                    "your previous tool-call serialization with HTTP "
                                    f"500: {serialization_error}. No MCP tools from "
                                    "that response were executed. All earlier tool "
                                    "calls and their results above remain completed. "
                                    "Regenerate only the next intended tool call "
                                    "using the advertised tool schema and the "
                                    "provider's required tool-call format. Close "
                                    "each parameter before closing the function. "
                                    "Do not repeat already completed tool calls."
                                ),
                            }
                        )
                        continue
                    repair_detail = (
                        "; provider tool serialization repair budget exhausted "
                        f"({_MAX_PROVIDER_TOOL_REPAIRS} feedback repairs)"
                        if serialization_error is not None
                        else ""
                    )
                    raise httpx.HTTPStatusError(
                        f"{exc}\n{stage} failed after {elapsed:.2f}s; "
                        f"response body: {detail}{repair_detail}",
                        request=exc.request,
                        response=exc.response,
                    ) from exc
                payload = response.json()
                choice = payload["choices"][0]
                message = choice["message"]
                assert message.get("role") == "assistant", message
                requested = message.get("tool_calls") or []
                reasoning = message.get("reasoning", message.get("reasoning_content"))
                reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
                _progress(
                    f"{stage}: response received in {elapsed:.2f}s"
                    f"{_usage_summary(payload)} reasoning_chars={reasoning_chars} "
                    f"tool_calls={len(requested)}"
                )
                if not requested:
                    assert choice.get("finish_reason") == "stop", choice
                    content = message.get("content")
                    final_text = (
                        _short_text(content) if isinstance(content, str) else ""
                    )
                    _progress(
                        f"{stage}: finished; "
                        f"provider_tool_repairs={provider_tool_repairs}; "
                        f"final text: {final_text or '(empty)'}"
                    )
                    messages.append(message)
                    if user_reply is None:
                        return calls
                    from tests.e2e.conversation import ModelTurn

                    reply = await user_reply(
                        ModelTurn(
                            turn_number,
                            content if isinstance(content, str) else "",
                            calls[turn_start:],
                        )
                    )
                    if reply is None:
                        return calls
                    assert reply.strip(), "Test user reply must not be empty"
                    assert turn_number < max_turns, (
                        f"Model exceeded {max_turns} conversation turns"
                    )
                    messages.append({"role": "user", "content": reply})
                    turn_number += 1
                    turn_start = len(calls)
                    continue
                assert choice.get("finish_reason") in {"tool_calls", "stop"}, choice
                assert len(calls) + len(requested) <= max_tool_calls, (
                    f"Model exceeded {max_tool_calls} MCP tool calls"
                )
                # Validate the entire batch before allowing any side effect.
                parsed = []
                ids: set[str] = set()
                for call in requested:
                    name = call["function"]["name"]
                    assert call["type"] == "function" and name in allowed, name
                    assert call["id"] and call["id"] not in ids, call
                    ids.add(call["id"])
                    arguments = json.loads(call["function"]["arguments"])
                    assert isinstance(arguments, dict), arguments
                    parsed.append((call["id"], name, arguments))
                messages.append(message)
                for call_id, name, arguments in parsed:
                    stage = f"MCP tool {name} (model round {round_number})"
                    _progress(f"{stage}: started")
                    started = time.monotonic()
                    result = await client.session.call_tool(name, arguments)
                    payload = result.model_dump(mode="json", by_alias=True)
                    call = ToolCall(name, arguments, tool_payload(payload))
                    calls.append(call)
                    summary = f"{_tool_summary(call)}; mcp_error={result.is_error}"
                    recent_results.append(summary)
                    _progress(
                        f"{stage}: finished in {time.monotonic() - started:.2f}s; "
                        f"{summary}"
                    )
                    tool_content = json.dumps(
                        compact_tool_result(payload), ensure_ascii=False
                    )
                    size = text_size(tool_content)
                    _progress(
                        f"{stage}: result_chars={size['characters']} "
                        f"result_utf8_bytes={size['utf8_bytes']}"
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": tool_content,
                        }
                    )
                # Avoid a tight polling loop while a real backtest worker starts.
                if parsed and all(name == "get_job_status" for _, name, _ in parsed):
                    stage = "backtest polling pause"
                    await asyncio.sleep(0.5)
    except (TimeoutError, httpx.TimeoutException) as exc:
        recent = " | ".join(recent_results[-3:]) or "none"
        msg = (
            f"Local model exceeded {timeout_seconds:g}s after {len(calls)} MCP calls; "
            f"stage: {stage}; recent tool results: {recent}"
        )
        raise AssertionError(msg) from exc
