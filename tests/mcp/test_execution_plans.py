"""Real MCP plan decisions and storage with substituted sample/worker execution."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import ValidationError, validate
from mcp import Client
from tests.mcp.execution_fixtures import plan_context

from tradingdev.app.preflight_service import PreflightError


async def _call(client: Client, name: str, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool(name, arguments)
    assert not result.is_error, result
    assert result.structured_content is not None
    schema = next(
        tool for tool in (await client.list_tools()).tools if tool.name == name
    ).output_schema
    assert schema is not None
    validate(result.structured_content, schema)
    payload = result.structured_content
    return dict(payload["result"] if set(payload) == {"result"} else payload)


@pytest.mark.parametrize("json_arguments", [False, True])
def test_prepare_and_read_wait_for_a_separate_explicit_decision(
    tmp_path: Path,
    json_arguments: bool,
) -> None:
    context = plan_context(tmp_path)
    if json_arguments:
        context.arguments["presentation"] = json.dumps(
            context.arguments["presentation"]
        )
        context.arguments["minimum_history_bars"] = "20"
        context.arguments["sample_bars"] = "128"
        context.arguments["parameters"] = "null"
        context.arguments["backtest_overrides"] = "null"

    async def check() -> None:
        # Product decisions use ordinary tools, without an elicitation callback.
        async with Client(context.server, mode="legacy") as client:
            prepared = await _call(client, "prepare_backtest", **context.arguments)
            assert prepared["success"] and prepared["status"] == "ready", prepared
            assert prepared["preflight"]["manifest_hash"] == prepared["manifest_hash"]
            assert prepared["confirmation_summary"]
            assert "confirmation_text" not in prepared
            assert "<html" in Path(prepared["html_path"]).read_text(encoding="utf-8")
            retained = await _call(
                client, "get_execution_plan", plan_id=prepared["plan_id"]
            )
            assert retained == prepared
            assert len(context.preflight.requests) == 1
            assert context.preflight.requests[0].minimum_history_bars == 20
            assert context.preflight.requests[0].sample_bars == 128
            assert context.runner.calls == []
            assert context.service.jobs.list_all_jobs() == []
            assert context.service.jobs.list_runs() == []

            started = await _call(
                client,
                "submit_execution_decision",
                plan_id=prepared["plan_id"],
                decision="approve",
            )
            assert started["success"]
            assert started["manifest_hash"] == prepared["manifest_hash"]
            assert started["plan_id"] == prepared["plan_id"]
            assert (
                context.service.jobs.load_manifest(started["job_id"]).manifest_hash
                == prepared["manifest_hash"]
            )
            repeated = await _call(
                client,
                "submit_execution_decision",
                plan_id=prepared["plan_id"],
                decision="approve",
            )
            assert repeated == started
            assert len(context.runner.calls) == 1

    asyncio.run(check())


@pytest.mark.parametrize(
    "decision,code",
    [
        ("decline", "confirmation_declined"),
        ("revise", "confirmation_adjustment_requested"),
    ],
)
def test_decline_or_revision_cancels_original_plan_without_a_job(
    tmp_path: Path, decision: str, code: str
) -> None:
    context = plan_context(tmp_path)

    async def check() -> None:
        async with Client(context.server, mode="legacy") as client:
            prepared = await _call(client, "prepare_backtest", **context.arguments)
            result = await _call(
                client,
                "submit_execution_decision",
                plan_id=prepared["plan_id"],
                decision=decision,
            )
            assert result["success"] is False and result["code"] == code
            retained = await _call(
                client, "get_execution_plan", plan_id=prepared["plan_id"]
            )
            assert retained["status"] == "cancelled"
            rejected = await _call(
                client,
                "submit_execution_decision",
                plan_id=prepared["plan_id"],
                decision="approve",
            )
            assert rejected["success"] is False
            assert context.runner.calls == []
            assert context.service.jobs.list_all_jobs() == []

            # A replacement is a distinct, unsubmitted plan requiring a decision.
            replacement = await _call(client, "prepare_backtest", **context.arguments)
            assert replacement["plan_id"] != prepared["plan_id"]
            assert replacement["status"] == "ready"
            assert context.runner.calls == []

    asyncio.run(check())


@pytest.mark.parametrize(
    "content",
    [
        {},
        {"approved": True},
        {"decision": None},
        {"decision": True},
        {"decision": 1},
        {"decision": ""},
        {"decision": "accept"},
        {"decision": "approve "},
        {"decision": "同意執行"},
        {"decision": "approve", "unexpected": True},
    ],
)
def test_missing_or_invalid_decision_cannot_consume_or_launch_plan(
    tmp_path: Path, content: dict[str, Any]
) -> None:
    context = plan_context(tmp_path)

    async def check() -> None:
        async with Client(context.server, mode="legacy") as client:
            prepared = await _call(client, "prepare_backtest", **context.arguments)
            result = await client.call_tool(
                "submit_execution_decision", {"plan_id": prepared["plan_id"], **content}
            )
            assert result.is_error
            assert (
                await _call(client, "get_execution_plan", plan_id=prepared["plan_id"])
            )["status"] == "ready"

    asyncio.run(check())
    assert context.runner.calls == []
    assert context.service.jobs.list_all_jobs() == []


def test_cancelled_plan_rejects_later_approval(tmp_path: Path) -> None:
    context = plan_context(tmp_path)

    async def check() -> None:
        async with Client(context.server, mode="legacy") as client:
            prepared = await _call(client, "prepare_backtest", **context.arguments)
            cancelled = await _call(
                client, "cancel_execution_plan", plan_id=prepared["plan_id"]
            )
            assert cancelled["status"] == "cancelled"
            result = await _call(
                client,
                "submit_execution_decision",
                plan_id=prepared["plan_id"],
                decision="approve",
            )
            assert result["success"] is False and result["code"] == "plan_not_ready"

    asyncio.run(check())
    assert context.runner.calls == []
    assert context.service.jobs.list_all_jobs() == []


def test_concurrent_decisions_cannot_duplicate_a_launch(tmp_path: Path) -> None:
    context = plan_context(tmp_path)

    async def check() -> None:
        async with (
            Client(context.server, mode="legacy") as client,
            Client(context.server, mode="legacy") as contender,
        ):
            prepared = await _call(client, "prepare_backtest", **context.arguments)
            results = await asyncio.gather(
                *(
                    _call(
                        connection,
                        "submit_execution_decision",
                        plan_id=prepared["plan_id"],
                        decision="approve",
                    )
                    for connection in (client, contender)
                )
            )
            started = [result for result in results if result["success"]]
            assert started
            assert {result["job_id"] for result in started} == {prepared["plan_id"]}
            assert len(context.runner.calls) == 1
            assert len(context.service.jobs.list_all_jobs()) == 1

    asyncio.run(check())


def test_prepare_failure_is_structured_through_real_mcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = plan_context(tmp_path)

    def failed_sample(_request: object) -> None:
        raise PreflightError(
            "invalid_execution_request", "The selected YAML cannot be read"
        )

    monkeypatch.setattr(context.preflight, "prepare", failed_sample)

    async def check() -> None:
        async with Client(context.server, mode="legacy") as client:
            result = await _call(client, "prepare_backtest", **context.arguments)
            assert result["success"] is False
            assert result["code"] == "invalid_execution_request"
            assert "YAML" in result["error"]

    asyncio.run(check())
    assert context.runner.calls == []
    assert context.service.jobs.list_all_jobs() == []


def test_decision_tool_schema_requires_one_explicit_choice(tmp_path: Path) -> None:
    context = plan_context(tmp_path)

    async def check() -> None:
        async with Client(context.server, mode="legacy") as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert "request_execution_confirmation" not in tools
            schema = tools["submit_execution_decision"].input_schema
            assert set(schema["required"]) == {"plan_id", "decision"}
            assert set(schema["properties"]) == {"plan_id", "decision"}
            for decision in ("approve", "decline", "revise"):
                validate({"plan_id": "plan", "decision": decision}, schema)
            with pytest.raises(ValidationError):
                validate({"plan_id": "plan", "decision": "unknown"}, schema)
            with pytest.raises(ValidationError):
                validate({"plan_id": "plan"}, schema)
            assert "default" not in schema["properties"]["decision"]

    asyncio.run(check())
