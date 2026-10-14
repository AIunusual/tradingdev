"""Real MCP elicitation and plan storage, with sample/worker execution substituted."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from jsonschema import validate
from mcp.server.fastmcp import Context
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import ElicitResult, ErrorData
from tests.integration.mcp_harness import SimulatedUserApproval
from tests.mcp.execution_fixtures import plan_context

from tradingdev.app.preflight_service import PreflightError

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.shared.context import RequestContext
    from mcp.types import ElicitRequestParams


async def _call(client: ClientSession, name: str, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool(name, arguments)
    assert not result.isError, result
    assert result.structuredContent is not None
    schema = next(
        tool for tool in (await client.list_tools()).tools if tool.name == name
    ).outputSchema
    assert schema is not None
    validate(result.structuredContent, schema)
    payload = result.structuredContent
    return dict(payload["result"] if set(payload) == {"result"} else payload)


@pytest.mark.parametrize("json_arguments", [False, True])
def test_prepare_read_and_user_confirmation_bind_one_manifest_and_one_job(
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
    approval = SimulatedUserApproval()

    async def check() -> None:
        async with create_connected_server_and_client_session(
            context.server,
            elicitation_callback=approval,
        ) as client:
            prepared = await _call(client, "prepare_backtest", **context.arguments)
            assert prepared["success"] and prepared["status"] == "ready", prepared
            assert context.runner.calls == []
            assert context.service.jobs.list_all_jobs() == []
            assert context.service.jobs.list_runs() == []
            assert prepared["preflight"]["manifest_hash"] == prepared["manifest_hash"]
            assert Path(prepared["html_path"]).is_file()
            retained = await _call(
                client, "get_execution_plan", plan_id=prepared["plan_id"]
            )
            assert retained == prepared
            assert len(context.preflight.requests) == 1
            assert context.preflight.requests[0].minimum_history_bars == 20
            assert context.preflight.requests[0].sample_bars == 128

            started = await _call(
                client, "request_execution_confirmation", plan_id=prepared["plan_id"]
            )
            assert (
                started["success"]
                and started["manifest_hash"] == prepared["manifest_hash"]
            )
            assert started["plan_id"] == prepared["plan_id"]
            assert len(approval.requests) == 1
            assert prepared["confirmation_text"] in approval.requests[0].message
            form = approval.requests[0]
            assert form.mode == "form"
            assert set(form.requestedSchema) <= {
                "$schema",
                "type",
                "properties",
                "required",
            }, "MCP elicitation supports only the restricted root schema fields"
            assert form.requestedSchema["type"] == "object"
            assert form.requestedSchema["properties"] == {
                "approved": {
                    "type": "boolean",
                    "default": False,
                    "title": "同意依上述設定開始正式執行",
                }
            }
            assert (
                context.service.jobs.load_manifest(started["job_id"]).manifest_hash
                == prepared["manifest_hash"]
            )
            repeated = await _call(
                client, "request_execution_confirmation", plan_id=prepared["plan_id"]
            )
            assert repeated == started
            assert len(approval.requests) == 1
            assert len(context.runner.calls) == 1

    asyncio.run(check())


@pytest.mark.parametrize(
    "action,approved", [("decline", True), ("cancel", True), ("accept", False)]
)
def test_unapproved_elicitation_cannot_create_a_job(
    tmp_path: Path, action: Any, approved: bool
) -> None:
    context = plan_context(tmp_path)
    approval = SimulatedUserApproval(action=action, approved=approved)

    async def check() -> None:
        async with create_connected_server_and_client_session(
            context.server, elicitation_callback=approval
        ) as client:
            prepared = await _call(client, "prepare_backtest", **context.arguments)
            result = await _call(
                client, "request_execution_confirmation", plan_id=prepared["plan_id"]
            )
            assert (
                result["success"] is False and result["code"] == "confirmation_declined"
            )
            retained = await _call(
                client, "get_execution_plan", plan_id=prepared["plan_id"]
            )
            assert retained["status"] == "cancelled"
            assert len(approval.requests) == 1

    asyncio.run(check())
    assert context.service.jobs.list_all_jobs() == []
    assert context.runner.calls == []


def test_unsupported_client_cannot_launch_or_consume_the_ready_plan(
    tmp_path: Path,
) -> None:
    context = plan_context(tmp_path)

    async def check() -> None:
        async with create_connected_server_and_client_session(context.server) as client:
            prepared = await _call(client, "prepare_backtest", **context.arguments)
            result = await _call(
                client, "request_execution_confirmation", plan_id=prepared["plan_id"]
            )
            assert (
                result["success"] is False
                and result["code"] == "confirmation_unsupported"
            )
            assert (
                await _call(client, "get_execution_plan", plan_id=prepared["plan_id"])
            )["status"] == "ready"

    asyncio.run(check())
    assert context.runner.calls == []
    assert context.service.jobs.list_all_jobs() == []


@pytest.mark.parametrize(
    "content",
    [{}, {"approved": "true"}, {"approved": 1}, {"approved": True, "unexpected": True}],
)
def test_malformed_user_answer_never_authorizes_execution(
    tmp_path: Path, content: dict[str, Any]
) -> None:
    context = plan_context(tmp_path)

    async def answer(
        context: RequestContext[ClientSession, Any], params: ElicitRequestParams
    ) -> ElicitResult:
        return ElicitResult(action="accept", content=content)

    async def check() -> None:
        async with create_connected_server_and_client_session(
            context.server, elicitation_callback=answer
        ) as client:
            prepared = await _call(client, "prepare_backtest", **context.arguments)
            result = await _call(
                client, "request_execution_confirmation", plan_id=prepared["plan_id"]
            )
            assert result["success"] is False
            assert result["code"] in {"confirmation_declined", "confirmation_failed"}

    asyncio.run(check())
    assert context.runner.calls == []
    assert context.service.jobs.list_all_jobs() == []


def test_client_error_releases_confirmation_for_another_attempt(tmp_path: Path) -> None:
    context = plan_context(tmp_path)

    async def answer(
        context: RequestContext[ClientSession, Any], params: ElicitRequestParams
    ) -> ErrorData:
        return ErrorData(code=-32603, message="Test UI failed before any user decision")

    async def check() -> None:
        async with create_connected_server_and_client_session(
            context.server, elicitation_callback=answer
        ) as client:
            prepared = await _call(client, "prepare_backtest", **context.arguments)
            result = await _call(
                client, "request_execution_confirmation", plan_id=prepared["plan_id"]
            )
            assert result["code"] == "confirmation_failed"
            assert (
                await _call(client, "get_execution_plan", plan_id=prepared["plan_id"])
            )["status"] == "ready"

    asyncio.run(check())
    assert context.runner.calls == []
    assert context.service.jobs.list_all_jobs() == []


def test_elicitation_timeout_releases_plan_without_launching_a_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = plan_context(tmp_path)
    approval = SimulatedUserApproval()

    async def timeout(self: Any, message: str, schema: type[Any]) -> Any:
        # Substitute only the timed-out transport await; MCP dispatch, persisted
        # plan state and the subsequent simulated user's response remain real.
        raise TimeoutError("Test confirmation transport timed out")

    async def check() -> None:
        async with create_connected_server_and_client_session(
            context.server, elicitation_callback=approval
        ) as client:
            prepared = await _call(client, "prepare_backtest", **context.arguments)
            with monkeypatch.context() as patch:
                patch.setattr(Context, "elicit", timeout)
                rejected = await _call(
                    client,
                    "request_execution_confirmation",
                    plan_id=prepared["plan_id"],
                )
            assert rejected["code"] == "confirmation_failed"
            assert context.runner.calls == []
            assert context.service.jobs.list_all_jobs() == []
            assert (
                await _call(client, "get_execution_plan", plan_id=prepared["plan_id"])
            )["status"] == "ready"
            started = await _call(
                client, "request_execution_confirmation", plan_id=prepared["plan_id"]
            )
            assert started["success"] is True
            assert len(approval.requests) == 1
            assert len(context.runner.calls) == 1

    asyncio.run(check())


def test_concurrent_confirmation_has_one_active_question_and_one_launch(
    tmp_path: Path,
) -> None:
    context = plan_context(tmp_path)

    async def check() -> None:
        requested, release = asyncio.Event(), asyncio.Event()

        async def answer(
            context: RequestContext[ClientSession, Any], params: ElicitRequestParams
        ) -> ElicitResult:
            requested.set()
            await release.wait()
            return ElicitResult(action="accept", content={"approved": True})

        async with (
            create_connected_server_and_client_session(
                context.server, elicitation_callback=answer
            ) as client,
            create_connected_server_and_client_session(
                context.server, elicitation_callback=SimulatedUserApproval()
            ) as contender,
        ):
            prepared = await _call(client, "prepare_backtest", **context.arguments)
            first = asyncio.create_task(
                _call(
                    client,
                    "request_execution_confirmation",
                    plan_id=prepared["plan_id"],
                )
            )
            try:
                await asyncio.wait_for(requested.wait(), 5)
                second = await asyncio.wait_for(
                    _call(
                        contender,
                        "request_execution_confirmation",
                        plan_id=prepared["plan_id"],
                    ),
                    timeout=5,
                )
                assert second["success"] is False and second["code"] == "plan_not_ready"
                assert context.runner.calls == []
            finally:
                release.set()
            assert (await first)["success"] is True

    asyncio.run(check())
    assert len(context.runner.calls) == 1


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
        async with create_connected_server_and_client_session(context.server) as client:
            result = await _call(client, "prepare_backtest", **context.arguments)
            assert (
                result["success"] is False
                and result["code"] == "invalid_execution_request"
            )
            assert "YAML" in result["error"]

    asyncio.run(check())
    assert context.runner.calls == []
    assert context.service.jobs.list_all_jobs() == []


def test_model_cannot_supply_approval_as_a_tool_argument(tmp_path: Path) -> None:
    context = plan_context(tmp_path)
    approval = SimulatedUserApproval()

    async def check() -> None:
        async with create_connected_server_and_client_session(
            context.server, elicitation_callback=approval
        ) as client:
            prepared = await _call(client, "prepare_backtest", **context.arguments)
            for argument in ("approved", "confirmed", "confirmation_token"):
                result = await client.call_tool(
                    "request_execution_confirmation",
                    {"plan_id": prepared["plan_id"], argument: True},
                )
                assert result.isError
            assert approval.requests == []

    asyncio.run(check())
    assert context.runner.calls == []
