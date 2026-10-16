"""User confirmation must reach HTTP clients before an execution can launch."""

from __future__ import annotations

import asyncio
import socket
from typing import TYPE_CHECKING, Any, Literal

import pytest
from mcp.server.fastmcp import Context
from mcp.types import ElicitResult
from tests.integration.mcp_harness import SimulatedUserApproval
from tests.mcp.http_harness import http_plan_server

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.server.elicitation import ElicitationResult
    from mcp.shared.context import RequestContext
    from mcp.types import ElicitRequestParams
    from pydantic import BaseModel

    from tradingdev.app.execution_plan_service import ConfirmationChallenge


@pytest.fixture
def bounded_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Context.elicit

    async def bounded(
        self: Context[Any, Any], message: str, schema: type[BaseModel]
    ) -> ElicitationResult[BaseModel]:
        # Preserve the actual request and response exchange, but bound a broken
        # transport without waiting the production five-minute human deadline.
        async with asyncio.timeout(2):
            return await original(self, message, schema)

    monkeypatch.setattr(Context, "elicit", bounded)


@pytest.mark.parametrize("action", ["accept", "decline"])
def test_http_confirmation_reaches_client_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    bounded_confirmation: None,
    action: Literal["accept", "decline"],
) -> None:
    approval = SimulatedUserApproval(action=action)

    async def check() -> None:
        async with http_plan_server(monkeypatch) as runtime:
            plan = runtime.context

            async def answer(
                context: RequestContext[ClientSession, Any], params: ElicitRequestParams
            ) -> ElicitResult:
                assert plan.runner.calls == []
                assert plan.service.jobs.list_all_jobs() == []
                return await approval(context, params)

            async with runtime.connect(elicitation_callback=answer) as client:
                prepared = await client.call("prepare_backtest", **plan.arguments)
                assert prepared["success"] and prepared["status"] == "ready"
                result = await client.call(
                    "request_execution_confirmation", plan_id=prepared["plan_id"]
                )
                assert len(approval.requests) == 1, result
                assert prepared["confirmation_text"] in approval.requests[0].message
                assert approval.requests[0].mode == "form"
                retained = await client.call(
                    "get_execution_plan", plan_id=prepared["plan_id"]
                )
                if action == "accept":
                    assert result["success"]
                    assert result["manifest_hash"] == prepared["manifest_hash"]
                    assert retained["job_id"] == result["job_id"]
                    repeated = await client.call(
                        "request_execution_confirmation", plan_id=prepared["plan_id"]
                    )
                    assert repeated == result
                    assert len(approval.requests) == 1
                    assert len(plan.runner.calls) == 1
                else:
                    assert result["code"] == "confirmation_declined"
                    assert retained["status"] == "cancelled"
                    assert plan.runner.calls == []
                    assert plan.service.jobs.list_all_jobs() == []

    asyncio.run(check())


def test_http_client_without_elicitation_leaves_plan_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def check() -> None:
        async with (
            http_plan_server(monkeypatch) as runtime,
            runtime.connect() as client,
        ):
            prepared = await client.call(
                "prepare_backtest", **runtime.context.arguments
            )
            result = await client.call(
                "request_execution_confirmation", plan_id=prepared["plan_id"]
            )
            assert result["code"] == "confirmation_unsupported"
            retained = await client.call(
                "get_execution_plan", plan_id=prepared["plan_id"]
            )
            assert retained["status"] == "ready"
            assert runtime.context.runner.calls == []
            assert runtime.context.service.jobs.list_all_jobs() == []

    asyncio.run(check())


def test_http_unanswered_confirmation_releases_plan_without_launch(
    monkeypatch: pytest.MonkeyPatch, bounded_confirmation: None
) -> None:
    async def check() -> None:
        requested = asyncio.Event()
        released = asyncio.Event()
        loop = asyncio.get_running_loop()

        async def unanswered(
            context: RequestContext[ClientSession, Any], params: ElicitRequestParams
        ) -> ElicitResult:
            requested.set()
            # The SDK waits for this callback in its receive loop. Keep the form
            # unanswered until the server has abandoned it, then release that
            # loop to receive the already-produced timeout response.
            await released.wait()
            return ElicitResult(action="cancel")

        async with http_plan_server(monkeypatch) as runtime:
            original_abandon = runtime.context.service.abandon_confirmation

            def observe_abandon(challenge: ConfirmationChallenge) -> None:
                original_abandon(challenge)
                loop.call_soon_threadsafe(released.set)

            monkeypatch.setattr(
                runtime.context.service, "abandon_confirmation", observe_abandon
            )
            async with runtime.connect(elicitation_callback=unanswered) as client:
                prepared = await client.call(
                    "prepare_backtest", **runtime.context.arguments
                )
                result = await client.call(
                    "request_execution_confirmation", plan_id=prepared["plan_id"]
                )
                assert requested.is_set(), result
                assert released.is_set()
                assert result["code"] == "confirmation_failed"
                retained = await client.call(
                    "get_execution_plan", plan_id=prepared["plan_id"]
                )
                assert retained["status"] == "ready"
                assert runtime.context.runner.calls == []
                assert runtime.context.service.jobs.list_all_jobs() == []

    asyncio.run(check())


@pytest.mark.parametrize("exit_mode", ["normal", "exception", "timeout"])
def test_http_harness_closes_listener_tasks_and_temporary_files(
    monkeypatch: pytest.MonkeyPatch, exit_mode: str
) -> None:
    async def check() -> None:
        try:
            async with (
                http_plan_server(monkeypatch) as runtime,
                runtime.connect() as client,
            ):
                prepared = await client.call(
                    "prepare_backtest", **runtime.context.arguments
                )
                assert prepared["success"]
                assert runtime.root.exists()
                if exit_mode == "exception":
                    raise ValueError("Test body failed")
                if exit_mode == "timeout":
                    async with asyncio.timeout(0):
                        await asyncio.Event().wait()
        except BaseException as error:
            # The SDK's task-group transports wrap errors raised by the caller.
            def leaves(exception: BaseException) -> list[BaseException]:
                if isinstance(exception, BaseExceptionGroup):
                    return [
                        leaf for child in exception.exceptions for leaf in leaves(child)
                    ]
                return [exception]

            expected = ValueError if exit_mode == "exception" else TimeoutError
            assert exit_mode != "normal"
            assert all(isinstance(leaf, expected) for leaf in leaves(error)), error
        else:
            assert exit_mode == "normal"
        assert runtime.task.done() and not runtime.task.cancelled()
        assert runtime.task.exception() is None
        assert not runtime.server.server_state.tasks
        assert not runtime.server.server_state.connections
        assert runtime.listener.fileno() == -1
        assert not runtime.root.exists()
        with socket.socket() as probe:
            probe.settimeout(0.5)
            assert probe.connect_ex(runtime.address) != 0

    asyncio.run(check())
