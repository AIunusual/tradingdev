"""HTTP preparation and document reads stay separate from explicit decisions."""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from typing import Any

import pytest
from tests.mcp.http_harness import http_plan_server


@pytest.mark.parametrize(
    "decision,code",
    [
        ("approve", None),
        ("decline", "confirmation_declined"),
        ("revise", "confirmation_adjustment_requested"),
    ],
)
def test_http_plan_waits_for_decision_after_reading_summary_and_html(
    monkeypatch: pytest.MonkeyPatch, decision: str, code: str | None
) -> None:
    async def check() -> None:
        async with (
            http_plan_server(monkeypatch) as runtime,
            runtime.connect() as client,
        ):
            plan = runtime.context
            prepared = await client.call("prepare_backtest", **plan.arguments)
            assert prepared["success"] and prepared["status"] == "ready"
            assert prepared["confirmation_summary"]
            assert "confirmation_text" not in prepared
            artifact = await client.call(
                "get_artifact",
                artifact_id=prepared["artifact_id"],
                include_content=True,
            )
            html = Path(prepared["html_path"]).read_text(encoding="utf-8")
            assert artifact["success"] and artifact["content"] == html
            assert "<html" in html
            retained = await client.call(
                "get_execution_plan", plan_id=prepared["plan_id"]
            )
            assert retained == prepared
            assert plan.runner.calls == []
            assert plan.service.jobs.list_all_jobs() == []
            result = await client.call(
                "submit_execution_decision",
                plan_id=prepared["plan_id"],
                decision=decision,
            )
            retained = await client.call(
                "get_execution_plan", plan_id=prepared["plan_id"]
            )
            if code is None:
                assert result["success"]
                assert result["manifest_hash"] == prepared["manifest_hash"]
                assert retained["job_id"] == result["job_id"]
                repeated = await client.call(
                    "submit_execution_decision",
                    plan_id=prepared["plan_id"],
                    decision="approve",
                )
                assert repeated == result
                assert len(plan.runner.calls) == 1
            else:
                assert result["code"] == code
                assert retained["status"] == "cancelled"
                assert plan.runner.calls == []
                assert plan.service.jobs.list_all_jobs() == []

    asyncio.run(check())


@pytest.mark.parametrize("content", [{}, {"decision": "invalid"}, {"approved": True}])
def test_http_invalid_decision_does_not_consume_or_launch_plan(
    monkeypatch: pytest.MonkeyPatch, content: dict[str, Any]
) -> None:
    async def check() -> None:
        async with (
            http_plan_server(monkeypatch) as runtime,
            runtime.connect() as client,
        ):
            prepared = await client.call(
                "prepare_backtest", **runtime.context.arguments
            )
            result = await client.session.call_tool(
                "submit_execution_decision", {"plan_id": prepared["plan_id"], **content}
            )
            assert result.isError
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
