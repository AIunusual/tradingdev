"""HTTP preparation and document reads stay separate from explicit decisions."""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from typing import Any

import httpx
import pytest
from sse_starlette.sse import AppStatus
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
            assert result.is_error
            retained = await client.call(
                "get_execution_plan", plan_id=prepared["plan_id"]
            )
            assert retained["status"] == "ready"
            assert runtime.context.runner.calls == []
            assert runtime.context.service.jobs.list_all_jobs() == []

    asyncio.run(check())


@pytest.mark.parametrize("exit_mode", ["normal", "exception", "timeout"])
@pytest.mark.parametrize("previous_shutdown", [False, True])
def test_http_harness_closes_listener_tasks_and_temporary_files(
    monkeypatch: pytest.MonkeyPatch, exit_mode: str, previous_shutdown: bool
) -> None:
    monkeypatch.setattr(AppStatus, "should_exit", previous_shutdown)

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
        assert runtime.shutdown_watchers
        assert all(watcher.done() for watcher in runtime.shutdown_watchers)
        assert all(not watcher.cancelled() for watcher in runtime.shutdown_watchers)
        assert all(watcher.exception() is None for watcher in runtime.shutdown_watchers)
        assert not runtime.server.server_state.tasks
        assert not runtime.server.server_state.connections
        assert runtime.listener.fileno() == -1
        assert not runtime.root.exists()
        assert AppStatus.should_exit is previous_shutdown
        with socket.socket() as probe:
            probe.settimeout(0.5)
            assert probe.connect_ex(runtime.address) != 0

    asyncio.run(check())


def test_http_harness_can_restart_within_one_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def check() -> None:
        for _ in range(2):
            async with (
                http_plan_server(monkeypatch) as runtime,
                runtime.connect() as client,
            ):
                prepared = await client.call(
                    "prepare_backtest", **runtime.context.arguments
                )
                assert prepared["success"]
            assert all(watcher.done() for watcher in runtime.shutdown_watchers)
            assert not runtime.root.exists()

    asyncio.run(check())


def test_http_request_size_limit_accepts_four_mib_and_rejects_larger_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def check() -> None:
        async with http_plan_server(monkeypatch) as runtime:
            host, port = runtime.address
            url = f"http://{host}:{port}/mcp"
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }
            request = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "body-limit-test", "version": "1"},
                    },
                }
            ).encode()
            body = request.ljust(4 * 1024 * 1024, b" ")
            async with httpx.AsyncClient(timeout=5) as client:
                accepted = await client.post(url, headers=headers, content=body)
                assert accepted.status_code == 200, accepted.text
                session_id = accepted.headers["mcp-session-id"]
                try:
                    rejected = await client.post(
                        url,
                        headers={
                            **headers,
                            "Mcp-Session-Id": session_id,
                            "MCP-Protocol-Version": "2025-11-25",
                        },
                        content=body + b" ",
                    )
                    assert rejected.status_code == 413
                finally:
                    deleted = await client.delete(
                        url,
                        headers={
                            **headers,
                            "Mcp-Session-Id": session_id,
                            "MCP-Protocol-Version": "2025-11-25",
                        },
                    )
                    assert deleted.status_code == 200
            assert runtime.context.runner.calls == []
            assert runtime.context.service.jobs.list_all_jobs() == []
        assert runtime.task.done()
        assert not runtime.root.exists()

    asyncio.run(check())
