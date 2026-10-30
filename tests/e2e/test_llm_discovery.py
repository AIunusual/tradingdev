"""Opt-in product discovery with natural intent and the complete tool surface."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

import pytest

from tests.e2e.codex_harness import run_codex, verify_generated_strategy
from tests.e2e.conversation_user import ObservedPlanUser
from tests.e2e.discovery_workflow import NATURAL_REQUEST, verify_discovery_outcome
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.e2e.strategy_scenarios import SCENARIOS, market_frame
from tests.e2e.workflow_diagnostics import workflow_diagnostics
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.job_store import JobStore

pytestmark = pytest.mark.live_llm

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn


def test_natural_request_discovers_strategy_and_report(
    pytestconfig: pytest.Config,
) -> None:
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}
    with (
        temporary_mcp_workspace() as workspace,
        workflow_diagnostics(workspace, SCENARIOS["sma"]),
    ):
        workspace.seed_market(market_frame())
        user = ObservedPlanUser(workspace)

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "model": pytestconfig.getoption("llm_model"),
                "user_reply": user,
            }
            async with workspace.connect() as client:
                tools = (await client.session.list_tools()).tools
                names = [tool.name for tool in tools]
                assert all(name not in NATURAL_REQUEST for name in names), (
                    "Natural user intent must not supply tool names"
                )
                print(
                    f"[discovery] Complete product tool set: {len(names)}", flush=True
                )
                options["tool_names"] = names
                if provider == "local":
                    return await run_local_model(
                        client,
                        NATURAL_REQUEST,
                        base_url=pytestconfig.getoption("llm_base_url"),
                        temperature=pytestconfig.getoption("llm_temperature"),
                        reasoning_effort=pytestconfig.getoption("llm_reasoning_effort"),
                        **options,
                    )
            return codex_calls(
                await run_codex(workspace.root, NATURAL_REQUEST, **options)
            )

        calls = asyncio.run(run())
        user.assert_complete()
        verify_discovery_outcome(workspace, calls, user.turns[-1])
        print(
            "[discovery] Final model reply (test evidence; links expire at cleanup):\n"
            + user.turns[-1].text,
            flush=True,
        )
        verify_generated_strategy(workspace.root, scenario="sma")


def test_natural_request_reads_historical_job_duration(
    pytestconfig: pytest.Config,
) -> None:
    with temporary_mcp_workspace() as workspace:
        store = JobStore(workspace=WorkspacePaths(workspace.workspace))
        for job_id, end in (
            ("timed-job", "2024-01-01T00:00:10+00:00"),
            ("legacy-job", None),
        ):
            record = store.create_job(job_id=job_id)
            store.store.upsert_job(
                {
                    **record,
                    "status": "failed",
                    "created_at": "2024-01-01T00:00:00+00:00",
                    "ended_at": end,
                    "error": "Historical fixture failure",
                }
            )
        request = (
            "請查我先前的 timed-job 和 legacy-job，各自失敗前經過幾秒？"
            "請每個工作各用一行回答。"
            "請區分有證據的耗時與無法判定的部分，缺少資料不要猜測。只查歷史，不要重新執行工作。"
        )
        turns: list[ModelTurn] = []

        async def user(turn: ModelTurn) -> None:
            turns.append(turn)

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "model": pytestconfig.getoption("llm_model"),
                "user_reply": user,
            }
            async with workspace.connect() as client:
                names = [
                    tool.name for tool in (await client.session.list_tools()).tools
                ]
                assert all(name not in request for name in names)
                options["tool_names"] = names
                if pytestconfig.getoption("llm_provider") == "local":
                    return await run_local_model(
                        client,
                        request,
                        base_url=pytestconfig.getoption("llm_base_url"),
                        temperature=pytestconfig.getoption("llm_temperature"),
                        reasoning_effort=pytestconfig.getoption("llm_reasoning_effort"),
                        **options,
                    )
            return codex_calls(await run_codex(workspace.root, request, **options))

        calls = asyncio.run(run())
        records: dict[str, dict[str, Any]] = {}
        for call in calls:
            assert call.name in {"list_jobs", "get_job_status"}, (
                "History request must not execute work"
            )
            if call.name == "list_jobs":
                records.update(
                    {row["job_id"]: row for row in call.result.get("items", [])}
                )
            else:
                records[str(call.arguments["job_id"])] = call.result
        assert records["timed-job"]["elapsed_seconds"] == 10.0
        assert records["legacy-job"]["elapsed_seconds"] is None
        assert (
            records["legacy-job"]["elapsed_unavailable_reason"]
            == "missing_or_invalid_timestamps"
        )
        assert turns
        known_lines = " ".join(
            line for line in turns[-1].text.splitlines() if "timed-job" in line
        )
        assert re.search(r"(?<!\d)10(?:\.0)?(?!\d)", known_lines)
        unknown_lines = " ".join(
            line for line in turns[-1].text.splitlines() if "legacy-job" in line
        )
        assert re.search(
            "無法|未知|缺|不足|不明|不能|unknown|unavailable",
            unknown_lines,
            re.IGNORECASE,
        )
        print("[historical-duration] Final model reply:\n" + turns[-1].text, flush=True)
