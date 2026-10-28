"""Opt-in discovery of fixed comparison from an ordinary research request."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest

from tests.e2e.codex_harness import run_codex
from tests.e2e.comparison_diagnostics import comparison_failure_diagnostics
from tests.e2e.llm_client import codex_calls, run_local_model
from tests.e2e.strategy_scenarios import market_frame
from tests.integration.mcp_harness import temporary_mcp_workspace
from tests.integration.test_comparison_workflow import (
    ComparisonUser,
    verify_comparison_delivery,
)

if TYPE_CHECKING:
    from tests.e2e.conversation_user import ObservedPlanUser
    from tests.e2e.llm_client import ToolCall
    from tests.integration.mcp_harness import MCPWorkspace


REQUEST = (
    "幫我測試內建的 KD 交叉策略，沿用它原本的買賣規則，"
    "比較 K 週期 9 根和 14 根兩組，其他指標設定保持原設定。"
    "標的是 BTC/USDT 永續合約，用一小時 K 棒，"
    "期間為 UTC 2024 年 1 月 1 日 00:00 到 1 月 8 日 00:00。"
    "這次只比較這兩組，不要擴大搜尋。"
)


async def _run(
    workspace: MCPWorkspace,
    pytestconfig: pytest.Config,
    user: ObservedPlanUser,
    calls: list[ToolCall],
    request: str = REQUEST,
) -> None:
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}
    options: dict[str, Any] = {
        "timeout_seconds": pytestconfig.getoption("llm_timeout"),
        "model": pytestconfig.getoption("llm_model"),
        "user_reply": user,
        "on_tool_call": calls.append,
    }
    async with workspace.connect() as client:
        names = list(client.output_schemas)
        assert all(name not in request for name in names)
        options["tool_names"] = names
        if provider == "local":
            await run_local_model(
                client,
                request,
                base_url=pytestconfig.getoption("llm_base_url"),
                temperature=pytestconfig.getoption("llm_temperature"),
                reasoning_effort=pytestconfig.getoption("llm_reasoning_effort"),
                **options,
            )
            return
    result = codex_calls(await run_codex(workspace.root, request, **options))
    assert calls == result, "The observer must retain all raw tool attempts"


@pytest.mark.live_llm
def test_natural_fixed_comparison_and_complete_report(
    pytestconfig: pytest.Config,
) -> None:
    with temporary_mcp_workspace() as workspace:
        workspace.seed_market(market_frame())
        user = ComparisonUser(workspace)
        calls: list[ToolCall] = []
        try:
            asyncio.run(_run(workspace, pytestconfig, user, calls))
            verify_comparison_delivery(workspace, user, calls)
        except BaseException as exc:
            for turn in user.turns:
                exc.add_note(f"Completed comparison turn {turn.number}:\n{turn.text}")
            exc.add_note(
                "Comparison tool diagnostics:\n"
                + json.dumps(
                    comparison_failure_diagnostics(calls), ensure_ascii=False, indent=2
                )
            )
            raise
