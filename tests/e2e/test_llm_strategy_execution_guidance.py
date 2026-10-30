"""Real model/MCP discovery and rejected preflights; no market fetch or training.

Only the two unsupported preparation modes are exposed. Their production workers
reject the request before fetching data or fitting QuantileStrategy. No worker or
service is substituted, and the final callback never grants execution approval.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import pytest

from tests.e2e.codex_harness import run_codex
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.e2e.report_delivery import verify_no_execution_attempts
from tests.e2e.strategy_execution_guidance import (
    GUIDANCE_PROMPT,
    GUIDANCE_TOOLS,
    assert_guidance_workflow,
    assert_guidance_workspace,
    seed_guidance_evidence,
)
from tests.integration.mcp_harness import temporary_mcp_workspace

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn


@pytest.mark.live_llm
def test_llm_strategy_execution_guidance(pytestconfig: pytest.Config) -> None:
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}, "Use scripts/check-llm.sh codex|local"
    with temporary_mcp_workspace() as workspace:
        seed_guidance_evidence(workspace)
        turns: list[ModelTurn] = []

        async def observe_final(turn: ModelTurn) -> None:
            turns.append(turn)

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "model": pytestconfig.getoption("llm_model"),
                "tool_names": GUIDANCE_TOOLS,
                "user_reply": observe_final,
                "max_tool_calls": 8,
                "max_turns": 1,
            }
            if provider == "codex":
                return codex_calls(
                    await run_codex(workspace.root, GUIDANCE_PROMPT, **options)
                )
            async with workspace.connect() as client:
                return await run_local_model(
                    client,
                    GUIDANCE_PROMPT,
                    base_url=pytestconfig.getoption("llm_base_url"),
                    temperature=pytestconfig.getoption("llm_temperature"),
                    reasoning_effort=pytestconfig.getoption("llm_reasoning_effort"),
                    **options,
                )

        calls = asyncio.run(run())
        try:
            assert_guidance_workflow(calls, turns)
            assert_guidance_workspace(workspace)
        except AssertionError as error:
            transcript = {
                "calls": [asdict(call) for call in calls],
                "turns": [asdict(turn) for turn in turns],
            }
            pytest.fail(
                f"Incomplete {provider}/strategy execution guidance: {error}\n"
                + json.dumps(transcript, ensure_ascii=False)
            )
    assert not workspace.root.exists()


@pytest.mark.live_llm
def test_llm_volume_monthly_loss_explanation(pytestconfig: pytest.Config) -> None:
    """Discover current policy from natural questions, including a follow-up."""
    with temporary_mcp_workspace() as workspace:
        turns: list[ModelTurn] = []

        async def user(turn: ModelTurn) -> str | None:
            turns.append(turn)
            if len(turns) == 1:
                return (
                    "若一月進場時已扣過手續費，跨月平倉時還會算進本月限制嗎？"
                    "如果二、三月都沒有行情，到四月才出現下一根 K 棒又怎麼算？"
                    "請沿用剛才查到的規則說明，不執行任何研究。"
                )
            return None

        request = (
            "請查明 TradingDev 的 Volume 模式如何計算每月虧損限制。"
            "假設一月買進後跨月持有，月底累積損益為 20，二月第一根收盤變成 10，"
            "每月最大虧損設 5，零手續費及滑價。會繼續持有嗎？"
            "交易明細應保留哪一段損益？只解釋目前規則，不建立或執行策略。"
        )

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "model": pytestconfig.getoption("llm_model"),
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "user_reply": user,
                "max_turns": 2,
            }
            async with workspace.connect() as client:
                options["tool_names"] = [
                    tool.name for tool in (await client.session.list_tools()).tools
                ]
                assert all(name not in request for name in options["tool_names"])
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
        verify_no_execution_attempts(calls)
        assert len(turns) == 2 and all(turn.text.strip() for turn in turns)
        for index, turn in enumerate(turns, 1):
            print(
                f"[monthly loss] Semantic review, turn {index}:\n{turn.text}",
                flush=True,
            )
        assert any(
            call.name == "get_strategy_contract"
            and call.result.get("section") == "configuration"
            and "previous observed month's last close equity" in json.dumps(call.result)
            and "Trade records retain whole-trade P&L and costs"
            in json.dumps(call.result)
            for call in turns[0].calls
        ), "The first explanation must be grounded in the current execution contract"
    assert not workspace.root.exists()
