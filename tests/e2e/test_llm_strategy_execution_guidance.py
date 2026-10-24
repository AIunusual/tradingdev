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
from tests.e2e.strategy_execution_guidance import (
    GUIDANCE_PROMPT,
    GUIDANCE_TOOLS,
    assert_guidance_workflow,
    assert_guidance_workspace,
)
from tests.integration.mcp_harness import temporary_mcp_workspace

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn


@pytest.mark.live_llm
def test_llm_strategy_execution_guidance(pytestconfig: pytest.Config) -> None:
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}, "Use scripts/check-llm.sh codex|local"
    with temporary_mcp_workspace() as workspace:
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
