"""Opt-in product discovery with natural intent and the complete tool surface."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.e2e.codex_harness import run_codex, verify_generated_strategy
from tests.e2e.conversation_user import ObservedPlanUser
from tests.e2e.discovery_workflow import NATURAL_REQUEST, verify_discovery_outcome
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.e2e.strategy_scenarios import SCENARIOS, market_frame
from tests.e2e.workflow_diagnostics import workflow_diagnostics
from tests.integration.mcp_harness import temporary_mcp_workspace

pytestmark = pytest.mark.live_llm


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
