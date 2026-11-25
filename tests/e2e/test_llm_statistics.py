"""Natural request reaches saved-run statistics without executing a strategy."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import pytest

from tests.app.test_statistics_analysis import _publish
from tests.e2e.codex_harness import run_codex
from tests.e2e.conversation import visible_text
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn

REQUEST = (
    "請分析已保存結果 statrun 的統計摘要與可用推論，說明不可用的原因；"
    "不要重新執行策略或抓行情。"
)
FORBIDDEN = {
    "prepare",
    "prepare_backtest",
    "prepare_comparison",
    "prepare_walk_forward",
    "prepare_optimization",
    "submit_execution_decision",
    "ensure_data",
    "save_strategy",
    "validate_strategy",
    "dry_run_strategy",
    "promote_strategy",
    "cleanup_strategy_drafts",
}


@pytest.mark.live_llm
def test_natural_request_uses_saved_statistics_analysis(
    pytestconfig: pytest.Config,
) -> None:
    with temporary_mcp_workspace() as harness:
        workspace = WorkspacePaths(harness.workspace)
        _publish(workspace, SQLiteStore(workspace), run_id="statrun")
        turns: list[ModelTurn] = []
        observed: list[ToolCall] = []

        async def observe_final(turn: ModelTurn) -> None:
            turns.append(turn)

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "model": pytestconfig.getoption("llm_model"),
                "reasoning_effort": pytestconfig.getoption("llm_reasoning_effort"),
                "user_reply": observe_final,
                "max_tool_calls": 12,
                "on_tool_call": observed.append,
            }
            async with harness.connect() as client:
                names = [
                    tool.name for tool in (await client.session.list_tools()).tools
                ]
                assert all(name not in REQUEST for name in names)
                options["tool_names"] = names
                if pytestconfig.getoption("llm_provider") == "local":
                    return await run_local_model(
                        client,
                        REQUEST,
                        base_url=pytestconfig.getoption("llm_base_url"),
                        temperature=pytestconfig.getoption("llm_temperature"),
                        **options,
                    )
            return codex_calls(await run_codex(harness.root, REQUEST, **options))

        try:
            calls = asyncio.run(run())
        except Exception as error:
            error.add_note(
                "Statistics calls before model failure:\n"
                + json.dumps([asdict(call) for call in observed], ensure_ascii=False)
            )
            raise
        print(
            json.dumps([asdict(call) for call in calls], ensure_ascii=False), flush=True
        )
        analysis = [call for call in calls if call.name == "analyze_run_statistics"]
        assert analysis
        assert analysis[0].arguments.get("run_id") == "statrun"
        assert not {call.name for call in calls} & FORBIDDEN
        assert len(turns) == 1
        text = visible_text(turns[0].text)
        assert "statrun" in text
        assert "不可用" in text or "unavailable" in text.lower()
    assert not harness.root.exists()
