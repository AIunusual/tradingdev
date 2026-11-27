"""Natural request reaches a paired comparison without starting an experiment."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import pytest

from tests.app.test_paired_comparison import _publish_run
from tests.e2e.codex_harness import run_codex
from tests.e2e.conversation import visible_text
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn

REQUEST = (
    "請用已保存的 run-a 與 run-b 做配對比較，說明差額、共同日期、"
    "排除原因與描述性限制；不要重新回測、訓練、抓行情或重新預測。"
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
    "train_model",
    "evaluate_signal",
    "predict",
}


@pytest.mark.live_llm
def test_natural_request_uses_saved_paired_comparison(
    pytestconfig: pytest.Config,
) -> None:
    with temporary_mcp_workspace() as harness:
        workspace = WorkspacePaths(harness.workspace)
        store = SQLiteStore(workspace)
        _publish_run(workspace, store, "run-a", returns=[0.01] * 30)
        _publish_run(workspace, store, "run-b", returns=[0.02] * 30)
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
                assert "compare_saved_series" in names
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
                "Paired comparison calls before model failure:\n"
                + json.dumps([asdict(call) for call in observed], ensure_ascii=False)
            )
            raise
        print(
            json.dumps([asdict(call) for call in calls], ensure_ascii=False), flush=True
        )
        compared = [call for call in calls if call.name == "compare_saved_series"]
        assert compared
        arguments = compared[0].arguments
        assert arguments.get("a", {}).get("run_id") == "run-a"
        assert arguments.get("b", {}).get("run_id") == "run-b"
        assert not {call.name for call in calls} & FORBIDDEN
        assert len(turns) == 1
        text = visible_text(turns[0].text)
        assert "run-a" in text
        assert "描述" in text or "descriptive" in text.lower()
    assert not harness.root.exists()
