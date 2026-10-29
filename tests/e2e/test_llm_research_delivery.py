"""Natural interpretation of saved results, without a tool-by-tool script."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from tests.app.test_report_service import _save, _scope
from tests.e2e.codex_harness import run_codex
from tests.e2e.llm_client import codex_calls, run_local_model
from tests.e2e.report_delivery import (
    verify_no_execution_attempts,
    verify_report_delivery,
)
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.report_service import ReportService
from tradingdev.domain.presentation.reports import REPORT_RECIPES

pytestmark = pytest.mark.live_llm

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn


def test_saved_research_interpretation(pytestconfig: pytest.Config) -> None:
    with temporary_mcp_workspace() as workspace:
        paths = WorkspacePaths(workspace.workspace)
        store = SQLiteStore(paths)
        _save(
            (paths, store, ReportService(workspace=paths, store=store)),
            scopes={
                "full": _scope(
                    values={
                        "total_return": 0.25,
                        "max_drawdown": 0.1,
                        "total_trades": 1,
                        "win_rate": 1.0,
                    }
                )
            },
            trades=[
                {
                    "entry_idx": 0,
                    "exit_idx": 2,
                    "status": "closed",
                    "direction": 1,
                    "entry_price": 100.0,
                    "exit_price": 125.0,
                    "net_pnl": 25.0,
                }
            ],
        )
        turns: list[ModelTurn] = []

        async def user(turn: ModelTurn) -> None:
            turns.append(turn)

        request = (
            "請幫我解讀已保存的 BTC 策略回測，結果代號是 run。"
            "我要拿給同事討論：這次有哪些觀察、哪些事情還不能確定，"
            "下一步值得研究什麼？請給我能獨立閱讀的完整報告。不要重新回測。"
        )

        async def run() -> Any:
            options = {
                "model": pytestconfig.getoption("llm_model"),
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "user_reply": user,
            }
            async with workspace.connect() as client:
                options["tool_names"] = [
                    t.name for t in (await client.session.list_tools()).tools
                ]
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
        assert turns
        verify_no_execution_attempts(calls)
        verify_report_delivery(
            paths,
            turns[-1],
            "run",
            expected_sections=list(REPORT_RECIPES["standard"]),
            expected_commentary=None,
            required_analysis_refs={("run", "full")},
        )
        assert len(store.list_runs()) == 1
        print("[research delivery] Final reply:\n" + turns[-1].text, flush=True)
