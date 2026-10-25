"""Natural user intent and outcome checks without a tool-by-tool answer script."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.e2e.report_delivery import verify_report_delivery
from tests.e2e.workflow_arguments import effective_workflow_calls
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.domain.presentation.reports import REPORT_RECIPES

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn
    from tests.e2e.llm_client import ToolCall
    from tests.integration.mcp_harness import MCPWorkspace


NATURAL_REQUEST = """請建立名為 llm_sma 的策略，使用 BTC/USDT 的 1 小時收盤價。
短期簡單均線高於長期時做多，低於時做空，相等或資料不足時空手。
參數 fast_period=5、slow_period=20，兩個週期都要能調整。
初始資金 10,000 USDT，手續費與滑價都為零，亂數種子 42，
期間為 UTC 2024-01-01 00:00:00 至 2024-01-08 00:00:00，365 日年化。
行情已放在後端快取，使用這份資料。
請完成回測並說明績效、風險與限制，讓我能查看這組參數的詳細交易紀錄。
"""


def verify_discovery_outcome(
    workspace: MCPWorkspace, calls: list[ToolCall], final_turn: ModelTurn
) -> str:
    """Bind saved execution and usable final report, independent of prose wording."""
    effective = effective_workflow_calls(calls)
    saves = [
        index
        for index, call in enumerate(effective)
        if call.name == "save_strategy"
        and call.arguments.get("strategy_id") == "llm_sma"
        and call.result.get("success") is True
    ]
    assert saves, "The natural request must successfully save its generated strategy"
    discovery = effective[: saves[0]]
    assert any(call.name == "list_strategies" for call in discovery), (
        "The natural request must discover existing strategies before authoring"
    )
    assert any(
        call.name == "get_strategy_contract"
        and call.result.get("section") == "core"
        and call.result.get("content")
        for call in discovery
    ), "The natural request must read the core contract before authoring"
    paths = WorkspacePaths(workspace.workspace)
    runs = SQLiteStore(paths).list_runs()
    assert len(runs) == 1, "Reporting must reuse exactly one completed execution"
    run = runs[0]
    assert run["strategy_id"] == "llm_sma"
    assert isinstance(run["revision_id"], str) and run["revision_id"]
    run_id = run["run_id"]
    assert any(
        call.name == "get_job_status"
        and call.result.get("status") == "done"
        and call.result.get("run_id") == run_id
        and call.result.get("revision_id") == run["revision_id"]
        for call in effective
    ), "The client must wait for this saved revision's execution"
    assert any(
        call.name in {"get_run", "get_run_metrics"}
        and call.arguments.get("run_id") == run_id
        and call.result.get("success") is True
        for call in effective
    ), "Interpretation must inspect the saved result"
    verify_report_delivery(
        paths,
        final_turn,
        run_id,
        expected_sections=list(REPORT_RECIPES["standard"]),
        expected_commentary=None,
    )
    return str(run_id)
