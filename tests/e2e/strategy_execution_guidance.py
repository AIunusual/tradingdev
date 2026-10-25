"""Evidence for discovery and recovery guidance without executing a strategy."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from typing import TYPE_CHECKING, Any

import yaml

from tests.e2e.llm_client import tool_payload

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn
    from tests.e2e.llm_client import ToolCall
    from tests.integration.mcp_harness import MCPWorkspace

STRATEGY_ID = "quantile_volume"
GUIDANCE_TOOLS = [
    "list_strategies",
    "get_strategy",
    "prepare_backtest",
    "prepare_optimization",
]
_CAPABILITIES = {
    "training_requirement": "required",
    "supported_execution_modes": ["walk_forward"],
}
_PRESENTATION = {
    "title": "策略入口相容性診斷",
    "summary": "只確認不支援的入口如何回覆，不啟動策略研究。",
    "parameter_descriptions": {},
}


def guidance_arguments(tool: str) -> dict[str, Any]:
    """Valid diagnostic requests which must be rejected before market access."""
    if tool == "list_strategies":
        return {}
    if tool == "get_strategy":
        return {"strategy_id": STRATEGY_ID}
    arguments: dict[str, Any] = {
        "strategy_id": STRATEGY_ID,
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "minimum_history_bars": 20,
        "sample_bars": 128,
        "presentation": _PRESENTATION,
    }
    if tool == "prepare_backtest":
        return {
            **arguments,
            "start_date": "2024-01-01",
            "end_date": "2024-01-08",
        }
    if tool == "prepare_optimization":
        return {
            **arguments,
            "param_ranges": {"horizon": [15, 30]},
            "optimization_metric": "total_return",
            "train_start": "2024-01-01",
            "train_end": "2024-01-03",
            "test_start": "2024-01-04",
            "test_end": "2024-01-08",
        }
    raise ValueError(f"Unsupported diagnostic tool: {tool}")


GUIDANCE_PROMPT = f"""請用 TradingDev MCP 診斷內建策略 {STRATEGY_ID} 的執行入口。
先 list_strategies，再 get_strategy(strategy_id="{STRATEGY_ID}")，讀取能力欄位與 YAML。
這次我明確要求測試兩個不支援入口的拒絕回覆，請依序各呼叫一次：
prepare_backtest，參數如下：
{json.dumps(guidance_arguments("prepare_backtest"), ensure_ascii=False)}
prepare_optimization，參數如下：
{json.dumps(guidance_arguments("prepare_optimization"), ensure_ascii=False)}
兩次呼叫都應保留原策略與設定；不要移除設定或更改參數來繞過拒絕。
只呼叫上述四次工具，不下載行情、不儲存或修改策略，不進行訓練或正式執行。
收到拒絕後只說明下一步，不呼叫建議的工具，也不提交執行決策；我尚未核准執行。
工具呼叫期間不要額外文字敘述或進度訊息，所有文字輸出只包含最後的 JSON。
最後只回覆一個 JSON 物件，不加 Markdown fence：strategy_id、execution_capabilities、
rejections、execution_started。execution_capabilities 原值引用探索結果；rejections
依兩次拒絕的順序，每筆包含 tool、next_tool、required_configuration，後兩項原值引用
各自的工具回覆，不猜測工具名或設定鍵。execution_started 如實表示是否已啟動執行。
完成後結束這個回合，等待我決定是否繼續。
"""


def _payload(call: ToolCall) -> Any:
    result = call.result
    if isinstance(result, dict) and (
        "structuredContent" in result or "content" in result
    ):
        assert result.get("isError") is not True, (
            "MCP transport/tool error is not guidance"
        )
        return tool_payload(result)
    return result


def assert_guidance_workflow(calls: list[ToolCall], turns: list[ModelTurn]) -> None:
    """Require actual ordered tool evidence and a report derived from its values."""
    assert [call.name for call in calls] == GUIDANCE_TOOLS, (
        "Expected exactly discovery, retrieval, and two rejected preparation calls"
    )
    assert len(turns) == 1 and turns[0].number == 1, (
        "Expected one unapproved model turn"
    )
    assert turns[0].calls == calls, "Final turn must contain the actual tool evidence"
    listed = _payload(calls[0])
    assert isinstance(listed, dict) and isinstance(listed.get("items"), list), (
        "list_strategies must return a page of items"
    )
    selected = [
        item
        for item in listed["items"]
        if isinstance(item, dict) and item.get("strategy_id") == STRATEGY_ID
    ]
    assert len(selected) == 1, "Discovery must identify the requested strategy"
    assert selected[0].get("execution_capabilities") == _CAPABILITIES, (
        "Discovery must advertise the training requirement and supported modes"
    )
    retrieved = _payload(calls[1])
    assert calls[1].arguments.get("strategy_id") == STRATEGY_ID
    assert isinstance(retrieved, dict) and retrieved.get("success") is True
    assert retrieved.get("strategy_id") == STRATEGY_ID
    assert retrieved.get("execution_capabilities") == _CAPABILITIES
    configuration = yaml.safe_load(retrieved.get("yaml_config", ""))
    assert isinstance(configuration, dict) and isinstance(
        configuration.get("validation"), dict
    ), "Retrieval must expose the existing validation configuration"

    expected_rejections = []
    for call in calls[2:]:
        assert call.arguments.get("strategy_id") == STRATEGY_ID
        result = _payload(call)
        assert isinstance(result, dict) and result.get("success") is False
        assert result.get("code") == "strategy_requires_training", (
            "Require the strategy lifecycle rejection, not another preparation error"
        )
        assert isinstance(result.get("error"), str) and result["error"].strip()
        assert not result.get("plan_id") and not result.get("job_id")
        guidance = result.get("execution_guidance")
        assert guidance == {
            "strategy_id": STRATEGY_ID,
            "capabilities": _CAPABILITIES,
            "recommended_mode": "walk_forward",
            "required_configuration": ["validation"],
        }, "Rejection must retain actionable guidance for this strategy"
        assert result.get("next_tool") == "prepare_walk_forward"
        expected_rejections.append(
            {
                "tool": call.name,
                "next_tool": result["next_tool"],
                "required_configuration": guidance["required_configuration"],
            }
        )

    try:
        report = json.loads(turns[0].text)
    except (TypeError, ValueError) as error:
        raise AssertionError("Final guidance must be one JSON object") from error
    assert isinstance(report, dict), "Final guidance must be one JSON object"
    assert report == {
        "strategy_id": STRATEGY_ID,
        "execution_capabilities": retrieved["execution_capabilities"],
        "rejections": expected_rejections,
        "execution_started": False,
    }, "Final guidance must quote the discovered capabilities and rejection next steps"
    assert report["execution_started"] is False


def assert_guidance_workspace(workspace: MCPWorkspace) -> None:
    """Independently reject persisted execution and market artifacts."""
    database = workspace.workspace / "tradingdev.sqlite"
    assert database.is_file(), "The real MCP server must initialize its workspace"
    with closing(
        sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    ) as connection:
        for table in ("execution_plans", "jobs", "runs", "artifacts"):
            count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count == 0, f"Diagnostic workflow unexpectedly persisted {table}"
    for directory in (
        workspace.workspace / "execution_plans",
        workspace.workspace / "runs",
        workspace.workspace / "models",
        workspace.workspace / "ml_datasets",
        workspace.workspace / "generated_strategies",
        workspace.workspace / "configs",
        workspace.workspace / "data",
        workspace.data_root,
    ):
        assert not any(path.is_file() for path in directory.rglob("*")), (
            f"Diagnostic workflow unexpectedly saved artifacts in {directory}"
        )
