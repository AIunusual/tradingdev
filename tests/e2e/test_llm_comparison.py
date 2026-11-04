"""Natural discovery of fixed comparisons and read-only saved-scope comparisons."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest
from mcp_types import PaginatedRequestParams

from tests.app.test_run_metrics import _publish, _scope
from tests.e2e.codex_harness import run_codex
from tests.e2e.comparison_diagnostics import comparison_failure_diagnostics
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.e2e.strategy_scenarios import market_frame
from tests.integration.mcp_harness import temporary_mcp_workspace
from tests.integration.test_comparison_workflow import (
    ComparisonUser,
    verify_comparison_delivery,
)
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.contracts.research import CompareRunsResponse
from tradingdev.app.run_service import RunService

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn, UserReply
    from tests.integration.mcp_harness import MCPClient, MCPWorkspace


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
    user: UserReply,
    calls: list[ToolCall],
    request: str = REQUEST,
) -> set[str]:
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}
    options: dict[str, Any] = {
        "timeout_seconds": pytestconfig.getoption("llm_timeout"),
        "model": pytestconfig.getoption("llm_model"),
        "reasoning_effort": pytestconfig.getoption("llm_reasoning_effort"),
        "user_reply": user,
        "on_tool_call": calls.append,
    }
    async with workspace.connect() as client:
        names = list(client.output_schemas)
        read_only_tools = await _read_only_tools(client)
        assert all(name not in request for name in names)
        options["tool_names"] = names
        if provider == "local":
            await run_local_model(
                client,
                request,
                base_url=pytestconfig.getoption("llm_base_url"),
                temperature=pytestconfig.getoption("llm_temperature"),
                **options,
            )
            return read_only_tools
    result = codex_calls(await run_codex(workspace.root, request, **options))
    assert calls == result, "The observer must retain all raw tool attempts"
    return read_only_tools


async def _read_only_tools(client: MCPClient) -> set[str]:
    names: set[str] = set()
    cursor = None
    while True:
        page = await client.session.list_tools(
            params=PaginatedRequestParams(cursor=cursor)
        )
        names.update(
            tool.name
            for tool in page.tools
            if tool.annotations and tool.annotations.read_only_hint is True
        )
        cursor = page.next_cursor
        if cursor is None:
            return names


def _seed_saved_scopes(workspace: MCPWorkspace) -> SQLiteStore:
    paths = WorkspacePaths(workspace.workspace)
    store = SQLiteStore(paths)
    scopes = {}
    for index, fees in enumerate((0.0006, 0.001)):
        scope = _scope(split="train", trial_index=index, values={"total_return": 0.11})
        context = scope.metadata["execution_context"]
        assert isinstance(context, dict)
        context["fees"] = fees
        scopes[f"trial/{index}/train"] = scope
    _publish(
        (paths, store, RunService(workspace=paths, store=store)),
        "scope_comparison",
        scopes=scopes,
        default_scope="trial/0/train",
    )
    return store


def _verify_saved_scope_calls(calls: list[ToolCall], read_only_tools: set[str]) -> None:
    attempted_writes = [call.name for call in calls if call.name not in read_only_tools]
    assert not attempted_writes, (
        f"Read-only request attempted writes: {attempted_writes}"
    )
    comparisons = [
        CompareRunsResponse.model_validate(call.result)
        for call in calls
        if call.name == "compare_runs" and call.result.get("success")
    ]
    assert any(
        {(row.run_id, row.scope) for row in comparison.runs}
        == {
            ("scope_comparison", "trial/0/train"),
            ("scope_comparison", "trial/1/train"),
        }
        and len(comparison.runs) == 2
        and all(
            row.metrics == {"total_return": 0.11} and row.metric_source == "bundle"
            for row in comparison.runs
        )
        and tuple(comparison.context_differences["execution_context.fees"].values())
        in ((0.0006, 0.001), (0.001, 0.0006))
        for comparison in comparisons
    ), "Use saved scope comparison without collapsing their identities"


def test_saved_scope_verification_rejects_failed_write_attempts() -> None:
    with temporary_mcp_workspace() as workspace:
        _seed_saved_scopes(workspace)

        async def read() -> tuple[list[ToolCall], set[str]]:
            async with workspace.connect() as client:
                read_only = await _read_only_tools(client)
                arguments = {
                    "selections": [
                        {"run_id": "scope_comparison", "scope": f"trial/{index}/train"}
                        for index in range(2)
                    ],
                    "metric_ids": ["total_return"],
                }
                result = await client.call("compare_runs", **arguments)
                return [ToolCall("compare_runs", arguments, result)], read_only

        calls, read_only = asyncio.run(read())
        _verify_saved_scope_calls(calls, read_only)
        for name in ("train_model", "ensure_data", "build_dataset", "unknown_tool"):
            failed = ToolCall(name, {}, {"success": False})
            with pytest.raises(
                AssertionError, match="Read-only request attempted writes"
            ):
                _verify_saved_scope_calls([*calls, failed], read_only)


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


@pytest.mark.live_llm
def test_natural_saved_scope_comparison(pytestconfig: pytest.Config) -> None:
    """Discover per-selection scopes through real MCP, without executing a run."""
    with temporary_mcp_workspace() as workspace:
        store = _seed_saved_scopes(workspace)
        request = (
            "請查已保存的 scope_comparison，將裡面兩組訓練期結果的總報酬並列，"
            "列出系統提供的可比較性判定、交易條件有哪些差異，"
            "以及這些數字能支持什麼結論。"
            "只讀取既有結果，不要重新執行研究。"
        )
        turns: list[ModelTurn] = []
        calls: list[ToolCall] = []

        async def user(turn: ModelTurn) -> None:
            turns.append(turn)

        try:
            read_only = asyncio.run(_run(workspace, pytestconfig, user, calls, request))
            _verify_saved_scope_calls(calls, read_only)
            assert store.list_jobs() == [] and len(store.list_runs()) == 1
            assert turns and turns[-1].text.strip()
            print(
                "[saved scope comparison] Final reply:\n" + turns[-1].text, flush=True
            )
        except BaseException as exc:
            exc.add_note(
                "Saved scope comparison attempts:\n"
                + json.dumps(
                    [
                        {
                            "name": call.name,
                            "arguments": call.arguments,
                            "result": call.result,
                        }
                        for call in calls
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise
