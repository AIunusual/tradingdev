"""Natural discovery of fixed comparisons and saved-scope comparisons/rankings."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from mcp_types import PaginatedRequestParams

from tests.app.test_run_metrics import _publish, _scope
from tests.app.test_run_ranking import _manifest, _publish_run
from tests.app.test_run_ranking import _scope as _ranking_scope
from tests.e2e.codex_harness import run_codex
from tests.e2e.comparison_diagnostics import comparison_failure_diagnostics
from tests.e2e.conversation import ModelTurn, links_to_file, visible_text
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model, tool_payload
from tests.e2e.report_delivery import verify_report_delivery
from tests.e2e.strategy_scenarios import market_frame
from tests.e2e.workflow_arguments import effective_workflow_calls
from tests.integration.mcp_harness import temporary_mcp_workspace
from tests.integration.test_comparison_workflow import (
    ComparisonUser,
    verify_comparison_delivery,
)
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.contracts.ml import (
    LeaderboardPageResponse,
    RankedRunRow,
    SkippedRunRow,
)
from tradingdev.app.contracts.reports import ReportResponse
from tradingdev.app.contracts.research import CompareRunsResponse
from tradingdev.app.run_service import RunService
from tradingdev.domain.execution import ExecutionManifest, OptimizationSpec
from tradingdev.domain.presentation.reports import REPORT_RECIPES

if TYPE_CHECKING:
    from tests.e2e.conversation import UserReply
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


def _verify_read_only_calls(calls: list[ToolCall], read_only_tools: set[str]) -> None:
    attempted_writes = [call.name for call in calls if call.name not in read_only_tools]
    assert not attempted_writes, (
        f"Read-only request attempted writes: {attempted_writes}"
    )


def _verify_saved_scope_calls(calls: list[ToolCall], read_only_tools: set[str]) -> None:
    _verify_read_only_calls(calls, read_only_tools)
    comparisons = [
        CompareRunsResponse.model_validate(call.result)
        for call in calls
        if call.name == "compare_runs"
        and isinstance(call.result, dict)
        and call.result.get("success") is True
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
        _verify_saved_scope_calls(
            [ToolCall("compare_runs", {}, None), *calls], read_only
        )
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


def _seed_saved_ranking(workspace: MCPWorkspace) -> SQLiteStore:
    paths = WorkspacePaths(workspace.workspace)
    store = SQLiteStore(paths)
    base = _manifest(window=99)
    manifest = ExecutionManifest.create(
        kind="optimization",
        config=base.config_copy(),
        strategy_execution=base.strategy_execution,
        optimization=OptimizationSpec(
            param_ranges={"window": [10, 20, 30]},
            optimization_metric="total_return",
            direction="maximize",
            train_start="2024-01-01",
            train_end="2024-03-01",
            test_start="2024-03-02",
            test_end="2024-03-03",
        ),
    )
    scopes = {
        f"trial/{index}/train": _ranking_scope({"sharpe_ratio": value}).model_copy(
            update={
                "parameters": {"window": 10 * (index + 1)},
                "split": "train",
                "trial_index": index,
            }
        )
        for index, value in enumerate((0.5, 1.5, None))
    }
    _publish_run(paths, store, "ranking_study", scope_models=scopes, manifest=manifest)
    return store


def _verify_ranking_operations(calls: list[ToolCall], read_only: set[str]) -> None:
    """Allow saved-result status/report operations without authorizing new research."""

    def selection(value: object) -> object:
        # FuncMetadata accepts top-level JSON arrays/null even on failed attempts.
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except ValueError:
                return value
            if isinstance(parsed, list) or parsed is None:
                return parsed
        return value

    for call in calls:
        if call.name == "get_job_status":
            assert call.arguments.get("job_id") == "job-ranking_study"
        elif call.name == "generate_report":
            assert selection(call.arguments.get("run_ids")) == ["ranking_study"]
            assert selection(call.arguments.get("comparison_job_id")) in (None, "")
            assert selection(call.arguments.get("ml_entry_ids")) in (None, [])
        else:
            _verify_read_only_calls([call], read_only)


def _verify_ranking_report(paths: WorkspacePaths, turn: ModelTurn) -> None:
    reports = [
        call
        for call in effective_workflow_calls(turn.calls)
        if call.name == "generate_report"
        and isinstance(call.result, dict)
        and call.result.get("success") is True
    ]
    if not reports:
        return
    linked = {
        report.report_id: call
        for call in reports
        if links_to_file(
            turn.text, Path((report := ReportResponse.model_validate(call.result)).path)
        )
    }
    assert len(linked) == 1, "Deliver exactly one of the generated saved-run reports"
    selected = next(iter(linked.values()))
    sections = selected.arguments.get("sections")
    verify_report_delivery(
        paths,
        turn,
        "ranking_study",
        expected_sections=sections
        if sections is not None
        else list(REPORT_RECIPES["standard"]),
        expected_commentary=None,
    )


def _ranking_parameter_evidence(calls: list[ToolCall]) -> set[str]:
    """Accept scoped result records or complete, consistently paged performance JSON."""
    found: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for index in range(3):
                scope = f"trial/{index}/train"
                if (
                    value.get("run_id") == "ranking_study"
                    and value.get("scope") == scope
                    and value.get("parameters") == {"window": 10 * (index + 1)}
                ):
                    found.add(scope)
            for item in value.values():
                visit(item)

    pages: dict[tuple[str, str], dict[int, str]] = {}
    totals: dict[tuple[str, str], int] = {}
    for call in calls:
        result = call.result
        if not isinstance(result, dict) or result.get("success") is not True:
            continue
        # Report arguments and their echoes are model output, not read evidence.
        if call.name != "generate_report":
            visit(result)
        if call.name == "get_artifact" and result.get("content_page"):
            artifact = result["artifact"]
            page = result["content_page"]
            if artifact["run_id"] != "ranking_study":
                continue
            key = (artifact["artifact_id"], page["content_sha256"])
            pages.setdefault(key, {})[page["offset"]] = result["content"]
            totals[key] = page["total_characters"]
    for key, parts in pages.items():
        content = ""
        for offset, text in sorted(parts.items()):
            if offset != len(content):
                break
            content += text
        if len(content) != totals[key]:
            continue
        try:
            bundle = json.loads(content)
        except ValueError:
            continue
        if isinstance(bundle, dict) and bundle.get("run_id") == "ranking_study":
            for scope, record in bundle.get("scopes", {}).items():
                visit({**record, "run_id": "ranking_study", "scope": scope})
    return found


def _verify_ranking_reply(text: str) -> None:
    """Require visible coverage; interpretation and attribution still need review."""
    prose = visible_text(text)
    assert "ranking_study" in prose and re.search(r"Sharpe|夏普", prose, re.I)
    for window in (10, 20, 30):
        assert re.search(rf"(?<![\w.]){window}(?:\.0+)?(?![\w.])", prose), (
            "Final reply must include every candidate's window setting"
        )
    for value in ("0.5", "1.5"):
        assert re.search(rf"(?<![\d.]){re.escape(value)}0*(?![\d.])", prose), (
            "Final reply must deliver the saved ranking values"
        )
    assert re.search(r"缺|空|無法|不可用|missing|null|unavailable", prose, re.I)
    assert re.search(r"限制|訓練|樣本|limit|train|sample", prose, re.I)


def _ranking_pages(calls: list[ToolCall]) -> list[LeaderboardPageResponse]:
    return [
        LeaderboardPageResponse.model_validate(call.result)
        for call in calls
        if call.name == "leaderboard"
        and isinstance(call.result, dict)
        and call.result.get("success") is True
    ]


def test_saved_ranking_evidence_and_operation_boundaries() -> None:
    with temporary_mcp_workspace() as workspace:
        store = _seed_saved_ranking(workspace)

        async def read() -> tuple[list[ToolCall], set[str]]:
            async with workspace.connect() as client:
                read_only = await _read_only_tools(client)
                calls = []
                for name, arguments in (
                    ("find_runs", {"strategy_id": "fixture"}),
                    (
                        "get_run_metrics",
                        {"run_id": "ranking_study", "scope": "trial/2/train"},
                    ),
                    ("get_job_status", {"job_id": "job-ranking_study"}),
                ):
                    result = await client.call(name, **arguments)
                    calls.append(ToolCall(name, arguments, result))
                artifact = next(
                    item
                    for item in store.list_artifacts("ranking_study")
                    if item["artifact_type"] == "performance_json"
                )
                offset = 0
                while True:
                    artifact_arguments: dict[str, Any] = {
                        "artifact_id": artifact["artifact_id"],
                        "include_content": True,
                        "offset": offset,
                        "limit": 1000,
                    }
                    result = await client.call("get_artifact", **artifact_arguments)
                    calls.append(ToolCall("get_artifact", artifact_arguments, result))
                    offset = result["content_page"]["next_offset"]
                    if offset is None:
                        break
                bad_arguments = {
                    "metric": "run.sharpe_ratio",
                    "runs": [["ranking_study", "trial/0/train"]],
                }
                failed = await client.session.call_tool("leaderboard", bad_arguments)
                assert failed.is_error
                payload = tool_payload(failed.model_dump(mode="json", by_alias=True))
                assert payload is None
                calls.append(ToolCall("leaderboard", bad_arguments, payload))
                ranking_arguments = {
                    "metric": "run.sharpe_ratio",
                    "runs": [
                        {"run_id": "ranking_study", "scope": f"trial/{index}/train"}
                        for index in range(3)
                    ],
                }
                ranked = await client.call("leaderboard", **ranking_arguments)
                calls.append(ToolCall("leaderboard", ranking_arguments, ranked))
                return calls, read_only

        calls, read_only = asyncio.run(read())
        all_scopes = {f"trial/{index}/train" for index in range(3)}
        assert _ranking_parameter_evidence(calls[:1]) == all_scopes
        assert _ranking_parameter_evidence(calls[1:2]) == {"trial/2/train"}
        artifacts = [call for call in calls if call.name == "get_artifact"]
        assert _ranking_parameter_evidence(artifacts) == all_scopes
        assert _ranking_parameter_evidence(artifacts[1:]) == set()
        pages = _ranking_pages(calls)
        assert len(pages) == 1 and pages[0].n_candidates == 3
        _verify_ranking_operations(calls, read_only)
        for name in ("train_model", "ensure_data", "build_dataset", "unknown_tool"):
            failed = ToolCall(name, {}, {"success": False})
            with pytest.raises(AssertionError, match="attempted writes"):
                _verify_ranking_operations([*calls, failed], read_only)
        for name, arguments in (
            ("get_job_status", {"job_id": "unrelated-job"}),
            ("generate_report", {"run_ids": ["unrelated-run"]}),
        ):
            with pytest.raises(AssertionError):
                _verify_ranking_operations(
                    [*calls, ToolCall(name, arguments, {"success": False})], read_only
                )
        assert store.list_jobs() == [] and len(store.list_runs()) == 1


def test_saved_ranking_reply_requires_visible_candidate_coverage() -> None:
    reply = (
        "ranking_study 的訓練 Sharpe 排名：window 20 為 1.50；"
        "window 10 為 0.50；window 30 缺值，不能排名。限制是僅有訓練結果。"
    )
    _verify_ranking_reply(reply)
    for incomplete in (
        "完成。",
        reply.replace("window 30 缺值，不能排名。", ""),
        reply.replace("0.50", "未知"),
        reply.replace("window 30", "window 99"),
        f"<span hidden>{reply}</span>完成。",
    ):
        with pytest.raises(AssertionError):
            _verify_ranking_reply(incomplete)


def test_saved_ranking_report_accepts_sdk_arguments_and_selected_link() -> None:
    with temporary_mcp_workspace() as workspace:
        _seed_saved_ranking(workspace)

        async def generate() -> tuple[list[ToolCall], set[str]]:
            async with workspace.connect() as client:
                read_only = await _read_only_tools(client)
                calls = []
                for sections in ('["overview","metrics"]', ["overview"]):
                    arguments: dict[str, Any] = {
                        "run_ids": '["ranking_study"]',
                        "ml_entry_ids": "null",
                        "sections": sections,
                        "commentary": [
                            {
                                "title": "保存結果",
                                "text": "只解讀訓練候選，未重跑研究。",
                            }
                        ],
                    }
                    result = await client.call("generate_report", **arguments)
                    assert result.get("success") is True
                    calls.append(ToolCall("generate_report", arguments, result))
                return calls, read_only

        calls, read_only = asyncio.run(generate())
        _verify_ranking_operations(calls, read_only)
        # Failed calls retain the same target guard; unrelated writes cannot hide.
        _verify_ranking_operations(
            [ToolCall("generate_report", calls[0].arguments, {"success": False})],
            read_only,
        )
        text = f"ranking_study 已整理訓練結果：[閱讀報告]({calls[0].result['path']})"
        _verify_ranking_report(
            WorkspacePaths(workspace.workspace), ModelTurn(1, text, calls)
        )


@pytest.mark.live_llm
def test_natural_saved_run_ranking(pytestconfig: pytest.Config) -> None:
    """Find the ranking surface, preserve missing candidates and avoid new work."""
    with temporary_mcp_workspace() as workspace:
        paths = WorkspacePaths(workspace.workspace)
        store = _seed_saved_ranking(workspace)
        request = (
            "請將已保存的 ranking_study 裡所有訓練候選按 Sharpe 排名，"
            "保留無法排名的候選並說明原因，列出各組設定和比較限制。"
            "只查既有成果，不要重新執行研究。"
        )
        turns: list[ModelTurn] = []
        calls: list[ToolCall] = []

        async def user(turn: ModelTurn) -> None:
            turns.append(turn)

        try:
            read_only = asyncio.run(_run(workspace, pytestconfig, user, calls, request))
            _verify_ranking_operations(calls, read_only)
            boards = _ranking_pages(calls)
            complete = [
                board
                for board in boards
                if board.source == "run"
                and board.metric == "run.sharpe_ratio"
                and board.n_candidates == 3
                and board.n_ranked == 2
                and board.n_skipped == 1
            ]
            assert complete, "Discover a ranking containing every requested candidate"
            assert all(
                board.direction == "maximize" and board.caveats for board in complete
            )
            ranked = {
                (row.run_id, row.scope): row
                for board in complete
                for row in board.ranked
                if isinstance(row, RankedRunRow)
            }
            assert set(ranked) == {
                ("ranking_study", "trial/0/train"),
                ("ranking_study", "trial/1/train"),
            }
            for index, rank, value in ((0, 2, 0.5), (1, 1, 1.5)):
                row = ranked[("ranking_study", f"trial/{index}/train")]
                assert row.rank == rank and row.value == value
                assert row.parameters == {"window": 10 * (index + 1)}
                assert row.metadata == _ranking_scope({}).metadata
            skipped = {
                (row.run_id, row.scope)
                for board in complete
                for row in board.skipped
                if isinstance(row, SkippedRunRow) and row.reason
            }
            assert skipped == {("ranking_study", "trial/2/train")}
            assert _ranking_parameter_evidence(calls) == {
                f"trial/{index}/train" for index in range(3)
            }, "Read all candidate settings, including the skipped candidate"
            assert store.list_jobs() == [] and len(store.list_runs()) == 1
            assert turns
            _verify_ranking_reply(turns[-1].text)
            _verify_ranking_report(paths, turns[-1])
            print("[saved run ranking] Final reply:\n" + turns[-1].text, flush=True)
        except BaseException as exc:
            for turn in turns:
                exc.add_note(f"Saved ranking turn {turn.number}:\n{turn.text}")
            exc.add_note(
                "Ranking tool diagnostics:\n"
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
