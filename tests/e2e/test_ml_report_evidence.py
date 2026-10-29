"""Offline checks over real MCP report artifacts and deliberately damaged evidence."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.e2e.conversation import ModelTurn
from tests.e2e.llm_client import ToolCall
from tests.e2e.ml_report_evidence import (
    ML_REPORT_REQUEST,
    seed_ml_report_candidates,
    verify_ml_report_outcome,
)
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.storage.filesystem import WorkspacePaths

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.integration.mcp_harness import MCPWorkspace
    from tradingdev.domain.ml.leaderboard import LeaderboardEntry


@dataclass(frozen=True)
class MLReportWorkflow:
    workspace: MCPWorkspace
    calls: list[ToolCall]
    turn: ModelTurn
    entries: tuple[LeaderboardEntry, ...]
    tool_names: list[str]

    def verify(self) -> dict[str, Any]:
        return verify_ml_report_outcome(
            self.workspace, self.calls, self.turn, self.entries
        )


@pytest.fixture(scope="module")
def saved_report() -> Iterator[MLReportWorkflow]:
    with temporary_mcp_workspace() as workspace:
        entries = seed_ml_report_candidates(WorkspacePaths(workspace.workspace))

        async def run() -> tuple[list[ToolCall], list[str]]:
            calls: list[ToolCall] = []
            async with workspace.connect() as client:
                names = list(client.output_schemas)

                async def call(name: str, **arguments: Any) -> Any:
                    result = await client.call(name, **arguments)
                    assert result.get("success") is True, result
                    calls.append(ToolCall(name, arguments, result))
                    return result

                await call("get_ml_contract")
                await call("leaderboard", metric="ic_mean", allow_parameter_axes=True)
                await call("get_report_sections")
                await call(
                    "generate_report",
                    ml_entry_ids=[entry.entry_id for entry in entries],
                    ml_allow_parameter_axes=True,
                )
            return calls, names

        calls, names = asyncio.run(run())
        report = calls[-1].result
        turn = ModelTurn(
            1,
            f"[查看橙海研究報告](<{report['path']}>)\n\n"
            "視窗 12 的平均訊號相關性為 0.2，視窗 24 的分數不可用，不能視為零。"
            "這是視窗敏感度研究，沒有提供交易成本與成交資料，"
            "不能當成含成本的策略回測。",
            calls,
        )
        yield MLReportWorkflow(workspace, calls, turn, entries, names)
    assert not workspace.root.exists()


@pytest.fixture
def report(saved_report: MLReportWorkflow) -> MLReportWorkflow:
    calls = deepcopy(saved_report.calls)
    return replace(
        saved_report, calls=calls, turn=replace(saved_report.turn, calls=calls)
    )


def test_natural_ml_request_does_not_supply_tools_ids_or_schema_answers(
    report: MLReportWorkflow,
) -> None:
    assert all(name not in ML_REPORT_REQUEST for name in report.tool_names)
    assert all(entry.entry_id not in ML_REPORT_REQUEST for entry in report.entries)
    assert all(
        field not in ML_REPORT_REQUEST
        for field in (
            "ml_entry_ids",
            "ml_ranking_metric",
            "ml_allow_parameter_axes",
            "allow_parameter_axes",
            "ic_mean",
            "sections",
        )
    )


def test_real_mcp_saved_ml_report_retains_unavailable_coordinate_and_delivery(
    report: MLReportWorkflow,
) -> None:
    manifest = report.verify()
    assert manifest["ml"]["scopes"][1]["values"]["ic_mean"] is None
    assert manifest["ml"]["scopes"][1]["metadata"]["window"] == 24
    assert manifest["ml"]["scopes"][1]["parameters"]["window"] == 7


@pytest.mark.parametrize(
    "damage", ["missing", "failed", "missing_skipped", "report_before_discovery"]
)
def test_ml_report_requires_discovery_of_both_ranked_and_unavailable_candidates(
    report: MLReportWorkflow, damage: str
) -> None:
    if damage == "missing":
        report.calls.pop(1)
    elif damage == "failed":
        report.calls[1].result["success"] = False
    elif damage == "missing_skipped":
        report.calls[1].result["skipped"] = []
    else:
        report.calls[1], report.calls[3] = report.calls[3], report.calls[1]
    with pytest.raises(AssertionError, match="discover"):
        report.verify()


@pytest.mark.parametrize(
    "name",
    [
        "prepare_backtest",
        "execute_backtest",
        "ensure_data",
        "train_model",
        "evaluate_signal",
        "save_strategy",
        "list_available_data",
    ],
)
@pytest.mark.parametrize("success", [False, True])
def test_ml_report_rejects_unnecessary_attempts_even_when_they_fail(
    report: MLReportWorkflow, name: str, success: bool
) -> None:
    report.calls.insert(0, ToolCall(name, {}, {"success": success}))
    with pytest.raises(AssertionError, match="must not attempt"):
        report.verify()


@pytest.mark.parametrize(
    "field,value",
    [
        ("ml_entry_ids", ["another-candidate"]),
        ("run_ids", ["invented-run"]),
        ("comparison_job_id", "invented-comparison"),
        ("ml_allow_parameter_axes", False),
        ("ml_ranking_metric", "hit_rate"),
        ("sections", ["overview"]),
    ],
)
def test_ml_report_requires_the_selected_evidence_and_settings(
    report: MLReportWorkflow, field: str, value: Any
) -> None:
    report.calls[-1].arguments[field] = value
    with pytest.raises(AssertionError):
        report.verify()


def test_ml_report_accepts_sdk_encoded_inputs_without_changing_raw_calls(
    report: MLReportWorkflow,
) -> None:
    report.calls[1].arguments["allow_parameter_axes"] = "true"
    for field in ("ml_entry_ids", "ml_allow_parameter_axes"):
        report.calls[-1].arguments[field] = json.dumps(
            report.calls[-1].arguments[field]
        )
    original = deepcopy(report.calls)
    report.verify()
    assert report.calls == original


@pytest.mark.parametrize("damage", ["bare_path", "wrong_report", "only_link", "fenced"])
def test_ml_report_requires_a_clickable_final_link_and_visible_interpretation(
    report: MLReportWorkflow, damage: str
) -> None:
    path = report.calls[-1].result["path"]
    if damage == "bare_path":
        text = report.turn.text.replace(f"[查看橙海研究報告](<{path}>)", path)
    elif damage == "wrong_report":
        text = report.turn.text.replace(path, path + ".other")
    elif damage == "only_link":
        text = f"[報告](<{path}>)"
    else:
        text = f"[報告](<{path}>)\n\n```text\n{report.turn.text}\n```"
    with pytest.raises(AssertionError):
        replace(report, turn=replace(report.turn, text=text)).verify()


def test_ml_report_accepts_actual_model_commission_and_slippage_explanation(
    report: MLReportWorkflow,
) -> None:
    # Verbatim explanatory paragraphs from the failed real-model run: the
    # meaning is correct even though the final reply does not use the word 成本.
    path = report.calls[-1].result["path"]
    text = (
        f"[橙海訊號研究離線報告](<{path}>)\n\n"
        "這是視窗敏感度比較。視窗 24 的缺值不等於零，保存原因僅為分數不可用或非有限值；"
        "只有一個有效平均 IC，無法判定視窗 12 較佳或結果穩健。IC 是相關係數，"
        "不能換算成報酬，也不能據此證明含手續費、滑價的策略可獲利。\n\n"
        "在「ML 訊號研究」查看候選排序與缺分數原因，展開候選證據及其他指標；"
        "ML 證據與回測各自獨立，不代表模型已用於該回測。"
    )
    replace(report, turn=replace(report.turn, text=text)).verify()


@pytest.mark.parametrize(
    "costs", ["交易成本", "手續費", "滑價", "commission", "slippage", "fees"]
)
def test_ml_report_accepts_equivalent_transaction_cost_terms(
    report: MLReportWorkflow, costs: str
) -> None:
    path = report.calls[-1].result["path"]
    text = (
        f"[報告](<{path}>)\n\n"
        "訊號評估視窗 12 有分數，視窗 24 分數不可用，不能視為零。"
        f"這份訊號研究不能證明含 {costs} 的策略回測可獲利。"
    )
    replace(report, turn=replace(report.turn, text=text)).verify()


@pytest.mark.parametrize(
    "explanation",
    [
        "仍需另行進行計入手續費與滑價的回測。",
        "尚需計入手續費及滑價的策略回測，才能推論淨收益。",
        "需要有含交易成本的策略回測，才可評估是否獲利。",
        "只有另外執行包含手續費與滑價的回測，才能確認結果。",
        "A backtest including commission and slippage is required "
        "to establish net profitability.",
    ],
)
def test_ml_report_accepts_required_further_cost_aware_backtesting(
    report: MLReportWorkflow, explanation: str
) -> None:
    path = report.calls[-1].result["path"]
    text = (
        f"[報告](<{path}>)\n\n"
        "訊號評估視窗 12 有分數，視窗 24 分數不可用，不能視為零。" + explanation
    )
    replace(report, turn=replace(report.turn, text=text)).verify()


@pytest.mark.parametrize(
    "explanation",
    [
        "這份訊號研究不能直接證明策略可獲利，還需要策略回測。",
        "含手續費與滑價的策略回測已證明可獲利。",
    ],
)
def test_ml_report_requires_an_explicit_transaction_cost_limitation(
    report: MLReportWorkflow, explanation: str
) -> None:
    path = report.calls[-1].result["path"]
    text = (
        f"[報告](<{path}>)\n\n"
        "訊號評估視窗 12 有分數，視窗 24 分數不可用，不能視為零。" + explanation
    )
    with pytest.raises(AssertionError, match="cost-aware backtest limits"):
        replace(report, turn=replace(report.turn, text=text)).verify()


@pytest.mark.parametrize("filename", ["report.html", "manifest.json"])
def test_ml_report_requires_unchanged_readable_artifacts(
    report: MLReportWorkflow, filename: str
) -> None:
    path = Path(report.calls[-1].result["path"]).with_name(filename)
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\nchanged")
        with pytest.raises(AssertionError, match="readable|bytes differ"):
            report.verify()
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize(
    "field,value",
    [("ml_entry_ids", ["another-candidate"]), ("artifact_id", "another-artifact")],
)
def test_ml_report_requires_the_final_response_identity(
    report: MLReportWorkflow, field: str, value: Any
) -> None:
    report.calls[-1].result[field] = value
    with pytest.raises(AssertionError):
        report.verify()
