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
from tradingdev.app.report_service import ReportService
from tradingdev.domain.ml.report import ML_SAMPLE_PROVENANCE_NOTE

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.integration.mcp_harness import MCPWorkspace
    from tradingdev.domain.ml.leaderboard import LeaderboardEntry

_UNKNOWN_PROVENANCE_CONTEXT = "現有紀錄未交代評估資料是否曾參與模型訓練。"


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
                    analysis={
                        "summary": "橙海視窗敏感度研究，尚不能判定最佳視窗。",
                        "observations": "視窗 12 平均 IC 為 0.2；視窗 24 缺值。",
                        "hypotheses_and_limits": ML_SAMPLE_PROVENANCE_NOTE,
                        "next_steps": "先確認缺值原因，再提出未執行的驗證設計。",
                        "references": [],
                    },
                )
            return calls, names

        calls, names = asyncio.run(run())
        report = calls[-1].result
        turn = ModelTurn(
            1,
            f"[查看橙海研究報告](<{report['path']}>)\n\n"
            "視窗 12 的平均訊號相關性為 0.2，視窗 24 的分數不可用，不能視為零。"
            "這是視窗敏感度研究，沒有提供交易成本與成交資料，"
            "不能當成含成本的策略回測。" + ML_SAMPLE_PROVENANCE_NOTE,
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
    assert "評估資料與訓練資料的關係" in ML_REPORT_REQUEST
    assert "沒有記錄的部分請明說未知" in ML_REPORT_REQUEST


def test_real_mcp_saved_ml_report_retains_unavailable_coordinate_and_delivery(
    report: MLReportWorkflow,
) -> None:
    manifest = report.verify()
    assert manifest["ml"]["scopes"][1]["values"]["ic_mean"] is None
    assert manifest["ml"]["scopes"][1]["metadata"]["window"] == 24
    assert manifest["ml"]["scopes"][1]["parameters"]["window"] == 7


def test_ml_report_cannot_leave_analysis_only_in_chat(report: MLReportWorkflow) -> None:
    report_call = report.calls[-1]
    old_path = report_call.result["path"]
    report_call.arguments.pop("analysis")
    report_call.result = ReportService(
        workspace=WorkspacePaths(report.workspace.workspace)
    ).generate_report(**report_call.arguments)
    assert report_call.result["success"] is True
    text = report.turn.text.replace(old_path, report_call.result["path"])
    with pytest.raises(AssertionError, match="HTML must retain research analysis"):
        replace(report, turn=replace(report.turn, text=text)).verify()


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
    # Add separate provenance context to isolate this cost-wording regression.
    path = report.calls[-1].result["path"]
    text = (
        f"[橙海訊號研究離線報告](<{path}>)\n\n"
        "這是視窗敏感度比較。視窗 24 的缺值不等於零，保存原因僅為分數不可用或非有限值；"
        "只有一個有效平均 IC，無法判定視窗 12 較佳或結果穩健。IC 是相關係數，"
        "不能換算成報酬，也不能據此證明含手續費、滑價的策略可獲利。\n\n"
        "在「ML 訊號研究」查看候選排序與缺分數原因，展開候選證據及其他指標；"
        "ML 證據與回測各自獨立，不代表模型已用於該回測。" + _UNKNOWN_PROVENANCE_CONTEXT
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
        + _UNKNOWN_PROVENANCE_CONTEXT
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
        "訊號評估視窗 12 有分數，視窗 24 分數不可用，不能視為零。"
        + explanation
        + _UNKNOWN_PROVENANCE_CONTEXT
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
        "訊號評估視窗 12 有分數，視窗 24 分數不可用，不能視為零。"
        + explanation
        + _UNKNOWN_PROVENANCE_CONTEXT
    )
    with pytest.raises(AssertionError, match="cost-aware backtest limits"):
        replace(report, turn=replace(report.turn, text=text)).verify()


@pytest.mark.parametrize(
    "claim",
    [
        "這份研究可支持樣本內訊號關聯的描述。",
        "這是樣本內評估。",
        "這些結果屬於樣本外驗證。",
        "這些結果僅是樣本內訊號關聯。",
        "這些分數僅能支持樣本內關聯。",
        "這是已證實的樣本外結果。",
        "無法證明樣本外效果，但這是樣本內評估。",
        "例如這些結果為樣本內評估。",
        "例如依照上述規則，這些結果為樣本內評估。",
        "例如依照上述規則這些結果為樣本內評估。",
        "如上面的規則，這些結果為樣本內評估。",
        "究竟是哪個模型未知，但這些結果是樣本內評估。",
        # Rule framing does not rescue a claim whose subject is the saved result.
        "依一般規則，這些結果屬於樣本內。",
        "一般規則：這些結果屬於樣本內。",
        "訓練資料上的評估是樣本內，因此這些結果屬於樣本內。",
        "這份在訓練資料上做的評估是樣本內。",
        # Naming the candidate makes the same words a claim about the saved rows.
        "兩個候選在訓練資料上做的評估是樣本內。",
        "模型在訓練資料上的評估是樣本內。",
        "視窗 12 在訓練資料上做的評估是樣本內。",
        "報告中的訓練資料評估是樣本內。",
        "候選評估是樣本內。",
        "在驗證資料上做的評估是樣本內。",
        "對這兩個候選而言，訓練資料上的評估是樣本內。",
        # An elided restatement must not ride on the exempted definition.
        "在訓練資料上做的評估是樣本內；候選結果亦然。",
        "在訓練資料上做的評估是樣本內；視窗 12 亦然。",
        "在訓練資料上做的評估是樣本內，這也適用於本次兩個候選。",
        "依定義，在訓練資料上做的評估是樣本內，兩個候選的評估也是如此。",
        "在訓練資料上做的評估是樣本內；候選結果亦然，雖然切分紀錄未保存。",
        "在訓練資料上做的評估是樣本內。這是定義。兩個候選的結果亦然。",
        "在訓練資料上做的評估是樣本內；這對兩個候選也成立。",
        "在訓練資料上做的評估是樣本內；兩個候選亦復如此。",
        # A reporting frame does not excuse a claim about the saved rows.
        "工具說明這些結果屬於樣本內。",
        "報告指出兩個候選在訓練資料上做的評估屬於樣本內。",
        # Only the product's definition is exempt: training-data evaluation is
        # in-sample. The opposite rule stays rejected as an inverted claim.
        "訓練資料上的評估是樣本外。",
    ],
)
def test_ml_report_rejects_affirmative_sample_labels_without_split_evidence(
    report: MLReportWorkflow, claim: str
) -> None:
    text = report.turn.text + "\n\n" + claim
    with pytest.raises(AssertionError, match="no evidence for an affirmative"):
        replace(report, turn=replace(report.turn, text=text)).verify()


@pytest.mark.parametrize(
    "explanation",
    [
        ML_SAMPLE_PROVENANCE_NOTE,
        "無法判定這些結果是否屬於樣本內或樣本外。",
        "不能認定這是樣本外驗證。",
        "樣本內或樣本外的關係仍未知。",
        "如果資料曾用於訓練，則屬於樣本內；目前沒有切分證據。",
        "是否屬於樣本內仍未知。",
        "結果屬於樣本內與否仍未知。",
        "這不是已證實的樣本外評估。",
        "契約的一般規則是：如評估用於訓練的資料則仍為樣本內；"
        "本次沒有證據可套用為既成事實。",
        "本研究究竟是樣本內、樣本外，或是否完成時間隔離與防洩漏檢查，均未知。",
        # General rule about training-data evaluation, candidate left unknown.
        "一般規則：在訓練資料上做的評估是樣本內；"
        "這些既有結果沒有切分證據，因此歸屬仍未知。",
        "訓練資料上做的評估屬於樣本內；這批保存結果缺少切分證據，來源仍未知。",
        "凡在訓練資料上做的評估都屬樣本內；本次候選是否符合則沒有紀錄。",
        "依定義，以訓練資料做的評估為樣本內；保存紀錄未交代候選的切分方式。",
        "「在訓練資料上做的評估」屬於樣本內；這些結果的切分證據沒有保存。",
        "訓練資料上的評估僅能支持樣本內關聯；候選是否如此仍未知。",
        "在訓練資料上做的評估是樣本內；視窗 24 的分數不可用，視窗 12 亦然。",
        "在訓練資料上做的評估是樣本內；視窗 12 的分數尚待確認，視窗 24 亦然。",
        "在訓練資料上做的評估是樣本內。視窗 12 的平均 IC 為 0.2。",
        "凡在訓練資料上做的評估，都屬於樣本內；候選是否如此仍未知。",
        "至於訓練資料上的評估，屬於樣本內；這些結果的切分證據沒有保存。",
        "在訓練資料上做的評估是樣本內；候選結果是否亦然未知。",
        # Reported rule: the model quotes the product rule instead of stating it.
        "工具一般規則說明訓練資料上的評估屬於樣本內，"
        "但不能由一般規則推定這兩筆實際屬於樣本內；同樣也不能宣稱樣本外。",
        "契約說明在訓練資料上做的評估是樣本內；這些結果的切分證據沒有保存。",
        "工具文件指出訓練資料上的評估屬於樣本內。",
    ],
)
def test_ml_report_accepts_unknown_or_conditional_sample_provenance(
    report: MLReportWorkflow, explanation: str
) -> None:
    text = report.turn.text + "\n\n" + explanation
    replace(report, turn=replace(report.turn, text=text)).verify()


@pytest.mark.parametrize(
    "explanation",
    [
        "保存紀錄缺乏訓練／評估切分證據，資料是否重疊、屬於樣本內或樣本外均未知。",
        "目前不知道評估資料是否曾參與訓練。",
        "模型擬合與這份評估的資料是否重疊，紀錄沒有交代。",
        "樣本內／外來源尚不明。",
        "無法確認這些結果的樣本內或樣本外歸屬。",
        _UNKNOWN_PROVENANCE_CONTEXT,
    ],
)
def test_ml_report_accepts_alternative_visible_unknown_provenance(
    report: MLReportWorkflow, explanation: str
) -> None:
    text = report.turn.text.replace(ML_SAMPLE_PROVENANCE_NOTE, explanation)
    replace(report, turn=replace(report.turn, text=text)).verify()


@pytest.mark.parametrize(
    "omission",
    ["absent", "score_unknown", "unproven_performance", "link", "fenced", "hidden"],
)
def test_ml_report_requires_visible_sample_provenance_in_the_final_reply(
    report: MLReportWorkflow, omission: str
) -> None:
    path = report.calls[-1].result["path"]
    replacements = {
        "absent": "",
        "score_unknown": "視窗 24 的分數仍未知。",
        "unproven_performance": "尚不足以證明樣本外有效。",
        "link": f"[{ML_SAMPLE_PROVENANCE_NOTE}](<{path}>)",
        "fenced": f"\n\n```text\n{ML_SAMPLE_PROVENANCE_NOTE}\n```",
        "hidden": f"\n\n<span hidden>{ML_SAMPLE_PROVENANCE_NOTE}</span>",
    }
    text = report.turn.text.replace(ML_SAMPLE_PROVENANCE_NOTE, replacements[omission])
    with pytest.raises(AssertionError, match="provenance is unknown"):
        replace(report, turn=replace(report.turn, text=text)).verify()


def test_ml_report_commentary_cannot_replace_final_provenance_explanation(
    report: MLReportWorkflow,
) -> None:
    report_call = report.calls[-1]
    old_path = report_call.result["path"]
    report_call.arguments["commentary"] = [
        {"title": "來源限制", "text": ML_SAMPLE_PROVENANCE_NOTE}
    ]
    report_call.result = ReportService(
        workspace=WorkspacePaths(report.workspace.workspace)
    ).generate_report(**report_call.arguments)
    assert report_call.result["success"] is True
    text = report.turn.text.replace(old_path, report_call.result["path"]).replace(
        ML_SAMPLE_PROVENANCE_NOTE, ""
    )
    with pytest.raises(AssertionError, match="provenance is unknown"):
        replace(report, turn=replace(report.turn, text=text)).verify()


@pytest.mark.parametrize(
    "claim",
    [
        None,
        "這些結果屬於樣本內評估。",
        "在訓練資料上做的評估是樣本內；候選結果亦然。",
    ],
)
@pytest.mark.parametrize("structured", [False, True])
def test_ml_report_commentary_also_retains_unknown_sample_provenance(
    report: MLReportWorkflow, claim: str | None, structured: bool
) -> None:
    report_call = report.calls[-1]
    old_path = report_call.result["path"]
    report_call.arguments["commentary"] = [
        {
            "title": "保存來源限制",
            "text": claim or ML_SAMPLE_PROVENANCE_NOTE,
        }
    ]
    if structured:
        note = report_call.arguments.pop("commentary")[0]
        report_call.arguments["analysis"] = {
            "summary": note["text"],
            "observations": "保存的視窗比較。",
            "hypotheses_and_limits": ML_SAMPLE_PROVENANCE_NOTE,
            "next_steps": "先核對資料。",
            "references": [],
        }
    report_call.result = ReportService(
        workspace=WorkspacePaths(report.workspace.workspace)
    ).generate_report(**report_call.arguments)
    assert report_call.result["success"] is True
    text = report.turn.text.replace(old_path, report_call.result["path"])
    delivered = replace(report, turn=replace(report.turn, text=text))
    if claim is not None:
        with pytest.raises(AssertionError, match="no evidence for an affirmative"):
            delivered.verify()
    else:
        delivered.verify()


@pytest.mark.parametrize("delivered_in", ["commentary", "analysis", "both"])
def test_ml_report_accepts_the_general_rule_in_each_delivered_entry_point(
    report: MLReportWorkflow, delivered_in: str
) -> None:
    """Commentary and structured analysis must accept the rule independently."""
    report_call = report.calls[-1]
    old_path = report_call.result["path"]
    rule = (
        "一般規則：在訓練資料上做的評估是樣本內；"
        "這些既有結果沒有切分證據，因此歸屬仍未知。"
    )
    if delivered_in != "analysis":
        report_call.arguments["commentary"] = [{"title": "來源限制", "text": rule}]
    if delivered_in != "commentary":
        report_call.arguments["analysis"] = {
            "summary": rule,
            "observations": "保存的視窗比較。",
            "hypotheses_and_limits": ML_SAMPLE_PROVENANCE_NOTE,
            "next_steps": "先核對資料。",
            "references": [],
        }
    report_call.result = ReportService(
        workspace=WorkspacePaths(report.workspace.workspace)
    ).generate_report(**report_call.arguments)
    assert report_call.result["success"] is True
    text = report.turn.text.replace(old_path, report_call.result["path"])
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
