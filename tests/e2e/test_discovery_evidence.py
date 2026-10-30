"""Offline checks for natural intent and execution-bound discovery evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.e2e.codex_harness import LIFECYCLE_TOOLS
from tests.e2e.conversation import ModelTurn
from tests.e2e.discovery_workflow import (
    NATURAL_REQUEST,
    verify_discovery_outcome,
    verify_dvol_query_reply,
    verify_market_cache_reply,
)
from tests.e2e.llm_client import ToolCall
from tests.integration.mcp_harness import MCPWorkspace
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore

if TYPE_CHECKING:
    from pathlib import Path


def test_natural_request_contains_requirements_without_tool_answers() -> None:
    assert all(name not in NATURAL_REQUEST for name in LIFECYCLE_TOOLS)
    assert "2024-01-01" in NATURAL_REQUEST and "2024-01-08" in NATURAL_REQUEST
    assert "fast_period=5" in NATURAL_REQUEST and "slow_period=20" in NATURAL_REQUEST
    assert "績效、風險與限制" in NATURAL_REQUEST
    assert "generate_report" not in NATURAL_REQUEST


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "contract",
        "wrong_section",
        "empty_contract",
        "save",
        "revision",
        "lookup",
        "extra_run",
    ],
)
def test_discovery_requires_saved_execution_identity_before_report_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str | None
) -> None:
    workspace = MCPWorkspace(tmp_path)
    store = SQLiteStore(WorkspacePaths(workspace.workspace))
    store.create_run(
        run_id="run",
        job_id="job",
        strategy_id="llm_sma",
        revision_id="revision",
        artifact_dir=tmp_path,
        metrics={},
    )
    calls = [
        ToolCall(
            "list_strategies",
            {},
            {"items": [], "offset": 0, "limit": 20, "total": 0, "next_offset": None},
        ),
        ToolCall(
            "get_strategy_contract",
            {},
            {"section": "core", "content": {"requirements": ["contract"]}},
        ),
        ToolCall("save_strategy", {"strategy_id": "llm_sma"}, {"success": True}),
        ToolCall(
            "get_job_status",
            {},
            {
                "status": "done",
                "run_id": "run",
                "revision_id": "revision",
            },
        ),
        ToolCall("get_run", {"run_id": "run"}, {"success": True}),
    ]
    if damage == "contract":
        calls.pop(1)
    elif damage == "wrong_section":
        calls[1].result["section"] = "example"
    elif damage == "empty_contract":
        calls[1].result["content"] = {}
    elif damage == "save":
        calls[2].result["success"] = False
    elif damage == "revision":
        calls[3].result["revision_id"] = "wrong-revision"
    elif damage == "lookup":
        calls.pop()
    elif damage == "extra_run":
        store.create_run(
            run_id="other",
            job_id="other",
            strategy_id="llm_sma",
            artifact_dir=tmp_path,
            metrics={},
        )
    verified: list[str] = []

    def verify(*args: object, **kwargs: object) -> str:
        verified.append(str(args[2]))
        assert kwargs["expected_commentary"] is None
        return "report"

    monkeypatch.setattr("tests.e2e.discovery_workflow.verify_report_delivery", verify)
    turn = ModelTurn(2, "Complete", calls)
    if damage is None:
        assert verify_discovery_outcome(workspace, calls, turn) == "run"
        assert verified == ["run"]
    else:
        with pytest.raises(AssertionError):
            verify_discovery_outcome(workspace, calls, turn)
        assert verified == []


EARLY_CACHE_REPLY = (
    "現貨 spot：3筆；可直接重用：否；須重新取得才符合重用政策：是；"
    "理由：取得時不足7天。\n"
    "期貨 futures/um：5筆；可直接重用：否；須重新取得才符合重用政策：是；"
    "理由：取得時不足7天。\n"
    "查詢成功不保證行情完整。"
)


@pytest.mark.parametrize("settled", [True, False])
@pytest.mark.parametrize("separator", ["；", "｜", "，", "。"])
def test_cache_reply_accepts_evidence_bound_decisions(
    settled: bool, separator: str
) -> None:
    reply = EARLY_CACHE_REPLY.replace("；", separator)
    if settled:
        reply = (
            reply.replace("可直接重用：否", "可直接重用：是")
            .replace("須重新取得才符合重用政策：是", "須重新取得才符合重用政策：否")
            .replace("取得時不足7天", "取得時已超過7天")
        )
    verify_market_cache_reply(reply, settled=settled)


@pytest.mark.parametrize(
    "reply",
    [
        "現貨 spot：3 筆，可直接重用。\n期貨 futures/um：5 筆，可直接重用。\n"
        "查詢當時未滿7天，但現在早已超過7天，因此不需重抓。成功查詢不保證行情完整。",
        EARLY_CACHE_REPLY.replace("可直接重用：否", "可直接重用：是", 1),
        EARLY_CACHE_REPLY.replace(
            "須重新取得才符合重用政策：是", "須重新取得才符合重用政策：否", 1
        )
        + "\n只需等待即可使用原本快取。",
        EARLY_CACHE_REPLY.replace("3筆", "4筆"),
        EARLY_CACHE_REPLY.splitlines()[0],
        EARLY_CACHE_REPLY + "\n" + EARLY_CACHE_REPLY.splitlines()[1],
        EARLY_CACHE_REPLY.replace("可直接重用：否", "不可直接重用：否", 1),
        EARLY_CACHE_REPLY.replace("須重新取得", "不須重新取得", 1),
        EARLY_CACHE_REPLY.replace("可直接重用：否", "可直接重用：否或是").replace(
            "須重新取得才符合重用政策：是", "須重新取得才符合重用政策：是或否"
        ),
        EARLY_CACHE_REPLY.replace("可直接重用：否", "可直接重用：否 或是"),
        EARLY_CACHE_REPLY.replace("理由：", ""),
    ],
    ids=[
        "unstructured-waiting",
        "wrong-reuse",
        "wait-without-reacquiring",
        "wrong-rows",
        "missing-market",
        "duplicate-market",
        "negated-reuse-label",
        "negated-reacquire-label",
        "unselected-decisions",
        "unselected-spaced-decision",
        "unlabeled-inline-reason",
    ],
)
def test_cache_reply_rejects_wrong_or_ambiguous_market_decisions(reply: str) -> None:
    with pytest.raises(AssertionError):
        verify_market_cache_reply(reply, settled=False)


@pytest.mark.parametrize("punctuation", ["，", ", ", "。", ". "])
@pytest.mark.parametrize(
    ("field", "opposite"),
    [("可直接重用：否", "是"), ("須重新取得才符合重用政策：是", "否")],
)
def test_cache_reply_rejects_punctuated_ambiguous_decisions(
    punctuation: str, field: str, opposite: str
) -> None:
    reply = EARLY_CACHE_REPLY.replace(field, field + punctuation + "或" + opposite)
    with pytest.raises(AssertionError):
        verify_market_cache_reply(reply, settled=False)


DVOL_REPLY = (
    "short.yaml｜查詢紀錄：有｜DVOL筆數：2筆｜可直接重用：是｜理由：已有對應期間查詢。\n"
    "extended.yaml｜查詢紀錄：無｜DVOL筆數：未知｜可直接重用：否｜理由：尚未查詢新增期間。"
)
DVOL_TABLE = (
    "| 設定 | 查詢紀錄 | DVOL筆數 | 可直接重用 |\n"
    "|---|---|---:|---|\n"
    "| short.yaml | 有 | 2 | 是 |\n"
    "| extended.yaml | 無 | 未知 | 否 |\n\n"
    "short.yaml：有對應的成功查詢紀錄，實存 2 筆。\n"
    "extended.yaml：沒有對應的 DVOL 查詢紀錄，筆數未知並非零筆。"
)


@pytest.mark.parametrize(
    "reply",
    [
        DVOL_REPLY,
        DVOL_TABLE,
        DVOL_REPLY.replace("｜", "，"),
        DVOL_REPLY.replace("｜", "。"),
        DVOL_TABLE.replace(
            "short.yaml", "[short.yaml](/tmp/research/short.yaml)"
        ).replace("extended.yaml", "[extended.yaml](/tmp/research/extended.yaml)"),
        "short.yaml：查詢紀錄：有，DVOL筆數：2筆，可直接重用：是。\n"
        "extended.yaml：查詢紀錄：無，DVOL筆數：未知，可直接重用：否。",
    ],
    ids=["lines", "table", "commas", "sentences", "linked-table", "without-reason"],
)
def test_dvol_reply_accepts_query_evidence_and_unknown_observations(reply: str) -> None:
    verify_dvol_query_reply(reply)


@pytest.mark.parametrize(
    "reply",
    [
        DVOL_REPLY.replace("DVOL筆數：未知", "DVOL筆數：0"),
        DVOL_REPLY.replace("查詢紀錄：無", "查詢紀錄：有"),
        DVOL_REPLY.replace("可直接重用：否", "可直接重用：是"),
        DVOL_REPLY.replace("DVOL筆數：2筆", "DVOL筆數：3筆"),
        DVOL_REPLY.splitlines()[0],
        DVOL_REPLY.replace("DVOL筆數：未知", "DVOL筆數：未知或0"),
        DVOL_REPLY + "\n" + DVOL_REPLY.splitlines()[1],
        DVOL_TABLE.replace("| 無 | 未知 | 否 |", "| 無 | 0 | 否 |"),
        DVOL_TABLE.replace("| 無 | 未知 | 否 |", "| 有 | 未知 | 否 |"),
        DVOL_TABLE.replace("| 無 | 未知 | 否 |", "| 無 | 未知 | 是 |"),
        DVOL_TABLE.replace("| 有 | 2 | 是 |", "| 有 | 3 | 是 |"),
        DVOL_TABLE.replace("| 無 | 未知 | 否 |", "| 無 | 未知 | 否或是 |"),
        DVOL_TABLE.replace("| 無 | 未知 | 否 |", "| 無 | 否 |"),
        DVOL_TABLE + "\n" + DVOL_REPLY.splitlines()[1],
        DVOL_TABLE.replace("| 無 | 未知 | 否 |", "| 無 | 未知 | 否，或是 |"),
        DVOL_TABLE.replace("| 無 | 未知 | 否 |", "| 無，或有 | 未知 | 否 |"),
        DVOL_TABLE.replace("| 無 | 未知 | 否 |", "| 無 | 未知，或0 | 否 |"),
        DVOL_REPLY.replace("可直接重用：否", "可直接重用：否，或是"),
        DVOL_REPLY.replace("DVOL筆數：未知", "DVOL筆數：未知，或0"),
        DVOL_TABLE.replace("short.yaml", "not-short.yaml"),
        DVOL_REPLY.replace("extended.yaml", "not-extended.yaml"),
        DVOL_TABLE.replace("short.yaml", "[short.yaml](/tmp/not-short.yaml)"),
        DVOL_REPLY.replace("可直接重用：否", "可直接重用：否。或是"),
        DVOL_REPLY.replace("理由：", ""),
    ],
    ids=[
        "invent-zero",
        "invent-query",
        "reuse",
        "wrong-rows",
        "missing",
        "ambiguous",
        "duplicate",
        "table-invent-zero",
        "table-invent-query",
        "table-reuse",
        "table-wrong-rows",
        "table-ambiguous",
        "table-incomplete",
        "table-duplicate",
        "table-punctuated-reuse",
        "table-punctuated-query",
        "table-punctuated-rows",
        "punctuated-reuse",
        "punctuated-rows",
        "table-wrong-config",
        "wrong-config",
        "misleading-link",
        "ambiguous-sentences",
        "unlabeled-inline-reason",
    ],
)
def test_dvol_reply_rejects_invented_or_ambiguous_query_evidence(reply: str) -> None:
    with pytest.raises(AssertionError):
        verify_dvol_query_reply(reply)
