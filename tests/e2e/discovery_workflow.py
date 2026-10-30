"""Natural user intent and outcome checks without a tool-by-tool answer script."""

from __future__ import annotations

import re
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


def verify_market_cache_reply(reply: str, *, settled: bool) -> None:
    """Check each market's explicit reuse and reacquisition decisions."""
    plain = re.sub(r"[*`_]", "", reply)
    for label, count in (("現貨|spot", 3), ("期貨|futures/um", 5)):
        lines = [
            line
            for line in plain.splitlines()
            if re.search(label, line) and "可直接重用" in line
        ]
        assert len(lines) == 1, "Each market needs one unambiguous decision line"
        line = lines[0]
        assert re.search(rf"(?<!\d){count}\s*筆", line)
        fields = list(
            re.finditer(
                r"(?<!\w)(可直接重用|[須需]重新取得(?:才符合重用政策)?|理由)\s*[:：]\s*",
                line,
            )
        )
        reuse, reacquire = [], []
        for index, field in enumerate(fields):
            end = fields[index + 1].start() if index + 1 < len(fields) else len(line)
            value = line[field.end() : end].strip(" \t；;|｜，,。.")
            if field[1] == "可直接重用":
                reuse.append(value)
            elif field[1] != "理由":
                reacquire.append(value)
        assert reuse == ["是" if settled else "否"], (
            "Reuse decision must match this market's acquisition evidence"
        )
        assert reacquire == ["否" if settled else "是"], (
            "Waiting cannot make an earlier acquisition reusable"
        )
    assert re.search("不保證|不代表|不能|不足以|無法保證", plain)
    if not settled:
        assert re.search(r"(?<!\d)7\s*天|七天|seven days", plain)


def verify_dvol_query_reply(reply: str) -> None:
    """Bind queried versus unqueried feature evidence to each requested config."""
    plain = re.sub(r"[*`_]", "", reply)
    records: dict[str, list[dict[str, str]]] = {"short.yaml": [], "extended.yaml": []}

    def config_link(match: re.Match[str]) -> str:
        label, target = (value.rsplit("/", 1)[-1] for value in match.groups())
        if label in records:
            assert target == label, "Config link label and target must agree"
        return target if target in records else label

    plain = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", config_link, plain)
    headers: list[str] = []
    fields = ("設定", "查詢紀錄", "DVOL筆數", "可直接重用")
    for line in plain.splitlines():
        if line.strip().startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if all(field in cells for field in fields):
                assert len(cells) == len(set(cells)), "Ambiguous table headers"
                headers = cells
            elif headers and not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                assert len(cells) == len(headers), "Incomplete evidence table row"
                record = dict(zip(headers, cells, strict=True))
                name = record["設定"].rsplit("/", 1)[-1]
                if name in records:
                    records[name].append(record)
        else:
            headers = []
            if re.search(r"查詢紀錄\s*[:：]", line):
                names = [
                    name
                    for name in records
                    if re.search(rf"(?<![\w.-]){re.escape(name)}(?![\w.-])", line)
                ]
                assert len(names) == 1, "Evidence must identify the requested config"
                boundaries = list(
                    re.finditer(
                        r"(?<!\w)(查詢紀錄|DVOL筆數|可直接重用|理由)\s*[:：]\s*",
                        line,
                    )
                )
                record = {}
                for index, boundary in enumerate(boundaries):
                    end = (
                        boundaries[index + 1].start()
                        if index + 1 < len(boundaries)
                        else len(line)
                    )
                    assert boundary[1] not in record, "Duplicate evidence field"
                    record[boundary[1]] = line[boundary.end() : end].strip(
                        " \t；;|｜，,。."
                    )
                assert all(field in record for field in fields[1:])
                records[names[0]].append(record)
    for label, queried, rows, reusable in (
        ("short.yaml", "有", "2", "是"),
        ("extended.yaml", "無", "未知", "否"),
    ):
        assert len(records[label]) == 1, "Each config needs one unambiguous record"
        record = records[label][0]
        assert record["查詢紀錄"] == queried
        assert re.fullmatch(rf"{rows}(?:\s*筆)?", record["DVOL筆數"]), (
            "Unqueried rows must stay unknown, never zero"
        )
        assert record["可直接重用"] == reusable
