"""Natural discovery and delivery of saved grids without new experiments."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from tooling.context_budget import json_size, tool_context_report

from tests.app.test_parameter_plateau import _record
from tests.app.test_selection_evidence import _publish
from tests.e2e.conversation import ModelTurn, links_to_file, visible_text
from tests.e2e.llm_client import ToolCall
from tests.e2e.test_llm_comparison import _run
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.plateau_service import PlateauService

if TYPE_CHECKING:
    from tests.e2e.conversation import UserReply
    from tests.integration.mcp_harness import MCPWorkspace


async def _initial_context(workspace: MCPWorkspace) -> dict[str, Any]:
    async with workspace.connect() as client:
        tools = (await client.session.list_tools()).tools
        assert len(tools) == len(client.output_schemas)
        return tool_context_report(
            client.instructions,
            [tool.model_dump(mode="json", by_alias=True) for tool in tools],
        )


@pytest.mark.live_llm
@pytest.mark.parametrize("source", ["signal", "optimization"])
def test_natural_saved_parameter_plateau(
    pytestconfig: pytest.Config, source: str
) -> None:
    with temporary_mcp_workspace() as workspace:
        paths = WorkspacePaths(workspace.workspace)
        sql = SQLiteStore(paths)
        if source == "signal":
            ids = _record(paths, [0.5, None, 0.6])
            selection = (
                f"訊號候選身分是 {', '.join(ids)}，"
                "已保存的 threshold 層級為 1、2、3，指標用平均 IC"
            )
        else:
            _publish(paths, sql, "saved-grid", 3)
            selection = (
                "最佳化結果身分是 saved-grid，參數 window 的層級為 1、2、3；"
                "只分析各組訓練結果，目標為回測 Sharpe"
            )
        request = (
            "這次 MCP 與客戶端在同一台電腦，工作區檔案可直接開啟。"
            f"請只用這個工作區已保存的證據，檢查參數最佳點是不是附近也穩定的平臺。{selection}。"
            "鄰域看附近一格；保留沒有分數的格點，不要只看排名前幾名。"
            "請每頁只取一格，第一輪先給摘要與第一格。"
            "只要這項分析和完整 JSON，不要其他 HTML 報告，"
            "不要下載行情、重新評估、訓練、回測或啟動任何新實驗。"
        )
        continuation = (
            "請繼續讀剩餘格點，沿用剛才固定集合，不要重新分析；"
            "交付包含全部格點和來源證據的 JSON 檔案連結，"
            "說明這是否能當樣本外或獲利證明。"
        )
        turns: list[ModelTurn] = []
        calls: list[ToolCall] = []
        context = asyncio.run(_initial_context(workspace))

        async def reply(turn: ModelTurn) -> str | None:
            turns.append(turn)
            print(turn.text, flush=True)
            return continuation if len(turns) == 1 else None

        readonly = asyncio.run(_run(workspace, pytestconfig, reply, calls, request))
        assert len(turns) == 2
        assert [call for turn in turns for call in turn.calls] == calls
        assert all(
            call.name in readonly | {"analyze_parameter_plateau"}
            and call.name != "predict"
            for call in calls
        ), calls
        created = [
            call.result
            for call in calls
            if call.name == "analyze_parameter_plateau"
            and isinstance(call.result, dict)
            and call.result.get("success")
        ]
        assert len(created) == 1, calls
        analysis = created[0]
        assert any(
            call.name == "analyze_parameter_plateau" and call.result == analysis
            for call in turns[0].calls
        )
        assert all(
            call.name != "analyze_parameter_plateau" for call in turns[1].calls
        ), turns[1].calls
        _assert_initial_delivery(turns[0].text, source)
        assert analysis["n_candidates"] == 3 and analysis["n_undefined"] == 1
        assert analysis["cells"]["limit"] == 1 and analysis["cells"]["offset"] == 0
        continued = [
            call for call in turns[1].calls if call.name == "get_parameter_plateau"
        ]
        assert continued, calls
        assert all(call.result.get("success") for call in continued), calls
        assert {call.result["cells"]["offset"] for call in continued} >= {1, 2}
        assert all(call.result["cells"]["limit"] == 1 for call in continued)
        assert all(
            call.arguments["analysis_id"] == analysis["analysis_id"]
            for call in continued
        )
        path = Path(analysis["artifact"]["path"])
        body = json.loads(path.read_text())
        assert body["source_hash"] == analysis["source_hash"]
        assert len(body["cells"]) == 3
        assert body["cells"][1]["status"] == "undefined"
        assert links_to_file(turns[-1].text, path), turns[-1].text
        _assert_limitations(turns[-1].text)
        initial = sum(
            context[key]["utf8_bytes"] for key in ("instructions", "model_tools")
        )
        history = json_size(
            {
                "user_messages": [request, continuation],
                "assistant_turns": [turn.text for turn in turns],
                "calls": [asdict(call) for call in calls],
            }
        )["utf8_bytes"]
        print(
            json.dumps(
                {
                    "client": pytestconfig.getoption("llm_provider"),
                    "measurement": (
                        "all TradingDev tools, Chat Completions input projection plus "
                        "observed application JSON / UTF-8"
                    ),
                    "initial_utf8_bytes": initial,
                    "largest_application_reply_utf8_bytes": max(
                        json_size(call.result)["utf8_bytes"] for call in calls
                    ),
                    "observed_history_estimate_utf8_bytes": initial + history,
                    "limitations": (
                        "Not native Codex request bytes or tokens; excludes host "
                        "instructions, private history, compression and output reserve."
                    ),
                }
            ),
            flush=True,
        )
        if source == "optimization":
            assert analysis["summary"]["best"]["identity"]["scope"] == "trial/2/train"
        assert len(sql.list_runs()) == int(source == "optimization")
    assert not workspace.root.exists()


def _assert_initial_delivery(text: str, source: str) -> None:
    """Require user-visible summary and first-cell facts, not tool output alone."""
    visible = re.sub(r"([-−+])\s+(?=\d)", r"\1", visible_text(text))
    axis = "threshold" if source == "signal" else "window"
    values = (
        (r"0\.5", r"0\.6") if source == "signal" else (r"0(?:\.0+)?", r"2(?:\.0+)?")
    )
    best_coordinate = rf"{axis}[\s\"'`]*[:=：]?\s*3(?:\.0+)?\b"
    assert re.search(
        rf"(?:最佳|最高|\bbest\b)[^\d。；;]{{0,60}}?{best_coordinate}"
        rf"|{best_coordinate}[^\d。；;]{{0,60}}?(?:最佳|最高|\bbest\b)",
        visible,
        re.IGNORECASE,
    ), visible
    for coordinate, value in zip((1, 3), values, strict=True):
        assert re.search(
            rf"{axis}[\s\"'`]*[:=：]?\s*{coordinate}(?:\.0+)?\b"
            rf".{{0,160}}?(?<![\w.+−-])\+?{value}(?![\w.])",
            visible,
        ), visible


def _assert_limitations(text: str) -> None:
    visible = visible_text(text)
    for concept in ("樣本外", "獲利"):
        assert re.search(
            rf"(?:不代表|不能|無法|不等於|不保證|不是|不構成|不可)[^。；;！？？，,]{{0,45}}{concept}"
            rf"|{concept}[^。；;！？？，,]{{0,30}}(?:不能|無法|不保證|不成立)",
            visible,
        ), visible


@pytest.mark.parametrize(
    "text",
    [
        "最佳 threshold=1，平均 IC 0.5。另有 threshold=3，平均 IC 0.6。",
        "最佳 threshold=3，平均 IC -0.6。第一格 threshold=1，平均 IC -0.5。",
    ],
    ids=["wrong-best", "negative-values"],
)
def test_initial_verifier_rejects_wrong_facts(text: str) -> None:
    with pytest.raises(AssertionError):
        _assert_initial_delivery(text, "signal")


@pytest.mark.parametrize(
    ("source", "text"),
    [
        (
            "signal",
            "最佳單點是 threshold 3，IC 為 0.6。第一格 threshold 1：平均 IC 0.5。",
        ),
        (
            "optimization",
            "window=3 的訓練 Sharpe 最高，為 2.0。第一格 window=1 的 Sharpe 為 0.0。",
        ),
    ],
)
def test_initial_verifier_accepts_correct_facts(source: str, text: str) -> None:
    _assert_initial_delivery(text, source)


@pytest.mark.parametrize(
    "fault", [None, "failed-reanalysis", "missing-initial-delivery", "early-pages"]
)
def test_natural_verifier_checks_each_turn(
    monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config, fault: str | None
) -> None:
    """Keep real collections/artifacts while injecting model workflow regressions."""
    import tests.e2e.test_llm_plateau as scenario

    async def initial(_workspace: MCPWorkspace) -> dict[str, Any]:
        return {"instructions": {"utf8_bytes": 0}, "model_tools": {"utf8_bytes": 0}}

    async def observer(
        workspace: MCPWorkspace,
        _config: pytest.Config,
        reply: UserReply,
        calls: list[ToolCall],
        _request: str,
    ) -> set[str]:
        paths = WorkspacePaths(workspace.workspace)
        service = PlateauService(workspace=paths, store=SQLiteStore(paths))
        args: dict[str, Any] = {
            "metric": "signal.ic_mean",
            "axes": {"/threshold": [1, 2, 3]},
            "entry_ids": ["0", "1", "2"],
            "limit": 1,
        }
        result = service.analyze(**args)
        assert result["success"], result
        first = [ToolCall("analyze_parameter_plateau", args, result)]
        pages = [
            ToolCall(
                "get_parameter_plateau",
                {"analysis_id": result["analysis_id"], "offset": offset, "limit": 1},
                service.read(result["analysis_id"], offset=offset, limit=1),
            )
            for offset in (1, 2)
        ]
        second = pages
        if fault == "failed-reanalysis":
            second = [
                ToolCall("analyze_parameter_plateau", {}, {"success": False}),
                *pages,
            ]
        if fault == "early-pages":
            first += pages
            second = []
        text = (
            "完成。"
            if fault == "missing-initial-delivery"
            else (
                "摘要：最佳 threshold=3，平均 IC 0.6。第一格 threshold=1，平均 IC 0.5。"
            )
        )
        calls.extend(first)
        assert await reply(ModelTurn(1, text, first))
        calls.extend(second)
        final = f"[完整 JSON]({result['artifact']['path']})。不能當作樣本外或獲利證明。"
        assert await reply(ModelTurn(2, final, second)) is None
        return {"get_parameter_plateau"}

    monkeypatch.setattr(scenario, "_initial_context", initial)
    monkeypatch.setattr(scenario, "_run", observer)
    if fault is None:
        scenario.test_natural_saved_parameter_plateau(pytestconfig, "signal")
    else:
        with pytest.raises(AssertionError):
            scenario.test_natural_saved_parameter_plateau(pytestconfig, "signal")


@pytest.mark.parametrize(
    "text",
    [
        "平臺穩定不代表樣本外有效，也不保證獲利。",
        "不能當成樣本外驗證或獲利證明。",
    ],
)
def test_limitations_verifier_accepts_explicit_limits(text: str) -> None:
    _assert_limitations(text)


@pytest.mark.parametrize(
    "text",
    [
        "平臺穩定代表樣本外有效並保證獲利。",
        "不能當成樣本外驗證，但已證明獲利。",
    ],
)
def test_limitations_verifier_rejects_positive_claims(text: str) -> None:
    with pytest.raises(AssertionError):
        _assert_limitations(text)
