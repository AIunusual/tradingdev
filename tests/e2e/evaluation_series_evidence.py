"""Saved evidence and outcome checks for natural evaluation-series requests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tests.data_cache import seed_market_cache
from tests.e2e.conversation import visible_text
from tests.e2e.test_llm_ml import (
    DATASET_ARGUMENTS,
    TRAINING_ARGUMENTS,
    seed_ml_markets,
)
from tradingdev.adapters.storage.evaluation_series import EvaluationSeriesStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.ml_datasets import MLDatasetStore
from tradingdev.app.ml_service import MLService

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn
    from tests.e2e.llm_client import ToolCall
    from tests.integration.mcp_harness import MCPWorkspace


SERIES_REQUEST = (
    "請找出這個工作區已保存的訊號評估和分折驗證明細。這次只能讀取既有研究，"
    "不要重新評估、訓練或修改資料。我想分批核對訊號明細，每批最多 100 筆；"
    "本回合只查看最前面的 100 筆，之後我會請你繼續。"
    "請用中文解釋原始第一列的 Pearson IC 和 Spearman IC，並附一個 JSON 物件："
    "signal_artifact_id、signal_row_count，以及保留全部原始欄位的 first_signal_row。"
)
SERIES_CONTINUATION = (
    "請從剛才停下的位置接著讀完訊號明細，每批仍最多 100 筆，不要跳過中間資料。"
    "核對總筆數，找出暖機結束後第一筆仍有 IC 不可用的列並解釋原因。"
    "再完整匯出那份已保存的分折驗證明細，"
    "解釋第一列的來源座標與折內序號有何不同、訓練和驗證範圍的端點是否包含在內。"
    "不用貼出所有列；請附一個 JSON 物件：signal_artifact_id、signal_row_count、"
    "last_signal_row、first_unavailable_after_warmup、"
    "fold_artifact_id、fold_row_count、first_fold_row、"
    "source_row_is_fold_local（是否為折內序號）、"
    "bounds_are_inclusive（是否包含端點）；後兩項用布林值，摘錄列保留全部原始欄位。"
)


@dataclass(frozen=True)
class SavedSeries:
    artifact_id: str
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class SeriesCase:
    signal: SavedSeries
    fold: SavedSeries


@dataclass(frozen=True)
class OversizedSeries:
    artifact_id: str
    total: int
    head: list[dict[str, Any]]
    tail: list[dict[str, Any]]

    @property
    def request(self) -> str:
        return (
            f"請嘗試一次完整匯出已保存的逐點研究序列 {self.artifact_id} 供我核對。"
            "如果遇到容量限制，改為查看最前面 2 筆和最後面 2 筆就好，"
            "不要把整份資料逐批讀完。請說明實際取得的範圍與限制，"
            "附一個 JSON 物件：artifact_id、total、complete（本次是否取得完整明細）、"
            "head、tail（各保留讀到的原始列）。只能讀取已保存資料，不要重算或修改研究。"
        )

    @property
    def recheck_request(self) -> str:
        return (
            "儲存裝置剛才出現異常，請重新確認同一份明細現在是否還能讀取；"
            "這次要以重新查到的狀態作準，不能沿用上一回合的內容當作目前仍可讀的證據。"
            "只回報可讀性和原因，不要重新計算、補資料或修改研究。"
            "請附 JSON 物件：artifact_id、available（目前能否讀取明細）。"
        )


async def prepare_series_case(workspace: MCPWorkspace) -> SeriesCase:
    """Publish both kinds through the real MCP producer tools, before the model."""
    seed_ml_markets(workspace)
    async with workspace.connect() as client:
        built = await client.call("build_dataset", **DATASET_ARGUMENTS)
        assert built["success"], built
        dataset_id = built["dataset"]["dataset_id"]
        paths = WorkspacePaths(workspace.workspace)
        frame, manifest = MLDatasetStore(paths.ml_datasets).load(dataset_id)
        column = manifest["feature_columns"][0]
        recorded = await client.call(
            "evaluate_signal",
            dataset_id=dataset_id,
            column=column,
            threshold=float(frame[column].median()),
            window=12,
            n_quantiles=2,
            record=True,
        )
        trained = await client.call(
            "train_model", dataset_id=dataset_id, **TRAINING_ARGUMENTS
        )
        assert recorded["success"] and trained["success"], (recorded, trained)
        signal_id = recorded["series_artifact_id"]
        fold_id = trained["model"]["validation"]["fold_series_artifact_id"]
        signal = await client.call("export_evaluation_series", artifact_id=signal_id)
        fold = await client.call("export_evaluation_series", artifact_id=fold_id)
        assert signal["success"] and fold["success"], (signal, fold)
        assert signal["manifest"]["producer_source"] == "backend_rule"
        assert fold["manifest"]["training_status"] == "fitted"
    assert len(signal["rows"]) > 200
    assert signal["rows"][0]["pearson_ic"] is None
    assert signal["rows"][0]["spearman_ic"] is None
    assert signal["rows"][0]["availability_reason"] == "ic_warmup"
    assert any(
        row["availability_reason"] == "constant_window" for row in signal["rows"]
    )
    assert fold["rows"][0]["source_row"] > 0
    return SeriesCase(
        SavedSeries(signal_id, signal["rows"]), SavedSeries(fold_id, fold["rows"])
    )


async def prepare_oversized_series(workspace: MCPWorkspace) -> OversizedSeries:
    """Publish the smallest row-overflow artifact without altering product limits."""
    # The returns feature set drops twelve warm-up bars and the final label bar.
    size = EvaluationSeriesStore.MAX_EXPORT_ROWS + 14
    positions = np.arange(size, dtype=float)
    close = 100 + positions * 0.001 + np.sin(positions / 7)
    market = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=size, freq="min", tz="UTC"
            ),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(size, 100.0),
        }
    )
    seed_market_cache(
        workspace.data_root / "processed", market, timeframe="1m", market_type="spot"
    )
    async with workspace.connect() as client:
        built = await client.call(
            "build_dataset",
            **{**DATASET_ARGUMENTS, "timeframe": "1m", "end_date": "2024-03-31"},
        )
        assert built["success"], built
        dataset_id = built["dataset"]["dataset_id"]
    paths = WorkspacePaths(workspace.workspace)
    frame, _ = MLDatasetStore(paths.ml_datasets).load(dataset_id)
    total = len(frame)
    assert total == EvaluationSeriesStore.MAX_EXPORT_ROWS + 1
    # Caller arrays have unknown provenance. The service still validates dataset
    # identity, returns, and publication; no tool response or limit is replaced.
    recorded = MLService(workspace=paths).evaluate_signal(
        np.sin(np.arange(total) / 5).tolist(),
        frame["forward_return"].tolist(),
        dataset_id=dataset_id,
        window=12,
        n_quantiles=2,
        record=True,
    )
    artifact_id = recorded["series_artifact_id"]
    async with workspace.connect() as client:
        head = await client.call(
            "get_evaluation_series_page", artifact_id=artifact_id, limit=2
        )
        tail = await client.call(
            "get_evaluation_series_page",
            artifact_id=artifact_id,
            offset=total - 2,
            limit=2,
        )
        oversized = await client.call(
            "export_evaluation_series", artifact_id=artifact_id
        )
        assert head["success"] and tail["success"], (head, tail)
        assert head["manifest"]["producer_source"] == "caller_array"
        assert oversized["success"] is False and "export bound" in oversized["error"]
    return OversizedSeries(artifact_id, total, head["rows"], tail["rows"])


def delivered_json(text: str) -> dict[str, Any]:
    """Read the requested delivery object, independent of surrounding prose."""
    candidates = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    candidates.append(text.strip())
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    raise AssertionError(f"No delivered JSON object: {text}")


def assert_readonly(calls: list[ToolCall], readonly_tools: set[str]) -> None:
    """Include failed attempts: a failed mutation is still an unauthorized attempt."""
    assert calls, "The model must consult saved evidence"
    assert all(call.name in readonly_tools for call in calls), [
        call.name for call in calls
    ]
    assert all(call.name != "predict" for call in calls), (
        "Reading saved series must not regenerate model predictions"
    )


def assert_signal_pages(
    calls: list[ToolCall], signal: SavedSeries, *, start: int, stop: int
) -> int | None:
    """Consume this turn's contiguous pages from its saved continuation cursor."""
    cursor = start
    next_offset: int | None = start
    for call in calls:
        result = call.result
        if (
            call.name != "get_evaluation_series_page"
            or not isinstance(result, dict)
            or result.get("success") is not True
            or result.get("artifact_id") != signal.artifact_id
            or result.get("mode") != "rows"
        ):
            continue
        offset = result["offset"]
        rows = result["rows"]
        assert next_offset is not None and offset == next_offset, (
            "Continuation restarted or skipped unread rows"
        )
        assert 0 < len(rows) <= 100
        assert result["total"] == len(signal.rows)
        assert rows == signal.rows[offset : offset + len(rows)]
        cursor = offset + len(rows)
        next_offset = cursor if cursor < len(signal.rows) else None
        assert result["next_offset"] == next_offset
    assert cursor == stop, "Incomplete or unexpected signal-page coverage"
    return next_offset


def assert_series_delivery(
    case: SeriesCase, turns: list[ModelTurn], readonly_tools: set[str]
) -> None:
    assert len(turns) == 2
    calls = [call for turn in turns for call in turn.calls]
    assert_readonly(calls, readonly_tools)
    discovered = {
        item["artifact_id"]
        for call in calls
        if call.name == "get_evaluation_series_page"
        and isinstance(call.result, dict)
        and call.result.get("success") is True
        and call.result.get("mode") == "discovery"
        for item in call.result["items"]
    }
    assert {case.signal.artifact_id, case.fold.artifact_id} <= discovered
    assert not any(
        call.name == "export_evaluation_series"
        and call.arguments.get("artifact_id") == case.signal.artifact_id
        for call in turns[0].calls
    ), "The first request permits only the initial 100 signal rows"
    next_offset = assert_signal_pages(turns[0].calls, case.signal, start=0, stop=100)
    assert next_offset is not None
    assert (
        assert_signal_pages(
            turns[1].calls, case.signal, start=next_offset, stop=len(case.signal.rows)
        )
        is None
    )
    first = delivered_json(turns[0].text)
    assert first["signal_artifact_id"] == case.signal.artifact_id
    assert first["signal_row_count"] == len(case.signal.rows)
    assert first["first_signal_row"] == case.signal.rows[0], (
        "Unavailable IC must stay null"
    )
    exported = [
        call.result
        for call in calls
        if call.name == "export_evaluation_series"
        and isinstance(call.result, dict)
        and call.result.get("success") is True
    ]
    assert any(
        result["manifest"]["artifact_id"] == case.fold.artifact_id
        and result["rows"] == case.fold.rows
        for result in exported
    )
    final = delivered_json(turns[1].text)
    assert final["signal_artifact_id"] == case.signal.artifact_id
    assert final["signal_row_count"] == len(case.signal.rows)
    assert final["last_signal_row"] == case.signal.rows[-1]
    assert final["first_unavailable_after_warmup"] == next(
        row
        for row in case.signal.rows
        if row["availability_reason"] != "ic_warmup"
        and (row["pearson_ic"] is None or row["spearman_ic"] is None)
    )
    assert final["fold_artifact_id"] == case.fold.artifact_id
    assert final["fold_row_count"] == len(case.fold.rows)
    assert final["first_fold_row"] == case.fold.rows[0], (
        "Keep original dataset coordinates"
    )
    assert final["source_row_is_fold_local"] is False
    assert final["bounds_are_inclusive"] is True


def assert_overflow_recovery(
    case: OversizedSeries, turns: list[ModelTurn], readonly_tools: set[str]
) -> None:
    assert len(turns) == 2
    assert_readonly([call for turn in turns for call in turn.calls], readonly_tools)
    failures = [
        index
        for index, call in enumerate(turns[0].calls)
        if call.name == "export_evaluation_series"
        and call.arguments.get("artifact_id") == case.artifact_id
        and isinstance(call.result, dict)
        and call.result.get("success") is False
        and "export bound" in call.result.get("error", "")
    ]
    assert failures, "The bounded export failure was not exercised"
    pages = [
        call.result
        for call in turns[0].calls[failures[0] + 1 :]
        if call.name == "get_evaluation_series_page"
        and isinstance(call.result, dict)
        and call.result.get("success") is True
        and call.result.get("artifact_id") == case.artifact_id
    ]
    assert {page["offset"] for page in pages} == {0, case.total - 2}
    assert all(len(page["rows"]) == 2 for page in pages)
    assert next(page["rows"] for page in pages if page["offset"] == 0) == case.head
    assert next(page["rows"] for page in pages if page["offset"] != 0) == case.tail
    final = delivered_json(turns[0].text)
    assert final["artifact_id"] == case.artifact_id
    assert final["total"] == case.total and final["complete"] is False
    assert final["head"] == case.head and final["tail"] == case.tail
    reads = [
        call
        for call in turns[1].calls
        if call.name in {"get_evaluation_series_page", "export_evaluation_series"}
        and call.arguments.get("artifact_id") == case.artifact_id
    ]
    assert reads, "A fresh read must detect the removed artifact"
    assert all(
        isinstance(call.result, dict)
        and call.result.get("success") is False
        and call.result.get("code") == "series_unavailable"
        for call in reads
    )
    missing = delivered_json(turns[1].text)
    assert missing["artifact_id"] == case.artifact_id and missing["available"] is False
    assert re.search(
        r"缺|不完整|不存在|不可用|無法|unavailable|incomplete",
        visible_text(turns[1].text),
        re.IGNORECASE,
    )
