"""Check live-test verifiers against real MCP reads and damaged transcripts."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from tests.e2e.conversation import ModelTurn
from tests.e2e.evaluation_series_evidence import (
    OversizedSeries,
    SeriesCase,
    assert_overflow_recovery,
    assert_series_delivery,
    delivered_json,
    prepare_oversized_series,
    prepare_series_case,
)
from tests.e2e.llm_client import ToolCall
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.storage.evaluation_series import EvaluationSeriesStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths

if TYPE_CHECKING:
    from collections.abc import Iterator


def _delivery(payload: dict[str, Any], wording: str = "") -> str:
    return wording + "\n```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


@dataclass(frozen=True)
class Transcript:
    case: SeriesCase | OversizedSeries
    turns: list[ModelTurn]
    readonly: set[str]

    def verify(self) -> None:
        if isinstance(self.case, SeriesCase):
            assert_series_delivery(self.case, self.turns, self.readonly)
        else:
            assert_overflow_recovery(self.case, self.turns, self.readonly)


@pytest.fixture(scope="module")
def series_transcript() -> Iterator[Transcript]:
    with temporary_mcp_workspace() as workspace:
        case = asyncio.run(prepare_series_case(workspace))

        async def run() -> Transcript:
            calls: list[ToolCall] = []
            async with workspace.connect() as client:
                tools = (await client.session.list_tools()).tools
                readonly = {
                    tool.name
                    for tool in tools
                    if tool.annotations and tool.annotations.read_only_hint is True
                }

                async def call(name: str, **arguments: Any) -> Any:
                    result = await client.call(name, **arguments)
                    calls.append(ToolCall(name, arguments, result))
                    return result

                await call("get_evaluation_series_page")
                await call(
                    "get_evaluation_series_page",
                    artifact_id=case.signal.artifact_id,
                    limit=100,
                )
                first = ModelTurn(
                    1,
                    _delivery(
                        {
                            "signal_artifact_id": case.signal.artifact_id,
                            "signal_row_count": len(case.signal.rows),
                            "first_signal_row": case.signal.rows[0],
                        },
                        "首列的 Pearson IC 與 Spearman IC 仍在暖機，"
                        "值不可用，不能視為零。",
                    ),
                    calls.copy(),
                )
                calls.clear()
                for offset in range(100, len(case.signal.rows), 100):
                    await call(
                        "get_evaluation_series_page",
                        artifact_id=case.signal.artifact_id,
                        offset=offset,
                        limit=100,
                    )
                await call(
                    "export_evaluation_series", artifact_id=case.fold.artifact_id
                )
                final = ModelTurn(
                    2,
                    _delivery(
                        {
                            "signal_artifact_id": case.signal.artifact_id,
                            "signal_row_count": len(case.signal.rows),
                            "last_signal_row": case.signal.rows[-1],
                            "first_unavailable_after_warmup": next(
                                row
                                for row in case.signal.rows
                                if row["availability_reason"] == "constant_window"
                            ),
                            "fold_artifact_id": case.fold.artifact_id,
                            "fold_row_count": len(case.fold.rows),
                            "first_fold_row": case.fold.rows[0],
                            "source_row_is_fold_local": False,
                            "bounds_are_inclusive": True,
                        },
                        "範圍包含首尾端點；source_row 是原始資料集列座標，"
                        "不是折內序號。",
                    ),
                    calls.copy(),
                )
            return Transcript(case, [first, final], readonly)

        yield asyncio.run(run())
    assert not workspace.root.exists()


@pytest.fixture(scope="module")
def recovery_transcript() -> Iterator[Transcript]:
    with temporary_mcp_workspace() as workspace:
        case = asyncio.run(prepare_oversized_series(workspace))

        async def run() -> Transcript:
            calls: list[ToolCall] = []
            async with workspace.connect() as client:
                tools = (await client.session.list_tools()).tools
                readonly = {
                    tool.name
                    for tool in tools
                    if tool.annotations and tool.annotations.read_only_hint is True
                }

                async def call(name: str, **arguments: Any) -> Any:
                    result = await client.call(name, **arguments)
                    calls.append(ToolCall(name, arguments, result))
                    return result

                await call("export_evaluation_series", artifact_id=case.artifact_id)
                for offset in (0, case.total - 2):
                    await call(
                        "get_evaluation_series_page",
                        artifact_id=case.artifact_id,
                        offset=offset,
                        limit=2,
                    )
                first = ModelTurn(
                    1,
                    _delivery(
                        {
                            "artifact_id": case.artifact_id,
                            "total": case.total,
                            "complete": False,
                            "head": case.head,
                            "tail": case.tail,
                        },
                        "超過匯出容量，只讀取首尾各兩列。",
                    ),
                    calls.copy(),
                )
                store = EvaluationSeriesStore(
                    WorkspacePaths(workspace.workspace).ml_evaluation_series
                )
                (store.path_for(case.artifact_id) / "series.parquet").unlink()
                calls.clear()
                await call(
                    "get_evaluation_series_page", artifact_id=case.artifact_id, limit=2
                )
                final = ModelTurn(
                    2,
                    _delivery(
                        {"artifact_id": case.artifact_id, "available": False},
                        "保存的明細缺檔，目前不可用。",
                    ),
                    calls.copy(),
                )
            return Transcript(case, [first, final], readonly)

        yield asyncio.run(run())
    assert not workspace.root.exists()


def test_series_verifier_accepts_real_pages_and_export(
    series_transcript: Transcript,
) -> None:
    series_transcript.verify()


@pytest.mark.parametrize(
    "damage",
    [
        "zero_ic",
        "zero_constant_ic",
        "renumber_fold",
        "exclusive_bound",
        "skipped_page",
        "second_turn_restarts",
        "fold_local_claim",
        "exclusive_bounds_claim",
        "failed_mutation",
        "regenerate_predictions",
    ],
)
def test_series_verifier_rejects_incorrect_delivery(
    series_transcript: Transcript, damage: str
) -> None:
    transcript = deepcopy(series_transcript)
    first, final = transcript.turns
    payload = delivered_json(first.text if damage == "zero_ic" else final.text)
    if damage == "zero_ic":
        payload["first_signal_row"]["pearson_ic"] = 0
        transcript.turns[0] = ModelTurn(1, _delivery(payload, "暖機"), first.calls)
    elif damage == "zero_constant_ic":
        payload["first_unavailable_after_warmup"]["pearson_ic"] = 0
        transcript.turns[1] = ModelTurn(
            2, _delivery(payload, "包含首尾端點，原始資料列。"), final.calls
        )
    elif damage in {"renumber_fold", "exclusive_bound"}:
        key = "source_row" if damage == "renumber_fold" else "eval_end_row"
        payload["first_fold_row"][key] -= 1
        transcript.turns[1] = ModelTurn(
            2, _delivery(payload, "包含首尾端點，原始資料列。"), final.calls
        )
    elif damage == "skipped_page":
        final.calls[:] = [
            call for call in final.calls if call.arguments.get("offset") != 100
        ]
    elif damage == "second_turn_restarts":
        final.calls[:0] = [
            deepcopy(call)
            for call in first.calls
            if call.name == "get_evaluation_series_page"
            and call.result.get("mode") == "rows"
        ]
    elif damage in {"fold_local_claim", "exclusive_bounds_claim"}:
        if damage == "fold_local_claim":
            payload["source_row_is_fold_local"] = True
            wording = "source_row 不是原始資料集列座標，而是折內序號。"
        else:
            payload["bounds_are_inclusive"] = False
            wording = "範圍不包含首尾端點。"
        transcript.turns[1] = ModelTurn(2, _delivery(payload, wording), final.calls)
    elif damage == "failed_mutation":
        final.calls.append(ToolCall("evaluate_signal", {}, {"success": False}))
    else:
        final.calls.append(ToolCall("predict", {}, {"success": False}))
    with pytest.raises(AssertionError):
        transcript.verify()


def test_recovery_verifier_accepts_real_overflow_and_missing_file(
    recovery_transcript: Transcript,
) -> None:
    recovery_transcript.verify()


@pytest.mark.parametrize(
    "damage", ["claims_complete", "misses_tail", "no_fresh_read", "claims_available"]
)
def test_recovery_verifier_rejects_incorrect_delivery(
    recovery_transcript: Transcript, damage: str
) -> None:
    transcript = deepcopy(recovery_transcript)
    first, final = transcript.turns
    if damage == "claims_complete":
        payload = delivered_json(first.text)
        payload["complete"] = True
        transcript.turns[0] = ModelTurn(1, _delivery(payload), first.calls)
    elif damage == "misses_tail":
        first.calls.pop()
    elif damage == "no_fresh_read":
        final.calls.clear()
    else:
        payload = delivered_json(final.text)
        payload["available"] = True
        transcript.turns[1] = ModelTurn(2, _delivery(payload, "不可用"), final.calls)
    with pytest.raises(AssertionError):
        transcript.verify()
