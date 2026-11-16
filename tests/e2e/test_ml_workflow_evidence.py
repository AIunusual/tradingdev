"""Offline evidence checks against a real MCP ML workflow, with no model calls."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from tests.e2e.llm_client import ToolCall
from tests.e2e.test_llm_ml import (
    DATASET_ARGUMENTS,
    PREDICTION_ARGUMENTS,
    TRAINING_ARGUMENTS,
    TrainingProvenanceCase,
    assert_ml_artifacts,
    assert_ml_workflow,
    assert_training_provenance,
    prepare_training_provenance,
    seed_ml_markets,
)
from tests.integration.mcp_harness import temporary_mcp_workspace

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from tests.integration.mcp_harness import MCPWorkspace


@pytest.fixture(scope="module")
def ml_workflow() -> Iterator[tuple[MCPWorkspace, list[ToolCall]]]:
    with temporary_mcp_workspace() as workspace:
        seed_ml_markets(workspace)

        async def run() -> list[ToolCall]:
            calls: list[ToolCall] = []
            async with workspace.connect() as client:

                async def call(name: str, **arguments: Any) -> Any:
                    result = await client.call(name, **arguments)
                    assert result.get("success") is True, result
                    calls.append(ToolCall(name, arguments, result))
                    return result

                await call("get_ml_contract")
                built = await call("build_dataset", **DATASET_ARGUMENTS)
                dataset_id = built["dataset"]["dataset_id"]
                trained = await call(
                    "train_model", dataset_id=dataset_id, **TRAINING_ARGUMENTS
                )
                model_id = trained["model"]["model_id"]
                await call(
                    "predict",
                    model_id=model_id,
                    dataset_id=dataset_id,
                    **PREDICTION_ARGUMENTS,
                )
                await call(
                    "evaluate_signal",
                    window=12,
                    n_quantiles=2,
                    record=True,
                    model_id=model_id,
                    dataset_id=dataset_id,
                )
                await call("leaderboard", metric="hit_rate", dataset_id=dataset_id)
            return calls

        calls = asyncio.run(run())
        yield workspace, calls
    assert not workspace.root.exists()


@pytest.fixture
def calls(ml_workflow: tuple[MCPWorkspace, list[ToolCall]]) -> list[ToolCall]:
    return deepcopy(ml_workflow[1])


def test_real_mcp_ml_workflow_matches_saved_prices_model_and_leaderboard(
    ml_workflow: tuple[MCPWorkspace, list[ToolCall]],
) -> None:
    workspace, calls = ml_workflow
    assert_ml_artifacts(workspace, assert_ml_workflow(calls))


@pytest.mark.parametrize("event", range(6))
def test_ml_requires_every_successful_step(calls: list[ToolCall], event: int) -> None:
    calls[event].result["success"] = False
    with pytest.raises(AssertionError, match="Missing successful ordered ML step"):
        assert_ml_workflow(calls)


@pytest.mark.parametrize("event", [4, 5])
def test_ml_rejects_lost_training_binding(calls: list[ToolCall], event: int) -> None:
    result = calls[event].result
    payload = result["recorded"] if event == 4 else result["ranked"][0]
    payload["training_dataset_id"] = "ds_unrelated"
    with pytest.raises(AssertionError):
        assert_ml_workflow(calls)


@pytest.mark.parametrize("projection", ["evaluation", "recorded", "leaderboard"])
@pytest.mark.parametrize("source", [None, "caller_array"])
def test_ml_rejects_lost_or_changed_producer_source(
    calls: list[ToolCall], projection: str, source: str | None
) -> None:
    payload = {
        "evaluation": calls[4].result,
        "recorded": calls[4].result["recorded"],
        "leaderboard": calls[5].result["ranked"][0],
    }[projection]
    if source is None:
        del payload["producer_source"]
    else:
        payload["producer_source"] = source
    with pytest.raises(AssertionError):
        assert_ml_workflow(calls)


@pytest.fixture(scope="module")
def provenance_workflow() -> Iterator[
    tuple[MCPWorkspace, TrainingProvenanceCase, list[ToolCall], str, set[str]]
]:
    with temporary_mcp_workspace() as workspace:
        case = asyncio.run(prepare_training_provenance(workspace))

        async def run() -> tuple[list[ToolCall], set[str]]:
            calls: list[ToolCall] = []
            async with workspace.connect() as client:
                product_tools = (await client.session.list_tools()).tools
                readonly = {
                    tool.name
                    for tool in product_tools
                    if tool.annotations and tool.annotations.read_only_hint is True
                }
                for name, args in [
                    ("get_ml_contract", {}),
                    *(
                        (
                            "evaluate_signal",
                            {
                                "dataset_id": dataset_id,
                                "model_id": case.model_id,
                                "record": True,
                            },
                        )
                        for dataset_id in (
                            case.training_dataset_id,
                            case.overlapping_dataset_id,
                        )
                    ),
                    (
                        "leaderboard",
                        {
                            "metric": "hit_rate",
                            "allow_mixed": True,
                        },
                    ),
                ]:
                    result = await client.call(name, **args)
                    assert result["success"], result
                    calls.append(ToolCall(name, args, result))
                for evaluation in calls[1:3]:
                    args = {"ml_entry_ids": [evaluation.result["recorded"]["entry_id"]]}
                    result = await client.call("generate_report", **args)
                    assert result["success"], result
                    calls.append(ToolCall("generate_report", args, result))
                assert calls[-1].result["path"] != calls[-2].result["path"]
            return calls, readonly

        calls, readonly = asyncio.run(run())
        first, second = (
            call.result["recorded"]["entry_id"]
            for call in calls
            if call.name == "evaluate_signal"
        )
        rule, unknown, undeclared, caller = case.historical
        reply = (
            "| 候選 | 訓練來源 | 評估來源 | 判讀 |\n| --- | --- | --- | --- |\n"
            f"| {first} | {case.training_dataset_id} | {case.training_dataset_id} | "
            "後端用訓練資料評估，屬樣本內。 |\n"
            f"| {second} | {case.training_dataset_id} | "
            f"{case.overlapping_dataset_id} | "
            "ID 不同不代表樣本外，資料仍可能重疊。 |\n"
            f"| {rule.entry_id} | 未知 | {rule.dataset_id} | 宣告為手設規則；"
            "訊號來源未知，無法確認是否涉及模型訓練。 |\n"
            f"| {unknown.entry_id} | 未知 | {unknown.dataset_id} | "
            "訓練來源未知，無法確認與評估資料的關係。 |\n"
            f"| {undeclared.entry_id} | 未知 | {undeclared.dataset_id} | "
            "未宣告訊號來源，無法判定是否曾訓練。 |\n"
            f"| {caller.entry_id} | {caller.training_dataset_id} | "
            f"{caller.dataset_id} | 呼叫端自備訊號；模型 ID 僅為宣告標籤，"
            "無法確認這組訊號的訓練與評估關係。 |"
        )
        reply += "\n" + calls[-1].result["navigation_instructions"]
        yield workspace, case, calls, reply, readonly
    assert not workspace.root.exists()


def test_real_mcp_training_provenance_evidence(
    provenance_workflow: tuple[
        MCPWorkspace, TrainingProvenanceCase, list[ToolCall], str, set[str]
    ],
) -> None:
    assert_training_provenance(*provenance_workflow)


def test_training_provenance_accepts_unknown_origin_in_either_word_order(
    provenance_workflow: tuple[
        MCPWorkspace, TrainingProvenanceCase, list[ToolCall], str, set[str]
    ],
) -> None:
    workspace, case, calls, reply, readonly = provenance_workflow
    assert "訊號來源未知" in reply
    assert_training_provenance(
        workspace, case, calls, reply.replace("訊號來源未知", "未知訊號來源"), readonly
    )


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_binding",
        "missing_evaluation",
        "unrequested_training",
        "missing_explanation",
        "wrong_training_source",
        "wrong_evaluation_source",
        "single_report_claim",
        "caller_as_model_output",
        "caller_as_in_sample",
        "changed_saved_source",
        "legacy_rule_as_no_training",
    ],
)
def test_provenance_evidence_rejects_broken_workflows(
    provenance_workflow: tuple[
        MCPWorkspace, TrainingProvenanceCase, list[ToolCall], str, set[str]
    ],
    fault: str,
) -> None:
    workspace, case, original, reply, readonly = provenance_workflow
    calls = deepcopy(original)
    expected_error = None
    if fault == "wrong_binding":
        calls[2].result["recorded"]["training_dataset_id"] = case.overlapping_dataset_id
    elif fault == "missing_evaluation":
        calls.pop(2)
    elif fault == "unrequested_training":
        calls.append(ToolCall("train_model", {}, {"success": False}))
    elif fault == "wrong_training_source":
        entry_id = calls[2].result["recorded"]["entry_id"]
        reply = reply.replace(
            f"| {entry_id} | {case.training_dataset_id} |",
            f"| {entry_id} | ds_WRONG_TRAIN |",
        )
        expected_error = "Missing or incorrect training dataset"
    elif fault == "wrong_evaluation_source":
        reply = reply.replace(case.overlapping_dataset_id, "ds_WRONG_EVAL")
        expected_error = "Missing or incorrect evaluation dataset"
    elif fault == "single_report_claim":
        reply += "\n所有內容保存在同一份離線 HTML。"
        expected_error = "Multiple report files"
    elif fault == "caller_as_model_output":
        reply = reply.replace(
            "呼叫端自備訊號；模型 ID 僅為宣告標籤，",
            "系統已確認這是模型實際產生的訊號，",
        )
        expected_error = "externally supplied"
    elif fault == "caller_as_in_sample":
        reply = reply.replace(
            "無法確認這組訊號的訓練與評估關係。",
            "兩個資料集 ID 相同，所以這筆候選屬樣本內。",
        )
        expected_error = "affirmative"
    elif fault == "changed_saved_source":
        caller = case.historical[-1]
        row = next(
            row
            for row in calls[3].result["ranked"]
            if row["entry_id"] == caller.entry_id
        )
        row["producer_source"] = "backend_model"
        expected_error = "saved producer source"
    elif fault == "legacy_rule_as_no_training":
        reply = reply.replace(
            "訊號來源未知，無法確認是否涉及模型訓練。",
            "訊號來源未知；模型訓練不適用。",
        )
        expected_error = "training/evaluation provenance is unknown"
    else:
        reply = "\n".join(
            line
            for line in reply.splitlines()
            if case.historical[-1].entry_id not in line
        )
    with pytest.raises(AssertionError, match=expected_error):
        assert_training_provenance(workspace, case, calls, reply, readonly)


def test_ml_requires_contract_before_building(calls: list[ToolCall]) -> None:
    calls[0], calls[1] = calls[1], calls[0]
    with pytest.raises(AssertionError, match="Missing successful ordered ML step"):
        assert_ml_workflow(calls)


@pytest.mark.parametrize("event", [2, 3, 4])
def test_ml_rejects_request_from_another_dataset(
    calls: list[ToolCall], event: int
) -> None:
    calls[event].arguments["dataset_id"] = "ds_other"
    with pytest.raises(AssertionError):
        assert_ml_workflow(calls)


@pytest.mark.parametrize("key", ["signal", "forward_returns", "horizon"])
def test_ml_evaluation_rejects_caller_arrays_and_horizon(
    calls: list[ToolCall], key: str
) -> None:
    calls[4].arguments[key] = 1 if key == "horizon" else [0.1]
    with pytest.raises(AssertionError, match="saved identities"):
        assert_ml_workflow(calls)


@pytest.mark.parametrize("key", ["signal", "forward_returns"])
def test_ml_requires_the_declared_preview_length(
    calls: list[ToolCall], key: str
) -> None:
    calls[3].result["prediction"][key].pop()
    with pytest.raises(AssertionError, match="preview must be bounded"):
        assert_ml_workflow(calls)


def test_ml_requires_full_evaluation_beyond_the_preview(calls: list[ToolCall]) -> None:
    calls[4].result["n_observations"] = 20
    with pytest.raises(AssertionError):
        assert_ml_workflow(calls)


def test_ml_sdk_encoded_arguments_preserve_raw_evidence(calls: list[ToolCall]) -> None:
    for key in ("parameters", "n_folds", "embargo", "min_train_bars"):
        calls[2].arguments[key] = json.dumps(calls[2].arguments[key])
    for key in ("offset", "limit"):
        calls[3].arguments[key] = json.dumps(calls[3].arguments[key])
    for key in ("record", "window", "n_quantiles"):
        calls[4].arguments[key] = json.dumps(calls[4].arguments[key])
    original = deepcopy(calls)
    assert_ml_workflow(calls)
    assert calls == original


def test_ml_rejects_leaderboard_from_another_evaluation(calls: list[ToolCall]) -> None:
    calls[5].result["ranked"][0]["entry_id"] = "unrelated"
    with pytest.raises(AssertionError):
        assert_ml_workflow(calls)


def test_ml_rejects_training_without_clean_folds(calls: list[ToolCall]) -> None:
    calls[2].result["model"]["validation"]["leakage_audit_clean"] = False
    with pytest.raises(AssertionError):
        assert_ml_workflow(calls)


def test_ml_saved_model_independently_checks_reported_prediction(
    ml_workflow: tuple[MCPWorkspace, list[ToolCall]], calls: list[ToolCall]
) -> None:
    evidence = assert_ml_workflow(calls)
    evidence.prediction["signal"][0] *= -1
    with pytest.raises(AssertionError):
        assert_ml_artifacts(ml_workflow[0], evidence)


def test_ml_saved_prices_independently_check_reported_returns(
    ml_workflow: tuple[MCPWorkspace, list[ToolCall]], calls: list[ToolCall]
) -> None:
    evidence = assert_ml_workflow(calls)
    evidence.prediction["forward_returns"][0] *= -1
    with pytest.raises(AssertionError):
        assert_ml_artifacts(ml_workflow[0], evidence)
