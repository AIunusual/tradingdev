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
    assert_ml_artifacts,
    assert_ml_workflow,
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
