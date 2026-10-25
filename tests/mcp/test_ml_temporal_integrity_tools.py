"""Temporal validation and single-asset evaluation through real MCP dispatch."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult
from tests.app.test_ml_temporal_integrity import (
    DATA_ARGUMENTS,
    MODEL_ARGUMENTS,
    SYMBOLS,
    seed_temporal_workspace,
)

from tradingdev.app.contracts.common import ErrorResponse
from tradingdev.app.contracts.ml import (
    BuildDatasetResponse,
    BuildPanelResponse,
    PredictionEvaluationResponse,
    PredictionResponse,
    TrainModelResponse,
)
from tradingdev.app.ml_service import MLService
from tradingdev.mcp.strict_server import StrictMCPServer
from tradingdev.mcp.tools import ml

if TYPE_CHECKING:
    from pathlib import Path

    from mcp.server import MCPServer

    from tradingdev.adapters.storage.filesystem import WorkspacePaths


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    return seed_temporal_workspace(tmp_path)


@pytest.fixture
def server(workspace: WorkspacePaths) -> MCPServer:
    mcp = StrictMCPServer("ml-temporal-integrity")
    ml.register(mcp, MLService(workspace))
    return mcp


def dispatch(server: MCPServer, name: str, **arguments: Any) -> dict[str, Any]:
    async def call() -> dict[str, Any]:
        result = await server.call_tool(name, arguments)
        assert isinstance(result, CallToolResult)
        assert result.structured_content is not None
        payload = result.structured_content
        assert isinstance(payload, dict)
        response = payload.get("result", payload)
        assert isinstance(response, dict)
        return response

    return asyncio.run(call())


def test_panel_validation_and_prediction_require_single_asset_for_evaluation(
    server: MCPServer, workspace: WorkspacePaths
) -> None:
    panel = BuildPanelResponse.model_validate(
        dispatch(server, "build_panel", symbols=SYMBOLS, **DATA_ARGUMENTS)
    ).panel
    trained = TrainModelResponse.model_validate(
        dispatch(
            server,
            "train_model",
            dataset_id=panel.dataset_id,
            validation="purged_walk_forward",
            n_folds=3,
            **MODEL_ARGUMENTS,
        )
    ).model
    assert trained.validation is not None
    assert trained.validation.leakage_audit_clean is True
    assert trained.validation.n_folds == 3
    prediction = PredictionResponse.model_validate(
        dispatch(
            server, "predict", dataset_id=panel.dataset_id, model_id=trained.model_id
        )
    ).prediction
    assert prediction.n_rows == panel.rows
    for record in (False, True):
        rejected = ErrorResponse.model_validate(
            dispatch(
                server,
                "evaluate_signal",
                dataset_id=panel.dataset_id,
                model_id=trained.model_id,
                record=record,
                window=12,
                n_quantiles=2,
            )
        )
        assert rejected.code == "invalid_signal"
        assert "single-symbol dataset" in rejected.error
    assert not (workspace.ml_leaderboard / "leaderboard.jsonl").exists()

    single = BuildDatasetResponse.model_validate(
        dispatch(server, "build_dataset", symbol="BTC/USDT", **DATA_ARGUMENTS)
    ).dataset
    single_prediction = PredictionResponse.model_validate(
        dispatch(
            server, "predict", dataset_id=single.dataset_id, model_id=trained.model_id
        )
    ).prediction
    assert single_prediction.n_rows == single.rows
    accepted = PredictionEvaluationResponse.model_validate(
        dispatch(
            server,
            "evaluate_signal",
            dataset_id=single.dataset_id,
            model_id=trained.model_id,
            record=True,
            window=12,
            n_quantiles=2,
        )
    )
    assert accepted.horizon == DATA_ARGUMENTS["horizon"]
    assert accepted.n_observations == single.rows
    assert accepted.recorded is not None and accepted.recorded.n_candidates == 1


def test_forward_filled_panel_validation_is_rejected_before_model_publication(
    server: MCPServer, workspace: WorkspacePaths
) -> None:
    panel = BuildPanelResponse.model_validate(
        dispatch(
            server,
            "build_panel",
            symbols=SYMBOLS,
            calendar_mode="union",
            missing_policy="forward_fill",
            **DATA_ARGUMENTS,
        )
    ).panel
    assert panel.calendar.forward_filled["ETH/USDT"] > 0
    rejected = ErrorResponse.model_validate(
        dispatch(
            server,
            "train_model",
            dataset_id=panel.dataset_id,
            validation="purged_kfold",
            n_folds=3,
            **MODEL_ARGUMENTS,
        )
    )
    assert rejected.code == "invalid_training_request"
    assert "intersection" in rejected.error and "missing_policy=drop" in rejected.error
    assert not list(workspace.models.glob("*.bin"))


@pytest.mark.parametrize("field", [None, "forward_returns", "horizon", "signal"])
def test_readonly_evaluation_uses_saved_rows_and_horizon(
    server: MCPServer, workspace: WorkspacePaths, field: str | None
) -> None:
    dataset = BuildDatasetResponse.model_validate(
        dispatch(server, "build_dataset", symbol="BTC/USDT", **DATA_ARGUMENTS)
    ).dataset
    trained = TrainModelResponse.model_validate(
        dispatch(
            server, "train_model", dataset_id=dataset.dataset_id, **MODEL_ARGUMENTS
        )
    ).model
    options: dict[str, Any] = {
        "model_id": trained.model_id,
        "dataset_id": dataset.dataset_id,
        "record": False,
        "window": 12,
        "n_quantiles": 2,
    }
    if field is not None:
        options[field] = 1 if field == "horizon" else [0.1]
        with pytest.raises(ToolError, match="Unknown arguments"):
            dispatch(server, "evaluate_signal", **options)
    else:
        accepted = PredictionEvaluationResponse.model_validate(
            dispatch(server, "evaluate_signal", **options)
        )
        assert accepted.horizon == dataset.horizon == 3
        assert accepted.recorded is None
        assert accepted.n_observations == dataset.rows
    assert not (workspace.ml_leaderboard / "leaderboard.jsonl").exists()
