"""Temporal validation and single-asset evaluation through real MCP dispatch."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from mcp.server import MCPServer
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
    PredictionResponse,
    SignalEvaluationResponse,
    TrainModelResponse,
)
from tradingdev.app.ml_service import MLService
from tradingdev.mcp.tools import ml

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.adapters.storage.filesystem import WorkspacePaths


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    return seed_temporal_workspace(tmp_path)


@pytest.fixture
def server(workspace: WorkspacePaths) -> MCPServer:
    mcp = MCPServer("ml-temporal-integrity")
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
                signal=prediction.signal,
                forward_returns=prediction.forward_returns,
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
    accepted = SignalEvaluationResponse.model_validate(
        dispatch(
            server,
            "evaluate_signal",
            dataset_id=single.dataset_id,
            model_id=trained.model_id,
            signal=single_prediction.signal,
            forward_returns=single_prediction.forward_returns,
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


@pytest.mark.parametrize("tamper", [None, "returns", "horizon", "row_count"])
def test_readonly_dataset_evaluation_enforces_saved_returns_and_horizon(
    server: MCPServer, workspace: WorkspacePaths, tamper: str | None
) -> None:
    dataset = BuildDatasetResponse.model_validate(
        dispatch(server, "build_dataset", symbol="BTC/USDT", **DATA_ARGUMENTS)
    ).dataset
    realized = pd.read_parquet(dataset.path)["forward_return"].tolist()
    signal = np.arange(len(realized), dtype=float).tolist()
    options: dict[str, Any] = {}
    if tamper == "returns":
        realized = [-value for value in realized]
    elif tamper == "horizon":
        options["horizon"] = 1
    elif tamper == "row_count":
        signal, realized = signal[:-1], realized[:-1]
    response = dispatch(
        server,
        "evaluate_signal",
        dataset_id=dataset.dataset_id,
        signal=signal,
        forward_returns=realized,
        record=False,
        window=12,
        n_quantiles=2,
        **options,
    )
    if tamper is None:
        accepted = SignalEvaluationResponse.model_validate(response)
        assert accepted.horizon == dataset.horizon == 3
        assert accepted.recorded is None
        assert accepted.n_observations == dataset.rows
    else:
        rejected = ErrorResponse.model_validate(response)
        assert rejected.code == "invalid_signal"
        expected = {
            "returns": "forward_returns do not match",
            "horizon": "does not match the dataset's horizon",
            "row_count": "would not describe this dataset",
        }
        assert expected[tamper] in rejected.error
    assert not (workspace.ml_leaderboard / "leaderboard.jsonl").exists()


def test_array_evaluation_uses_nonoverlapping_horizon_compounding(
    server: MCPServer,
) -> None:
    response = SignalEvaluationResponse.model_validate(
        dispatch(
            server,
            "evaluate_signal",
            signal=np.arange(10, dtype=float).tolist(),
            forward_returns=[0.1] * 10,
            horizon=3,
            window=2,
            n_quantiles=2,
        )
    )
    assert response.horizon == 3
    assert response.simulated.long_only_cumulative == pytest.approx(1.1**3 - 1)
    assert response.simulated.long_short_cumulative == pytest.approx(1.1**3 - 1)
    assert response.simulated.long_only_mean == pytest.approx(0.03)
