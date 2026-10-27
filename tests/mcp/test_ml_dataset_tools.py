"""build_dataset / train_model contracts through in-process MCP dispatch."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from mcp.server import MCPServer
from mcp_types import CallToolResult

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.contracts.ml import (
    BuildDatasetResponse,
    MlContractResponse,
    PredictionResponse,
    TrainModelResponse,
)
from tradingdev.app.ml_service import MLService
from tradingdev.mcp.tools import ml

if TYPE_CHECKING:
    from pathlib import Path


def _write_cached_bars(workspace: WorkspacePaths, rows: int = 300) -> None:
    index = np.arange(rows, dtype=float)
    close = 100.0 + 10.0 * np.sin(index / 5.0) + index * 0.01
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 10.0),
        }
    )
    path = workspace.processed_data / "btcusdt_1h_2024.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _dispatch(
    mcp: MCPServer, name: str, arguments: dict[str, object]
) -> dict[str, object]:
    async def call() -> dict[str, object]:
        result = await mcp.call_tool(name, arguments)
        assert isinstance(result, CallToolResult)
        assert result.structured_content is not None
        payload = result.structured_content
        assert isinstance(payload, dict)
        response = payload.get("result", payload)
        assert isinstance(response, dict)
        return response

    return asyncio.run(call())


@pytest.fixture
def service(tmp_path: Path) -> MLService:
    paths = WorkspacePaths(tmp_path / "workspace")
    paths.ensure()
    _write_cached_bars(paths)
    return MLService(workspace=paths)


def test_build_dataset_then_train_model_through_mcp(service: MLService) -> None:
    mcp = MCPServer("ml-dataset-test")
    ml.register(mcp, service)

    built = BuildDatasetResponse.model_validate(
        _dispatch(
            mcp,
            "build_dataset",
            {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "feature_set": "returns",
                "horizon": 1,
            },
        )
    )
    assert built.dataset.dataset_id.startswith("ds_")
    assert built.dataset.rows > 0
    assert built.dataset.n_features == len(built.dataset.feature_columns)

    trained = TrainModelResponse.model_validate(
        _dispatch(
            mcp,
            "train_model",
            {
                "dataset_id": built.dataset.dataset_id,
                "model_kind": "xgboost_direction",
                "parameters": {"n_estimators": 5, "max_depth": 2},
            },
        )
    )
    assert trained.model.model_id
    assert trained.model.dataset_id == built.dataset.dataset_id
    assert trained.model.n_train_samples == built.dataset.rows
    assert trained.model.feature_importance


def test_build_train_predict_through_mcp(service: MLService) -> None:
    mcp = MCPServer("ml-loop-test")
    ml.register(mcp, service)

    built = BuildDatasetResponse.model_validate(
        _dispatch(
            mcp,
            "build_dataset",
            {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "feature_set": "returns",
                "horizon": 1,
            },
        )
    )
    trained = TrainModelResponse.model_validate(
        _dispatch(
            mcp,
            "train_model",
            {
                "dataset_id": built.dataset.dataset_id,
                "model_kind": "xgboost_direction",
                "parameters": {"n_estimators": 60, "max_depth": 3},
            },
        )
    )
    predicted = PredictionResponse.model_validate(
        _dispatch(
            mcp,
            "predict",
            {
                "model_id": trained.model.model_id,
                "dataset_id": built.dataset.dataset_id,
            },
        )
    )
    assert predicted.prediction.model_id == trained.model.model_id
    assert predicted.prediction.n_rows == built.dataset.rows
    assert (
        len(predicted.prediction.signal)
        == len(predicted.prediction.forward_returns)
        == 100
    )
    assert predicted.prediction.next_offset == 100

    verdict = _dispatch(
        mcp,
        "evaluate_signal",
        {
            "model_id": trained.model.model_id,
            "dataset_id": built.dataset.dataset_id,
        },
    )
    assert verdict["success"] is True
    assert verdict["n_observations"] == built.dataset.rows


def test_predict_rejects_a_mismatched_dataset(service: MLService) -> None:
    mcp = MCPServer("ml-mismatch-test")
    ml.register(mcp, service)

    returns_dataset = BuildDatasetResponse.model_validate(
        _dispatch(
            mcp,
            "build_dataset",
            {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "feature_set": "returns",
                "horizon": 1,
            },
        )
    ).dataset
    technical_dataset = BuildDatasetResponse.model_validate(
        _dispatch(
            mcp,
            "build_dataset",
            {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "feature_set": "technical",
                "horizon": 1,
            },
        )
    ).dataset
    model_id = TrainModelResponse.model_validate(
        _dispatch(
            mcp,
            "train_model",
            {
                "dataset_id": returns_dataset.dataset_id,
                "model_kind": "xgboost_direction",
                "parameters": {"n_estimators": 60, "max_depth": 3},
            },
        )
    ).model.model_id

    payload = _dispatch(
        mcp,
        "predict",
        {"model_id": model_id, "dataset_id": technical_dataset.dataset_id},
    )
    assert payload["success"] is False
    assert payload["code"] == "invalid_prediction_request"
    assert isinstance(payload["error"], str)
    assert "feature contract mismatch" in payload["error"]


def test_predict_rejects_a_malformed_model_id(service: MLService) -> None:
    mcp = MCPServer("ml-bad-model-id")
    ml.register(mcp, service)
    dataset_id = BuildDatasetResponse.model_validate(
        _dispatch(
            mcp,
            "build_dataset",
            {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "feature_set": "returns",
                "horizon": 1,
            },
        )
    ).dataset.dataset_id
    payload = _dispatch(
        mcp, "predict", {"model_id": "../../etc/passwd", "dataset_id": dataset_id}
    )
    assert payload["success"] is False
    assert payload["code"] == "invalid_prediction_request"


def test_build_dataset_reports_missing_cache(tmp_path: Path) -> None:
    paths = WorkspacePaths(tmp_path / "empty")
    paths.ensure()
    mcp = MCPServer("ml-missing-cache")
    ml.register(mcp, MLService(workspace=paths))
    payload = _dispatch(
        mcp,
        "build_dataset",
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "feature_set": "returns",
        },
    )
    assert payload["success"] is False
    assert payload["code"] == "invalid_dataset_request"


def test_train_model_rejects_an_unknown_kind(service: MLService) -> None:
    mcp = MCPServer("ml-bad-kind")
    ml.register(mcp, service)
    payload = _dispatch(
        mcp,
        "train_model",
        {"dataset_id": "ds_" + "0" * 16, "model_kind": "magic_forest"},
    )
    assert payload["success"] is False
    assert payload["code"] == "invalid_training_request"


def _train(mcp: MCPServer, dataset_id: str, **overrides: object) -> TrainModelResponse:
    arguments: dict[str, object] = {
        "dataset_id": dataset_id,
        "model_kind": "xgboost_direction",
        "parameters": {"n_estimators": 5, "max_depth": 2},
    }
    arguments.update(overrides)
    return TrainModelResponse.model_validate(_dispatch(mcp, "train_model", arguments))


def _build(
    mcp: MCPServer, feature_set: str = "returns", horizon: int = 5
) -> BuildDatasetResponse:
    return BuildDatasetResponse.model_validate(
        _dispatch(
            mcp,
            "build_dataset",
            {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "feature_set": feature_set,
                "horizon": horizon,
            },
        )
    )


def test_train_model_returns_a_purged_validation_block(service: MLService) -> None:
    mcp = MCPServer("ml-validation-test")
    ml.register(mcp, service)
    dataset_id = _build(mcp).dataset.dataset_id

    trained = _train(
        mcp,
        dataset_id,
        validation="purged_walk_forward",
        n_folds=3,
        embargo=2,
    )
    validation = trained.model.validation
    assert validation is not None
    assert validation.scheme == "purged_walk_forward"
    assert validation.n_folds == 3
    assert len(validation.folds) == 3
    assert validation.leakage_audit_clean is True
    assert validation.artifact_trained_on == "all_bars"
    # The purge is the dataset's horizon (5) on every walk-forward fold.
    assert validation.purged_bars == 15
    assert [fold.purged_bars for fold in validation.folds] == [5, 5, 5]
    assert validation.embargoed_bars > 0
    assert all(fold.test_bars > 0 for fold in validation.folds)
    assert validation.aggregate.accuracy is not None


def test_train_model_without_a_scheme_reports_no_validation(
    service: MLService,
) -> None:
    mcp = MCPServer("ml-no-validation-test")
    ml.register(mcp, service)
    trained = _train(mcp, _build(mcp).dataset.dataset_id)
    assert trained.model.validation is None


def test_every_scheme_the_contract_offers_is_accepted(service: MLService) -> None:
    mcp = MCPServer("ml-scheme-menu-test")
    ml.register(mcp, service)
    contract = MlContractResponse.model_validate(_dispatch(mcp, "get_ml_contract", {}))
    advertised = [item.name for item in contract.validation_schemes]
    assert advertised
    dataset_id = _build(mcp).dataset.dataset_id
    for name in advertised:
        trained = _train(mcp, dataset_id, validation=name, n_folds=3)
        assert trained.model.validation is not None
        assert trained.model.validation.scheme == name


def test_train_model_rejects_an_unknown_scheme(service: MLService) -> None:
    mcp = MCPServer("ml-bad-scheme")
    ml.register(mcp, service)
    payload = _dispatch(
        mcp,
        "train_model",
        {
            "dataset_id": _build(mcp).dataset.dataset_id,
            "model_kind": "xgboost_direction",
            "validation": "rolling",
        },
    )
    assert payload["success"] is False
    assert payload["code"] == "invalid_training_request"
    assert isinstance(payload["error"], str)
    assert "unknown validation scheme" in payload["error"]


def test_train_model_rejects_a_zero_fold_count(service: MLService) -> None:
    mcp = MCPServer("ml-zero-folds")
    ml.register(mcp, service)
    payload = _dispatch(
        mcp,
        "train_model",
        {
            "dataset_id": _build(mcp).dataset.dataset_id,
            "model_kind": "xgboost_direction",
            "validation": "purged_kfold",
            "n_folds": 0,
        },
    )
    assert payload["success"] is False
    assert payload["code"] == "invalid_training_request"
    assert isinstance(payload["error"], str)
    assert "n_folds" in payload["error"]


def test_build_dataset_reports_feature_reuse(service: MLService) -> None:
    mcp = MCPServer("ml-cache-test")
    ml.register(mcp, service)
    arguments = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "feature_set": "returns",
        "horizon": 1,
    }
    computed = BuildDatasetResponse.model_validate(
        _dispatch(mcp, "build_dataset", arguments)
    ).dataset
    reused = BuildDatasetResponse.model_validate(
        _dispatch(mcp, "build_dataset", arguments)
    ).dataset
    assert computed.features_from_cache is False
    assert reused.features_from_cache is True
    assert reused.dataset_id == computed.dataset_id
    assert reused.content_hash == computed.content_hash
