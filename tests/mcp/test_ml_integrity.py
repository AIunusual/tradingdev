"""Recorded ML evaluations bind to the stored returns through real MCP dispatch."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from mcp.server import MCPServer
from mcp_types import CallToolResult

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.contracts.common import ErrorResponse
from tradingdev.app.contracts.ml import (
    BuildDatasetResponse,
    LeaderboardPageResponse,
    PredictionEvaluationResponse,
)
from tradingdev.app.ml_service import MLService
from tradingdev.mcp.tools import ml

if TYPE_CHECKING:
    from pathlib import Path


def _dispatch(
    server: MCPServer, name: str, arguments: dict[str, object]
) -> dict[str, object]:
    async def call() -> dict[str, object]:
        result = await server.call_tool(name, arguments)
        assert isinstance(result, CallToolResult)
        assert result.structured_content is not None
        payload = result.structured_content
        assert isinstance(payload, dict)
        response = payload.get("result", payload)
        assert isinstance(response, dict)
        return response

    return asyncio.run(call())


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    paths = WorkspacePaths(tmp_path / "workspace")
    paths.ensure()
    positions = np.arange(300, dtype=float)
    close = 100.0 + 10.0 * np.sin(positions / 5.0) + positions * 0.01
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(300, 10.0),
        }
    ).to_parquet(paths.processed_data / "btcusdt_1h_2024.parquet", index=False)
    return paths


@pytest.mark.parametrize("tamper", ["model", "dataset"])
def test_record_rejects_modified_artifacts_without_overwriting_the_leaderboard(
    workspace: WorkspacePaths, tamper: str
) -> None:
    server = MCPServer("ml-return-integrity-test")
    ml.register(server, MLService(workspace=workspace))
    built = BuildDatasetResponse.model_validate(
        _dispatch(
            server,
            "build_dataset",
            {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "feature_set": "returns",
            },
        )
    )
    dataset_id = built.dataset.dataset_id
    trained = _dispatch(
        server,
        "train_model",
        {
            "dataset_id": dataset_id,
            "model_kind": "lightgbm_direction",
            "parameters": {"n_estimators": 12, "max_depth": 3, "n_jobs": 1},
        },
    )
    model = trained["model"]
    assert isinstance(model, dict)
    model_id = str(model["model_id"])
    arguments: dict[str, object] = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "window": 20,
        "record": True,
    }
    recorded = PredictionEvaluationResponse.model_validate(
        _dispatch(server, "evaluate_signal", arguments)
    )
    assert recorded.recorded is not None
    assert recorded.recorded.n_candidates == 1
    leaderboard_path = workspace.ml_leaderboard / "leaderboard.jsonl"
    before = leaderboard_path.read_bytes()
    if tamper == "model":
        blob = workspace.models / f"{model_id}.bin"
        blob.write_bytes(blob.read_bytes() + b"modified")
    else:
        dataset = workspace.ml_datasets / dataset_id / "dataset.parquet"
        frame = pd.read_parquet(dataset)
        frame["forward_return"] *= -1
        frame.to_parquet(dataset, index=False)

    rejected = ErrorResponse.model_validate(
        _dispatch(
            server,
            "evaluate_signal",
            arguments,
        )
    )

    assert rejected.success is False
    assert rejected.code == "invalid_signal"
    assert "tampered" in rejected.error or "hash mismatch" in rejected.error
    assert leaderboard_path.read_bytes() == before
    leaderboard = LeaderboardPageResponse.model_validate(
        _dispatch(server, "leaderboard", {"metric": "rank_ic_mean"})
    )
    assert leaderboard.n_candidates == leaderboard.n_ranked == 1
    assert leaderboard.ranked[0].entry_id == recorded.recorded.entry_id
    assert leaderboard.ranked[0].value == recorded.metrics["rank_ic_mean"]
