"""Concurrent real MCP dispatch cannot expose a partial model publication."""

from __future__ import annotations

import asyncio
import os
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import pandas as pd
import pytest
from mcp_types import CallToolResult

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.contracts.ml import (
    BuildDatasetResponse,
    PredictionResponse,
    TrainModelResponse,
)
from tradingdev.app.ml_service import MLService
from tradingdev.mcp.strict_server import StrictMCPServer
from tradingdev.mcp.tools import ml

_WAIT_SECONDS = 15


async def _dispatch(
    server: StrictMCPServer, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    result = await server.call_tool(name, arguments)
    assert isinstance(result, CallToolResult)
    assert result.structured_content is not None
    payload = result.structured_content
    assert isinstance(payload, dict)
    response = payload.get("result", payload)
    assert isinstance(response, dict)
    return response


def _seed(workspace: WorkspacePaths) -> None:
    workspace.ensure()
    index = np.arange(300, dtype=float)
    close = 100.0 + 10.0 * np.sin(index / 5.0) + index * 0.01
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(300, 10.0),
        }
    ).to_parquet(workspace.processed_data / "btcusdt_1h_2024.parquet", index=False)


@pytest.mark.parametrize("outcome", ["normal", "failure", "timeout"])
def test_predict_during_retraining_keeps_verified_model_and_reaps_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    _seed(workspace)
    server = StrictMCPServer("concurrent-model-publication")
    ml.register(server, MLService(workspace))
    paused, release, finished = Event(), Event(), Event()

    async def scenario() -> None:
        built = BuildDatasetResponse.model_validate(
            await _dispatch(
                server,
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
        training_args = {
            "dataset_id": built.dataset.dataset_id,
            "model_kind": "xgboost_direction",
            "parameters": {"n_estimators": 5, "max_depth": 2},
        }
        trained = TrainModelResponse.model_validate(
            await _dispatch(server, "train_model", training_args)
        )
        model_id = trained.model.model_id
        prediction_args = {
            "model_id": model_id,
            "dataset_id": built.dataset.dataset_id,
        }
        expected = PredictionResponse.model_validate(
            await _dispatch(server, "predict", prediction_args)
        )
        blob_path = workspace.models / f"{model_id}.bin"
        meta_path = workspace.models / f"{model_id}.json"
        original_write, original_replace = Path.write_bytes, os.replace

        def pause() -> None:
            paused.set()
            if not release.wait(_WAIT_SECONDS):
                raise TimeoutError("test did not release concurrent training")

        def write(path: Path, data: bytes) -> int:
            if path == blob_path:
                # Preserve write_bytes semantics while pausing after truncation.
                # This also reproduces the unsafe old store, rather than merely
                # requiring the new store's private implementation to exist.
                with path.open("wb") as stream:
                    pause()
                    return stream.write(data)
            return original_write(path, data)

        def publish(source: Path, target: Path) -> None:
            if target == meta_path:
                pause()
            original_replace(source, target)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "write_bytes", write)
            patch.setattr(os, "replace", publish)
            retraining = asyncio.create_task(
                _dispatch(server, "train_model", training_args)
            )
            try:
                assert await asyncio.to_thread(paused.wait, _WAIT_SECONDS)
                if outcome == "failure":
                    raise RuntimeError("injected assertion failure")
                if outcome == "timeout":
                    # Keep the real request alive so cleanup must release and
                    # await it; a timeout must not leave its worker writing.
                    await asyncio.wait_for(asyncio.shield(retraining), timeout=0)
                response = await asyncio.wait_for(
                    _dispatch(server, "predict", prediction_args),
                    timeout=_WAIT_SECONDS,
                )
                assert response["success"] is True, response
                assert PredictionResponse.model_validate(response) == expected
            finally:
                release.set()
                retrained = TrainModelResponse.model_validate(
                    await asyncio.wait_for(retraining, timeout=_WAIT_SECONDS)
                )
                assert retrained.model.model_id == model_id
                finished.set()
        assert (
            PredictionResponse.model_validate(
                await _dispatch(server, "predict", prediction_args)
            )
            == expected
        )

    expected_exit = (
        nullcontext()
        if outcome == "normal"
        else pytest.raises(RuntimeError if outcome == "failure" else TimeoutError)
    )
    with expected_exit:
        asyncio.run(scenario())
    assert paused.is_set() and finished.is_set()
    assert list(workspace.models.glob("*.tmp")) == []
    assert len(list(workspace.models.glob("*.bin"))) == 1
    assert len(list(workspace.models.glob("*.json"))) == 1
