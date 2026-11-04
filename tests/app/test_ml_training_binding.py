"""A recorded model names the dataset it trained on, not the one it was scored on."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from tests.data_cache import seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.ml_service import MLService
from tradingdev.domain.ml.registry import LeaderboardStore

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.domain.ml.leaderboard import LeaderboardEntry

_ROWS = 600
_TRAIN = ("2024-01-01", "2024-01-12")
_LATER = ("2024-01-13", "2024-01-25")


def _write_bars(workspace: WorkspacePaths) -> None:
    index = np.arange(_ROWS, dtype=float)
    close = 100.0 + 10.0 * np.sin(index / 5.0) + index * 0.01
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=_ROWS, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(_ROWS, 10.0),
        }
    )
    seed_market_cache(workspace.processed_data, frame, symbol="BTC/USDT")


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    paths = WorkspacePaths(tmp_path / "workspace")
    paths.ensure()
    _write_bars(paths)
    return paths


@pytest.fixture
def service(workspace: WorkspacePaths) -> MLService:
    return MLService(workspace=workspace)


def _build(service: MLService, period: tuple[str, str]) -> str:
    result = service.build_dataset(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date=period[0],
        end_date=period[1],
        feature_set="technical",
    )
    dataset_id: str = result["dataset"]["dataset_id"]
    return dataset_id


def _train(service: MLService, dataset_id: str) -> str:
    result = service.train_model(
        dataset_id=dataset_id,
        model_kind="xgboost_direction",
        parameters={"n_estimators": 5, "max_depth": 2},
    )
    model_id: str = result["model"]["model_id"]
    return model_id


def _recorded_entry(workspace: WorkspacePaths, entry_id: str) -> LeaderboardEntry:
    return next(
        entry
        for entry in LeaderboardStore(workspace.ml_leaderboard).entries()
        if entry.entry_id == entry_id
    )


class TestTrainingDatasetBinding:
    def test_an_in_sample_evaluation_says_so_by_naming_one_dataset(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service, _TRAIN)
        model_id = _train(service, dataset_id)

        result = service.evaluate_prediction(
            model_id=model_id, dataset_id=dataset_id, window=20, record=True
        )

        recorded = result["recorded"]
        assert recorded["training_dataset_id"] == dataset_id
        entry = _recorded_entry(workspace, recorded["entry_id"])
        # This is the verified path: the backend generated the signal from this
        # artifact, so equal ids mean the score is in-sample. The record names the
        # dataset rather than implying it, so a reader sees it without the artifact.
        assert entry.dataset_id == entry.training_dataset_id == dataset_id

    def test_scoring_on_another_dataset_keeps_both_identities(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        trained_on = _build(service, _TRAIN)
        scored_on = _build(service, _LATER)
        model_id = _train(service, trained_on)

        result = service.evaluate_prediction(
            model_id=model_id, dataset_id=scored_on, window=20, record=True
        )

        recorded = result["recorded"]
        assert recorded["training_dataset_id"] == trained_on
        entry = _recorded_entry(workspace, recorded["entry_id"])
        assert entry.dataset_id == scored_on
        assert entry.training_dataset_id == trained_on
        assert entry.dataset_id != entry.training_dataset_id

    def test_a_rule_records_no_training_dataset(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service, _TRAIN)
        frame = pd.read_parquet(workspace.ml_datasets / dataset_id / "dataset.parquet")
        level = float(frame["rsi_14"].median())

        result = service.evaluate_rule(
            dataset_id=dataset_id,
            column="rsi_14",
            threshold=level,
            direction="below",
            window=20,
            record=True,
        )

        assert result["recorded"]["training_dataset_id"] == ""
        entry = _recorded_entry(workspace, result["recorded"]["entry_id"])
        assert entry.training_dataset_id == ""
        assert entry.dataset_id == dataset_id

    @pytest.mark.parametrize("artifact_state", ["missing", "invalid_schema"])
    def test_an_unloadable_artifact_records_no_training_dataset(
        self, service: MLService, workspace: WorkspacePaths, artifact_state: str
    ) -> None:
        """An absent or invalid artifact leaves training data unknown."""
        dataset_id = _build(service, _TRAIN)
        frame = pd.read_parquet(workspace.ml_datasets / dataset_id / "dataset.parquet")
        model_id = "c" * 16
        if artifact_state == "invalid_schema":
            model_id = _train(service, dataset_id)
            path = workspace.models / f"{model_id}.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["schema_version"] = []
            path.write_text(json.dumps(metadata), encoding="utf-8")

        result = service.evaluate_signal(
            frame["forward_return"].astype(float).tolist(),
            frame["forward_return"].astype(float).tolist(),
            window=20,
            record=True,
            model_id=model_id,
            dataset_id=dataset_id,
        )

        assert result["recorded"]["training_dataset_id"] == ""
        assert result["recorded"]["method_family"] == "ml_model"
        entry = _recorded_entry(workspace, result["recorded"]["entry_id"])
        assert entry.model_id == model_id and entry.training_dataset_id == ""
