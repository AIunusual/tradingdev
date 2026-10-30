"""Dataset building and training against isolated cached market fixtures."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from tests.data_cache import seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.ml_service import MLService

if TYPE_CHECKING:
    from pathlib import Path

_DATASET_ID = re.compile(r"^ds_[0-9a-f]{16}$")


def _write_cached_bars(workspace: WorkspacePaths, rows: int = 300) -> None:
    """Write one year of oscillating OHLCV bars under the canonical name."""
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
    seed_market_cache(workspace.processed_data, frame)


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    paths = WorkspacePaths(tmp_path / "workspace")
    paths.ensure()
    _write_cached_bars(paths)
    return paths


@pytest.fixture
def service(workspace: WorkspacePaths) -> MLService:
    return MLService(workspace=workspace)


def _build(service: MLService, **overrides: object) -> dict[str, Any]:
    kwargs: dict[str, object] = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "feature_set": "returns",
        "horizon": 1,
    }
    kwargs.update(overrides)
    return service.build_dataset(**kwargs)  # type: ignore[arg-type]


class TestBuildDataset:
    def test_emits_the_declared_feature_contract(self, service: MLService) -> None:
        result = _build(service)
        dataset = result["dataset"]
        assert dataset["feature_set"] == "returns"
        assert dataset["n_features"] == len(dataset["feature_columns"])
        assert dataset["label_column"] == "target"
        assert _DATASET_ID.match(str(dataset["dataset_id"]))
        assert dataset["rows"] > 0

    def test_rows_are_dropped_for_warmup_and_label_tail(
        self, service: MLService
    ) -> None:
        dataset = _build(service)["dataset"]
        assert dataset["dropped_rows"] > 0
        assert dataset["rows"] + dataset["dropped_rows"] == 300

    def test_is_deterministic(self, service: MLService) -> None:
        first = _build(service)["dataset"]["dataset_id"]
        second = _build(service)["dataset"]["dataset_id"]
        assert first == second

    def test_horizon_changes_the_identity(self, service: MLService) -> None:
        assert (
            _build(service)["dataset"]["dataset_id"]
            != _build(service, horizon=5)["dataset"]["dataset_id"]
        )

    def test_feature_set_changes_the_identity(self, service: MLService) -> None:
        assert (
            _build(service)["dataset"]["dataset_id"]
            != _build(service, feature_set="technical")["dataset"]["dataset_id"]
        )

    def test_requires_cached_market_data(self, tmp_path: Path) -> None:
        empty = WorkspacePaths(tmp_path / "empty")
        empty.ensure()
        with pytest.raises(ValueError, match="ensure_data"):
            MLService(workspace=empty).build_dataset(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date="2024-01-01",
                end_date="2024-01-31",
                feature_set="returns",
            )

    def test_unknown_feature_set_is_rejected(self, service: MLService) -> None:
        with pytest.raises(ValueError, match="unknown feature set"):
            _build(service, feature_set="not_a_set")

    def test_horizon_must_be_positive(self, service: MLService) -> None:
        with pytest.raises(ValueError, match="horizon must be at least 1"):
            _build(service, horizon=0)


class TestTrainModel:
    @staticmethod
    def _train(
        service: MLService, dataset_id: str, **overrides: object
    ) -> dict[str, Any]:
        kwargs: dict[str, object] = {
            "dataset_id": dataset_id,
            "model_kind": "xgboost_direction",
            "parameters": {"n_estimators": 5, "max_depth": 2},
        }
        kwargs.update(overrides)
        return service.train_model(**kwargs)  # type: ignore[arg-type]

    def test_persists_an_artifact(
        self, service: MLService, workspace: WorkspacePaths
    ) -> None:
        dataset_id = _build(service)["dataset"]["dataset_id"]
        model = self._train(service, dataset_id)["model"]
        assert model["model_id"]
        assert model["framework"] == "xgboost"
        assert model["n_train_samples"] > 0
        assert (workspace.models / f"{model['model_id']}.bin").exists()
        assert (workspace.models / f"{model['model_id']}.json").exists()

    def test_is_deterministic(self, service: MLService) -> None:
        dataset_id = _build(service)["dataset"]["dataset_id"]
        assert (
            self._train(service, dataset_id)["model"]["model_id"]
            == self._train(service, dataset_id)["model"]["model_id"]
        )

    def test_parameters_change_the_identity(self, service: MLService) -> None:
        dataset_id = _build(service)["dataset"]["dataset_id"]
        assert (
            self._train(service, dataset_id, parameters={"n_estimators": 5})["model"][
                "model_id"
            ]
            != self._train(service, dataset_id, parameters={"n_estimators": 9})[
                "model"
            ]["model_id"]
        )

    def test_feature_importance_covers_every_feature(self, service: MLService) -> None:
        dataset_id = _build(service)["dataset"]["dataset_id"]
        model = self._train(service, dataset_id)["model"]
        assert len(model["feature_importance"]) == model["n_features"]
        values = [item["importance"] for item in model["feature_importance"]]
        assert values == sorted(values, reverse=True)

    def test_unknown_parameter_is_rejected(self, service: MLService) -> None:
        dataset_id = _build(service)["dataset"]["dataset_id"]
        with pytest.raises(ValueError, match="unknown parameters"):
            self._train(service, dataset_id, parameters={"n_estimator": 5})

    def test_unknown_model_kind_is_rejected(self, service: MLService) -> None:
        dataset_id = _build(service)["dataset"]["dataset_id"]
        with pytest.raises(ValueError, match="unknown model kind"):
            self._train(service, dataset_id, model_kind="magic_forest")

    def test_unknown_dataset_id_is_rejected(self, service: MLService) -> None:
        with pytest.raises(ValueError, match="unknown dataset_id"):
            self._train(service, "ds_" + "0" * 16)

    def test_malformed_dataset_id_cannot_escape_the_store(
        self, service: MLService
    ) -> None:
        for bad in ("../../etc/passwd", "ds_xyz", "ds_" + "0" * 15, ""):
            with pytest.raises(ValueError, match="malformed dataset_id"):
                self._train(service, bad)

    def test_every_zoo_kind_trains(self, service: MLService) -> None:
        dataset_id = _build(service)["dataset"]["dataset_id"]
        for kind in ("xgboost_direction", "lightgbm_direction", "catboost_direction"):
            model = self._train(service, dataset_id, model_kind=kind)["model"]
            assert model["framework"] == kind.split("_")[0]
            assert model["model_id"]
