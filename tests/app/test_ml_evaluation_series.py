"""Recorded evaluations remain readable after their producer disappears."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from tests.data_cache import seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.ml_service import MLService
from tradingdev.domain.ml.registry import LeaderboardStore

if TYPE_CHECKING:
    from pathlib import Path

_ROWS = 300


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    paths = WorkspacePaths(tmp_path / "workspace")
    paths.ensure()
    positions = np.arange(_ROWS, dtype=float)
    close = 100.0 + 10.0 * np.sin(positions / 5.0) + positions * 0.01
    seed_market_cache(
        paths.processed_data,
        pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=_ROWS, freq="h", tz="UTC"
                ),
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.full(_ROWS, 10.0),
            }
        ),
    )
    return paths


def _dataset(service: MLService) -> str:
    result = service.build_dataset(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-31",
        feature_set="returns",
        horizon=1,
    )
    return str(result["dataset"]["dataset_id"])


def test_publication_and_leaderboard_update_share_the_series_lock(
    workspace: WorkspacePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = MLService(workspace=workspace)
    dataset_id = _dataset(service)
    frame = pd.read_parquet(workspace.ml_datasets / dataset_id / "dataset.parquet")
    realized = frame["forward_return"].astype(float).tolist()
    original = LeaderboardStore.record

    def record(self: LeaderboardStore, entry: object) -> bool:
        assert service.series_store.lock.is_locked
        return original(self, entry)  # type: ignore[arg-type]

    monkeypatch.setattr(LeaderboardStore, "record", record)

    assert service.evaluate_signal(
        [value * 0.5 for value in realized],
        realized,
        dataset_id=dataset_id,
        window=20,
        record=True,
    )["recorded"]["series_artifact_id"]


def test_leaderboard_response_exposes_the_current_series_id(
    workspace: WorkspacePaths,
) -> None:
    service = MLService(workspace=workspace)
    dataset_id = _dataset(service)
    frame = pd.read_parquet(workspace.ml_datasets / dataset_id / "dataset.parquet")
    realized = frame["forward_return"].astype(float).tolist()
    recorded = service.evaluate_signal(
        [value * 0.5 for value in realized],
        realized,
        dataset_id=dataset_id,
        window=20,
        record=True,
    )
    artifact_id = str(recorded["series_artifact_id"])

    payload = service.leaderboard("ic_mean")
    rows = [*payload["ranked"], *payload["skipped"]]
    row = next(
        item for item in rows if item["entry_id"] == recorded["recorded"]["entry_id"]
    )

    assert row["series_artifact_id"] == artifact_id


def test_saved_series_matches_the_dataset_rows_independently(
    workspace: WorkspacePaths,
) -> None:
    service = MLService(workspace=workspace)
    dataset_id = _dataset(service)
    frame = pd.read_parquet(workspace.ml_datasets / dataset_id / "dataset.parquet")
    realized = frame["forward_return"].astype(float).tolist()
    signal = [value * 0.5 for value in realized]

    recorded = service.evaluate_signal(
        signal,
        realized,
        dataset_id=dataset_id,
        window=20,
        n_quantiles=5,
        record=True,
    )
    artifact_id = recorded["series_artifact_id"]
    assert recorded["producer_source"] == "caller_array"
    exported = service.evaluation_series_export(str(artifact_id))
    assert exported["manifest"]["producer_source"] == "caller_array"
    assert exported["manifest"]["entry_id"] == recorded["recorded"]["entry_id"]
    assert exported["manifest"]["training_status"] == "unknown"
    assert exported["manifest"]["holdout_relationship"] == "unknown"

    rows = exported["rows"]
    assert len(rows) == len(frame)
    assert [row["source_row"] for row in rows] == list(range(len(frame)))
    assert [row["timestamp"] for row in rows] == [
        timestamp.isoformat()
        for timestamp in pd.to_datetime(frame["timestamp"], utc=True)
    ]
    assert [row["signal"] for row in rows] == pytest.approx(signal)
    assert [row["realized_return"] for row in rows] == pytest.approx(realized)
    assert all(row["pearson_ic"] is None for row in rows[:19])
    assert rows[19]["pearson_ic"] is not None


def test_manifest_binds_candidate_identity_and_declared_settings(
    workspace: WorkspacePaths,
) -> None:
    service = MLService(workspace=workspace)
    dataset_id = _dataset(service)
    frame = pd.read_parquet(workspace.ml_datasets / dataset_id / "dataset.parquet")
    realized = frame["forward_return"].astype(float).tolist()
    recorded = service.evaluate_signal(
        [value * 0.5 for value in realized],
        realized,
        dataset_id=dataset_id,
        window=20,
        method_family="rule",
        method_name="rsi14",
        method_parameters={"threshold": 50.0},
        record=True,
    )
    manifest = service.evaluation_series_export(str(recorded["series_artifact_id"]))[
        "manifest"
    ]

    assert manifest["settings"]["candidate"] == {
        "model_id": "",
        "method_name": "rsi14",
        "parameters": {"threshold": 50.0},
    }


def test_caller_array_with_model_id_keeps_training_relationship_unknown(
    workspace: WorkspacePaths,
) -> None:
    service = MLService(workspace=workspace)
    dataset_id = _dataset(service)
    trained = service.train_model(
        dataset_id=dataset_id,
        model_kind="lightgbm_direction",
        parameters={"n_estimators": 12, "max_depth": 3, "n_jobs": 1},
    )
    frame = pd.read_parquet(workspace.ml_datasets / dataset_id / "dataset.parquet")
    realized = frame["forward_return"].astype(float).tolist()
    recorded = service.evaluate_signal(
        [value * 0.5 for value in realized],
        realized,
        model_id=str(trained["model"]["model_id"]),
        dataset_id=dataset_id,
        window=20,
        record=True,
    )
    manifest = service.evaluation_series_export(str(recorded["series_artifact_id"]))[
        "manifest"
    ]

    assert recorded["producer_source"] == "caller_array"
    assert manifest["training_status"] == "unknown"
    assert manifest["holdout_relationship"] == "unknown"


def test_series_survives_model_removal_and_reports_legacy_absence(
    workspace: WorkspacePaths,
) -> None:
    service = MLService(workspace=workspace)
    dataset_id = _dataset(service)
    trained = service.train_model(
        dataset_id=dataset_id,
        model_kind="lightgbm_direction",
        parameters={"n_estimators": 12, "max_depth": 3, "n_jobs": 1},
    )
    model_id = str(trained["model"]["model_id"])
    model_path = workspace.models / f"{model_id}.bin"
    recorded: dict[str, Any] = service.evaluate_prediction(
        model_id=model_id, dataset_id=dataset_id, window=20, record=True
    )
    artifact_id = str(recorded["series_artifact_id"])
    manifest = service.evaluation_series_export(artifact_id)["manifest"]

    assert manifest["training_dataset_id"] == dataset_id
    assert manifest["training_status"] == "fitted"
    assert manifest["holdout_relationship"] == "in_sample"

    model_path.unlink()
    loaded = service.evaluation_series_export(artifact_id)
    assert len(loaded["rows"]) == len(
        pd.read_parquet(workspace.ml_datasets / dataset_id / "dataset.parquet")
    )
    with pytest.raises(ValueError, match="unavailable"):
        service.evaluation_series_export("0" * 64)
