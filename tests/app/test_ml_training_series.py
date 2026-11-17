"""Fold validation saves the actual fold predictions with real split ranges."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from tests.data_cache import seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.ml_service import MLService
from tradingdev.domain.ml.validation.temporal import temporal_folds

if TYPE_CHECKING:
    from pathlib import Path

_ROWS = 300
_FOLDS = 3


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


def test_fold_rows_use_real_split_coordinates(workspace: WorkspacePaths) -> None:
    service = MLService(workspace=workspace)
    built: dict[str, Any] = service.build_dataset(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-31",
        feature_set="returns",
        horizon=5,
    )["dataset"]
    dataset_id = str(built["dataset_id"])
    model: dict[str, Any] = service.train_model(
        dataset_id=dataset_id,
        model_kind="lightgbm_direction",
        parameters={"n_estimators": 8, "max_depth": 2, "n_jobs": 1},
        validation="purged_walk_forward",
        n_folds=_FOLDS,
    )["model"]
    validation: dict[str, Any] = model["validation"]
    artifact_id = str(validation["fold_series_artifact_id"])
    exported = service.evaluation_series_export(artifact_id)

    frame = pd.read_parquet(workspace.ml_datasets / dataset_id / "dataset.parquet")
    min_train = max(1, frame["timestamp"].nunique() // (2 * (_FOLDS + 1)))
    splits = temporal_folds(
        frame,
        scheme="purged_walk_forward",
        horizon=5,
        n_folds=_FOLDS,
        embargo=0,
        min_train_dates=min_train,
    )
    expected_rows = [int(row) for split in splits for row in split.test_rows]

    rows = exported["rows"]
    manifest = exported["manifest"]
    assert manifest["kind"] == "fold_validation"
    assert manifest["training_status"] == "fitted"
    assert [row["source_row"] for row in rows] == expected_rows
    assert [row["fold"] for row in rows] == [
        fold for fold, split in enumerate(splits) for _ in split.test_rows
    ]
    targets = frame["target"].to_numpy()
    for row in rows:
        assert row["target"] == pytest.approx(float(targets[row["source_row"]]))
        assert row["prediction"] in (-1.0, 0.0, 1.0)
    for fold, split in enumerate(splits):
        fold_rows = [row for row in rows if row["fold"] == fold]
        assert fold_rows[0]["train_start_row"] == int(split.train_rows[0])
        assert fold_rows[0]["train_end_row"] == int(split.train_rows[-1])
        assert fold_rows[0]["eval_start_row"] == int(split.test_rows[0])
        assert fold_rows[0]["eval_end_row"] == int(split.test_rows[-1])
