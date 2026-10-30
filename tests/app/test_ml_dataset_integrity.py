"""Dataset rebuilds and recorded evaluation must preserve their actual inputs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from tests.data_cache import seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.data_manager import DataManager
from tradingdev.app.ml_service import MLService

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


def _build(service: MLService) -> dict[str, Any]:
    result = service.build_dataset(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-10",
        feature_set="returns",
    )
    dataset: dict[str, Any] = result["dataset"]
    return dataset


def _seed(tmp_path: Path, frame: pd.DataFrame) -> tuple[WorkspacePaths, Path]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    workspace.ensure()
    path = seed_market_cache(workspace.processed_data, frame)
    return workspace, path


def test_rebuild_preserves_old_dataset_and_evaluation(
    tmp_path: Path,
    sample_ohlcv_df: pd.DataFrame,
) -> None:
    workspace, source = _seed(tmp_path, sample_ohlcv_df)
    service = MLService(workspace)
    first = _build(service)
    old_path = workspace.ml_datasets / first["dataset_id"] / "dataset.parquet"
    original = old_path.read_bytes()
    returns = pd.read_parquet(old_path)["forward_return"].tolist()
    saved = service.evaluate_signal(
        returns, returns, window=20, record=True, dataset_id=first["dataset_id"]
    )
    changed = sample_ohlcv_df.copy()
    changed["close"] *= np.linspace(1.0, 1.2, len(changed))
    changed.to_parquet(source, index=False)
    second = _build(service)
    assert second["dataset_id"] != first["dataset_id"]
    assert old_path.read_bytes() == original
    assert _build(service)["dataset_id"] == second["dataset_id"]
    assert (
        service.leaderboard("ic_mean")["ranked"][0]["entry_id"]
        == saved["recorded"]["entry_id"]
    )
    new_returns = pd.read_parquet(second["path"])["forward_return"].tolist()
    service.evaluate_signal(
        new_returns,
        new_returns,
        window=20,
        record=True,
        dataset_id=second["dataset_id"],
    )
    with pytest.raises(ValueError, match="compar|context"):
        service.leaderboard("ic_mean")


def test_rejects_changed_returns_and_modified_dataset(
    tmp_path: Path,
    sample_ohlcv_df: pd.DataFrame,
) -> None:
    workspace, _ = _seed(tmp_path, sample_ohlcv_df)
    service = MLService(workspace)
    dataset = _build(service)
    path = workspace.ml_datasets / dataset["dataset_id"] / "dataset.parquet"
    frame = pd.read_parquet(path)
    returns = frame["forward_return"].tolist()
    with pytest.raises(ValueError, match="do not match"):
        service.evaluate_signal(
            returns,
            [-value for value in returns],
            window=20,
            record=True,
            dataset_id=dataset["dataset_id"],
        )
    assert not (workspace.ml_leaderboard / "leaderboard.jsonl").exists()
    frame.loc[0, "forward_return"] = 1234.0
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="content hash mismatch"):
        service.train_model(dataset_id=dataset["dataset_id"], model_kind="xgboost")


def test_metadata_tampering_is_rejected(
    tmp_path: Path, sample_ohlcv_df: pd.DataFrame
) -> None:
    workspace, _ = _seed(tmp_path, sample_ohlcv_df)
    service = MLService(workspace)
    dataset = _build(service)
    path = workspace.ml_datasets / dataset["dataset_id"] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["horizon"] = 12
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="metadata identity"):
        service.train_model(dataset_id=dataset["dataset_id"], model_kind="xgboost")


def test_cached_only_build_preserves_generation_and_never_downloads(
    tmp_path: Path,
    sample_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, complete = _seed(tmp_path, sample_ohlcv_df)
    original = complete.read_bytes()

    def refuse_download(*args: object, **kwargs: object) -> pd.DataFrame:
        pytest.fail("cached-only build attempted a download")

    monkeypatch.setattr(DataManager, "load", refuse_download)
    assert _build(MLService(workspace))["rows"] > 0
    assert complete.read_bytes() == original


def test_dataset_identity_includes_feature_implementation(
    tmp_path: Path,
    sample_ohlcv_df: pd.DataFrame,
) -> None:
    workspace, _ = _seed(tmp_path, sample_ohlcv_df)
    before = _build(MLService(workspace, feature_code_fingerprint="v1"))
    after = _build(MLService(workspace, feature_code_fingerprint="v2"))
    assert before["dataset_id"] != after["dataset_id"]
