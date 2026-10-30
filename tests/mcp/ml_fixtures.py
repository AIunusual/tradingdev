"""Small saved model/dataset pairs for MCP identity and paging tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tests.data_cache import seed_market_cache

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.app.ml_service import MLService


def seed_bars(workspace: WorkspacePaths, rows: int = 300) -> None:
    workspace.ensure()
    index = np.arange(rows, dtype=float)
    close = 100.0 + 10.0 * np.sin(index / 5.0) + index * 0.01
    seed_market_cache(
        workspace.processed_data,
        pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=rows, freq="h", tz="UTC"
                ),
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.full(rows, 10.0),
            }
        ),
    )


def model_pair(service: MLService, *, horizon: int = 1) -> dict[str, str]:
    dataset = service.build_dataset(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-31",
        feature_set="returns",
        horizon=horizon,
    )["dataset"]
    model = service.train_model(
        dataset_id=dataset["dataset_id"],
        model_kind="lightgbm_direction",
        parameters={"n_estimators": 12, "max_depth": 3, "n_jobs": 1},
    )["model"]
    return {
        "model_id": str(model["model_id"]),
        "dataset_id": str(dataset["dataset_id"]),
    }
