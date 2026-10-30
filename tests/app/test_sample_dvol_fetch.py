"""Missing sample features retain sparse exact matches with bounded providers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pandas as pd
from tests.data_cache import seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.data_service import DataService
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.data.crawlers.deribit_dvol import DeribitDVOLCrawler

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_missing_dvol_samples_all_sparse_market_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = WorkspacePaths(tmp_path / "source")
    service = DataService(workspace)
    timestamps = pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-20"], utc=True)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            **dict.fromkeys(("open", "high", "low", "close", "volume"), 100.0),
        }
    )
    cache = seed_market_cache(workspace.processed_data, frame)
    original = cache.read_bytes()
    source_files = set(workspace.processed_data.rglob("*"))
    raw_files = set(workspace.raw_data.rglob("*"))
    config = BacktestConfig(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-31",
        init_cash=10000,
    )
    requests: list[tuple[int, int]] = []
    origin = int(timestamps[0].timestamp() * 1000)
    hour = 3_600_000

    def respond(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["start_timestamp"])
        end = int(request.url.params["end_timestamp"])
        requests.append((start, end))
        rows = [
            [stamp, 1.0, 1.0, 1.0, 100.0 + (stamp - origin) / hour]
            for stamp in range(start, end + 1, hour)
        ]
        assert len(rows) <= 64
        return httpx.Response(
            200,
            content=json.dumps({"result": {"data": rows, "continuation": None}}),
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        crawler = DeribitDVOLCrawler()
        crawler._client.close()
        crawler._client = client
        monkeypatch.setattr(
            "tradingdev.app.data_service.DeribitDVOLCrawler", lambda: crawler
        )
        result = service.load_sample(
            {
                "data": {
                    "requirements": {
                        "market": {"symbol": "BTC/USDT", "timeframe": "1h"},
                        "features": [
                            {"type": "dvol", "source": "deribit", "column": "dvol"}
                        ],
                    }
                }
            },
            config,
            max_rows=64,
            output_dir=tmp_path / "sample",
        )
    assert result.frame["dvol"].tolist() == [100.0, 196.0, 556.0]
    assert len(requests) == 3
    assert all(end - start < 64 * hour for start, end in requests)
    saved = pd.read_parquet(tmp_path / "sample/feature-0.parquet")
    assert saved["timestamp"].tolist() == list(timestamps)
    assert cache.read_bytes() == original
    assert set(workspace.processed_data.rglob("*")) == source_files
    assert set(workspace.raw_data.rglob("*")) == raw_files
