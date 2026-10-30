"""Preflight samples never populate, remove, or replace ordinary data caches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
from tests.data_cache import seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.data_service import DataService
from tradingdev.domain.backtest.schemas import BacktestConfig

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path


def _config() -> BacktestConfig:
    return BacktestConfig.model_validate(
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2025-12-31",
            "init_cash": 10000.0,
        }
    )


def test_sample_reads_year_cache_without_fetching_or_modifying_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_ohlcv_df: pd.DataFrame,
) -> None:
    workspace = WorkspacePaths(tmp_path / "source")
    data = DataService(workspace)
    cached = seed_market_cache(workspace.processed_data, sample_ohlcv_df)
    original = cached.read_bytes()
    cache_files = set(workspace.processed_data.rglob("*"))

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("A sample must not use the yearly downloader or fetch cached data")

    class Crawler:
        fetch_sample = staticmethod(unexpected)

    monkeypatch.setattr(
        "tradingdev.app.data_service.create_crawler", lambda *_args: Crawler()
    )
    monkeypatch.setattr("tradingdev.app.data_service.DataManager.load", unexpected)
    result = data.load_sample(
        {}, _config(), max_rows=64, output_dir=tmp_path / "sample"
    )
    assert len(result.frame) == 64
    assert result.processed_path == tmp_path / "sample" / "sample.parquet"
    assert cached.read_bytes() == original
    assert set(workspace.processed_data.rglob("*")) == cache_files


def test_sample_cache_miss_fetches_short_interval_without_creating_year_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_ohlcv_df: pd.DataFrame,
) -> None:
    workspace = WorkspacePaths(tmp_path / "source")
    data = DataService(workspace)
    requested: list[tuple[datetime, datetime]] = []

    class Crawler:
        def fetch_sample(
            self,
            symbol: str,
            timeframe: str,
            start: datetime,
            end: datetime,
            *,
            max_rows: int,
        ) -> pd.DataFrame:
            requested.append((start, end))
            return sample_ohlcv_df.head(max_rows).copy()

    monkeypatch.setattr(
        "tradingdev.app.data_service.create_crawler", lambda *_args: Crawler()
    )
    result = data.load_sample(
        {}, _config(), max_rows=64, output_dir=tmp_path / "sample"
    )
    assert len(result.frame) == 64
    assert len(requested) == 1
    assert (requested[0][1] - requested[0][0]).total_seconds() <= 64 * 3 * 3600
    assert list(workspace.processed_data.iterdir()) == []
    assert list(workspace.raw_data.iterdir()) == []


def test_sample_checks_provider_even_when_bars_are_cached(
    tmp_path: Path,
    sample_ohlcv_df: pd.DataFrame,
) -> None:
    workspace = WorkspacePaths(tmp_path / "source")
    data = DataService(workspace)
    seed_market_cache(workspace.processed_data, sample_ohlcv_df)
    with pytest.raises(ValueError, match="unsupported_provider"):
        data.load_sample(
            {"data": {"source": "unsupported_provider"}},
            _config(),
            max_rows=64,
            output_dir=tmp_path / "sample",
        )


def test_sample_rejects_provider_that_exceeds_requested_row_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_ohlcv_df: pd.DataFrame,
) -> None:
    workspace = WorkspacePaths(tmp_path / "source")
    data = DataService(workspace)

    class Crawler:
        def fetch_sample(self, *_args: Any, **_kwargs: Any) -> pd.DataFrame:
            return sample_ohlcv_df.head(65).copy()

    monkeypatch.setattr(
        "tradingdev.app.data_service.create_crawler", lambda *_args: Crawler()
    )
    output = tmp_path / "sample"
    with pytest.raises(ValueError, match="provider exceeded the sample row budget"):
        data.load_sample({}, _config(), max_rows=64, output_dir=output)
    assert not output.exists()
    assert not list(workspace.processed_data.iterdir())


def test_sample_missing_feature_is_fetched_only_into_sample_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_ohlcv_df: pd.DataFrame,
) -> None:
    workspace = WorkspacePaths(tmp_path / "source")
    data = DataService(workspace)
    seed_market_cache(workspace.processed_data, sample_ohlcv_df)
    missing = workspace.processed_data / "dvol.parquet"
    raw: dict[str, Any] = {
        "data": {
            "requirements": {
                "market": {"symbol": "BTC/USDT", "timeframe": "1h"},
                "features": [
                    {
                        "type": "dvol",
                        "source": "deribit",
                        "column": "dvol",
                        "path": str(missing),
                    }
                ],
            }
        }
    }

    class Crawler:
        def fetch(self, **_kwargs: Any) -> pd.DataFrame:
            return pd.DataFrame(
                {"timestamp": sample_ohlcv_df["timestamp"], "dvol_close": 50.0}
            )

        def fetch_sample(self, *, max_rows: int, **kwargs: Any) -> pd.DataFrame:
            return self.fetch(**kwargs).head(max_rows)

        def save_raw(self, frame: pd.DataFrame, path: Path) -> None:
            path.write_text(frame.to_csv(index=False))

    monkeypatch.setattr("tradingdev.app.data_service.DeribitDVOLCrawler", Crawler)
    output = tmp_path / "sample"
    result = data.load_sample(raw, _config(), max_rows=64, output_dir=output)
    assert len(result.frame) == 64
    assert result.frame["dvol"].eq(50.0).all()
    assert not missing.exists()
    assert (output / "feature-0.parquet").exists()
    assert (output / "feature-0.csv").exists()
