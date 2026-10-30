"""One cache identity across download, inspection, sampling and read-only ML input."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.data_service import DataService
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.data.crawlers.base import BaseCrawler

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.domain.data.schemas import DataConfig


def test_sources_markets_symbols_and_custom_roots_never_share_prices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TRADINGDEV_DATA_ROOT", raising=False)
    service = DataService(WorkspacePaths(tmp_path / "workspace"))
    fetches: list[tuple[str, str | None]] = []
    selected_price = [100.0]

    def provider(source: str, config: DataConfig) -> Mock:
        assert source == config.source
        crawler = Mock(spec=BaseCrawler)

        def fetch(**kwargs: Any) -> pd.DataFrame:
            fetches.append(
                (source, config.market_type if source == "binance_vision" else None)
            )
            return pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2024-01-01", periods=48, freq="h", tz="UTC"
                    ),
                    **dict.fromkeys(
                        ("open", "high", "low", "close"), selected_price[0]
                    ),
                    "volume": 1.0,
                }
            )

        crawler.fetch.side_effect = fetch
        crawler.fetch_sample.side_effect = AssertionError(
            "Covered sample fetched unexpectedly"
        )
        crawler.save_raw.side_effect = lambda frame, path: frame.to_csv(
            path, index=False
        )
        return crawler

    monkeypatch.setattr("tradingdev.app.data_service.create_crawler", provider)
    ids, paths = set(), set()
    cases = [
        ("binance_vision", "futures/um", "BTC/USDT"),
        ("binance_vision", "spot", "BTC/USDT"),
        ("binance_api", "spot", "BTC/USDT"),
        ("binance_api", "spot", "BTCUSDT"),
        ("binance_api", "spot", "btc/usdt"),
    ]
    for index, (source, market, symbol) in enumerate(cases, 1):
        selected_price[0] = index * 100.0
        config: dict[str, Any] = {
            "data": {
                "source": "unused-fallback",
                "market_type": market,
                "raw_dir": str(tmp_path / "custom-raw"),
                "processed_dir": str(tmp_path / "custom-processed"),
                "requirements": {
                    "market": {"source": source, "symbol": symbol, "timeframe": "1h"}
                },
            }
        }
        backtest = BacktestConfig(
            symbol=symbol,
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-02",
            init_cash=10000.0,
        )
        loaded = service.load(config, backtest)
        assert loaded.frame.close.eq(selected_price[0]).all()
        assert loaded.processed_path.is_relative_to(tmp_path / "custom-processed")
        ids.add(loaded.dataset_id)
        paths.add(loaded.processed_path)
        cached = service.load_cached(
            symbol=symbol,
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-02",
            raw_config=config,
        )
        pd.testing.assert_frame_equal(cached.frame, loaded.frame)
        assert cached.dataset_id == loaded.dataset_id
        assert service.data_available(
            symbol, "1h", "2024-01-01", "2024-01-02", raw_config=config
        )
        sample = service.load_sample(
            config, backtest, max_rows=4, output_dir=tmp_path / f"sample-{index}"
        )
        assert sample.frame.close.tolist() == [selected_price[0]] * 4
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            yaml.safe_dump({**config, "backtest": backtest.model_dump(mode="json")}),
            encoding="utf-8",
        )
        inspected = service.inspect_dataset(yaml_path)
        assert inspected["market_available"] is True
        assert inspected["processed_dir"] == str(tmp_path / "custom-processed")
        assert inspected["market"]["source"] == source
        assert inspected["market"]["market_type"] == (
            market if source == "binance_vision" else None
        )
        assert inspected["market"]["paths"] == [str(loaded.processed_path)]
        assert inspected["market"]["queries"]
    assert len(ids) == len(paths) == len(fetches) == len(cases)
    assert not list(service._workspace.processed_data.iterdir())
    assert len(service.list_available_data(config)) == len(cases)


def test_request_identity_preserves_utc_intraday_bounds_and_effective_market(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.data_cache import seed_market_cache

    workspace = WorkspacePaths(tmp_path)
    service = DataService(workspace)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"),
            **dict.fromkeys(("open", "high", "low", "close", "volume"), 1.0),
        }
    )
    seed_market_cache(workspace.processed_data, frame, source="binance_api")

    def read(start: str, end: str, market: str) -> str:
        return service.load_cached(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
            source="binance_api",
            market_type=market,
        ).dataset_id

    first = read("2024-01-01T00:00Z", "2024-01-01T01:00Z", "spot")
    assert first == read("2024-01-01T08:00+08:00", "2024-01-01T09:00+08:00", "ignored")
    assert first != read("2024-01-01T02:00Z", "2024-01-01T03:00Z", "spot")


def test_legacy_flat_cache_is_preserved_but_never_advertised_as_identified(
    tmp_path: Path,
) -> None:
    service = DataService(WorkspacePaths(tmp_path))
    old = service._workspace.processed_data / "btcusdt_1h_2024.parquet"
    old.write_bytes(b"unknown source cache")
    assert service.list_available_data() == []
    assert not service.data_available("BTC/USDT", "1h", "2024-01-01", "2024-01-02")
    with pytest.raises(ValueError, match="ensure_data"):
        service.load_cached(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-02",
        )
    assert old.read_bytes() == b"unknown source cache"
