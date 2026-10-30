"""Data service tests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pandas as pd
import pytest
from tests.data_cache import seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.data_service import DataService
from tradingdev.domain.backtest.schemas import BacktestConfig


@pytest.mark.parametrize(
    ("source", "timeframe", "market_type", "error"),
    [
        ("misspelled", "1h", "futures/um", "Unknown data source"),
        ("yahoo_finance", "3h", "futures/um", "timeframe '3h'"),
        ("binance_vision", "1h", "unsupported", "market_type 'unsupported'"),
        ("yahoo_finance", "1h", "unused", None),
        ("binance_api", "3h", "unused", None),
        ("binance_vision", "3h", "spot", None),
    ],
)
def test_execution_checks_source_catalog_without_market_access(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    source: str,
    timeframe: str,
    market_type: str,
    error: str | None,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Preparing settings must not construct a crawler or load prices")

    monkeypatch.setattr("tradingdev.app.data_service.create_crawler", forbidden)
    monkeypatch.setattr(DataService, "load", forbidden)
    monkeypatch.setattr(DataService, "load_sample", forbidden)
    workspace = WorkspacePaths(tmp_path / "source-check")
    service = DataService(workspace)
    backtest = BacktestConfig(
        symbol="EXPLICIT/INSTRUMENT",
        timeframe=timeframe,
        start_date="2024-01-01",
        end_date="2024-01-31",
        init_cash=10000,
    )
    # The actual requirement takes precedence over an unused fallback source.
    raw = {
        "data": {
            "source": "unused-fallback",
            "market_type": market_type,
            "requirements": {
                "market": {
                    "source": source,
                    "symbol": backtest.symbol,
                    "timeframe": timeframe,
                }
            },
        }
    }
    if error is not None:
        with pytest.raises(ValueError, match=error):
            service.execution_config(raw, backtest)
    else:
        result = service.execution_config(raw, backtest)
        assert result["requirements"]["market"]["source"] == source
        assert result["requirements"]["market"]["timeframe"] == timeframe
    assert not list(workspace.processed_data.rglob("*.parquet"))


if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from pytest import MonkeyPatch


@pytest.mark.parametrize(
    ("start", "end", "utc_start", "utc_end", "years"),
    [
        (
            "2024-01-01T00:00:00+08:00",
            "2024-01-01T01:00:00+08:00",
            "2023-12-31T16:00:00Z",
            "2023-12-31T17:00:00Z",
            [2023],
        ),
        (
            "2024-12-31T20:00:00-08:00",
            "2024-12-31T22:00:00-08:00",
            "2025-01-01T04:00:00Z",
            "2025-01-01T06:00:00Z",
            [2025],
        ),
        (
            "2024-01-01T07:00:00+08:00",
            "2024-01-01T09:00:00+08:00",
            "2023-12-31T23:00:00Z",
            "2024-01-01T01:00:00Z",
            [2023, 2024],
        ),
    ],
)
def test_ensure_fetches_utc_years_for_offset_dates(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    start: str,
    end: str,
    utc_start: str,
    utc_end: str,
    years: list[int],
) -> None:
    monkeypatch.delenv("TRADINGDEV_DATA_ROOT", raising=False)
    service = DataService(WorkspacePaths(tmp_path / "year-boundaries"))

    def hourly_data(
        *, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        assert symbol == "BTC/USDT" and timeframe == "1h"
        return pd.DataFrame(
            {
                "timestamp": pd.date_range(start, end, freq="h"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            }
        )

    with (
        patch(
            "tradingdev.domain.data.crawlers.binance_api.BinanceAPICrawler.fetch",
            side_effect=hourly_data,
        ) as fetch,
    ):
        selected = service.ensure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
            source="binance_api",
        )
        assert [call.kwargs["start"].year for call in fetch.call_args_list] == years
        assert selected.frame["timestamp"].tolist() == list(
            pd.date_range(utc_start, utc_end, freq="h")
        )
        repeated = service.ensure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=utc_start,
            end_date=utc_end,
            source="binance_api",
        )
        assert fetch.call_count == len(years)
    pd.testing.assert_frame_equal(selected.frame, repeated.frame)
    assert selected.dataset_id == repeated.dataset_id


def test_ensure_rejects_reversed_dates_before_constructing_provider(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Invalid request bounds must fail before provider access")

    monkeypatch.setattr("tradingdev.app.data_service.create_crawler", forbidden)
    service = DataService(WorkspacePaths(tmp_path / "invalid-period"))
    with pytest.raises(ValueError, match="end_date must not precede start_date"):
        service.ensure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-01T07:00:00+08:00",
        )


def test_data_service_uses_shared_data_root(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    sample_ohlcv_df: pd.DataFrame,
) -> None:
    data_root = tmp_path / "shared_data"
    processed = data_root / "processed"
    processed.mkdir(parents=True)
    seed_market_cache(processed, sample_ohlcv_df)
    monkeypatch.setenv("TRADINGDEV_DATA_ROOT", str(data_root))

    service = DataService(WorkspacePaths(tmp_path / "workspace"))
    config = service.data_config(
        {
            "data": {
                "requirements": {
                    "market": {
                        "source": "binance_api",
                        "symbol": "BTC/USDT",
                        "timeframe": "1h",
                    },
                    "features": [],
                }
            }
        }
    )

    assert config.raw_dir == str(data_root.resolve() / "raw")
    assert config.processed_dir == str(processed.resolve())
    assert service.list_available_data() == [
        {
            "source": "binance_vision",
            "market_type": "futures/um",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "years_available": [2024],
        }
    ]
    assert service.data_available("BTC/USDT", "1h", "2024-01-01", "2024-01-31")


def test_inspect_dataset_reports_feature_sources_and_missing_values(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    workspace.processed_data.mkdir(parents=True, exist_ok=True)
    seed_market_cache(
        workspace.processed_data,
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [10.0, 11.0, 12.0],
            }
        ),
        source="binance_api",
    )
    feature_path = tmp_path / "funding_rate.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "funding_rate": [0.01, None, 0.02],
        }
    ).to_parquet(feature_path, index=False)
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text(
        f"""\
strategy:
  id: fixture
  class_name: FixtureStrategy
backtest:
  symbol: BTC/USDT
  timeframe: 1h
  start_date: "2024-01-01"
  end_date: "2024-01-31"
  init_cash: 10000.0
data:
  requirements:
    market:
      source: binance_api
      symbol: BTC/USDT
      timeframe: 1h
    features:
      - type: funding_rate
        source: local
        column: funding_rate
        path: "{feature_path}"
""",
        encoding="utf-8",
    )

    report = DataService(workspace).inspect_dataset(config_path)

    assert report["market_available"] is True
    assert report["market"]["rows"] == 3
    assert report["market"]["columns"] == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert report["market"]["timezone"] == "UTC"
    assert report["dataset_fingerprint"]["dataset_id"].startswith("query-sha256:")
    assert report["requirements"]["market"]["symbol"] == "BTC/USDT"
    assert report["features"] == [
        {
            "type": "funding_rate",
            "source": "local",
            "column": "funding_rate",
            "path": str(feature_path),
            "exists": True,
            "rows": 3,
            "missing_values": 1,
        }
    ]


def test_requirements_default_to_market_data_only(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = DataService(workspace)
    backtest_config = BacktestConfig(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-31",
        init_cash=10000.0,
    )

    requirements = service.requirements(
        {"data": {"source": "binance_api"}},
        backtest_config,
    )

    assert requirements.market.symbol == "BTC/USDT"
    assert requirements.market.timeframe == "1h"
    assert requirements.market.source == "binance_api"
    assert requirements.features == []


def test_requirements_market_inherits_data_source_when_omitted(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = DataService(workspace)
    backtest_config = BacktestConfig(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-31",
        init_cash=10000.0,
    )

    requirements = service.requirements(
        {
            "data": {
                "source": "binance_api",
                "requirements": {
                    "market": {"symbol": "BTC/USDT", "timeframe": "1h"},
                    "features": [],
                },
            }
        },
        backtest_config,
    )

    assert requirements.market.source == "binance_api"

    explicit = service.requirements(
        {
            "data": {
                "source": "binance_api",
                "requirements": {
                    "market": {
                        "source": "binance_vision",
                        "symbol": "BTC/USDT",
                        "timeframe": "1h",
                    },
                    "features": [],
                },
            }
        },
        backtest_config,
    )

    assert explicit.market.source == "binance_vision"
