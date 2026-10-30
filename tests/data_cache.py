"""Seed explicit successful synthetic-provider queries, never legacy flat caches."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tradingdev.adapters.storage.data_cache import DataCache
from tradingdev.domain.data.cache import DataIdentity
from tradingdev.domain.data.schemas import DataConfig, MarketDataRequest

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


def seed_market_cache(
    processed_dir: Path,
    frame: pd.DataFrame,
    *,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    source: str = "binance_vision",
    market_type: str = "futures/um",
    year: int = 2024,
) -> Path:
    """A synthetic provider successfully queried a whole historical year."""
    request = MarketDataRequest(
        symbol=symbol,
        timeframe=timeframe,
        start_date=datetime(year, 1, 1, tzinfo=UTC),
        end_date=datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=UTC),
    )
    config = DataConfig(source=source, market_type=market_type)
    cache = DataCache(
        processed_dir.parent / "raw",
        processed_dir,
        DataIdentity.market(config, request),
    )
    with cache.lock(str(year)):
        receipt = cache.publish(
            str(year),
            raw=frame,
            frame=frame,
            query_start=request.start_date,
            query_end=request.end_date,
            fetched_at=datetime(year + 1, 1, 8, tzinfo=UTC),
            save_raw=lambda data, path: data.to_csv(path, index=False),
        )
    return cache.processed_path(receipt)


def cached_market_path(
    processed_dir: Path,
    *,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    source: str = "binance_vision",
    market_type: str = "futures/um",
    year: int = 2024,
) -> Path:
    identity = DataIdentity.market(
        DataConfig(source=source, market_type=market_type),
        MarketDataRequest(
            symbol=symbol,
            timeframe=timeframe,
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
        ),
    )
    cache = DataCache(processed_dir.parent / "raw", processed_dir, identity)
    receipt = cache.receipt(str(year))
    assert receipt is not None
    return cache.processed_path(receipt)
