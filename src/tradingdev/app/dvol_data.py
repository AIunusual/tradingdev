"""Exact-range DVOL query caching using the shared publication and reuse policy."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tradingdev.adapters.storage.data_cache import DataCache
from tradingdev.domain.data.cache import CacheReceipt, DataIdentity
from tradingdev.domain.data.loader import DataLoader

if TYPE_CHECKING:
    from collections.abc import Callable

    from tradingdev.domain.data.crawlers.base import BaseCrawler
    from tradingdev.domain.data.schemas import DataConfig, MarketDataRequest


def _now_utc() -> datetime:
    return datetime.now(UTC)


class DVOLData:
    """Keep automatic feature cache evidence separate from user-supplied files."""

    def __init__(
        self,
        config: DataConfig,
        request: MarketDataRequest,
        *,
        now_fn: Callable[[], datetime] = _now_utc,
    ) -> None:
        self.request = request
        self._now_fn = now_fn
        self.cache = DataCache(
            Path(config.raw_dir),
            Path(config.processed_dir),
            DataIdentity(
                kind="dvol",
                source="deribit",
                market_type=None,
                symbol=request.symbol.split("/")[0],
                timeframe=request.timeframe,
            ),
        )
        bounds = json.dumps(
            [
                request.start_date.isoformat(),
                request.end_date.isoformat(),
            ]
        )
        self.key = "query-" + hashlib.sha256(bounds.encode("utf-8")).hexdigest()

    def receipt(self) -> CacheReceipt | None:
        return self.cache.receipt(self.key)

    def reusable_path(self) -> Path | None:
        receipt = self.receipt()
        if receipt is None or not receipt.reusable(
            self.request.start_date,
            self.request.end_date,
        ):
            return None
        return self.cache.processed_path(receipt)

    def load(self, crawler_factory: Callable[[], BaseCrawler]) -> pd.DataFrame:
        request = self.request
        if request.end_date > self._now_fn():
            raise ValueError("DVOL request includes future timestamps")
        with self.cache.lock(self.key):
            path = self.reusable_path()
            if path is None:
                crawler = crawler_factory()
                raw = crawler.fetch(
                    symbol=self.cache.identity.symbol,
                    timeframe=request.timeframe,
                    start=request.start_date,
                    end=request.end_date,
                ).copy()
                raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
                frame = raw.loc[
                    raw["timestamp"].between(request.start_date, request.end_date)
                ].reset_index(drop=True)
                receipt = self.cache.publish(
                    self.key,
                    raw=raw,
                    frame=frame,
                    query_start=request.start_date,
                    query_end=request.end_date,
                    fetched_at=self._now_fn(),
                    save_raw=crawler.save_raw,
                )
                path = self.cache.processed_path(receipt)
            return DataLoader().load_parquet(path)
