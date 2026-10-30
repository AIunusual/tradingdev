"""Coordinate yearly provider queries with versioned storage and reuse policy."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tradingdev.adapters.storage.data_cache import DataCache
from tradingdev.domain.data.cache import DataIdentity
from tradingdev.domain.data.crawlers.registry import create_crawler
from tradingdev.domain.data.loader import DataLoader
from tradingdev.domain.data.processor import DataProcessor

if TYPE_CHECKING:
    from collections.abc import Callable

    from tradingdev.domain.data.crawlers.base import BaseCrawler
    from tradingdev.domain.data.schemas import DataConfig, MarketDataRequest


def _now_utc() -> datetime:
    return datetime.now(UTC)


class DataManager:
    """A successful query is coverage evidence, never proof of complete candles."""

    def __init__(
        self,
        data_config: DataConfig,
        request: MarketDataRequest,
        *,
        crawler: BaseCrawler | None = None,
        now_fn: Callable[[], datetime] = _now_utc,
    ) -> None:
        self._request = request
        self._now_fn = now_fn
        self._crawler = crawler or create_crawler(data_config.source, data_config)
        self.cache = DataCache(
            Path(data_config.raw_dir),
            Path(data_config.processed_dir),
            DataIdentity.market(data_config, request),
        )

    def load(self) -> tuple[pd.DataFrame, Path]:
        if self._request.end_date > self._now_fn():
            raise ValueError("Market request includes future timestamps")
        frames: list[pd.DataFrame] = []
        paths: list[Path] = []
        for year in range(
            self._request.start_date.year, self._request.end_date.year + 1
        ):
            with self.cache.lock(str(year)):
                start = datetime(year, 1, 1, tzinfo=UTC)
                year_end = datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)
                requested_start = max(start, self._request.start_date)
                requested_end = min(year_end, self._request.end_date)
                receipt = self.cache.receipt(str(year))
                if receipt is None or not receipt.reusable(
                    requested_start, requested_end
                ):
                    now = self._now_fn()
                    end = min(year_end, now)
                    raw = self._crawler.fetch(
                        symbol=self._request.symbol,
                        timeframe=self._request.timeframe,
                        start=start,
                        end=end,
                    )
                    raw = raw.copy()
                    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
                    frame = DataProcessor().process(raw)
                    frame = frame.loc[
                        frame["timestamp"].between(start, end)
                    ].reset_index(drop=True)
                    receipt = self.cache.publish(
                        str(year),
                        raw=raw,
                        frame=frame,
                        query_start=start,
                        query_end=end,
                        fetched_at=self._now_fn(),
                        save_raw=self._crawler.save_raw,
                    )
                path = self.cache.processed_path(receipt)
                frames.append(DataLoader().load_parquet(path))
                paths.append(path)
        combined = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        return self._trim_to_range(combined), paths[0]

    def _trim_to_range(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[
            frame["timestamp"].between(self._request.start_date, self._request.end_date)
        ].reset_index(drop=True)
