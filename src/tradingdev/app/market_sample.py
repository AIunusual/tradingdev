"""Find historical samples across market closures using bounded fetch windows."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tradingdev.adapters.storage.data_cache import DataCache
from tradingdev.adapters.storage.market_samples import read_market_sample
from tradingdev.domain.data.cache import DataIdentity
from tradingdev.domain.data.processor import DataProcessor

if TYPE_CHECKING:
    from tradingdev.domain.data.crawlers.base import BaseCrawler
    from tradingdev.domain.data.schemas import DataConfig, MarketDataRequest


def load_market_sample(
    request: MarketDataRequest,
    config: DataConfig,
    crawler: BaseCrawler,
    *,
    max_rows: int,
) -> pd.DataFrame:
    """Search only the requested interval; the supervising worker owns the deadline."""
    if request.end_date > datetime.now(UTC):
        raise ValueError("Market request includes future timestamps")
    match = re.fullmatch(r"(\d+)(s|m|h|d|w|wk|mo|M)", request.timeframe)
    if match is None:
        raise ValueError(f"Unsupported sample timeframe: {request.timeframe}")
    units = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
        "wk": 604800,
        "mo": 2678400,
        "M": 2678400,
    }
    seconds = int(match[1]) * units[match[2]]
    if seconds <= 0:
        raise ValueError("Sample timeframe must be positive")
    start = request.start_date
    start = start.replace(tzinfo=UTC) if start.tzinfo is None else start.astimezone(UTC)
    end = request.end_date
    end = end.replace(tzinfo=UTC) if end.tzinfo is None else end.astimezone(UTC)
    cache = DataCache(
        Path(config.raw_dir),
        Path(config.processed_dir),
        DataIdentity.market(config, request),
    )
    selected = pd.DataFrame()
    cursor = start
    width = seconds * max_rows * 3
    while cursor <= end:
        # Compare the requested range before constructing a timedelta or adding it.
        stop = (
            end
            if width >= (end - cursor).total_seconds()
            else cursor + timedelta(seconds=width)
        )
        for year in range(cursor.year, stop.year + 1):
            year_start = max(cursor, datetime(year, 1, 1, tzinfo=UTC))
            year_end = min(stop, datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=UTC))
            receipt = cache.receipt(str(year))
            complete = receipt is not None and receipt.reusable(year_start, year_end)
            if complete and receipt is not None:
                frame = read_market_sample(
                    cache.processed_path(receipt),
                    start=year_start,
                    end=year_end,
                    max_rows=max_rows,
                )
                selected = _accumulate(selected, frame, max_rows)
            if len(selected) >= max_rows:
                break
            if not complete:
                # Providers sample only this missing year, so a result cap cannot
                # be consumed by an earlier year whose complete cache takes priority.
                raw = crawler.fetch_sample(
                    request.symbol,
                    request.timeframe,
                    year_start,
                    year_end,
                    max_rows=max_rows,
                )
                if len(raw) > max_rows:
                    raise ValueError("Market provider exceeded the sample row budget")
                if not raw.empty:
                    raw = raw.copy()
                    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
                    frame = DataProcessor().process(raw)
                    frame = frame.loc[frame["timestamp"].between(year_start, year_end)]
                    selected = _accumulate(selected, frame, max_rows)
            if len(selected) >= max_rows:
                break
        if len(selected) >= max_rows or stop == end:
            break
        # Overlap inclusive boundaries to preserve candles; deduplicate above.
        cursor = stop
    if selected.empty:
        raise ValueError(
            "No historical bars are available in the requested sample interval"
        )
    return selected.reset_index(drop=True)


def _accumulate(
    selected: pd.DataFrame, frame: pd.DataFrame, max_rows: int
) -> pd.DataFrame:
    if frame.empty:
        return selected
    return (
        pd.concat([selected, frame], ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .head(max_rows)
    )
