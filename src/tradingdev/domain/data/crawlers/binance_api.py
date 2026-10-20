"""Binance public API crawler using ccxt."""

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import ccxt
import pandas as pd

from tradingdev.domain.data.crawlers.base import BaseCrawler
from tradingdev.domain.data.crawlers.sampling import sample_frame, sample_range
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logger(__name__)

_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
_BATCH_LIMIT = 1000


class BinanceAPICrawler(BaseCrawler):
    """Fetch OHLCV data from Binance via ccxt (no API key required)."""

    def __init__(self) -> None:
        self._exchange: ccxt.binance = ccxt.binance({"enableRateLimit": True})

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles with automatic pagination.

        Args:
            symbol: Trading pair (e.g. ``"BTC/USDT"``).
            timeframe: Candle interval (e.g. ``"1h"``).
            start: Start time (UTC).
            end: End time (UTC).

        Returns:
            DataFrame with columns ``[timestamp, open, high, low, close, volume]``.
        """
        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        all_candles: list[list[object]] = []

        logger.info(
            "Fetching %s %s candles from %s to %s",
            symbol,
            timeframe,
            start.isoformat(),
            end.isoformat(),
        )

        while since_ms < end_ms:
            candles: list[list[object]] = self._exchange.fetch_ohlcv(
                symbol, timeframe, since=since_ms, limit=_BATCH_LIMIT
            )
            if not candles:
                break

            all_candles.extend(candles)
            last_ts = int(candles[-1][0])  # type: ignore[call-overload]
            logger.info(
                "Fetched %d candles, last timestamp: %s",
                len(candles),
                datetime.fromtimestamp(last_ts / 1000, tz=UTC).isoformat(),
            )

            if len(candles) < _BATCH_LIMIT:
                break

            since_ms = last_ts + 1
            time.sleep(self._exchange.rateLimit / 1000)

        df = pd.DataFrame(all_candles, columns=_OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Filter to requested range
        start_utc = start.replace(tzinfo=UTC)
        end_utc = end.replace(tzinfo=UTC)
        df = df[(df["timestamp"] >= start_utc) & (df["timestamp"] <= end_utc)]
        df = df.reset_index(drop=True)

        logger.info("Total candles fetched: %d", len(df))
        return df

    def fetch_sample(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        max_rows: int,
    ) -> pd.DataFrame:
        """Fetch the earliest unique candles using bounded CCXT pages."""
        start_utc, end_utc = sample_range(start, end, max_rows)
        since_ms = -(-pd.Timestamp(start_utc).value // 1_000_000)
        end_ms = pd.Timestamp(end_utc).value // 1_000_000
        selected: dict[int, list[object]] = {}
        while since_ms <= end_ms and len(selected) < max_rows:
            limit = min(max_rows - len(selected), _BATCH_LIMIT)
            candles: list[list[object]] = self._exchange.fetch_ohlcv(
                symbol, timeframe, since=since_ms, limit=limit
            )
            if len(candles) > limit:
                raise ValueError(
                    "Binance sample response exceeds the requested row limit"
                )
            if not candles:
                break
            last_ms = since_ms - 1
            for candle in candles:
                if len(candle) != len(_OHLCV_COLUMNS):
                    raise ValueError("Invalid Binance sample candle")
                timestamp = int(candle[0])  # type: ignore[call-overload]
                last_ms = max(last_ms, timestamp)
                if since_ms <= timestamp <= end_ms:
                    selected.setdefault(timestamp, candle)
            if last_ms < since_ms:
                raise ValueError("Binance sample pagination did not advance")
            if len(candles) < limit or last_ms >= end_ms or len(selected) == max_rows:
                break
            since_ms = last_ms + 1
            time.sleep(self._exchange.rateLimit / 1000)
        return sample_frame(list(selected.values()), _OHLCV_COLUMNS, unit="ms")

    def save_raw(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save raw OHLCV data as CSV.

        Args:
            df: OHLCV DataFrame.
            output_path: Path to write CSV file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved raw data to %s", output_path)
