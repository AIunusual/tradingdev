"""Binance Data Vision batch downloader for historical K-line data."""

from __future__ import annotations

import calendar
import io
import zipfile
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import httpx
import pandas as pd

from tradingdev.domain.data.crawlers.base import BaseCrawler
from tradingdev.domain.data.crawlers.binance_vision_archive import (
    SampleDownloadBudget,
    normalize_vision_rows,
    read_sample_archive,
)
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logger(__name__)

_BASE_URL = "https://data.binance.vision/data"
_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _to_vision_symbol(symbol: str) -> str:
    """Convert 'BTC/USDT' -> 'BTCUSDT'."""
    return symbol.replace("/", "").upper()


def _month_range(start: datetime, end: datetime) -> list[tuple[int, int]]:
    """Return list of (year, month) tuples covering [start, end]."""
    months: list[tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _parse_zip_csv(content: bytes) -> pd.DataFrame:
    """Parse a Binance Vision ZIP file and return a standard OHLCV DataFrame."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            raw = pd.read_csv(f, header=None, usecols=range(6))
    return normalize_vision_rows(raw)


class BinanceVisionCrawler(BaseCrawler):
    """Download historical K-line data from data.binance.vision.

    Uses monthly ZIP files for complete past months and daily ZIPs for
    the current partial month. No API key or geo-restriction required.

    Args:
        market_type: ``'futures/um'`` for USD-M perpetuals (default),
                     ``'spot'`` for spot markets.
    """

    def __init__(self, market_type: str = "futures/um") -> None:
        self._market_type = market_type
        self._client = httpx.Client(timeout=60.0, follow_redirects=True)

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles from data.binance.vision.

        Args:
            symbol: Trading pair (e.g. ``"BTC/USDT"``).
            timeframe: Candle interval (e.g. ``"1d"``, ``"1h"``).
            start: Start time (UTC).
            end: End time (UTC).

        Returns:
            DataFrame with columns ``[timestamp, open, high, low, close, volume]``.
        """
        vision_symbol = _to_vision_symbol(symbol)
        today = datetime.now(UTC).date()
        dfs: list[pd.DataFrame] = []

        logger.info(
            "Fetching %s %s candles from %s to %s via binance.vision",
            symbol,
            timeframe,
            start.isoformat(),
            end.isoformat(),
        )

        for year, month in _month_range(start, end):
            last_day = calendar.monthrange(year, month)[1]
            month_end_date = date(year, month, last_day)

            if month_end_date < today:
                df = self._fetch_monthly(vision_symbol, timeframe, year, month)
            else:
                month_start_date = date(year, month, 1)
                start_date = (
                    start.date() if start.tzinfo else start.replace(tzinfo=UTC).date()
                )
                day_start = max(month_start_date, start_date)
                day_end = min(month_end_date, today - timedelta(days=1))
                df = self._fetch_daily_range(
                    vision_symbol, timeframe, day_start, day_end
                )

            if df is not None and not df.empty:
                dfs.append(df)

        if not dfs:
            logger.warning("No data fetched for %s %s", symbol, timeframe)
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values("timestamp").drop_duplicates("timestamp")

        start_utc = start if start.tzinfo else start.replace(tzinfo=UTC)
        end_utc = end if end.tzinfo else end.replace(tzinfo=UTC)
        combined = combined[
            (combined["timestamp"] >= start_utc) & (combined["timestamp"] <= end_utc)
        ]

        logger.info("Total candles fetched: %d", len(combined))
        return combined.reset_index(drop=True)

    def fetch_sample(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        max_rows: int,
    ) -> pd.DataFrame:
        """Read capped daily archives, falling back once per month on a 404."""
        if max_rows <= 0:
            raise ValueError("Sample max_rows must be positive")
        start = (
            start.replace(tzinfo=UTC) if start.tzinfo is None else start.astimezone(UTC)
        )
        end = end.replace(tzinfo=UTC) if end.tzinfo is None else end.astimezone(UTC)
        last_day = min(end.date(), datetime.now(UTC).date() - timedelta(days=1))
        current = start.date()
        vision_symbol = _to_vision_symbol(symbol)
        selected = pd.DataFrame(columns=_OHLCV_COLUMNS)
        budget = SampleDownloadBudget()
        fallback_months: set[tuple[int, int]] = set()

        while current <= last_day and start <= end:
            next_day = current + timedelta(days=1)
            day_start = datetime.combine(current, datetime.min.time(), UTC)
            day_end = datetime.combine(next_day, datetime.min.time(), UTC) - timedelta(
                microseconds=1
            )
            suffix = f"{current.year}-{current.month:02d}-{current.day:02d}"
            root = f"{_BASE_URL}/{self._market_type}"
            filename = f"{vision_symbol}-{timeframe}"
            daily_url = (
                f"{root}/daily/klines/{vision_symbol}/{timeframe}/"
                f"{filename}-{suffix}.zip"
            )
            remaining = max_rows - len(selected)
            frame = read_sample_archive(
                self._client,
                daily_url,
                start=max(start, day_start),
                end=min(end, day_end),
                max_rows=remaining,
                budget=budget,
            )
            month = (current.year, current.month)
            if frame is None and month not in fallback_months:
                fallback_months.add(month)
                monthly_url = (
                    f"{root}/monthly/klines/{vision_symbol}/{timeframe}/"
                    f"{filename}-{current.year}-{current.month:02d}.zip"
                )
                month_end = current.replace(
                    day=calendar.monthrange(current.year, current.month)[1]
                )
                next_month = month_end + timedelta(days=1)
                month_end_time = datetime.combine(
                    next_month, datetime.min.time(), UTC
                ) - timedelta(microseconds=1)
                frame = read_sample_archive(
                    self._client,
                    monthly_url,
                    start=max(start, day_start),
                    end=min(end, month_end_time),
                    max_rows=remaining,
                    budget=budget,
                )
                if frame is not None:
                    next_day = next_month
            if frame is not None and not frame.empty:
                selected = (
                    (
                        frame
                        if selected.empty
                        else pd.concat([selected, frame], ignore_index=True)
                    )
                    .drop_duplicates("timestamp")
                    .sort_values("timestamp")
                    .head(max_rows)
                )
                if len(selected) >= max_rows:
                    break
            current = next_day
        return selected.reset_index(drop=True)

    def save_raw(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save raw OHLCV data as CSV.

        Args:
            df: OHLCV DataFrame.
            output_path: Path to write CSV file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved raw data to %s", output_path)

    def _fetch_monthly(
        self,
        symbol: str,
        timeframe: str,
        year: int,
        month: int,
    ) -> pd.DataFrame | None:
        url = (
            f"{_BASE_URL}/{self._market_type}/monthly/klines/"
            f"{symbol}/{timeframe}/{symbol}-{timeframe}-{year}-{month:02d}.zip"
        )
        return self._download_and_parse(url)

    def _fetch_daily_range(
        self,
        symbol: str,
        timeframe: str,
        day_start: date,
        day_end: date,
    ) -> pd.DataFrame | None:
        dfs: list[pd.DataFrame] = []
        current = day_start
        while current <= day_end:
            url = (
                f"{_BASE_URL}/{self._market_type}/daily/klines/"
                f"{symbol}/{timeframe}/"
                f"{symbol}-{timeframe}-{current.year}-{current.month:02d}-{current.day:02d}.zip"
            )
            df = self._download_and_parse(url)
            if df is not None:
                dfs.append(df)
            current += timedelta(days=1)
        return pd.concat(dfs, ignore_index=True) if dfs else None

    def _download_and_parse(self, url: str) -> pd.DataFrame | None:
        logger.info("Downloading %s", url)
        # Only an explicit 404 means no archive. Transport failures must abort
        # the fetch so a partial result cannot become a trusted yearly cache.
        resp = self._client.get(url)
        if resp.status_code == 404:
            logger.warning("Not found: %s", url)
            return None
        resp.raise_for_status()
        return _parse_zip_csv(resp.content)
