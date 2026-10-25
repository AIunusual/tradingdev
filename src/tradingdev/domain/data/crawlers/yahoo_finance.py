"""Yahoo Finance chart API crawler for stocks, futures, indices, and FX.

Uses the public v8 chart endpoint (no API key). Symbols follow Yahoo
conventions: ``AAPL`` (US stock), ``2330.TW`` (non-US stock), ``ES=F``
(futures), ``^GSPC`` (index), ``EURUSD=X`` (FX).

Yahoo Finance is an unofficial free API without a published quota;
intraday intervals are limited to roughly the last 730 days. Keep
request ranges small and rely on the yearly cache to avoid re-fetching.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import pandas as pd

from tradingdev.domain.data.crawlers.base import BaseCrawler
from tradingdev.domain.data.crawlers.registry import YAHOO_INTERVALS
from tradingdev.domain.data.crawlers.sampling import (
    SAMPLE_BATCH_ROWS,
    sample_frame,
    sample_range,
    stream_sample_json,
)
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logger(__name__)

_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

# Calendar candles follow the exchange timezone. Use the shortest calendar
# period, allowing a one-hour DST change, rather than interpreting 1mo as 30d.
# Half-open windows of these lengths cannot contain more than their row budget.
_SAMPLE_INTERVAL_SECONDS = {
    "1m": 60,
    "2m": 120,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "60m": 3600,
    "90m": 5400,
    "1d": 23 * 3600,
    "1wk": (7 * 24 - 1) * 3600,
    "1mo": (28 * 24 - 1) * 3600,
}


def _to_yahoo_interval(timeframe: str) -> str:
    """Map a project timeframe to a Yahoo chart interval."""
    interval = YAHOO_INTERVALS.get(timeframe)
    if interval is None:
        supported = ", ".join(sorted(YAHOO_INTERVALS))
        msg = (
            f"Timeframe {timeframe!r} is not supported by yahoo_finance. "
            f"Supported timeframes: {supported}"
        )
        raise ValueError(msg)
    return interval


def _parse_chart_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert a v8 chart JSON payload into a standard OHLCV DataFrame."""
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        msg = f"Yahoo Finance chart error: {error}"
        raise ValueError(msg)
    results = chart.get("result") or []
    if not results:
        return pd.DataFrame(columns=_OHLCV_COLUMNS)
    result = results[0]
    timestamps = result.get("timestamp") or []
    if not timestamps:
        return pd.DataFrame(columns=_OHLCV_COLUMNS)
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
            "open": pd.to_numeric(pd.Series(quote["open"]), errors="coerce"),
            "high": pd.to_numeric(pd.Series(quote["high"]), errors="coerce"),
            "low": pd.to_numeric(pd.Series(quote["low"]), errors="coerce"),
            "close": pd.to_numeric(pd.Series(quote["close"]), errors="coerce"),
            "volume": pd.to_numeric(pd.Series(quote["volume"]), errors="coerce"),
        }
    )
    return df.dropna(subset=["close"]).reset_index(drop=True)


def _sample_chart_rows(payload: dict[str, Any], row_limit: int) -> list[list[Any]]:
    """Validate bounded arrays and project OHLCV before frame construction."""
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ValueError(f"Yahoo Finance chart error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        return []
    if len(results) != 1:
        raise ValueError("Expected one Yahoo Finance sample result")
    result = results[0]
    timestamps = result.get("timestamp") or []
    if not isinstance(timestamps, list) or len(timestamps) > row_limit:
        raise ValueError("Yahoo sample response exceeds the requested row limit")
    if not timestamps:
        return []
    quote = result["indicators"]["quote"][0]
    projected: dict[str, Any] = {}
    for name in _OHLCV_COLUMNS[1:]:
        values = quote[name]
        if not isinstance(values, list) or len(values) != len(timestamps):
            raise ValueError(f"Invalid Yahoo sample {name} array length")
        projected[name] = pd.to_numeric(values, errors="coerce")
    return [
        [timestamp, *(projected[name][index] for name in _OHLCV_COLUMNS[1:])]
        for index, timestamp in enumerate(timestamps)
        if not pd.isna(projected["close"][index])
    ]


class YahooFinanceCrawler(BaseCrawler):
    """Fetch OHLCV data from the Yahoo Finance chart API (no API key)."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles for [start, end].

        Args:
            symbol: Yahoo Finance symbol (e.g. ``"AAPL"``, ``"ES=F"``).
            timeframe: Candle interval (e.g. ``"1d"``, ``"1h"``).
            start: Start time (UTC).
            end: End time (UTC).

        Returns:
            DataFrame with columns ``[timestamp, open, high, low, close, volume]``.
        """
        interval = _to_yahoo_interval(timeframe)
        start_utc = start if start.tzinfo else start.replace(tzinfo=UTC)
        end_utc = end if end.tzinfo else end.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        if start_utc > now:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)
        end_utc = min(end_utc, now)

        logger.info(
            "Fetching %s %s candles from %s to %s via Yahoo Finance",
            symbol,
            timeframe,
            start_utc.isoformat(),
            end_utc.isoformat(),
        )
        response = self._client.get(
            f"{_BASE_URL}/{symbol}",
            params={
                "period1": int(start_utc.timestamp()),
                "period2": int(end_utc.timestamp()),
                "interval": interval,
                "events": "history",
            },
        )
        response.raise_for_status()
        df = _parse_chart_payload(response.json())
        if df.empty:
            logger.warning("No data fetched for %s %s", symbol, timeframe)
            return df

        df = df[(df["timestamp"] >= start_utc) & (df["timestamp"] <= end_utc)]
        df = df.sort_values("timestamp").drop_duplicates("timestamp")
        logger.info("Total candles fetched: %d", len(df))
        return df.reset_index(drop=True)

    def fetch_sample(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        max_rows: int,
    ) -> pd.DataFrame:
        """Fetch earliest candles through bounded, half-open chart windows."""
        start_utc, end_utc = sample_range(start, end, max_rows)
        interval = _to_yahoo_interval(timeframe)
        end_utc = min(end_utc, datetime.now(UTC))
        cursor = -(-pd.Timestamp(start_utc).value // 1_000_000_000)
        # Yahoo period2 is exclusive; adding one second includes an end candle
        # and also makes a start == end request nonempty.
        stop = pd.Timestamp(end_utc).value // 1_000_000_000 + 1
        selected: dict[int, list[Any]] = {}
        while cursor < stop and len(selected) < max_rows:
            limit = min(max_rows - len(selected), SAMPLE_BATCH_ROWS)
            window_stop = min(stop, cursor + limit * _SAMPLE_INTERVAL_SECONDS[interval])
            payload = stream_sample_json(
                self._client,
                f"{_BASE_URL}/{symbol}",
                {
                    "period1": cursor,
                    "period2": window_stop,
                    "interval": interval,
                    "events": "history",
                },
            )
            for row in _sample_chart_rows(payload, limit):
                timestamp = int(row[0])
                if cursor <= timestamp < window_stop:
                    selected.setdefault(timestamp, row)
            cursor = window_stop
        return sample_frame(list(selected.values()), _OHLCV_COLUMNS, unit="s")

    def save_raw(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save raw OHLCV data as CSV.

        Args:
            df: OHLCV DataFrame.
            output_path: Path to write CSV file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved raw data to %s", output_path)
