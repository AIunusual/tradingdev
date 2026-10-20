"""Deribit DVOL (implied volatility index) crawler."""

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import pandas as pd

from tradingdev.domain.data.crawlers.base import BaseCrawler
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

_DVOL_COLUMNS = [
    "timestamp",
    "dvol_open",
    "dvol_high",
    "dvol_low",
    "dvol_close",
]
_BASE_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
_REQUEST_DELAY = 0.5  # seconds between paginated requests

# Deribit resolution parameter (seconds): 1m = "60"
_TIMEFRAME_MAP: dict[str, str] = {
    "1s": "1",
    "1m": "60",
    "1h": "3600",
    "12h": "43200",
    "1d": "1D",
}


class DeribitDVOLCrawler(BaseCrawler):
    """Fetch Deribit DVOL data via public API (no key required).

    The DVOL index measures the 30-day implied volatility of BTC
    derived from Deribit's options market, analogous to the VIX.
    Values are annualized volatility in percent (e.g. 45 = 45%).
    """

    def __init__(self) -> None:
        self._client = httpx.Client(timeout=30.0)

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch DVOL candles with automatic pagination.

        Args:
            symbol: Currency (e.g. ``"BTC"``).
            timeframe: Resolution (``"1m"``, ``"1h"``, ``"1d"``).
            start: Start time (UTC).
            end: End time (UTC).

        Returns:
            DataFrame with columns
            ``[timestamp, dvol_open, dvol_high, dvol_low, dvol_close]``.
        """
        resolution = _TIMEFRAME_MAP.get(timeframe)
        if resolution is None:
            msg = (
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported: {list(_TIMEFRAME_MAP)}"
            )
            raise ValueError(msg)

        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        all_rows: list[list[float]] = []

        logger.info(
            "Fetching %s DVOL (%s) from %s to %s",
            symbol,
            timeframe,
            start.isoformat(),
            end.isoformat(),
        )

        current_end_ms = end_ms
        while current_end_ms > start_ms:
            params: dict[str, str | int] = {
                "currency": symbol.upper(),
                "start_timestamp": start_ms,
                "end_timestamp": current_end_ms,
                "resolution": resolution,
            }
            resp = self._client.get(_BASE_URL, params=params)
            resp.raise_for_status()
            body = resp.json()

            data: list[list[float]] = body["result"]["data"]
            if not data:
                break

            all_rows.extend(data)
            continuation = body["result"].get("continuation")

            logger.info(
                "Fetched %d DVOL candles, total so far: %d",
                len(data),
                len(all_rows),
            )

            if continuation is None:
                break

            current_end_ms = int(continuation)
            time.sleep(_REQUEST_DELAY)

        if not all_rows:
            logger.warning("No DVOL data returned for %s", symbol)
            return pd.DataFrame(columns=_DVOL_COLUMNS)

        df = pd.DataFrame(all_rows, columns=_DVOL_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Sort chronologically and remove duplicates
        df = df.sort_values("timestamp").drop_duplicates(
            subset=["timestamp"], keep="first"
        )

        # Filter to requested range
        start_utc = start.replace(tzinfo=UTC)
        end_utc = end.replace(tzinfo=UTC)
        df = df[(df["timestamp"] >= start_utc) & (df["timestamp"] <= end_utc)]
        df = df.reset_index(drop=True)

        logger.info("Total DVOL candles fetched: %d", len(df))
        return df

    def fetch_sample(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        max_rows: int,
        timestamps: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """Fetch earliest DVOL rows, optionally restricted to exact timestamps.

        Deribit paginates backwards. Finish each bounded chronological window
        before advancing so a small sample cannot select only its newest bars.
        Target timestamps let sparse market samples skip unrelated DVOL windows.
        """
        start_utc, end_utc = sample_range(start, end, max_rows)
        resolution = _TIMEFRAME_MAP.get(timeframe)
        if resolution is None:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported: {list(_TIMEFRAME_MAP)}"
            )
        interval_ms = (86_400 if resolution == "1D" else int(resolution)) * 1000
        cursor = -(-pd.Timestamp(start_utc).value // 1_000_000)
        stop = pd.Timestamp(end_utc).value // 1_000_000
        targets: set[int] | None = None
        if timestamps is not None:
            if len(timestamps) > max_rows:
                raise ValueError("DVOL sample target count exceeds max_rows")
            normalized = pd.to_datetime(timestamps, utc=True).as_unit("ns")
            # Preserve exact joins: a nanosecond target with no millisecond
            # representation cannot match any provider timestamp.
            targets = {
                timestamp.value // 1_000_000
                for timestamp in normalized
                if not pd.isna(timestamp)
                and timestamp.value % 1_000_000 == 0
                and cursor <= timestamp.value // 1_000_000 <= stop
            }
        selected: dict[int, list[Any]] = {}
        while cursor <= stop and len(selected) < max_rows:
            if targets is not None:
                pending = [timestamp for timestamp in targets if timestamp >= cursor]
                if not pending:
                    break
                cursor = min(pending)
            limit = min(max_rows - len(selected), SAMPLE_BATCH_ROWS)
            window_end = min(stop, cursor + limit * interval_ms - 1)
            rows = self._fetch_sample_window(
                symbol, resolution, cursor, window_end, limit
            )
            for timestamp, row in rows.items():
                if targets is None or timestamp in targets:
                    selected.setdefault(timestamp, row)
            cursor = window_end + 1
        return sample_frame(list(selected.values()), _DVOL_COLUMNS, unit="ms")

    def _fetch_sample_window(
        self,
        symbol: str,
        resolution: str,
        start_ms: int,
        end_ms: int,
        row_limit: int,
    ) -> dict[int, list[Any]]:
        """Read a finite window, bounding both pages and retained unique rows."""
        selected: dict[int, list[Any]] = {}
        page_end = end_ms
        while page_end >= start_ms:
            payload = stream_sample_json(
                self._client,
                _BASE_URL,
                {
                    "currency": symbol.upper(),
                    "start_timestamp": start_ms,
                    "end_timestamp": page_end,
                    "resolution": resolution,
                },
            )
            if payload.get("error"):
                raise ValueError(f"Deribit sample error: {payload['error']}")
            result = payload["result"]
            data = result["data"]
            if not isinstance(data, list) or len(data) > row_limit:
                raise ValueError(
                    "Deribit sample response exceeds the requested row limit"
                )
            for row in data:
                if not isinstance(row, list) or len(row) != len(_DVOL_COLUMNS):
                    raise ValueError("Invalid Deribit sample candle")
                timestamp = int(row[0])
                if start_ms <= timestamp <= page_end:
                    if timestamp not in selected and len(selected) == row_limit:
                        raise ValueError("Deribit sample window exceeds its row limit")
                    selected.setdefault(timestamp, row)
            continuation = result.get("continuation")
            if continuation is None:
                break
            next_end = int(continuation)
            if next_end >= page_end:
                raise ValueError("Deribit sample pagination did not advance")
            page_end = next_end
            if page_end >= start_ms:
                time.sleep(_REQUEST_DELAY)
        return selected

    def save_raw(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save raw DVOL data as CSV.

        Args:
            df: DVOL DataFrame.
            output_path: Path to write CSV file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved raw DVOL data to %s", output_path)
