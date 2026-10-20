"""Binance USD-M Futures derivatives data crawler.

Fetches funding rate historical data from Binance public API
(no API key required).

Decision note: this crawler is intentionally kept separate from
``binance_api.py``.  The Binance API crawler owns OHLCV market candles, while
this module owns derivatives-specific feature series declared through
``data.requirements.features``.  Keeping the adapters separate avoids mixing
market-candle cache semantics with funding-rate feature ingestion.

Note: Open Interest and Long/Short Ratio historical data is only
available for the most recent ~2 weeks via Binance API, making them
unsuitable for backtesting.  They can still be used in live trading.
"""

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import ccxt
import pandas as pd

from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logger(__name__)

_REQUEST_DELAY = 0.3
_BATCH_LIMIT = 1000


class BinanceDerivativesCrawler:
    """Fetch derivatives data from Binance USD-M Futures.

    Currently supports:
    - **Funding Rate**: 8-hour settlement rate with full history.

    Open Interest and Long/Short Ratio are available for real-time
    use but NOT for historical backtesting (Binance only keeps
    ~2 weeks of history for these endpoints).
    """

    def __init__(self) -> None:
        self._exchange = ccxt.binanceusdm({"enableRateLimit": True})
        self._exchange.load_markets()

    def fetch_funding_rate(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch historical funding rates with pagination.

        Funding rates settle every 8 hours (00:00, 08:00, 16:00 UTC).
        The rate represents the cost of holding a long position; a
        positive rate means longs pay shorts.

        Args:
            symbol: Trading pair (e.g. ``"BTC/USDT"``).
            start: Start time (UTC).
            end: End time (UTC).

        Returns:
            DataFrame with columns ``[timestamp, funding_rate]``.
        """
        futures_symbol = _to_futures_symbol(symbol)
        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        all_rows: list[dict[str, object]] = []

        logger.info(
            "Fetching funding rate for %s: %s ~ %s",
            symbol,
            start.isoformat(),
            end.isoformat(),
        )

        while since_ms < end_ms:
            records = self._exchange.fetch_funding_rate_history(
                futures_symbol,
                since=since_ms,
                limit=_BATCH_LIMIT,
            )
            if not records:
                break

            for r in records:
                all_rows.append(
                    {
                        "timestamp": r["datetime"],
                        "funding_rate": r["fundingRate"],
                    }
                )

            last_ts = int(records[-1]["timestamp"])
            logger.info(
                "Fetched %d records, total so far: %d",
                len(records),
                len(all_rows),
            )

            if len(records) < _BATCH_LIMIT:
                break
            since_ms = last_ts + 1
            time.sleep(_REQUEST_DELAY)

        df = pd.DataFrame(all_rows)
        if len(df) > 0:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").drop_duplicates(
                subset=["timestamp"],
                keep="first",
            )
            start_utc = start.replace(tzinfo=UTC)
            end_utc = end.replace(tzinfo=UTC)
            df = df[(df["timestamp"] >= start_utc) & (df["timestamp"] <= end_utc)]
            df = df.reset_index(drop=True)

        logger.info("Fetched %d funding rate records", len(df))
        return df

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save derivatives data as Parquet."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info("Saved derivatives data to %s", output_path)


def _to_futures_symbol(symbol: str) -> str:
    """Convert spot symbol to futures symbol for ccxt."""
    if ":" not in symbol:
        quote = symbol.split("/")[1] if "/" in symbol else "USDT"
        return f"{symbol}:{quote}"
    return symbol
