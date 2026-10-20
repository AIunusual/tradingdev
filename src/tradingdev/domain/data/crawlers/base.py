"""Abstract base class for all data crawlers."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    import pandas as pd


class BaseCrawler(ABC):
    """Base interface for OHLCV data crawlers.

    All crawlers must return a DataFrame with columns:
    ``[timestamp, open, high, low, close, volume]``
    where ``timestamp`` is a timezone-aware UTC datetime.
    """

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from the source.

        Args:
            symbol: Trading pair symbol (e.g. ``"BTC/USDT"``).
            timeframe: Candle timeframe (e.g. ``"1h"``).
            start: Start datetime (UTC).
            end: End datetime (UTC).

        Returns:
            DataFrame with standard OHLCV columns.
        """
        ...

    @abstractmethod
    def fetch_sample(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        max_rows: int,
    ) -> pd.DataFrame:
        """Fetch at most max_rows earliest distinct bars in the inclusive interval.

        Implementations must bound provider requests and parsing before building
        result frames; fetching an entire history and truncating it is not a
        sample implementation. A single timestamp (start == end) is supported.
        Naive bounds are interpreted as UTC. Do not populate ordinary caches.
        """
        ...

    @abstractmethod
    def save_raw(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save raw fetched data to disk.

        Args:
            df: OHLCV DataFrame to save.
            output_path: File path to write to.
        """
        ...
