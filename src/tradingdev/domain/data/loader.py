"""Data loading utilities for CSV and Parquet files."""

from typing import TYPE_CHECKING

import pandas as pd

from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logger(__name__)


class DataLoader:
    """Load OHLCV data from local files."""

    def load_parquet(self, path: Path) -> pd.DataFrame:
        """Load a Parquet file and ensure proper dtypes.

        Args:
            path: Path to the Parquet file.

        Returns:
            DataFrame with UTC-aware timestamp column.
        """
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        logger.info("Loaded %d rows from %s", len(df), path)
        return df

    def load_csv(self, path: Path) -> pd.DataFrame:
        """Load a CSV file containing raw OHLCV data.

        Args:
            path: Path to the CSV file.

        Returns:
            DataFrame with parsed timestamp column.
        """
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        logger.info("Loaded %d rows from %s", len(df), path)
        return df
