"""Data cleaning and transformation pipeline."""

from typing import TYPE_CHECKING

from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

logger = setup_logger(__name__)


class DataProcessor:
    """Clean and transform raw OHLCV data for backtesting."""

    def process(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw OHLCV data.

        Steps:
            1. Sort by timestamp.
            2. Remove duplicate timestamps.
            3. Ensure UTC timezone on timestamp column.
            4. Forward-fill missing OHLCV values; zero-fill volume.
            5. Detect and log gaps in the time series.

        Args:
            raw_df: Raw OHLCV DataFrame.

        Returns:
            Cleaned DataFrame ready for backtesting.
        """
        df = raw_df.copy()

        # 1. Sort
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 2. Remove duplicates
        n_before = len(df)
        df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(
            drop=True
        )
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.warning("Removed %d duplicate timestamps", n_dropped)

        # 3. Ensure UTC
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

        # 4. Handle missing values
        ohlc_cols = ["open", "high", "low", "close"]
        n_missing = df[ohlc_cols].isna().sum().sum()
        if n_missing > 0:
            logger.warning("Forward-filling %d missing OHLC values", n_missing)
            df[ohlc_cols] = df[ohlc_cols].ffill()

        n_missing_vol = df["volume"].isna().sum()
        if n_missing_vol > 0:
            logger.warning("Zero-filling %d missing volume values", n_missing_vol)
            df["volume"] = df["volume"].fillna(0.0)

        # 5. Detect gaps
        if len(df) > 1:
            time_diffs = df["timestamp"].diff().dropna()
            median_diff = time_diffs.median()
            gaps = time_diffs[time_diffs > median_diff * 1.5]
            if len(gaps) > 0:
                logger.warning(
                    "Detected %d time gaps (expected interval: %s)",
                    len(gaps),
                    median_diff,
                )

        logger.info("Processed data: %d rows", len(df))
        return df

    def save_processed(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save cleaned data as Parquet.

        Args:
            df: Processed OHLCV DataFrame.
            output_path: Path to write Parquet file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info("Saved processed data to %s", output_path)
