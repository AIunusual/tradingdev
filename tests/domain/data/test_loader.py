"""Tests for the data loader."""

from typing import TYPE_CHECKING

from tradingdev.domain.data.loader import DataLoader

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


class TestDataLoader:
    def setup_method(self) -> None:
        self.loader = DataLoader()

    def test_parquet_round_trip(
        self, sample_ohlcv_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = tmp_path / "test.parquet"
        sample_ohlcv_df.to_parquet(path, index=False)

        result = self.loader.load_parquet(path)
        assert len(result) == len(sample_ohlcv_df)
        assert result["timestamp"].dt.tz is not None

    def test_csv_round_trip(
        self, sample_ohlcv_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = tmp_path / "test.csv"
        sample_ohlcv_df.to_csv(path, index=False)

        result = self.loader.load_csv(path)
        assert len(result) == len(sample_ohlcv_df)
        assert result["timestamp"].dt.tz is not None
