"""Tests for the Binance API crawler (using mocked ccxt)."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pandas as pd

from tradingdev.domain.data.crawlers.binance_api import BinanceAPICrawler

if TYPE_CHECKING:
    from pathlib import Path


class TestBinanceAPICrawler:
    def _make_mock_candles(
        self, start_ms: int, count: int, interval_ms: int = 3_600_000
    ) -> list[list[object]]:
        """Generate mock OHLCV candle data."""
        candles: list[list[object]] = []
        for i in range(count):
            ts = start_ms + i * interval_ms
            candles.append([ts, 40000.0, 40100.0, 39900.0, 40050.0, 500.0])
        return candles

    @patch("tradingdev.domain.data.crawlers.binance_api.ccxt.binance")
    def test_fetch_returns_dataframe(self, mock_binance_cls: MagicMock) -> None:
        mock_exchange = MagicMock()
        mock_exchange.rateLimit = 100
        mock_binance_cls.return_value = mock_exchange

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 1, 3, tzinfo=UTC)
        start_ms = int(start.timestamp() * 1000)

        mock_exchange.fetch_ohlcv.return_value = self._make_mock_candles(start_ms, 3)

        crawler = BinanceAPICrawler()
        df = crawler.fetch("BTC/USDT", "1h", start, end)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    @patch("tradingdev.domain.data.crawlers.binance_api.ccxt.binance")
    def test_fetch_timestamps_are_utc(self, mock_binance_cls: MagicMock) -> None:
        mock_exchange = MagicMock()
        mock_exchange.rateLimit = 100
        mock_binance_cls.return_value = mock_exchange

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 1, 2, tzinfo=UTC)
        start_ms = int(start.timestamp() * 1000)

        mock_exchange.fetch_ohlcv.return_value = self._make_mock_candles(start_ms, 2)

        crawler = BinanceAPICrawler()
        df = crawler.fetch("BTC/USDT", "1h", start, end)

        assert df["timestamp"].dt.tz is not None

    @patch("tradingdev.domain.data.crawlers.binance_api.ccxt.binance")
    def test_save_raw_creates_csv(
        self, mock_binance_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_binance_cls.return_value = MagicMock()

        crawler = BinanceAPICrawler()
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz=UTC),
                "open": [1.0, 2.0, 3.0],
                "high": [1.0, 2.0, 3.0],
                "low": [1.0, 2.0, 3.0],
                "close": [1.0, 2.0, 3.0],
                "volume": [100.0, 200.0, 300.0],
            }
        )

        output = tmp_path / "test.csv"
        crawler.save_raw(df, output)
        assert output.exists()
