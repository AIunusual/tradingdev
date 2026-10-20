"""Tests for the Deribit DVOL crawler (using mocked httpx)."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingdev.domain.data.crawlers.deribit_dvol import DeribitDVOLCrawler

if TYPE_CHECKING:
    from pathlib import Path


class TestDeribitDVOLCrawler:
    def _make_mock_response(
        self,
        start_ms: int,
        count: int,
        interval_ms: int = 60_000,
        continuation: int | None = None,
    ) -> dict[str, object]:
        """Generate a mock Deribit API response."""
        data: list[list[float]] = []
        for i in range(count):
            ts = float(start_ms + i * interval_ms)
            data.append([ts, 45.0, 46.0, 44.0, 45.5])
        return {
            "result": {
                "data": data,
                "continuation": continuation,
            },
        }

    @patch("tradingdev.domain.data.crawlers.deribit_dvol.httpx.Client")
    def test_fetch_returns_dataframe(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 1, 0, 5, tzinfo=UTC)
        start_ms = int(start.timestamp() * 1000)

        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_mock_response(start_ms, 5)
        mock_client.get.return_value = mock_resp

        crawler = DeribitDVOLCrawler()
        df = crawler.fetch("BTC", "1m", start, end)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == [
            "timestamp",
            "dvol_open",
            "dvol_high",
            "dvol_low",
            "dvol_close",
        ]

    @patch("tradingdev.domain.data.crawlers.deribit_dvol.httpx.Client")
    def test_fetch_timestamps_are_utc(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 1, 0, 3, tzinfo=UTC)
        start_ms = int(start.timestamp() * 1000)

        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_mock_response(start_ms, 3)
        mock_client.get.return_value = mock_resp

        crawler = DeribitDVOLCrawler()
        df = crawler.fetch("BTC", "1m", start, end)

        assert df["timestamp"].dt.tz is not None

    @patch("tradingdev.domain.data.crawlers.deribit_dvol.httpx.Client")
    @patch("tradingdev.domain.data.crawlers.deribit_dvol.time.sleep")
    def test_fetch_pagination(
        self,
        mock_sleep: MagicMock,
        mock_client_cls: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 1, 0, 6, tzinfo=UTC)
        start_ms = int(start.timestamp() * 1000)

        # First page: 3 candles with continuation
        page1 = self._make_mock_response(
            start_ms, 3, continuation=start_ms + 3 * 60_000
        )
        # Second page: 3 more candles, no continuation
        page2 = self._make_mock_response(start_ms + 3 * 60_000, 3)

        mock_resp1 = MagicMock()
        mock_resp1.json.return_value = page1
        mock_resp2 = MagicMock()
        mock_resp2.json.return_value = page2

        mock_client.get.side_effect = [mock_resp1, mock_resp2]

        crawler = DeribitDVOLCrawler()
        df = crawler.fetch("BTC", "1m", start, end)

        assert len(df) == 6
        assert mock_client.get.call_count == 2
        mock_sleep.assert_called_once()

    @patch("tradingdev.domain.data.crawlers.deribit_dvol.httpx.Client")
    def test_fetch_empty_response(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": {"data": [], "continuation": None}}
        mock_client.get.return_value = mock_resp

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 1, 0, 5, tzinfo=UTC)

        crawler = DeribitDVOLCrawler()
        df = crawler.fetch("BTC", "1m", start, end)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_fetch_unsupported_timeframe(self) -> None:
        crawler = DeribitDVOLCrawler()
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            crawler.fetch("BTC", "5m", start, end)

    @patch("tradingdev.domain.data.crawlers.deribit_dvol.httpx.Client")
    def test_save_raw_creates_csv(
        self, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_client_cls.return_value = MagicMock()

        crawler = DeribitDVOLCrawler()
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01",
                    periods=3,
                    freq="min",
                    tz=UTC,
                ),
                "dvol_open": [45.0, 46.0, 44.0],
                "dvol_high": [46.0, 47.0, 45.0],
                "dvol_low": [44.0, 45.0, 43.0],
                "dvol_close": [45.5, 46.5, 44.5],
            }
        )

        output = tmp_path / "test_dvol.csv"
        crawler.save_raw(df, output)
        assert output.exists()

    @patch("tradingdev.domain.data.crawlers.deribit_dvol.httpx.Client")
    def test_fetch_deduplicates_timestamps(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 1, 0, 3, tzinfo=UTC)
        start_ms = int(start.timestamp() * 1000)

        # Response with duplicate timestamps from overlapping pages
        data: list[list[float]] = [
            [float(start_ms), 45.0, 46.0, 44.0, 45.5],
            [float(start_ms + 60_000), 45.0, 46.0, 44.0, 45.5],
            [float(start_ms + 60_000), 45.0, 46.0, 44.0, 45.5],
            [float(start_ms + 120_000), 45.0, 46.0, 44.0, 45.5],
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": {"data": data, "continuation": None}}
        mock_client.get.return_value = mock_resp

        crawler = DeribitDVOLCrawler()
        df = crawler.fetch("BTC", "1m", start, end)

        assert len(df) == 3
