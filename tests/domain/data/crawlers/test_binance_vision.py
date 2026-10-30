"""Tests for the Binance Vision crawler (mocked HTTP responses)."""

from __future__ import annotations

import io
import re
import zipfile
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from tradingdev.domain.data.crawlers.binance_vision import (
    BinanceVisionCrawler,
    _month_range,
    _parse_zip_csv,
    _to_vision_symbol,
)
from tradingdev.domain.data.data_manager import DataManager
from tradingdev.domain.data.schemas import DataConfig, MarketDataRequest

if TYPE_CHECKING:
    from pathlib import Path

# ------------------------------------------------------------------ #
# Helper utilities                                                    #
# ------------------------------------------------------------------ #


def _make_zip_csv(
    rows: list[tuple[object, ...]], *, include_header: bool = False
) -> bytes:
    """Build a fake Binance Vision ZIP with a single K-line CSV."""
    lines = [",".join(str(v) for v in row) for row in rows]
    if include_header:
        header = "open_time,open,high,low,close,volume,close_time,quote_volume,count"
        lines = [header] + lines
    csv_bytes = "\n".join(lines).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BTCUSDT-1d-2024-01.csv", csv_bytes)
    return buf.getvalue()


def _candle_row(ts_ms: int) -> tuple[object, ...]:
    return (
        ts_ms,
        40000.0,
        40100.0,
        39900.0,
        40050.0,
        500.0,
        ts_ms + 86_399_999,
        1.0,
        10,
        1.0,
        1.0,
        0,
    )


def _jan_2024_zip() -> bytes:
    rows = [_candle_row(1704067200000 + i * 86_400_000) for i in range(31)]
    return _make_zip_csv(rows)


def _make_mock_response(content: bytes, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


def test_monthly_transport_failure_is_not_published_as_complete_cache(
    tmp_path: Path,
) -> None:
    outage = True
    months: list[int] = []

    def respond(request: httpx.Request) -> httpx.Response:
        match = re.search(r"2024-(\d{2})\.zip$", request.url.path)
        assert match is not None
        month = int(match[1])
        months.append(month)
        if month == 2 and outage:
            raise httpx.ReadTimeout("February unavailable", request=request)
        timestamp = int(datetime(2024, month, 1, tzinfo=UTC).timestamp() * 1000)
        return httpx.Response(200, content=_make_zip_csv([_candle_row(timestamp)]))

    crawler = BinanceVisionCrawler()
    crawler._client.close()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        crawler._client = client
        manager = DataManager(
            DataConfig(
                raw_dir=str(tmp_path / "raw"), processed_dir=str(tmp_path / "processed")
            ),
            MarketDataRequest(
                symbol="BTC/USDT",
                timeframe="1d",
                start_date="2024-01-01",
                end_date="2024-12-31",
            ),
            crawler=crawler,
            now_fn=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        )
        with pytest.raises(httpx.ReadTimeout, match="February unavailable"):
            manager.load()
        assert not list(tmp_path.rglob("*.parquet"))
        outage = False
        months.clear()
        frame, path = manager.load()
        assert frame.timestamp.dt.month.tolist() == list(range(1, 13))
        assert months == list(range(1, 13))
        assert path.is_file()


def test_daily_transport_failure_propagates_but_404_remains_no_data() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("-01.zip"):
            return httpx.Response(404)
        raise httpx.ConnectTimeout("daily outage", request=request)

    crawler = BinanceVisionCrawler()
    crawler._client.close()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        crawler._client = client
        with patch("tradingdev.domain.data.crawlers.binance_vision.datetime") as clock:
            clock.now.return_value = datetime(2026, 1, 3, tzinfo=UTC)
            with pytest.raises(httpx.ConnectTimeout, match="daily outage"):
                crawler.fetch(
                    "BTC/USDT",
                    "1d",
                    datetime(2026, 1, 1, tzinfo=UTC),
                    datetime(2026, 1, 2, tzinfo=UTC),
                )


# ------------------------------------------------------------------ #
# Pure utility tests                                                  #
# ------------------------------------------------------------------ #


class TestToVisionSymbol:
    def test_slash_removed(self) -> None:
        assert _to_vision_symbol("BTC/USDT") == "BTCUSDT"

    def test_already_no_slash(self) -> None:
        assert _to_vision_symbol("ETHUSDT") == "ETHUSDT"

    def test_uppercased(self) -> None:
        assert _to_vision_symbol("btc/usdt") == "BTCUSDT"


class TestMonthRange:
    def test_single_month(self) -> None:
        start = datetime(2024, 3, 5, tzinfo=UTC)
        end = datetime(2024, 3, 28, tzinfo=UTC)
        assert _month_range(start, end) == [(2024, 3)]

    def test_two_months(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 2, 28, tzinfo=UTC)
        assert _month_range(start, end) == [(2024, 1), (2024, 2)]

    def test_year_boundary(self) -> None:
        start = datetime(2024, 11, 1, tzinfo=UTC)
        end = datetime(2025, 2, 1, tzinfo=UTC)
        assert _month_range(start, end) == [
            (2024, 11),
            (2024, 12),
            (2025, 1),
            (2025, 2),
        ]


class TestParseZipCsv:
    def test_columns_and_types(self) -> None:
        ts_ms = 1704067200000
        content = _make_zip_csv([_candle_row(ts_ms)])
        df = _parse_zip_csv(content)
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert df["timestamp"].dt.tz is not None
        assert df["open"].dtype == float

    def test_timestamp_value(self) -> None:
        ts_ms = 1704067200000
        content = _make_zip_csv([_candle_row(ts_ms)])
        df = _parse_zip_csv(content)
        expected = pd.Timestamp("2024-01-01", tz=UTC)
        assert df["timestamp"].iloc[0] == expected

    def test_csv_with_header_row(self) -> None:
        # Newer Binance Vision CSVs include a header row starting with "open_time"
        ts_ms = 1704067200000
        content = _make_zip_csv([_candle_row(ts_ms)], include_header=True)
        df = _parse_zip_csv(content)
        assert len(df) == 1
        assert df["timestamp"].iloc[0] == pd.Timestamp("2024-01-01", tz=UTC)
        assert df["close"].iloc[0] == 40050.0


# ------------------------------------------------------------------ #
# BinanceVisionCrawler integration tests (mocked HTTP)               #
# ------------------------------------------------------------------ #


class TestBinanceVisionCrawler:
    @pytest.fixture
    def crawler(self) -> BinanceVisionCrawler:
        return BinanceVisionCrawler(market_type="futures/um")

    def test_fetch_complete_past_month_uses_monthly_url(
        self, crawler: BinanceVisionCrawler
    ) -> None:
        zip_content = _jan_2024_zip()
        mock_resp = _make_mock_response(zip_content)

        with (
            patch.object(crawler._client, "get", return_value=mock_resp) as mock_get,
            patch("tradingdev.domain.data.crawlers.binance_vision.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = datetime(2026, 5, 1, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            df = crawler.fetch(
                "BTC/USDT",
                "1d",
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 31, tzinfo=UTC),
            )

        called_url = mock_get.call_args[0][0]
        assert "monthly" in called_url
        assert "BTCUSDT-1d-2024-01.zip" in called_url
        assert len(df) == 31
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    def test_fetch_filters_to_requested_range(
        self, crawler: BinanceVisionCrawler
    ) -> None:
        zip_content = _jan_2024_zip()
        mock_resp = _make_mock_response(zip_content)

        with (
            patch.object(crawler._client, "get", return_value=mock_resp),
            patch("tradingdev.domain.data.crawlers.binance_vision.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = datetime(2026, 5, 1, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            df = crawler.fetch(
                "BTC/USDT",
                "1d",
                datetime(2024, 1, 5, tzinfo=UTC),
                datetime(2024, 1, 10, tzinfo=UTC),
            )

        assert df["timestamp"].min() >= pd.Timestamp("2024-01-05", tz=UTC)
        assert df["timestamp"].max() <= pd.Timestamp("2024-01-10", tz=UTC)

    def test_fetch_404_returns_empty_dataframe(
        self, crawler: BinanceVisionCrawler
    ) -> None:
        mock_resp = _make_mock_response(b"", status_code=404)

        with (
            patch.object(crawler._client, "get", return_value=mock_resp),
            patch("tradingdev.domain.data.crawlers.binance_vision.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = datetime(2026, 5, 1, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            df = crawler.fetch(
                "BTC/USDT",
                "1d",
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 31, tzinfo=UTC),
            )

        assert df.empty
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    def test_market_type_in_url(self, crawler: BinanceVisionCrawler) -> None:
        zip_content = _jan_2024_zip()
        mock_resp = _make_mock_response(zip_content)

        with (
            patch.object(crawler._client, "get", return_value=mock_resp) as mock_get,
            patch("tradingdev.domain.data.crawlers.binance_vision.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = datetime(2026, 5, 1, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            crawler.fetch(
                "BTC/USDT",
                "1d",
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 31, tzinfo=UTC),
            )

        called_url = mock_get.call_args[0][0]
        assert "futures/um" in called_url

    def test_spot_market_type_in_url(self) -> None:
        spot_crawler = BinanceVisionCrawler(market_type="spot")
        zip_content = _jan_2024_zip()
        mock_resp = _make_mock_response(zip_content)

        with (
            patch.object(
                spot_crawler._client, "get", return_value=mock_resp
            ) as mock_get,
            patch("tradingdev.domain.data.crawlers.binance_vision.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = datetime(2026, 5, 1, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            spot_crawler.fetch(
                "BTC/USDT",
                "1d",
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 31, tzinfo=UTC),
            )

        called_url = mock_get.call_args[0][0]
        assert "spot" in called_url

    def test_save_raw_creates_csv(
        self, crawler: BinanceVisionCrawler, tmp_path: Path
    ) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz=UTC),
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
