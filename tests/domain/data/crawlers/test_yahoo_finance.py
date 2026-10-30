"""Tests for the Yahoo Finance crawler (mocked HTTP responses)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingdev.domain.data.crawlers.yahoo_finance import (
    YahooFinanceCrawler,
    _parse_chart_payload,
    _to_yahoo_interval,
)

if TYPE_CHECKING:
    from pathlib import Path


def _chart_payload(
    timestamps: list[int],
    closes: list[float | None],
) -> dict[str, Any]:
    return {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": "AAPL", "timezone": "EST"},
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": [
                                    c - 1 if c is not None else None for c in closes
                                ],
                                "high": [
                                    c + 1 if c is not None else None for c in closes
                                ],
                                "low": [
                                    c - 2 if c is not None else None for c in closes
                                ],
                                "close": closes,
                                "volume": [
                                    1000 if c is not None else None for c in closes
                                ],
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }


def _mock_response(payload: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


class TestToYahooInterval:
    def test_hourly_maps_to_60m(self) -> None:
        assert _to_yahoo_interval("1h") == "60m"

    def test_daily_passthrough(self) -> None:
        assert _to_yahoo_interval("1d") == "1d"

    def test_weekly_alias(self) -> None:
        assert _to_yahoo_interval("1w") == "1wk"

    def test_unsupported_timeframe_raises(self) -> None:
        with pytest.raises(ValueError, match="not supported by yahoo_finance"):
            _to_yahoo_interval("4h")


class TestParseChartPayload:
    def test_parses_standard_payload(self) -> None:
        day = 86_400
        base = 1704153600  # 2024-01-02 00:00 UTC
        payload = _chart_payload(
            [base, base + day, base + 2 * day],
            [100.0, 101.0, 102.0],
        )

        df = _parse_chart_payload(payload)

        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert len(df) == 3
        assert df["timestamp"].dt.tz is not None
        assert df["close"].tolist() == [100.0, 101.0, 102.0]

    def test_drops_rows_with_null_close(self) -> None:
        base = 1704153600
        payload = _chart_payload([base, base + 60], [100.0, None])

        df = _parse_chart_payload(payload)

        assert len(df) == 1

    def test_error_payload_raises(self) -> None:
        payload = {
            "chart": {
                "result": None,
                "error": {"code": "Not Found", "description": "No data found"},
            }
        }

        with pytest.raises(ValueError, match="Yahoo Finance chart error"):
            _parse_chart_payload(payload)

    def test_empty_result_returns_empty_frame(self) -> None:
        payload: dict[str, Any] = {"chart": {"result": [], "error": None}}

        df = _parse_chart_payload(payload)

        assert df.empty


class TestYahooFinanceCrawler:
    def test_fetch_builds_request_and_trims_range(self) -> None:
        day = 86_400
        base = 1704153600  # 2024-01-02 00:00 UTC
        payload = _chart_payload(
            [base - day, base, base + day, base + 10 * day],
            [99.0, 100.0, 101.0, 110.0],
        )
        crawler = YahooFinanceCrawler()

        with patch.object(
            crawler._client, "get", return_value=_mock_response(payload)
        ) as mock_get:
            df = crawler.fetch(
                symbol="AAPL",
                timeframe="1d",
                start=datetime(2024, 1, 2, tzinfo=UTC),
                end=datetime(2024, 1, 5, tzinfo=UTC),
            )

        url = mock_get.call_args.args[0]
        params = mock_get.call_args.kwargs["params"]
        assert url.endswith("/chart/AAPL")
        assert params["interval"] == "1d"
        assert params["period1"] == base
        assert df["close"].tolist() == [100.0, 101.0]

    def test_fetch_clamps_future_end_to_now(self) -> None:
        base = 1704153600
        payload = _chart_payload([base], [100.0])
        crawler = YahooFinanceCrawler()

        with patch.object(
            crawler._client, "get", return_value=_mock_response(payload)
        ) as mock_get:
            crawler.fetch(
                symbol="AAPL",
                timeframe="1d",
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2999, 1, 1, tzinfo=UTC),
            )

        params = mock_get.call_args.kwargs["params"]
        assert params["period2"] <= int(datetime.now(UTC).timestamp()) + 1

    @pytest.mark.parametrize(
        "end",
        [
            "2024-01-02T00:00:01+00:00",
            "2024-01-02T08:00:01+08:00",
            "2024-01-02T00:00:01.500000+00:00",
        ],
    )
    def test_fetch_keeps_the_inclusive_end_with_exclusive_provider(
        self, end: str
    ) -> None:
        base = 1704153600
        crawler = YahooFinanceCrawler()

        def respond(_url: str, *, params: dict[str, Any]) -> MagicMock:
            # Simulate the real API boundary instead of returning the tail
            # regardless of the requested period2.
            stamps = [
                t
                for t in range(base, base + 4)
                if params["period1"] <= t < params["period2"]
            ]
            return _mock_response(_chart_payload(stamps, [100.0] * len(stamps)))

        with patch.object(crawler._client, "get", side_effect=respond):
            frame = crawler.fetch(
                "^GSPC",
                "1m",
                datetime.fromtimestamp(base, UTC),
                datetime.fromisoformat(end),
            )
        assert frame["timestamp"].tolist() == list(
            pd.to_datetime([base, base + 1], unit="s", utc=True)
        )

    def test_save_raw_writes_csv(self, tmp_path: Path) -> None:
        crawler = YahooFinanceCrawler()
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-02", periods=2, freq="D", tz="UTC"),
                "open": [99.0, 100.0],
                "high": [101.0, 102.0],
                "low": [98.0, 99.0],
                "close": [100.0, 101.0],
                "volume": [1000.0, 1100.0],
            }
        )
        output = tmp_path / "raw" / "aapl_1d_2024.csv"

        crawler.save_raw(df, output)

        assert output.exists()
        loaded = pd.read_csv(output)
        assert len(loaded) == 2
