"""Bounded provider sampling with fake CCXT and HTTP transports only."""

import gzip
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import pytest

from tradingdev.domain.data.crawlers.binance_api import BinanceAPICrawler
from tradingdev.domain.data.crawlers.deribit_dvol import DeribitDVOLCrawler
from tradingdev.domain.data.crawlers.sampling import stream_sample_json
from tradingdev.domain.data.crawlers.yahoo_finance import YahooFinanceCrawler

if TYPE_CHECKING:
    from collections.abc import Iterator

_START = datetime(2024, 1, 1, tzinfo=UTC)
_START_MS = int(_START.timestamp() * 1000)


def _candle(timestamp: int) -> list[object]:
    return [timestamp, 10.0, 12.0, 9.0, 11.0, 100.0]


def _chart(
    timestamps: list[int], closes: list[float | None] | None = None
) -> dict[str, Any]:
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": [10.0] * len(timestamps),
                                "high": [12.0] * len(timestamps),
                                "low": [9.0] * len(timestamps),
                                "close": closes
                                if closes is not None
                                else [11.0] * len(timestamps),
                                "volume": [100.0] * len(timestamps),
                                # Non-OHLCV provider fields must never expand the frame.
                                "unrelated": list(range(100)),
                            }
                        ]
                    },
                }
            ],
        }
    }


def _dvol(timestamps: list[int], continuation: int | None = None) -> dict[str, Any]:
    return {
        "result": {
            "data": [[timestamp, 40.0, 42.0, 39.0, 41.0] for timestamp in timestamps],
            "continuation": continuation,
        }
    }


class _TrackedStream(httpx.SyncByteStream):
    def __init__(self, content: bytes, *, timeout: bool = False) -> None:
        self.content = content
        self.timeout = timeout
        self.closed = False
        self.read_started = False

    def __iter__(self) -> Iterator[bytes]:
        self.read_started = True
        yield self.content
        if self.timeout:
            raise httpx.ReadTimeout("simulated stream timeout")

    def close(self) -> None:
        self.closed = True


def test_binance_sample_limit_is_remaining_rows_and_stops_early() -> None:
    exchange = MagicMock(rateLimit=0)
    exchange.fetch_ohlcv.side_effect = [
        [_candle(_START_MS + index * 60_000) for index in range(1000)],
        [_candle(_START_MS + 1000 * 60_000)],
    ]
    with patch(
        "tradingdev.domain.data.crawlers.binance_api.ccxt.binance",
        return_value=exchange,
    ):
        crawler = BinanceAPICrawler()
    with patch.object(
        crawler, "fetch", side_effect=AssertionError("full fetch called")
    ):
        frame = crawler.fetch_sample(
            "BTC/USDT", "1m", _START, _START + timedelta(days=10), max_rows=1001
        )
    assert len(frame) == 1001
    assert [call.kwargs["limit"] for call in exchange.fetch_ohlcv.call_args_list] == [
        1000,
        1,
    ]
    assert frame["timestamp"].is_unique
    assert str(frame["timestamp"].dt.tz) == "UTC"


def test_binance_sample_deduplicates_and_includes_end() -> None:
    exchange = MagicMock(rateLimit=0)
    exchange.fetch_ohlcv.side_effect = [
        [_candle(_START_MS + 60_000), _candle(_START_MS), _candle(_START_MS + 60_000)],
        [_candle(_START_MS + 120_000)],
    ]
    with patch(
        "tradingdev.domain.data.crawlers.binance_api.ccxt.binance",
        return_value=exchange,
    ):
        frame = BinanceAPICrawler().fetch_sample(
            "BTC/USDT", "1m", _START, _START + timedelta(minutes=2), max_rows=3
        )
    assert frame["timestamp"].tolist() == list(
        pd.date_range(_START, periods=3, freq="min")
    )
    assert exchange.fetch_ohlcv.call_args.kwargs["limit"] == 1


def test_binance_sample_single_instant_and_offset_timezone() -> None:
    exchange = MagicMock(rateLimit=0)
    exchange.fetch_ohlcv.return_value = [_candle(_START_MS)]
    local_start = _START.astimezone(timezone(timedelta(hours=8)))
    with patch(
        "tradingdev.domain.data.crawlers.binance_api.ccxt.binance",
        return_value=exchange,
    ):
        frame = BinanceAPICrawler().fetch_sample(
            "BTC/USDT", "1m", local_start, local_start, max_rows=1
        )
    assert frame["timestamp"].tolist() == [_START]
    assert exchange.fetch_ohlcv.call_args.kwargs == {"since": _START_MS, "limit": 1}


def test_binance_sample_rejects_oversize_before_frame_construction() -> None:
    exchange = MagicMock(rateLimit=0)
    exchange.fetch_ohlcv.return_value = [
        _candle(_START_MS),
        _candle(_START_MS + 60_000),
    ]
    with patch(
        "tradingdev.domain.data.crawlers.binance_api.ccxt.binance",
        return_value=exchange,
    ):
        crawler = BinanceAPICrawler()
    with patch("tradingdev.domain.data.crawlers.binance_api.sample_frame") as construct:
        with pytest.raises(ValueError, match="row limit"):
            crawler.fetch_sample("BTC/USDT", "1m", _START, _START, max_rows=1)
        construct.assert_not_called()


def test_binance_sample_empty_and_nonadvancing_pages() -> None:
    exchange = MagicMock(rateLimit=0)
    with patch(
        "tradingdev.domain.data.crawlers.binance_api.ccxt.binance",
        return_value=exchange,
    ):
        crawler = BinanceAPICrawler()
    exchange.fetch_ohlcv.return_value = []
    assert crawler.fetch_sample("BTC/USDT", "1m", _START, _START, max_rows=1).empty
    exchange.fetch_ohlcv.return_value = [_candle(_START_MS - 1)]
    with pytest.raises(ValueError, match="did not advance"):
        crawler.fetch_sample("BTC/USDT", "1m", _START, _START, max_rows=1)


@pytest.mark.parametrize("provider", ["yahoo", "deribit"])
def test_http_sample_zero_length_range_includes_one_candle(provider: str) -> None:
    requests: list[httpx.Request] = []
    responses: list[httpx.Response] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = (
            _chart([_START_MS // 1000]) if provider == "yahoo" else _dvol([_START_MS])
        )
        response = httpx.Response(200, json=body)
        responses.append(response)
        return response

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with patch("httpx.Client", return_value=client):
            crawler = (
                YahooFinanceCrawler() if provider == "yahoo" else DeribitDVOLCrawler()
            )
        local_start = _START.astimezone(timezone(timedelta(hours=-5)))
        with patch.object(
            crawler, "fetch", side_effect=AssertionError("full fetch called")
        ):
            frame = crawler.fetch_sample(
                "BTC", "1d", local_start, local_start, max_rows=1
            )
    assert frame["timestamp"].tolist() == [_START]
    assert str(frame["timestamp"].dt.tz) == "UTC"
    assert len(requests) == 1
    assert responses[0].is_closed
    params = requests[0].url.params
    if provider == "yahoo":
        assert int(params["period2"]) == int(params["period1"]) + 1
    else:
        assert int(params["end_timestamp"]) == _START_MS


def test_yahoo_sample_skips_holiday_and_null_close_windows() -> None:
    timestamps = [
        int(datetime(2024, 1, day, 14, 30, tzinfo=UTC).timestamp()) for day in (1, 2, 3)
    ]
    requests: list[tuple[int, int]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        lower, upper = (int(request.url.params[key]) for key in ("period1", "period2"))
        requests.append((lower, upper))
        rows = [timestamp for timestamp in timestamps if lower <= timestamp < upper]
        closes = [None if timestamp < timestamps[2] else 11.0 for timestamp in rows]
        return httpx.Response(200, json=_chart(rows, closes))

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with patch("httpx.Client", return_value=client):
            crawler = YahooFinanceCrawler()
        frame = crawler.fetch_sample(
            "AAPL", "1d", _START, _START + timedelta(days=7), max_rows=1
        )
    assert frame["timestamp"].tolist() == [
        pd.Timestamp(timestamps[2], unit="s", tz=UTC)
    ]
    assert all(upper - lower <= 23 * 3600 for lower, upper in requests)
    assert all(
        first[1] == second[0]
        for first, second in zip(requests, requests[1:], strict=False)
    )
    assert len(frame.columns) == 6


@pytest.mark.parametrize("interval", ["1d", "1wk", "1mo"])
def test_yahoo_sample_calendar_windows_across_dst_and_leap_month(interval: str) -> None:
    local = ZoneInfo("America/New_York")
    dates = {
        "1d": [datetime(2024, 3, day, tzinfo=local) for day in (9, 10, 11)],
        "1wk": [datetime(2024, 3, day, tzinfo=local) for day in (3, 10, 17)],
        "1mo": [datetime(2024, month, 1, tzinfo=local) for month in (2, 3, 4)],
    }[interval]
    timestamps = [int(date.timestamp()) for date in dates]
    row_counts: list[int] = []

    def handle(request: httpx.Request) -> httpx.Response:
        lower, upper = (int(request.url.params[key]) for key in ("period1", "period2"))
        rows = [timestamp for timestamp in timestamps if lower <= timestamp < upper]
        row_counts.append(len(rows))
        return httpx.Response(200, json=_chart(rows))

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with patch("httpx.Client", return_value=client):
            crawler = YahooFinanceCrawler()
        frame = crawler.fetch_sample("AAPL", interval, dates[0], dates[-1], max_rows=3)
    assert frame["timestamp"].tolist() == [date.astimezone(UTC) for date in dates]
    assert all(count <= 3 for count in row_counts)


@pytest.mark.parametrize("failure", ["rows", "columns", "provider"])
def test_yahoo_sample_rejects_invalid_response_before_frame(failure: str) -> None:
    body = _chart([_START_MS // 1000])
    if failure == "rows":
        body = _chart([_START_MS // 1000] * 2)
    elif failure == "columns":
        body["chart"]["result"][0]["indicators"]["quote"][0]["volume"] = [1, 2]
    else:
        body = {"chart": {"error": "unavailable"}}
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    ) as client:
        with patch("httpx.Client", return_value=client):
            crawler = YahooFinanceCrawler()
        with patch(
            "tradingdev.domain.data.crawlers.yahoo_finance.sample_frame"
        ) as construct:
            with pytest.raises(ValueError):
                crawler.fetch_sample("AAPL", "1d", _START, _START, max_rows=1)
            construct.assert_not_called()


def test_deribit_sample_finishes_reverse_pages_and_keeps_earliest() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200, json=_dvol([_START_MS + 120_000], _START_MS + 60_000)
            )
        return httpx.Response(200, json=_dvol([_START_MS + 60_000, _START_MS]))

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with patch("httpx.Client", return_value=client):
            crawler = DeribitDVOLCrawler()
        with patch("tradingdev.domain.data.crawlers.deribit_dvol.time.sleep"):
            frame = crawler.fetch_sample(
                "BTC", "1m", _START, _START + timedelta(days=1), max_rows=3
            )
    assert frame["timestamp"].tolist() == list(
        pd.date_range(_START, periods=3, freq="min")
    )
    assert len(requests) == 2
    assert int(requests[0].url.params["end_timestamp"]) == _START_MS + 180_000 - 1
    assert int(requests[1].url.params["end_timestamp"]) == _START_MS + 60_000


def test_deribit_sample_empty_windows_and_inclusive_final_midnight() -> None:
    end = _START + timedelta(days=2)
    timestamp = int(end.timestamp() * 1000)
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        lower, upper = (
            int(request.url.params[key]) for key in ("start_timestamp", "end_timestamp")
        )
        return httpx.Response(
            200, json=_dvol([timestamp] if lower <= timestamp <= upper else [])
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with patch("httpx.Client", return_value=client):
            crawler = DeribitDVOLCrawler()
        frame = crawler.fetch_sample("BTC", "1d", _START, end, max_rows=1)
    assert frame["timestamp"].tolist() == [end]
    assert len(requests) == 3


def test_deribit_sample_sparse_targets_skip_unrelated_windows() -> None:
    end = _START + timedelta(days=300)
    wanted = pd.DatetimeIndex(
        [_START, end, pd.Timestamp(end) + pd.Timedelta(nanoseconds=1)]
    )
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        lower = int(request.url.params["start_timestamp"])
        return httpx.Response(200, json=_dvol([lower, lower + 60_000]))

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with patch("httpx.Client", return_value=client):
            crawler = DeribitDVOLCrawler()
        frame = crawler.fetch_sample(
            "BTC", "1m", _START, end, max_rows=3, timestamps=wanted
        )
    assert frame["timestamp"].tolist() == [_START, end]
    assert len(requests) == 2
    assert int(requests[1].url.params["start_timestamp"]) == int(end.timestamp() * 1000)


def test_deribit_sample_rejects_accumulated_window_overflow() -> None:
    pages = iter(
        [
            _dvol([_START_MS + 60_000, _START_MS + 100_000], _START_MS + 59_999),
            _dvol([_START_MS]),
        ]
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=next(pages)))
    ) as client:
        with patch("httpx.Client", return_value=client):
            crawler = DeribitDVOLCrawler()
        with (
            patch("tradingdev.domain.data.crawlers.deribit_dvol.time.sleep"),
            pytest.raises(ValueError, match="window exceeds"),
        ):
            crawler.fetch_sample(
                "BTC", "1m", _START, _START + timedelta(minutes=2), max_rows=2
            )


@pytest.mark.parametrize("provider", ["binance", "yahoo", "deribit"])
def test_sample_fractional_instant_does_not_round_outside_range(provider: str) -> None:
    instant = pd.Timestamp(_START) + pd.Timedelta(nanoseconds=1)
    with (
        patch("httpx.Client") as client,
        patch("tradingdev.domain.data.crawlers.binance_api.ccxt.binance") as exchange,
    ):
        crawler = {
            "binance": BinanceAPICrawler,
            "yahoo": YahooFinanceCrawler,
            "deribit": DeribitDVOLCrawler,
        }[provider]()
        frame = crawler.fetch_sample("BTC", "1m", instant, instant, max_rows=1)
    assert frame.empty
    assert str(frame["timestamp"].dt.tz) == "UTC"
    client.return_value.stream.assert_not_called()
    exchange.return_value.fetch_ohlcv.assert_not_called()


@pytest.mark.parametrize("provider", ["yahoo", "deribit"])
def test_sample_unsupported_timeframe_does_not_request(provider: str) -> None:
    with patch("httpx.Client") as client:
        crawler = YahooFinanceCrawler() if provider == "yahoo" else DeribitDVOLCrawler()
        with pytest.raises(ValueError, match="[Uu]nsupported|not supported"):
            crawler.fetch_sample("BTC", "4h", _START, _START, max_rows=1)
    client.return_value.stream.assert_not_called()


@pytest.mark.parametrize("failure", ["rows", "continuation", "provider"])
def test_deribit_sample_invalid_response(failure: str) -> None:
    body = _dvol([_START_MS])
    if failure == "rows":
        body = _dvol([_START_MS, _START_MS + 1])
    elif failure == "continuation":
        body = _dvol([_START_MS], _START_MS)
    else:
        body = {"error": {"message": "provider unavailable"}}
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    ) as client:
        with patch("httpx.Client", return_value=client):
            crawler = DeribitDVOLCrawler()
        with patch(
            "tradingdev.domain.data.crawlers.deribit_dvol.sample_frame"
        ) as construct:
            with pytest.raises(ValueError):
                crawler.fetch_sample("BTC", "1m", _START, _START, max_rows=1)
            construct.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["success", "bytes", "json", "http", "timeout", "compressed"]
)
def test_sample_stream_closes_success_failure_and_timeout(failure: str) -> None:
    content = b'{"result": {}}'
    if failure == "bytes":
        content = b" " * 129
    elif failure == "json":
        content = b"not-json"
    stream = _TrackedStream(content, timeout=failure == "timeout")
    response = httpx.Response(
        503 if failure == "http" else 200,
        stream=stream,
        headers={"Content-Encoding": "gzip"} if failure == "compressed" else {},
    )
    with (
        httpx.Client(transport=httpx.MockTransport(lambda _: response)) as client,
        patch("tradingdev.domain.data.crawlers.sampling.SAMPLE_RESPONSE_BYTES", 128),
    ):
        if failure == "success":
            assert stream_sample_json(client, "https://example.test/sample", {}) == {
                "result": {}
            }
        else:
            with pytest.raises((ValueError, httpx.HTTPError)):
                stream_sample_json(client, "https://example.test/sample", {})
        assert stream.closed
        assert response.is_closed
        assert response.request.headers["Accept-Encoding"] == "identity"


@pytest.mark.parametrize("status", [302, 307])
@pytest.mark.parametrize("compressed", [False, True])
def test_sample_stream_rejects_redirect_without_reading_body(
    status: int, compressed: bool
) -> None:
    content = b" " * 640
    stream = _TrackedStream(gzip.compress(content) if compressed else content)
    headers = {"Location": "https://example.test/followed"}
    if compressed:
        headers["Content-Encoding"] = "gzip"
    response = httpx.Response(status, stream=stream, headers=headers)
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/sample":
            return response
        return httpx.Response(200, json={"result": {}})

    with (
        httpx.Client(
            transport=httpx.MockTransport(handle), follow_redirects=True
        ) as client,
        patch("tradingdev.domain.data.crawlers.sampling.SAMPLE_RESPONSE_BYTES", 128),
    ):
        with pytest.raises(httpx.HTTPStatusError) as error:
            stream_sample_json(client, "https://example.test/sample", {})
        assert error.value.response.status_code == status
        assert len(requests) == 1
        assert not stream.read_started
        assert not response.is_stream_consumed
        assert stream.closed
        assert response.is_closed


@pytest.mark.parametrize("provider", ["binance", "yahoo", "deribit"])
def test_sample_rejects_invalid_limits_without_network(provider: str) -> None:
    with (
        patch("httpx.Client") as client,
        patch("tradingdev.domain.data.crawlers.binance_api.ccxt.binance") as exchange,
    ):
        crawler = {
            "binance": BinanceAPICrawler,
            "yahoo": YahooFinanceCrawler,
            "deribit": DeribitDVOLCrawler,
        }[provider]()
        for limit in (0, -1):
            with pytest.raises(ValueError, match="positive integer"):
                crawler.fetch_sample("BTC", "1m", _START, _START, max_rows=limit)
        with pytest.raises(ValueError, match="after end"):
            crawler.fetch_sample(
                "BTC", "1m", _START + timedelta(days=1), _START, max_rows=1
            )
    client.return_value.stream.assert_not_called()
    exchange.return_value.fetch_ohlcv.assert_not_called()
