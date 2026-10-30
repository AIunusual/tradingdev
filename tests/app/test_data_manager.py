"""Versioned market query reuse and failure-safe publication."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.data_manager import DataManager
from tradingdev.app.data_service import DataService
from tradingdev.domain.data.crawlers.base import BaseCrawler
from tradingdev.domain.data.processor import DataProcessor
from tradingdev.domain.data.schemas import DataConfig, MarketDataRequest
from tradingdev.domain.research import NewResearchSpec


@pytest.mark.parametrize(
    ("start", "end", "first", "last", "rows"),
    [
        ("2024-01-01", "2024-01-08", "2024-01-01", "2024-01-08", 169),
        (
            "2024-01-01T08:00:00+08:00",
            "2024-01-08T08:00:00+08:00",
            "2024-01-01",
            "2024-01-08",
            169,
        ),
        (
            "2024-01-01T00:30:00Z",
            "2024-01-08T00:30:00Z",
            "2024-01-01T01:00:00",
            "2024-01-08",
            168,
        ),
    ],
    ids=["date-only-midnight", "equivalent-offset", "no-exact-boundary-candle"],
)
def test_research_period_filters_candle_timestamps_inclusively(
    tmp_path: Path, start: str, end: str, first: str, last: str, rows: int
) -> None:
    research = NewResearchSpec.model_validate(
        {
            "strategy_outline": "Buy a rising trend.",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": start,
            "end_date": end,
        }
    )
    manager = DataManager(
        data_config=DataConfig(
            source="binance_vision",
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
        ),
        request=MarketDataRequest(
            symbol=research.symbol,
            timeframe=research.timeframe,
            start_date=research.start_date,
            end_date=research.end_date,
        ),
    )
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2023-12-31", "2024-01-09", freq="h", tz=UTC)}
    )
    with patch.object(manager._crawler, "fetch") as fetch:
        selected = manager._trim_to_range(frame)
    fetch.assert_not_called()
    assert len(selected) == rows
    assert selected["timestamp"].iloc[0] == pd.Timestamp(first, tz=UTC)
    assert selected["timestamp"].iloc[-1] == pd.Timestamp(last, tz=UTC)


def _crawler() -> Mock:
    crawler = Mock(spec=BaseCrawler)

    def fetch(
        symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": pd.date_range(start, end, freq="D"),
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1.0,
            }
        )

    crawler.fetch.side_effect = fetch
    crawler.save_raw.side_effect = lambda frame, path: frame.to_csv(path, index=False)
    return crawler


def _manager(
    tmp_path: Path,
    crawler: Mock,
    now: list[datetime],
    *,
    start: str = "2026-01-01",
    end: str = "2026-01-02",
) -> DataManager:
    return DataManager(
        DataConfig(
            source="binance_api",
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
        ),
        MarketDataRequest(
            symbol="BTC/USDT", timeframe="1d", start_date=start, end_date=end
        ),
        crawler=crawler,
        now_fn=lambda: now[0],
    )


def test_extended_and_still_forming_intervals_refresh_until_settled(
    tmp_path: Path,
) -> None:
    now = [datetime(2026, 1, 2, tzinfo=UTC)]
    crawler = _crawler()
    first, old_path = _manager(tmp_path, crawler, now).load()
    assert first.timestamp.max() == pd.Timestamp("2026-01-02", tz=UTC)
    now[0] = datetime(2026, 1, 3, tzinfo=UTC)
    manager = _manager(tmp_path, crawler, now, end="2026-01-03")
    later, path = manager.load()
    assert later.timestamp.max() == pd.Timestamp("2026-01-03", tz=UTC)
    assert crawler.fetch.call_count == 2 and path != old_path
    assert old_path.is_file()  # Existing readers retain an immutable generation.
    manager.load()
    assert crawler.fetch.call_count == 3  # Even the same recent interval can change.
    now[0] = datetime(2026, 1, 10, tzinfo=UTC)
    manager.load()
    assert crawler.fetch.call_count == 4  # Earlier acquisition was not settled.
    manager.load()
    _manager(tmp_path, crawler, now).load()
    assert crawler.fetch.call_count == 4


@pytest.mark.parametrize(
    "failure", ["fetch", "timeout", "process", "raw", "parquet", "publish"]
)
def test_refresh_failure_preserves_previous_generation_and_cleans_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    now = [datetime(2026, 1, 2, tzinfo=UTC)]
    crawler = _crawler()
    manager = _manager(tmp_path, crawler, now)
    _, path = manager.load()
    receipt = manager.cache.receipt("2026")
    assert receipt is not None
    raw_path = manager.cache.raw_path(receipt)
    before = {
        file: file.read_bytes()
        for file in (path, raw_path, manager.cache.processed_dir / "2026.json")
    }
    files_before = set(tmp_path.rglob("*"))

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise (
            TimeoutError("injected failure")
            if failure == "timeout"
            else OSError("injected failure")
        )

    now[0] = datetime(2026, 1, 3, tzinfo=UTC)
    if failure in {"fetch", "timeout"}:
        crawler.fetch.side_effect = fail
    elif failure == "process":
        monkeypatch.setattr(DataProcessor, "process", fail)
    elif failure == "raw":
        crawler.save_raw.side_effect = fail
    elif failure == "parquet":
        monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    else:
        replace = Path.replace

        def fail_pointer(file: Path, target: Path) -> Path:
            if target.suffix == ".json":
                fail()
            return replace(file, target)

        monkeypatch.setattr(Path, "replace", fail_pointer)
    with pytest.raises(OSError, match="injected failure"):
        manager.load()
    assert {file: file.read_bytes() for file in before} == before
    assert set(tmp_path.rglob("*")) == files_before
    with manager.cache.lock("2026").acquire(timeout=0):
        pass


def test_empty_success_records_query_coverage_and_reuses_history(
    tmp_path: Path,
) -> None:
    crawler = _crawler()
    crawler.fetch.side_effect = None
    crawler.fetch.return_value = pd.DataFrame(
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    now = [datetime(2026, 1, 10, tzinfo=UTC)]
    manager = _manager(tmp_path, crawler, now, start="2025-01-01", end="2025-12-31")
    frame, _ = manager.load()
    assert frame.empty
    receipt = manager.cache.receipt("2025")
    assert receipt is not None and receipt.query_end == datetime(
        2025, 12, 31, 23, 59, 59, 999999, tzinfo=UTC
    )
    manager.load()
    assert crawler.fetch.call_count == 1


def test_year_boundary_refresh_keeps_previous_readable_version(tmp_path: Path) -> None:
    now = [datetime(2025, 12, 31, tzinfo=UTC)]
    crawler = _crawler()
    manager = _manager(tmp_path, crawler, now, start="2025-01-01", end="2025-12-31")
    _, old = manager.load()
    now[0] = datetime(2026, 1, 8, tzinfo=UTC)
    _, latest = manager.load()
    assert latest != old and latest.is_file() and old.is_file()
    manager.load()
    assert crawler.fetch.call_count == 2


def test_future_request_fails_before_provider_or_cache_write(tmp_path: Path) -> None:
    crawler = _crawler()
    manager = _manager(tmp_path, crawler, [datetime(2026, 1, 1, tzinfo=UTC)])
    with pytest.raises(ValueError, match="future"):
        manager.load()
    crawler.fetch.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_cached_only_recent_query_explains_why_immediate_retry_cannot_help(
    tmp_path: Path,
) -> None:
    now = [datetime(2026, 1, 3, tzinfo=UTC)]
    crawler = _crawler()
    manager = _manager(tmp_path, crawler, now)
    service = DataService(WorkspacePaths(tmp_path / "workspace"))
    config = {
        "data": {
            "source": "binance_api",
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
        }
    }
    for _ in range(2):
        manager.load()
        with pytest.raises(ValueError, match="seven days.*wait"):
            service.load_cached(
                symbol="BTC/USDT",
                timeframe="1d",
                start_date="2026-01-01",
                end_date="2026-01-02",
                raw_config=config,
            )
    assert crawler.fetch.call_count == 2


def test_inspection_distinguishes_missing_observations_from_empty_query(
    tmp_path: Path,
) -> None:
    crawler = _crawler()
    crawler.fetch.side_effect = None
    crawler.fetch.return_value = pd.DataFrame(
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    manager = _manager(tmp_path, crawler, [datetime(2026, 1, 10, tzinfo=UTC)])
    service = DataService(WorkspacePaths(tmp_path / "workspace"))
    config = DataConfig(
        source="binance_api",
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
    )
    before = service._inspect_market(manager._request, config)
    assert before["rows"] is None and before["queries"] == []
    manager.load()
    after = service._inspect_market(manager._request, config)
    assert after["rows"] == 0 and len(after["queries"]) == 1


def test_concurrent_requests_do_not_publish_out_of_order_or_lose_coverage(
    tmp_path: Path,
) -> None:
    first, second = _crawler(), _crawler()
    entered, release, second_started = Event(), Event(), Event()
    now = [datetime(2026, 1, 10, tzinfo=UTC)]
    original = first.fetch.side_effect

    def blocked(**kwargs: Any) -> pd.DataFrame:
        entered.set()
        assert release.wait(10)
        result = original(**kwargs)
        assert isinstance(result, pd.DataFrame)
        return result

    first.fetch.side_effect = blocked
    manager = _manager(tmp_path, first, now, end="2026-01-02")
    later = _manager(tmp_path, second, now, end="2026-01-03")

    def load_later() -> tuple[pd.DataFrame, Path]:
        second_started.set()
        return later.load()

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(manager.load)
        try:
            assert entered.wait(10)
            other = pool.submit(load_later)
            assert second_started.wait(10)
        finally:
            release.set()
        _, first_path = pending.result(timeout=10)
        result, second_path = other.result(timeout=10)
    assert first_path == second_path
    assert result.timestamp.max() == pd.Timestamp("2026-01-03", tz=UTC)
    second.fetch.assert_not_called()
    assert not list(tmp_path.rglob(".pending-*"))
