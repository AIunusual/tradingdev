"""Regression coverage for calendar gaps, timestamp storage, and sample limits."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import pytest
from tests.data_cache import cached_market_path, seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.data_service import DataService
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.data.crawlers.binance_api import BinanceAPICrawler

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path


def _frame(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            **dict.fromkeys(("open", "high", "low", "close", "volume"), 100.0),
        }
    )


def _service(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    frame: pd.DataFrame,
    *,
    timeframe: str,
    start: str,
    end: str,
    cached: bool,
) -> tuple[DataService, BacktestConfig, list[tuple[datetime, datetime]]]:
    workspace = WorkspacePaths(root / "workspace")
    service = DataService(workspace)
    calls: list[tuple[datetime, datetime]] = []

    class Crawler:
        def fetch(
            self, _symbol: str, _timeframe: str, begin: datetime, stop: datetime
        ) -> pd.DataFrame:
            if cached:
                pytest.fail("A complete yearly cache must avoid network requests")
            calls.append((begin, stop))
            normalized = frame.assign(
                timestamp=pd.to_datetime(frame["timestamp"], utc=True)
            )
            result = normalized.loc[normalized["timestamp"].between(begin, stop)].copy()
            # Providers return untyped empty frames on non-trading windows.
            return result if not result.empty else pd.DataFrame(columns=frame.columns)

        def fetch_sample(
            self,
            symbol: str,
            timeframe: str,
            begin: datetime,
            stop: datetime,
            *,
            max_rows: int,
        ) -> pd.DataFrame:
            return self.fetch(symbol, timeframe, begin, stop).head(max_rows)

    monkeypatch.setattr(
        "tradingdev.app.data_service.create_crawler", lambda *_: Crawler()
    )
    if cached:
        for year in range(int(start[:4]), int(end[:4]) + 1):
            selected = frame.loc[
                pd.to_datetime(frame["timestamp"], utc=True).dt.year == year
            ]
            seed_market_cache(
                workspace.processed_data,
                selected,
                symbol="AAPL",
                timeframe=timeframe,
                year=year,
            )
    config = BacktestConfig(
        symbol="AAPL",
        timeframe=timeframe,
        start_date=start,
        end_date=end,
        init_cash=10000,
    )
    return service, config, calls


@pytest.mark.parametrize("max_rows", [1024, 4096])
@pytest.mark.parametrize("cached", [False, True])
def test_monthly_sample_clips_to_requested_dates_before_arithmetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_rows: int, cached: bool
) -> None:
    frame = _frame(pd.date_range("2024-01-01", periods=24, freq="MS", tz="UTC"))
    service, config, calls = _service(
        tmp_path,
        monkeypatch,
        frame,
        timeframe="1mo",
        start="2024-01-01",
        end="2025-12-31",
        cached=cached,
    )
    result = service.load_sample(
        {}, config, max_rows=max_rows, output_dir=tmp_path / "sample"
    )
    pd.testing.assert_frame_equal(result.frame.reset_index(drop=True), frame)
    if not cached:
        assert len(calls) == 2
        assert calls[0][0] == pd.Timestamp("2024-01-01", tz="UTC")
        assert calls[-1][1] == pd.Timestamp("2025-12-31", tz="UTC")


@pytest.mark.parametrize("timezone", [None, "UTC", "America/New_York"])
def test_sample_normalizes_cached_timestamps_like_regular_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timezone: str | None
) -> None:
    timestamps = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    stored = (
        timestamps.tz_localize(None)
        if timezone is None
        else timestamps.tz_convert(timezone)
    )
    frame = _frame(stored)
    service, config, _ = _service(
        tmp_path,
        monkeypatch,
        frame,
        timeframe="1h",
        start="2024-01-01",
        end="2024-01-08",
        cached=True,
    )
    original = service.load({}, config).frame
    result = service.load_sample(
        {}, config, max_rows=64, output_dir=tmp_path / "sample"
    )
    pd.testing.assert_frame_equal(
        result.frame.reset_index(drop=True), original.head(64).reset_index(drop=True)
    )


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("max_rows", [64, 1024])
def test_sample_searches_past_holiday_without_expanding_each_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached: bool, max_rows: int
) -> None:
    frame = _frame(pd.date_range("2023-01-03 14:30", periods=390, freq="min", tz="UTC"))
    service, config, calls = _service(
        tmp_path,
        monkeypatch,
        frame,
        timeframe="1m",
        start="2023-01-01",
        end="2023-01-08",
        cached=cached,
    )
    result = service.load_sample(
        {}, config, max_rows=max_rows, output_dir=tmp_path / "sample"
    )
    pd.testing.assert_frame_equal(
        result.frame.reset_index(drop=True), frame.head(max_rows)
    )
    assert all(
        (end - start).total_seconds() <= max_rows * 3 * 60 for start, end in calls
    )
    assert all(
        pd.Timestamp("2023-01-01", tz="UTC")
        <= start
        <= end
        <= pd.Timestamp("2023-01-08", tz="UTC")
        for start, end in calls
    )
    if not cached:
        assert len(calls) > 1


def test_empty_provider_windows_exhaust_interval_with_clear_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _frame(pd.DatetimeIndex([], tz="UTC"))
    service, config, calls = _service(
        tmp_path,
        monkeypatch,
        frame,
        timeframe="1m",
        start="2024-01-01",
        end="2024-01-02",
        cached=False,
    )
    with pytest.raises(ValueError, match="No historical bars"):
        service.load_sample({}, config, max_rows=64, output_dir=tmp_path / "sample")
    assert calls[-1][1] == pd.Timestamp("2024-01-02", tz="UTC")
    assert len(calls) < 10
    assert not (tmp_path / "sample").exists()


def test_samples_accumulate_across_windows_and_deduplicate_boundary_bars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamps = pd.DatetimeIndex(
        [
            pd.Timestamp("2023-12-31 03:12", tz="UTC"),
            *pd.date_range("2023-12-31 06:23", periods=90, freq="min", tz="UTC"),
        ]
    )
    frame = _frame(timestamps)
    service, config, calls = _service(
        tmp_path,
        monkeypatch,
        frame,
        timeframe="1m",
        start="2023-12-31",
        end="2024-01-02",
        cached=False,
    )
    result = service.load_sample(
        {}, config, max_rows=64, output_dir=tmp_path / "sample"
    )
    pd.testing.assert_frame_equal(result.frame.reset_index(drop=True), frame.head(64))
    assert len(calls) >= 3
    assert result.frame["timestamp"].is_unique


def test_provider_errors_are_not_treated_as_closed_market(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _frame(pd.DatetimeIndex([], tz="UTC"))
    service, config, _ = _service(
        tmp_path,
        monkeypatch,
        frame,
        timeframe="1m",
        start="2024-01-01",
        end="2024-01-02",
        cached=False,
    )

    class BrokenCrawler:
        def fetch_sample(self, *_args: Any, **_kwargs: Any) -> pd.DataFrame:
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "tradingdev.app.data_service.create_crawler", lambda *_: BrokenCrawler()
    )
    with pytest.raises(RuntimeError, match="provider unavailable"):
        service.load_sample({}, config, max_rows=64, output_dir=tmp_path / "sample")


def test_partial_cache_does_not_hide_available_provider_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _frame(pd.date_range("2024-02-01", periods=100, freq="h", tz="UTC"))
    service, config, calls = _service(
        tmp_path,
        monkeypatch,
        frame,
        timeframe="1h",
        start="2024-02-01",
        end="2024-02-10",
        cached=False,
    )
    cache = tmp_path / "workspace/data/processed/aapl_1h_2024_partial.parquet"
    _frame(pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")).to_parquet(
        cache, index=False
    )
    before = cache.read_bytes()
    result = service.load_sample(
        {}, config, max_rows=64, output_dir=tmp_path / "sample"
    )
    pd.testing.assert_frame_equal(result.frame, frame.head(64))
    assert len(calls) == 1
    assert calls[0][0] == pd.Timestamp("2024-02-01", tz="UTC")
    assert (calls[0][1] - calls[0][0]).total_seconds() <= 64 * 3 * 3600
    assert cache.read_bytes() == before


def test_sufficient_earlier_year_cache_avoids_fetching_later_year(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _frame(pd.date_range("2024-12-27", periods=120, freq="h", tz="UTC"))
    service, config, calls = _service(
        tmp_path,
        monkeypatch,
        frame,
        timeframe="1h",
        start="2024-12-27",
        end="2025-01-15",
        cached=True,
    )
    # Only the earlier year exists, and it already satisfies the sample budget.
    cached_market_path(
        tmp_path / "workspace/data/processed", symbol="AAPL", year=2025
    ).unlink()
    result = service.load_sample(
        {}, config, max_rows=64, output_dir=tmp_path / "sample"
    )
    pd.testing.assert_frame_equal(result.frame, frame.head(64))
    assert not calls


@pytest.mark.parametrize("as_strings", [False, True])
def test_unsorted_cache_batches_keep_earliest_unique_utc_bars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, as_strings: bool
) -> None:
    expected = _frame(pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"))
    stored = pd.concat([expected, expected.head(10)], ignore_index=True).sample(
        frac=1, random_state=42
    )
    if as_strings:
        stored["timestamp"] = stored["timestamp"].astype(str)
    service, config, _ = _service(
        tmp_path,
        monkeypatch,
        stored,
        timeframe="1h",
        start="2024-01-01",
        end="2024-01-08",
        cached=True,
    )
    path = cached_market_path(tmp_path / "workspace/data/processed", symbol="AAPL")
    stored.to_parquet(path, index=False, row_group_size=13)
    before = path.read_bytes()
    result = service.load_sample(
        {}, config, max_rows=64, output_dir=tmp_path / "sample"
    )
    pd.testing.assert_frame_equal(result.frame, expected.head(64))
    assert path.read_bytes() == before


def test_cache_sampling_crosses_year_boundary_without_losing_midnight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _frame(pd.date_range("2023-12-31 23:50", periods=90, freq="min", tz="UTC"))
    service, config, _ = _service(
        tmp_path,
        monkeypatch,
        frame,
        timeframe="1m",
        start="2023-12-31",
        end="2024-01-02",
        cached=True,
    )
    result = service.load_sample(
        {}, config, max_rows=64, output_dir=tmp_path / "sample"
    )
    pd.testing.assert_frame_equal(result.frame, frame.head(64))
    assert pd.Timestamp("2024-01-01", tz="UTC") in result.frame["timestamp"].tolist()


@pytest.mark.parametrize("earlier_year_cached", [False, True])
@pytest.mark.parametrize("timestamp_unit", ["ms", "us", "ns"])
def test_provider_sampling_includes_midnight_at_year_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    earlier_year_cached: bool,
    timestamp_unit: Literal["ms", "us", "ns"],
) -> None:
    timestamps = pd.date_range(
        "2024-12-31 22:57", periods=64, freq="min", tz="UTC"
    ).as_unit(timestamp_unit)
    frame = _frame(timestamps)
    service, config, _ = _service(
        tmp_path,
        monkeypatch,
        frame,
        timeframe="1m",
        start="2024-12-31 22:57",
        end="2025-01-01 00:00",
        cached=False,
    )
    if earlier_year_cached:
        seed_market_cache(
            tmp_path / "workspace/data/processed",
            frame.iloc[:-1],
            symbol="AAPL",
            timeframe="1m",
        )
    milliseconds = [int(timestamp.timestamp() * 1000) for timestamp in timestamps]

    class Exchange:
        def fetch_ohlcv(
            self, _symbol: str, _timeframe: str, *, since: int, limit: int
        ) -> list[list[object]]:
            candles: list[list[object]] = [
                [stamp, 100.0, 100.0, 100.0, 100.0, 100.0]
                for stamp in milliseconds
                if stamp >= since
            ]
            return candles[:limit]

    monkeypatch.setattr(
        "tradingdev.domain.data.crawlers.binance_api.ccxt.binance",
        lambda *_args: Exchange(),
    )
    monkeypatch.setattr(
        "tradingdev.app.data_service.create_crawler",
        lambda *_args: BinanceAPICrawler(),
    )
    result = service.load_sample(
        {}, config, max_rows=64, output_dir=tmp_path / "sample"
    )
    pd.testing.assert_frame_equal(
        result.frame.assign(timestamp=result.frame["timestamp"].dt.as_unit("ns")),
        frame.assign(timestamp=frame["timestamp"].dt.as_unit("ns")),
    )


@pytest.mark.parametrize("cached_year", [2024, 2025])
def test_complete_year_cache_wins_over_overlapping_provider_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached_year: int
) -> None:
    start = "2024-12-31 22:57" if cached_year == 2024 else "2024-12-31 23:59"
    timestamps = pd.date_range(start, periods=64, freq="min", tz="UTC")
    remote = _frame(timestamps)
    columns = ["open", "high", "low", "close", "volume"]
    remote[columns] = 200.0
    expected = remote.copy()
    expected.loc[expected["timestamp"].dt.year == cached_year, columns] = 100.0
    service, config, _ = _service(
        tmp_path,
        monkeypatch,
        remote,
        timeframe="1m",
        start=start,
        end=timestamps[-1].strftime("%Y-%m-%d %H:%M"),
        cached=False,
    )
    cache = seed_market_cache(
        tmp_path / "workspace/data/processed",
        expected.loc[expected["timestamp"].dt.year == cached_year],
        symbol="AAPL",
        timeframe="1m",
        year=cached_year,
    )
    before = cache.read_bytes()
    result = service.load_sample(
        {}, config, max_rows=64, output_dir=tmp_path / "sample"
    )
    pd.testing.assert_frame_equal(result.frame, expected)
    assert cache.read_bytes() == before
