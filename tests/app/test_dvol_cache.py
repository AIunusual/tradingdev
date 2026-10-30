"""Exact feature queries cannot hide unfetched dates behind forward filling."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml
from tests.data_cache import seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.data_service import DataService
from tradingdev.app.dvol_data import DVOLData
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.data.crawlers.base import BaseCrawler
from tradingdev.domain.data.requirements import FeatureSpec
from tradingdev.domain.data.schemas import DataConfig, MarketDataRequest


def _crawler() -> Mock:
    crawler = Mock(spec=BaseCrawler)

    def fetch(**kwargs: Any) -> pd.DataFrame:
        timestamps = kwargs.get("timestamps")
        if timestamps is None:
            timestamps = pd.date_range(kwargs["start"], kwargs["end"], freq="D")
        return pd.DataFrame(
            {"timestamp": timestamps, "dvol_close": timestamps.day * 10.0}
        )

    crawler.fetch.side_effect = fetch
    crawler.fetch_sample.side_effect = fetch
    crawler.save_raw.side_effect = lambda frame, path: frame.to_csv(path, index=False)
    return crawler


def _request(start: str = "2024-01-01", end: str = "2024-01-02") -> MarketDataRequest:
    return MarketDataRequest(
        symbol="BTC/USDT", timeframe="1d", start_date=start, end_date=end
    )


def _backtest(start: str, end: str) -> BacktestConfig:
    return BacktestConfig(**_request(start, end).model_dump(), init_cash=10000)


def _service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[DataService, dict[str, Any], Mock]:
    service = DataService(WorkspacePaths(tmp_path / "workspace"))
    config: dict[str, Any] = {
        "data": {
            "raw_dir": str(tmp_path / "custom/raw"),
            "processed_dir": str(tmp_path / "custom/processed"),
            "requirements": {
                "market": {"symbol": "BTC/USDT", "timeframe": "1d"},
                "features": [{"type": "dvol", "source": "deribit", "column": "dvol"}],
            },
        }
    }
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-12-30", "2025-01-03", freq="D", tz="UTC"),
            **dict.fromkeys(("open", "high", "low", "close", "volume"), 100.0),
        }
    )
    for year in (2023, 2024, 2025):
        seed_market_cache(
            Path(config["data"]["processed_dir"]),
            frame.loc[frame.timestamp.dt.year == year],
            timeframe="1d",
            year=year,
        )
    crawler = _crawler()
    monkeypatch.setattr(
        "tradingdev.app.data_service.DeribitDVOLCrawler", lambda: crawler
    )
    return service, config, crawler


def test_automatic_dvol_queries_exact_extensions_disjoint_and_cross_year_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, config, crawler = _service(tmp_path, monkeypatch)
    identities = set()
    for start, end in (
        ("2024-01-01", "2024-01-02"),
        ("2024-01-01", "2024-01-03"),
        ("2023-12-31", "2024-01-03"),
        ("2024-02-01", "2024-02-02"),
        ("2024-12-31", "2025-01-02"),
    ):
        backtest = _backtest(start, end)
        prepared = {"data": service.execution_config(config, backtest)}
        assert prepared["data"]["requirements"]["features"][0]["path"] is None
        result = service.load(prepared, backtest)
        assert (
            result.frame.dvol.tolist()
            == (result.frame.timestamp.dt.day * 10.0).tolist()
        )
        managed = DVOLData(service.data_config(config), _request(start, end))
        identities.add(managed.key)
        receipt = managed.receipt()
        assert (
            receipt is not None
            and receipt.query_start == _request(start, end).start_date
        )
        assert receipt.query_end == _request(start, end).end_date
        service.load(prepared, backtest)
    assert len(identities) == crawler.fetch.call_count == 5
    assert not list(service._workspace.processed_data.iterdir())


def test_sample_extension_fetches_new_values_without_publishing_partial_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, config, crawler = _service(tmp_path, monkeypatch)
    service.load(config, _backtest("2024-01-01", "2024-01-02"))
    original = {
        path: path.read_bytes()
        for path in (tmp_path / "custom").rglob("*")
        if path.is_file()
    }
    backtest = _backtest("2024-01-01", "2024-01-03")
    sample = service.load_sample(
        config, backtest, max_rows=3, output_dir=tmp_path / "sample"
    )
    assert sample.frame.dvol.tolist() == [10.0, 20.0, 30.0]
    assert crawler.fetch_sample.call_count == 1
    assert {
        path: path.read_bytes()
        for path in (tmp_path / "custom").rglob("*")
        if path.is_file()
    } == original
    managed = DVOLData(service.data_config(config), _request(end="2024-01-03"))
    assert managed.receipt() is None
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({**config, "backtest": backtest.model_dump(mode="json")})
    )
    before = service.inspect_dataset(path)["features"][0]
    assert before["query"] is None and before["query_reusable"] is False
    assert before["rows"] is None
    loaded = service.load(config, backtest)
    assert loaded.frame.dvol.tolist() == [10.0, 20.0, 30.0]
    assert crawler.fetch.call_count == 2
    after = service.inspect_dataset(path)["features"][0]
    assert after["query"]["query_end"].startswith("2024-01-03")
    assert after["query_reusable"] is True and after["rows"] == 3
    service.load_sample(
        config, backtest, max_rows=3, output_dir=tmp_path / "cached-sample"
    )
    assert crawler.fetch_sample.call_count == 1


@pytest.mark.parametrize("failure", ["fetch", "timeout", "parquet"])
def test_recent_feature_refresh_failure_preserves_published_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    now = [datetime(2026, 1, 2, tzinfo=UTC)]
    crawler = _crawler()
    managed = DVOLData(
        DataConfig(
            raw_dir=str(tmp_path / "raw"), processed_dir=str(tmp_path / "processed")
        ),
        _request("2026-01-01", "2026-01-02"),
        now_fn=lambda: now[0],
    )
    managed.load(lambda: crawler)
    original = {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }
    now[0] = datetime(2026, 1, 3, tzinfo=UTC)

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise (
            TimeoutError("feature refresh failed")
            if failure == "timeout"
            else OSError("feature refresh failed")
        )

    if failure == "parquet":
        monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    else:
        crawler.fetch.side_effect = fail
    with pytest.raises(OSError, match="feature refresh failed"):
        managed.load(lambda: crawler)
    assert {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    } == original
    assert not list(tmp_path.rglob(".pending-*"))
    with managed.cache.lock(managed.key).acquire(timeout=0):
        pass


def test_successful_empty_feature_query_is_distinct_from_never_queried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, config, crawler = _service(tmp_path, monkeypatch)
    crawler.fetch.side_effect = None
    crawler.fetch.return_value = pd.DataFrame(columns=["timestamp", "dvol_close"])
    for _ in range(2):
        result = service.load(config, _backtest("2024-01-01", "2024-01-03"))
        assert result.frame.dvol.isna().all()
    assert crawler.fetch.call_count == 1
    assert (
        DVOLData(service.data_config(config), _request(end="2024-01-03")).receipt()
        is not None
    )


def test_legacy_automatic_feature_file_is_preserved_without_inventing_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, config, crawler = _service(tmp_path, monkeypatch)
    path = Path(config["data"]["processed_dir"]) / "btc_dvol_1d_2024_2024.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, tz="UTC"),
            "dvol_close": 999.0,
        }
    ).to_parquet(path)
    before = path.read_bytes()
    result = service.load(config, _backtest("2024-01-01", "2024-01-03"))
    assert result.frame.dvol.tolist() == [10.0, 20.0, 30.0]
    assert crawler.fetch.call_count == 1 and path.read_bytes() == before


def test_automatic_feature_raw_path_cannot_bypass_managed_storage() -> None:
    with pytest.raises(ValueError, match="raw_path requires path"):
        FeatureSpec(type="dvol", source="deribit", column="dvol", raw_path="custom.csv")


def test_recent_feature_query_only_reuses_an_acquisition_after_settlement(
    tmp_path: Path,
) -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    crawler = _crawler()
    managed = DVOLData(
        DataConfig(
            raw_dir=str(tmp_path / "raw"), processed_dir=str(tmp_path / "processed")
        ),
        _request("2026-01-01", "2026-01-02"),
        now_fn=lambda: now[0],
    )
    with pytest.raises(ValueError, match="future"):
        managed.load(lambda: crawler)
    assert not list(tmp_path.iterdir())
    now[0] = datetime(2026, 1, 2, tzinfo=UTC)
    managed.load(lambda: crawler)
    managed.load(lambda: crawler)
    assert crawler.fetch.call_count == 2
    now[0] = datetime(2026, 1, 9, tzinfo=UTC)
    managed.load(lambda: crawler)
    managed.load(lambda: crawler)
    assert crawler.fetch.call_count == 3


def test_failed_extended_sample_preserves_source_feature_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, config, crawler = _service(tmp_path, monkeypatch)
    service.load(config, _backtest("2024-01-01", "2024-01-02"))
    original = {
        path: path.read_bytes()
        for path in (tmp_path / "custom").rglob("*")
        if path.is_file()
    }
    crawler.fetch_sample.side_effect = TimeoutError("sample feature unavailable")
    with pytest.raises(TimeoutError, match="sample feature unavailable"):
        service.load_sample(
            config,
            _backtest("2024-01-01", "2024-01-03"),
            max_rows=3,
            output_dir=tmp_path / "sample-failed",
        )
    assert {
        path: path.read_bytes()
        for path in (tmp_path / "custom").rglob("*")
        if path.is_file()
    } == original
    assert not list((tmp_path / "sample-failed").rglob("*"))


@pytest.mark.parametrize("empty", [False, True])
def test_real_dvol_provider_queries_inclusive_point_before_recording_success(
    tmp_path: Path,
    empty: bool,
) -> None:
    import httpx

    from tradingdev.domain.data.crawlers.deribit_dvol import DeribitDVOLCrawler

    request = _request("2024-01-01", "2024-01-01")
    stamp = int(request.start_date.timestamp() * 1000)
    calls = []

    def respond(query: httpx.Request) -> httpx.Response:
        assert int(query.url.params["start_timestamp"]) == stamp
        assert int(query.url.params["end_timestamp"]) == stamp
        calls.append(query)
        return httpx.Response(
            200,
            json={
                "result": {
                    "data": [] if empty else [[stamp, 10.0, 10.0, 10.0, 10.0]],
                    "continuation": None,
                }
            },
        )

    managed = DVOLData(
        DataConfig(
            raw_dir=str(tmp_path / "raw"), processed_dir=str(tmp_path / "processed")
        ),
        request,
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        crawler = DeribitDVOLCrawler()
        crawler._client.close()
        crawler._client = client
        frame = managed.load(lambda: crawler)
        managed.load(lambda: crawler)
    assert len(calls) == 1
    assert len(frame) == (0 if empty else 1)
    receipt = managed.receipt()
    assert receipt is not None and receipt.reusable(
        request.start_date, request.end_date
    )


@pytest.mark.parametrize("stalled", [False, True])
def test_dvol_pagination_queries_left_endpoint_and_rejects_stalled_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stalled: bool,
) -> None:
    import httpx

    from tradingdev.domain.data.crawlers.deribit_dvol import DeribitDVOLCrawler

    request = MarketDataRequest(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01T00:00Z",
        end_date="2024-01-01T01:00Z",
    )
    start, end = (
        int(value.timestamp() * 1000)
        for value in (request.start_date, request.end_date)
    )
    cursors = []

    def respond(query: httpx.Request) -> httpx.Response:
        cursor = int(query.url.params["end_timestamp"])
        cursors.append(cursor)
        continuation = end if stalled else start if cursor == end else None
        return httpx.Response(
            200,
            json={
                "result": {
                    "data": [[cursor, 10.0, 10.0, 10.0, 10.0]],
                    "continuation": continuation,
                }
            },
        )

    monkeypatch.setattr(
        "tradingdev.domain.data.crawlers.deribit_dvol.time.sleep", lambda *_: None
    )
    managed = DVOLData(
        DataConfig(
            raw_dir=str(tmp_path / "raw"), processed_dir=str(tmp_path / "processed")
        ),
        request,
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        crawler = DeribitDVOLCrawler()
        crawler._client.close()
        crawler._client = client
        if stalled:
            with pytest.raises(ValueError, match="did not move backward"):
                managed.load(lambda: crawler)
            assert cursors == [end] and managed.receipt() is None
            assert not list(tmp_path.rglob("*.parquet"))
        else:
            frame = managed.load(lambda: crawler)
            assert cursors == [end, start]
            assert frame.timestamp.tolist() == [
                pd.Timestamp(request.start_date),
                pd.Timestamp(request.end_date),
            ]
