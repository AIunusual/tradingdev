"""Crawler registry tests."""

from __future__ import annotations

import pytest

from tradingdev.domain.data.crawlers.base import BaseCrawler
from tradingdev.domain.data.crawlers.binance_vision import BinanceVisionCrawler
from tradingdev.domain.data.crawlers.registry import (
    YAHOO_INTERVALS,
    SourceDetails,
    available_sources,
    create_crawler,
    describe_sources,
    register_crawler,
)
from tradingdev.domain.data.crawlers.yahoo_finance import _to_yahoo_interval
from tradingdev.domain.data.schemas import DataConfig


def test_builtin_sources_are_registered() -> None:
    sources = available_sources()

    assert "binance_vision" in sources
    assert "binance_api" in sources


def test_create_crawler_builds_binance_vision_with_market_type() -> None:
    crawler = create_crawler(
        "binance_vision",
        DataConfig(source="binance_vision", market_type="spot"),
    )

    assert isinstance(crawler, BinanceVisionCrawler)
    assert isinstance(crawler, BaseCrawler)
    assert crawler._market_type == "spot"


def test_create_crawler_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="Unknown data source 'nope'"):
        create_crawler("nope", DataConfig())


def test_source_discovery_matches_yahoo_interval_handling() -> None:
    descriptions = {item["source"]: item for item in describe_sources()}
    assert set(descriptions) == set(available_sources())
    yahoo = descriptions["yahoo_finance"]
    assert set(yahoo["timeframes"]) == set(YAHOO_INTERVALS)
    for timeframe in yahoo["timeframes"]:
        assert _to_yahoo_interval(timeframe) == YAHOO_INTERVALS[timeframe]
    assert descriptions["binance_vision"]["default_market_type"] == "futures/um"
    assert descriptions["binance_vision"]["market_types"] == ("futures/um", "spot")
    assert descriptions["binance_api"]["timeframes"] is None


def test_registered_metadata_is_available_without_constructing_a_crawler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tradingdev.domain.data.crawlers.registry._REGISTRY", {})

    def no_construction(_config: DataConfig) -> BaseCrawler:
        pytest.fail("Discovery must not construct a crawler or access the network")

    details = SourceDetails(
        description="A source supplied by this test",
        asset_classes=("test",),
        symbol_examples=("TEST",),
        timeframes=("1h",),
        market_types=(),
        default_market_type=None,
        limitations=("Fixture only",),
    )
    register_crawler("test_source", no_construction, details)
    assert available_sources() == ["test_source"]
    result = describe_sources()
    assert result[0]["source"] == "test_source"
    assert result[0]["description"] == details.description
    assert result[0]["timeframes"] == details.timeframes
