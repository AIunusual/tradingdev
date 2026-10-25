"""Registered crawler factories and their discoverable source capabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from tradingdev.domain.data.crawlers.base import BaseCrawler
    from tradingdev.domain.data.schemas import DataConfig

# Both Yahoo's adapter and discovery use this exact set of accepted spellings.
YAHOO_INTERVALS = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "60m": "60m",
    "90m": "90m",
    "1d": "1d",
    "1w": "1wk",
    "1wk": "1wk",
    "1mo": "1mo",
}


@dataclass(frozen=True)
class SourceDetails:
    """Source-specific choices, not a promise that a requested period has data."""

    description: str
    asset_classes: tuple[str, ...]
    symbol_examples: tuple[str, ...]
    timeframes: tuple[str, ...] | None
    market_types: tuple[str, ...]
    default_market_type: str | None
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class _RegisteredSource:
    factory: Callable[[DataConfig], BaseCrawler]
    details: SourceDetails


_REGISTRY: dict[str, _RegisteredSource] = {}


def register_crawler(
    name: str,
    factory: Callable[[DataConfig], BaseCrawler],
    details: SourceDetails,
) -> None:
    """Register a crawler and the information needed to choose it."""
    _REGISTRY[name] = _RegisteredSource(factory, details)


def create_crawler(source: str, data_config: DataConfig) -> BaseCrawler:
    """Create the crawler registered for a data source name."""
    registration = _REGISTRY.get(source)
    if registration is None:
        known = ", ".join(available_sources())
        msg = f"Unknown data source {source!r}. Known sources: {known}"
        raise ValueError(msg)
    return registration.factory(data_config)


def available_sources() -> list[str]:
    """Return the registered data source names."""
    return sorted(_REGISTRY)


def describe_sources() -> list[dict[str, Any]]:
    """Describe registered choices without constructing crawlers or fetching data."""
    return [
        {"source": name, **asdict(_REGISTRY[name].details)}
        for name in available_sources()
    ]


def _binance_vision_factory(data_config: DataConfig) -> BaseCrawler:
    from tradingdev.domain.data.crawlers.binance_vision import BinanceVisionCrawler

    return BinanceVisionCrawler(market_type=data_config.market_type)


def _binance_api_factory(_data_config: DataConfig) -> BaseCrawler:
    from tradingdev.domain.data.crawlers.binance_api import BinanceAPICrawler

    return BinanceAPICrawler()


def _yahoo_finance_factory(_data_config: DataConfig) -> BaseCrawler:
    from tradingdev.domain.data.crawlers.yahoo_finance import YahooFinanceCrawler

    return YahooFinanceCrawler()


register_crawler(
    "binance_vision",
    _binance_vision_factory,
    SourceDetails(
        description="Binance Data Vision archived crypto OHLCV; default source.",
        asset_classes=("crypto",),
        symbol_examples=("BTC/USDT", "ETH/USDT"),
        timeframes=None,
        market_types=("futures/um", "spot"),
        default_market_type="futures/um",
        limitations=(
            "Select data.requirements.market.source=binance_vision and set "
            "data.market_type for USD-M futures or spot.",
            "Timeframe and historical coverage depend on available Binance archives; "
            "null timeframes means no exhaustive local capability list.",
        ),
    ),
)
register_crawler(
    "binance_api",
    _binance_api_factory,
    SourceDetails(
        description="Binance public OHLCV through the configured ccxt client.",
        asset_classes=("crypto",),
        symbol_examples=("BTC/USDT", "ETH/USDT"),
        timeframes=None,
        market_types=(),
        default_market_type=None,
        limitations=(
            "Uses the ccxt Binance client's default market; data.market_type is "
            "not applied by this adapter. Prefer binance_vision for explicit "
            "USD-M futures/spot selection.",
            "Accepted symbols, timeframes and history depend on the upstream "
            "exchange; no exhaustive local timeframe list is advertised.",
        ),
    ),
)
register_crawler(
    "yahoo_finance",
    _yahoo_finance_factory,
    SourceDetails(
        description="Public Yahoo Finance chart OHLCV using Yahoo symbol conventions.",
        asset_classes=("stocks", "ETFs", "futures", "indices", "FX"),
        symbol_examples=("AAPL", "2330.TW", "ES=F", "^GSPC", "EURUSD=X"),
        timeframes=tuple(YAHOO_INTERVALS),
        market_types=(),
        default_market_type=None,
        limitations=(
            "data.market_type does not apply. Intraday history and availability "
            "are limited by Yahoo; a locally supported interval does not guarantee "
            "the requested range is available.",
        ),
    ),
)
