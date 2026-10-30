"""Lossless data identities and explicit cache query evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from tradingdev.domain.data.crawlers.registry import describe_sources

if TYPE_CHECKING:
    from tradingdev.domain.data.schemas import DataConfig, MarketDataRequest

# A reuse boundary, not a guarantee about exchange corrections or missing bars.
SETTLEMENT_DELAY = timedelta(days=7)


class DataIdentity(BaseModel):
    """The effective provider and exact symbols; never strip punctuation/case."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["ohlcv", "dvol"] = "ohlcv"
    source: str
    market_type: str | None
    symbol: str
    timeframe: str

    @property
    def key(self) -> str:
        encoded = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @classmethod
    def market(cls, config: DataConfig, request: MarketDataRequest) -> Self:
        details = next(
            (row for row in describe_sources() if row["source"] == config.source), None
        )
        if details is None:
            raise ValueError(f"Unknown data source: {config.source}")
        markets = details["market_types"]
        if markets and config.market_type not in markets:
            raise ValueError(
                f"Unsupported market_type for {config.source}: {config.market_type}"
            )
        return cls(
            source=config.source,
            market_type=config.market_type if markets else None,
            symbol=request.symbol,
            timeframe=request.timeframe,
        )


class CacheReceipt(BaseModel):
    """One successful provider query, independent of returned candle timestamps."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[2] = 2
    identity: DataIdentity
    generation: str
    query_start: datetime
    query_end: datetime
    fetched_at: datetime

    @field_validator("generation")
    @classmethod
    def safe_generation(cls, value: str) -> str:
        if len(value) != 32 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("Invalid cache generation")
        return value

    @field_validator("query_start", "query_end", "fetched_at")
    @classmethod
    def aware_utc(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("Cache query times must have a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if not self.query_start <= self.query_end <= self.fetched_at:
            raise ValueError("Cache query bounds must precede acquisition time")
        return self

    def covers(self, start: datetime, end: datetime) -> bool:
        return self.query_start <= start <= end <= self.query_end

    def reusable(self, start: datetime, end: datetime) -> bool:
        return self.covers(start, end) and end <= self.fetched_at - SETTLEMENT_DELAY
