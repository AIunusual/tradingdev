"""Domain schemas for market data."""

from datetime import UTC, datetime
from typing import Self

from pydantic import BaseModel, field_validator, model_validator


class DVOLBar(BaseModel):
    """Single Deribit DVOL (implied volatility index) data point."""

    timestamp: datetime
    dvol_open: float
    dvol_high: float
    dvol_low: float
    dvol_close: float

    @field_validator("dvol_close")
    @classmethod
    def dvol_close_positive(cls, v: float) -> float:
        """Validate dvol_close > 0."""
        if v <= 0:
            msg = "dvol_close must be positive"
            raise ValueError(msg)
        return v


class OHLCVBar(BaseModel):
    """Single OHLCV candle bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @field_validator("volume")
    @classmethod
    def volume_non_negative(cls, v: float) -> float:
        """Validate that volume is non-negative."""
        if v < 0:
            msg = "volume must be non-negative"
            raise ValueError(msg)
        return v


class DataConfig(BaseModel):
    """Data source and path configuration."""

    source: str = "binance_vision"
    market_type: str = "futures/um"
    raw_dir: str = "workspace/data/raw"
    processed_dir: str = "workspace/data/processed"


class MarketDataRequest(BaseModel):
    """A concrete market data slice to fetch, independent of backtesting."""

    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime

    @field_validator("start_date", "end_date")
    @classmethod
    def utc_instant(cls, value: datetime) -> datetime:
        """Select cache years and provider bounds using the same UTC instants."""
        return (
            value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        )

    @model_validator(mode="after")
    def ordered_interval(self) -> Self:
        """Allow one inclusive instant, but reject reversed requests before I/O."""
        if self.end_date < self.start_date:
            raise ValueError("end_date must not precede start_date")
        return self
