"""Explicit data requirement schemas."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class MarketDataSpec(BaseModel):
    """Primary market data requirement."""

    source: str = "binance_vision"
    symbol: str
    timeframe: str


class FeatureSpec(BaseModel):
    """Additional feature data requirement."""

    type: Literal["dvol", "funding_rate", "custom"]
    source: str
    column: str
    path: str | None = None
    raw_path: str | None = None

    @model_validator(mode="after")
    def raw_path_requires_output(self) -> Self:
        if self.type == "dvol" and self.raw_path and not self.path:
            raise ValueError(
                "DVOL raw_path requires path; omit both for managed caching"
            )
        return self


class DataSourceSpec(BaseModel):
    """External data source descriptor."""

    name: str
    kind: str
    metadata: dict[str, str] = Field(default_factory=dict)


class DataRequirement(BaseModel):
    """Full data requirement section for a strategy run."""

    market: MarketDataSpec
    features: list[FeatureSpec] = []
