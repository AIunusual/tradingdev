"""Typed read contracts for market-data discovery and cache inspection."""

from typing import Literal

from tradingdev.app.contracts.common import ContractModel


class DataSourceDescription(ContractModel):
    """Registered source choices and limits available without a network request."""

    source: str
    description: str
    asset_classes: list[str]
    symbol_examples: list[str]
    timeframes: list[str] | None
    market_types: list[str]
    default_market_type: str | None
    limitations: list[str]


class CachedDataset(ContractModel):
    """Available yearly cache files for one market and timeframe."""

    symbol: str
    timeframe: str
    source: str
    market_type: str | None
    years_available: list[int]


class MarketRequirement(ContractModel):
    """Resolved primary market requirement."""

    source: str
    symbol: str
    timeframe: str


class FeatureRequirement(ContractModel):
    """Resolved feature source configuration."""

    type: Literal["dvol", "funding_rate", "custom"]
    source: str
    column: str
    path: str | None
    raw_path: str | None


class DatasetRequirements(ContractModel):
    """Market and feature requirements declared by a run configuration."""

    market: MarketRequirement
    features: list[FeatureRequirement]


class QueryEvidence(ContractModel):
    """Successful query bounds and time; not a guarantee of complete observations."""

    query_start: str
    query_end: str
    fetched_at: str


class MarketInspection(ContractModel):
    """Observed local market-file contents and read errors."""

    symbol: str
    timeframe: str
    source: str
    market_type: str | None
    queries: list[QueryEvidence]
    paths: list[str]
    exists: bool
    rows: int | None
    columns: list[str]
    start_timestamp: str | None
    end_timestamp: str | None
    timezone: str | None
    missing_values: dict[str, int]
    errors: list[str]


class FeatureInspection(ContractModel):
    """Observed availability and missing values for one local feature."""

    type: Literal["dvol", "funding_rate", "custom"]
    source: str
    column: str
    path: str | None
    exists: bool
    rows: int | None
    missing_values: int | None


class DatasetFingerprint(ContractModel):
    """Identity and fingerprint computed from the inspected data files."""

    dataset_id: str
    fingerprint: str


class DatasetInspection(ContractModel):
    """Workspace cache inventory, optionally inspected against a run config."""

    data_root: str
    raw_dir: str
    processed_dir: str
    datasets: list[CachedDataset]
    requirements: DatasetRequirements | None
    market_available: bool | None
    market: MarketInspection | None
    dataset_fingerprint: DatasetFingerprint | None
    features: list[FeatureInspection]


class EnsureDataResponse(ContractModel):
    """A successful market query was cached; zero observations remain possible."""

    success: Literal[True]
    rows: int
    processed_path: str
    dataset_id: str
