"""Domain schemas for backtest execution."""

from __future__ import annotations

import datetime as dt  # noqa: TC003
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from tradingdev.domain.data.requirements import FeatureSpec, MarketDataSpec
from tradingdev.domain.data.schemas import DataConfig
from tradingdev.domain.research import ResearchMetadata
from tradingdev.domain.research_policy import DEFAULT_FEES, DEFAULT_SLIPPAGE
from tradingdev.domain.strategies.capabilities import TrainingRequirement
from tradingdev.domain.validation.config import WalkForwardConfig
from tradingdev.shared.utils.json_values import strict_json_value


class BacktestConfig(BaseModel):
    """Backtest execution configuration."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    timeframe: str
    start_date: dt.datetime
    end_date: dt.datetime
    init_cash: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    fees: float = DEFAULT_FEES
    slippage: float = DEFAULT_SLIPPAGE
    position_size: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    signal_as_position: bool = Field(
        default=False,
        description=(
            "Volume mode only: true makes signal=0 close an existing position; "
            "false retains it while preventing new entries. Signal mode always "
            "uses position states (1 long, -1 short, 0 flat) and ignores this flag. "
            "In both modes -1 selects short, not a sell-only exit."
        ),
    )
    re_entry_after_sl: bool = True
    mode: Literal["signal", "volume"] = "signal"
    monthly_max_loss: float = Field(
        default=1500.0,
        description=(
            "Volume mode only; requires timestamps. Quote-currency loss limit for "
            "each UTC calendar month. Compare cumulative P&L equity with the "
            "previous observed month's last close equity (zero in the first month). "
            "Missing months are not filled; gaps belong to the next observed month. "
            "Count current-month costs and unrealised P&L changes, not prior-month "
            "P&L or entry costs again. Trade records retain whole-trade P&L and costs. "
            "Check after stop exits before re-entry and after ordinary trading at "
            "close. Reaching the loss limit closes any remaining position at close "
            "and pauses until the next observed month; exit costs may exceed the limit."
        ),
    )
    periods_per_year: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    risk_free_rate: float = Field(default=0.0, gt=-1, allow_inf_nan=False)
    required_return: float = Field(default=0.0, gt=-1, allow_inf_nan=False)

    @model_validator(mode="after")
    def end_after_start(self) -> Self:
        """Validate that end_date is after start_date."""
        if self.end_date <= self.start_date:
            msg = "end_date must be after start_date"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def signal_mode_requires_init_cash(self) -> Self:
        """Signal mode requires init_cash to be set."""
        if self.mode == "signal" and self.init_cash is None:
            msg = "init_cash is required when mode is 'signal'"
            raise ValueError(msg)
        return self


class ParallelConfig(BaseModel):
    """Parallel execution configuration."""

    model_config = ConfigDict(extra="forbid")

    reserve_cores: int = 2
    safety_factor: float = 0.6
    overhead_multiplier: float = Field(default=3.0, gt=0, allow_inf_nan=False)

    @field_validator("reserve_cores")
    @classmethod
    def reserve_cores_non_negative(cls, v: int) -> int:
        """Validate reserve_cores >= 0."""
        if v < 0:
            msg = "reserve_cores must be non-negative"
            raise ValueError(msg)
        return v

    @field_validator("safety_factor")
    @classmethod
    def safety_factor_range(cls, v: float) -> float:
        """Validate 0 < safety_factor <= 1."""
        if not 0 < v <= 1:
            msg = "safety_factor must be between 0 (exclusive) and 1 (inclusive)"
            raise ValueError(msg)
        return v


class StrategyRunConfig(BaseModel):
    """Known strategy metadata, with explicit dynamic constructor parameters."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    revision_id: str | None = None
    version: str | None = None
    class_name: str | None = None
    description: str | None = None
    display_name: str | None = None
    training_requirement: TrainingRequirement = Field(
        default="unknown",
        exclude=True,
        description=(
            "Bundled catalog declaration, not an execution override or model state. "
            "Generated strategy capabilities remain unknown."
        ),
    )
    source_path: str | None = None
    source_hash: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    fit: dict[str, Any] | None = None


class RunMarketDataSpec(MarketDataSpec):
    """Reject misspelled market requirement fields before execution."""

    model_config = ConfigDict(extra="forbid")


class RunFeatureSpec(FeatureSpec):
    """Reject misspelled feature requirement fields before execution."""

    model_config = ConfigDict(extra="forbid")


class RunDataRequirement(BaseModel):
    """Strict requirements within a strategy execution configuration."""

    model_config = ConfigDict(extra="forbid")

    market: RunMarketDataSpec
    features: list[RunFeatureSpec] = Field(default_factory=list)


class RunDataConfig(DataConfig):
    """Data settings whose environment-dependent defaults resolve at submission."""

    model_config = ConfigDict(extra="forbid")

    requirements: RunDataRequirement | None = None


class BacktestRunConfig(BaseModel):
    """Complete YAML shape shared by validation and execution boundaries."""

    model_config = ConfigDict(extra="forbid")

    strategy: StrategyRunConfig
    backtest: BacktestConfig
    data: RunDataConfig = Field(default_factory=RunDataConfig)
    research: ResearchMetadata | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    validation: WalkForwardConfig | None = None
    parallel: ParallelConfig | None = None
    random_seed: int | None = Field(
        default=None,
        strict=True,
        ge=0,
        le=2**32 - 1,
        description=(
            "Run seed for isolated Python and NumPy generators exposed by "
            "tradingdev.domain.randomness. This belongs at the YAML root, not "
            "under backtest. Null uses independent entropy. Global RNGs and "
            "third-party model seeds are not changed."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def validate_raw_json(cls, value: object) -> object:
        """Reject lossy raw values before Pydantic serializers can normalize them."""
        strict_json_value(value, allow_dates=True)
        return value

    @model_validator(mode="after")
    def validate_resolved_json(self) -> Self:
        """Reject non-finite typed values introduced by coercion or defaults."""
        strict_json_value(self.model_dump(mode="python"), allow_dates=True)
        return self

    @field_serializer("strategy", "data")
    def preserve_declared_fields(
        self, value: StrategyRunConfig | RunDataConfig
    ) -> dict[str, Any]:
        """Do not fill identity or environment-dependent data defaults here."""
        return value.model_dump(mode="python", exclude_unset=True)

    @property
    def is_walk_forward(self) -> bool:
        """Return whether the config requests walk-forward validation."""
        return self.validation is not None
