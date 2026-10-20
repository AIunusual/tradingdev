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
from tradingdev.domain.validation.date_windows import utc_calendar_day
from tradingdev.shared.utils.json_values import strict_json_value


class BacktestConfig(BaseModel):
    """Backtest execution configuration."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    timeframe: str
    start_date: dt.datetime
    end_date: dt.datetime
    init_cash: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    fees: float = 0.0006
    slippage: float = 0.0005
    position_size: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    signal_as_position: bool = False
    re_entry_after_sl: bool = True
    mode: Literal["signal", "volume"] = "signal"
    monthly_max_loss: float = 1500.0
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


class WalkForwardConfig(BaseModel):
    """Walk-forward validation configuration."""

    model_config = ConfigDict(extra="forbid")

    train_start: dt.datetime | None = None
    train_end: dt.datetime | None = None
    test_start: dt.datetime | None = None
    test_end: dt.datetime | None = None
    n_splits: int = 1
    train_ratio: float = 0.8
    expanding: bool = False
    target_metric: str = "sharpe_ratio"
    label_horizon: int = 0
    """Bars a label looks ahead; >0 enables the leakage gate on the folds.

    With ``0`` no fold geometry is audited (a plain price-series walk-forward
    has no forward-looking label). Set it to the prediction horizon when the
    folds train on interval-based labels, so bars whose answer is computed
    from test-window prices are rejected before any fold runs.
    """
    embargo_bars: int = 0
    """Extra bars removed around each test window, on top of the purge.

    Purge width is derivable from ``label_horizon``; this buffer is not, and
    exists because financial features are serially correlated. Only used by
    the purged splitter.
    """
    purged_split: bool = False
    """Use :class:`PurgedDataSplitter` instead of :class:`DataSplitter`.

    Requires ``label_horizon >= 1``. Folds are then purged and embargoed by
    construction, so the leakage gate passes instead of blocking the run.
    """

    @model_validator(mode="after")
    def explicit_dates_ordered(self) -> Self:
        """Require active explicit dates to be ordered and disjoint by UTC day.

        Every boundary is an inclusive UTC calendar day, so a train window
        ending on the day the test window starts shares that day's bars with
        the test set. ``train_end`` and ``test_start`` must therefore name
        different days; reversed or partially specified periods are rejected
        instead of silently falling back to the automatic split. Purged mode
        does not use these dates, so their completeness and order are irrelevant.
        """
        if self.purged_split:
            return self
        train_start, train_end = self.train_start, self.train_end
        test_start, test_end = self.test_start, self.test_end
        if (
            train_start is None
            or train_end is None
            or test_start is None
            or test_end is None
        ):
            missing = [
                name
                for name, value in (
                    ("train_start", train_start),
                    ("train_end", train_end),
                    ("test_start", test_start),
                    ("test_end", test_end),
                )
                if value is None
            ]
            if len(missing) == 4:
                return self
            raise ValueError(
                "Explicit walk-forward dates must be given all together; "
                f"missing: {', '.join(missing)}"
            )

        days = tuple(
            utc_calendar_day(value)
            for value in (train_start, train_end, test_start, test_end)
        )
        if not (days[0] <= days[1] < days[2] <= days[3]):
            raise ValueError(
                "Explicit walk-forward dates must satisfy "
                "train_start <= train_end < test_start <= test_end as inclusive "
                f"UTC calendar days (got train {days[0]}..{days[1]}, "
                f"test {days[2]}..{days[3]}); the train window's last day must "
                "precede the test window's first day"
            )
        return self


class ParallelConfig(BaseModel):
    """Parallel execution configuration."""

    model_config = ConfigDict(extra="forbid")

    reserve_cores: int = 2
    safety_factor: float = 0.6
    overhead_multiplier: float = 3.0

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
