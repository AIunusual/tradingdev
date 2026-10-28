"""Disjoint walk-forward modes: every accepted field affects the chosen mode."""

from __future__ import annotations

import datetime as dt  # noqa: TC003
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from tradingdev.domain.validation.date_windows import utc_calendar_day


class _WalkForwardBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_metric: str = Field(
        default="sharpe_ratio",
        description="Fold log/summary metric; does not configure strategy training.",
    )
    label_horizon: int = Field(
        default=0,
        ge=0,
        strict=True,
        description="Forward label bars; zero disables the fold leakage audit.",
    )


class AutomaticWalkForwardConfig(_WalkForwardBase):
    """Rolling or expanding windows computed from the available row count."""

    mode: Literal["automatic"]
    n_splits: int = Field(default=1, ge=1, strict=True)
    train_ratio: float = Field(
        default=0.8, gt=0, lt=1, allow_inf_nan=False, strict=True
    )
    expanding: bool = Field(default=False, strict=True)


class ExplicitWalkForwardConfig(_WalkForwardBase):
    """One training/test pair, with inclusive nonoverlapping UTC calendar days."""

    mode: Literal["explicit"]
    train_start: dt.datetime
    train_end: dt.datetime
    test_start: dt.datetime
    test_end: dt.datetime

    @model_validator(mode="after")
    def dates_ordered(self) -> Self:
        days = tuple(
            utc_calendar_day(value)
            for value in (
                self.train_start,
                self.train_end,
                self.test_start,
                self.test_end,
            )
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


class PurgedWalkForwardConfig(_WalkForwardBase):
    """Anchored expanding windows with label purge and an optional extra embargo."""

    mode: Literal["purged"]
    n_splits: int = Field(default=1, ge=1, strict=True)
    label_horizon: int = Field(
        gt=0,
        strict=True,
        description="Required positive forward label horizon in bars.",
    )
    embargo_bars: int = Field(default=0, ge=0, strict=True)


type WalkForwardConfig = Annotated[
    AutomaticWalkForwardConfig | ExplicitWalkForwardConfig | PurgedWalkForwardConfig,
    Field(discriminator="mode"),
]

_CONFIG_ADAPTER: TypeAdapter[WalkForwardConfig] = TypeAdapter(WalkForwardConfig)


def parse_walk_forward_config(value: object) -> WalkForwardConfig:
    """Parse new configuration without inferring a mode or dropping unused fields."""
    return _CONFIG_ADAPTER.validate_python(value)
