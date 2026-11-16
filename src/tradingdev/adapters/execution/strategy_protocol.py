"""Explicit data-only requests and responses for an isolated strategy instance."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from tradingdev.adapters.execution.frame_codec import FramePayload
from tradingdev.domain.strategies.schemas import StrategyDiagnostic

type Operation = Literal[
    "capture", "validate", "construct", "fit", "signals", "parameters", "trace"
]


class StrategyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    protocol: Literal[1] = 1
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    operation: Operation
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    class_name: str = Field(min_length=1, max_length=256)
    source: str | None = None
    kwargs: dict[str, JsonValue] = Field(default_factory=dict)
    random_seed: int | None = Field(default=None, ge=0, le=2**32 - 1, strict=True)
    frame: FramePayload | None = None


class StrategyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    protocol: Literal[1] = 1
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    operation: Operation
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    class_name: str
    value: JsonValue = None
    frame: FramePayload | None = None
    error: str | None = None
    error_kind: Literal["strategy_callback"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    diagnostics: list[StrategyDiagnostic] = Field(default_factory=list)
    signal_analysis: dict[str, JsonValue] = Field(default_factory=dict)
