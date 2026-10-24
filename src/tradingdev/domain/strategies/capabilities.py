"""Declared strategy requirements, independent of model readiness and clients."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

type ExecutionKind = Literal["backtest", "walk_forward", "optimization"]
type TrainingRequirement = Literal["required", "optional", "unknown"]


def _supported_modes(requirement: TrainingRequirement) -> list[ExecutionKind] | None:
    if requirement == "required":
        return ["walk_forward"]
    if requirement == "optional":
        return ["backtest", "walk_forward", "optimization"]
    return None


def _validation_configuration() -> list[Literal["validation"]]:
    return ["validation"]


class StrategyCapabilities(BaseModel):
    """Intrinsic execution support; configuration and validation still apply.

    Unknown is not permission or evidence of a fitted model. Discovery reads
    declarations without importing or executing strategy source.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    training_requirement: TrainingRequirement = "unknown"
    supported_execution_modes: list[ExecutionKind] | None = None

    @model_validator(mode="after")
    def consistent_modes(self) -> Self:
        if self.supported_execution_modes != _supported_modes(
            self.training_requirement
        ):
            raise ValueError("Execution modes must match the training requirement")
        return self

    @classmethod
    def from_training_requirement(cls, requirement: TrainingRequirement) -> Self:
        return cls(
            training_requirement=requirement,
            supported_execution_modes=_supported_modes(requirement),
        )


class StrategyExecutionGuidance(BaseModel):
    """Recovery prerequisites for a strategy requiring training before scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_id: str
    capabilities: StrategyCapabilities
    recommended_mode: Literal["walk_forward"] = "walk_forward"
    required_configuration: list[Literal["validation"]] = Field(
        default_factory=_validation_configuration
    )

    @model_validator(mode="after")
    def training_is_required(self) -> Self:
        if self.capabilities.training_requirement != "required":
            raise ValueError(
                "Training guidance requires an explicit training requirement"
            )
        if self.required_configuration != ["validation"]:
            raise ValueError(
                "Walk-forward guidance must require validation configuration"
            )
        return self


class StrategyExecutionModeError(ValueError):
    """A known training requirement cannot be met by the requested mode."""

    code = "strategy_requires_training"

    def __init__(
        self,
        strategy_id: str,
        capabilities: StrategyCapabilities,
        kind: ExecutionKind,
    ) -> None:
        self.guidance = StrategyExecutionGuidance(
            strategy_id=strategy_id, capabilities=capabilities
        )
        super().__init__(
            f"Strategy '{strategy_id}' requires training before scoring; "
            f"'{kind}' does not fit the strategy. Use walk-forward validation "
            "with a valid validation configuration defining training and test windows. "
            "Removing validation does not make an untrained strategy executable."
        )


def ensure_execution_mode(
    strategy_id: str, capabilities: StrategyCapabilities, kind: ExecutionKind
) -> None:
    """Reject explicitly unsupported modes without guessing unknown capabilities."""
    supported = capabilities.supported_execution_modes
    if supported is not None and kind not in supported:
        raise StrategyExecutionModeError(strategy_id, capabilities, kind)
