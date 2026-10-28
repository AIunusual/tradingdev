"""Research-bound requests shared by MCP and supervised preparation workers."""

from typing import Self

from pydantic import Field, JsonValue, field_serializer, model_validator

from tradingdev.app.contracts.common import ContractModel
from tradingdev.domain.comparison import (
    MAX_COMPARISON_CANDIDATES,
    CandidateLabel,
    CandidateRationale,
)
from tradingdev.domain.execution import OptimizationSpec
from tradingdev.domain.research import ResearchSpec, research_payload
from tradingdev.domain.validation.config import WalkForwardConfig


class ResearchExecutionInput(ContractModel):
    """The one shared strategy identity, research context and execution policy."""

    strategy_id: str
    research: ResearchSpec
    revision_id: str | None = None
    backtest_overrides: dict[str, JsonValue] | None = None

    @field_serializer("research")
    def serialize_research(self, value: ResearchSpec) -> dict[str, JsonValue]:
        return research_payload(value)

    @model_validator(mode="after")
    def require_selected_revision_for_inheritance(self) -> Self:
        if self.research.kind == "inherit_selected_revision" and not self.revision_id:
            raise ValueError(
                "inherit_selected_revision requires an explicit revision_id; "
                "read get_strategy and select the immutable revision to reuse"
            )
        return self


class BacktestInput(ResearchExecutionInput):
    """Select one runnable strategy with optional fixed parameter overrides."""

    parameters: dict[str, JsonValue] | None = None


class ParameterCandidateInput(ContractModel):
    """A labeled parameter proposal captured before performance is available."""

    label: CandidateLabel
    rationale: CandidateRationale
    parameters: dict[str, JsonValue]


class ComparisonInput(ResearchExecutionInput):
    """An explicit ordered list of fixed candidates over one shared research."""

    candidates: list[ParameterCandidateInput] = Field(
        min_length=1, max_length=MAX_COMPARISON_CANDIDATES
    )


class WalkForwardInput(BacktestInput):
    """Optionally replace all validation settings for the selected research period."""

    validation_config: WalkForwardConfig | None = None


class OptimizationInput(BacktestInput):
    """An explicitly requested search within the research's complete period."""

    param_ranges: dict[str, list[JsonValue]]
    optimization_metric: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    def optimization_spec(self) -> OptimizationSpec:
        return OptimizationSpec.model_validate(
            self.model_dump(
                include={
                    "param_ranges",
                    "optimization_metric",
                    "train_start",
                    "train_end",
                    "test_start",
                    "test_end",
                }
            )
        )
