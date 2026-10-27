"""Research-bound requests shared by MCP and supervised preparation workers."""

from typing import Self

from pydantic import JsonValue, field_serializer, model_validator

from tradingdev.app.contracts.common import ContractModel
from tradingdev.domain.backtest.schemas import WalkForwardConfig
from tradingdev.domain.execution import OptimizationSpec
from tradingdev.domain.research import ResearchSpec, research_payload


class BacktestInput(ContractModel):
    """Select a runnable strategy and an explicit research context."""

    strategy_id: str
    research: ResearchSpec
    revision_id: str | None = None
    parameters: dict[str, JsonValue] | None = None
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
