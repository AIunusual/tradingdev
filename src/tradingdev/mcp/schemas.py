"""MCP strategy authoring input contract."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from tradingdev.domain.research import ResearchSpec  # noqa: TC001


class SaveStrategyInput(BaseModel):
    """Create a new strategy or revise an explicitly selected current revision."""

    model_config = ConfigDict(extra="forbid")

    code: str
    yaml_config: str
    research: ResearchSpec
    strategy_id: str | None = None
    expected_revision_id: str | None = None
    display_name: str | None = None
