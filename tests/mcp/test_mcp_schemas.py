"""MCP DTO tests."""

import pytest
from pydantic import ValidationError
from tests.integration.execution_fixtures import research_input

from tradingdev.domain.research import NewResearchSpec
from tradingdev.mcp.schemas import SaveStrategyInput


def test_save_strategy_input_defaults() -> None:
    dto = SaveStrategyInput(
        code="class X: ...",
        yaml_config="strategy: {}",
        research=NewResearchSpec.model_validate(research_input()),
    )

    assert dto.strategy_id is None
    assert dto.expected_revision_id is None
    assert dto.display_name is None
    assert dto.research.strategy_outline


def test_save_requires_research_and_rejects_a_second_summary() -> None:
    with pytest.raises(ValidationError, match="research"):
        SaveStrategyInput.model_validate({"code": "pass", "yaml_config": "{}"})
    with pytest.raises(ValidationError, match="request_summary"):
        SaveStrategyInput.model_validate(
            {
                "code": "pass",
                "yaml_config": "{}",
                "research": research_input(),
                "request_summary": "A second, conflicting summary",
            }
        )
