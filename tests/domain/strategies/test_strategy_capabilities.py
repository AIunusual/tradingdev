"""Declared training requirements constrain modes without implying readiness."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from tradingdev.domain.backtest.schemas import StrategyRunConfig
from tradingdev.domain.strategies.capabilities import (
    StrategyCapabilities,
    StrategyExecutionGuidance,
    StrategyExecutionModeError,
    ensure_execution_mode,
)
from tradingdev.domain.strategies.catalog import BundledStrategyCatalog

if TYPE_CHECKING:
    from tradingdev.domain.strategies.capabilities import ExecutionKind


@pytest.mark.parametrize("kind", ["backtest", "optimization"])
def test_required_training_refuses_modes_that_do_not_fit(kind: ExecutionKind) -> None:
    capabilities = StrategyCapabilities.from_training_requirement("required")
    with pytest.raises(StrategyExecutionModeError) as caught:
        ensure_execution_mode("different_strategy", capabilities, kind)
    assert caught.value.code == "strategy_requires_training"
    assert caught.value.guidance.strategy_id == "different_strategy"
    assert caught.value.guidance.recommended_mode == "walk_forward"
    assert caught.value.guidance.required_configuration == ["validation"]
    assert "Removing validation" in str(caught.value)


@pytest.mark.parametrize("kind", ["backtest", "walk_forward", "optimization"])
def test_unknown_capabilities_do_not_guess_training_or_support(
    kind: ExecutionKind,
) -> None:
    capabilities = StrategyCapabilities()
    assert capabilities.supported_execution_modes is None
    ensure_execution_mode("generated_strategy", capabilities, kind)


def test_required_training_allows_walk_forward() -> None:
    capabilities = StrategyCapabilities.from_training_requirement("required")
    ensure_execution_mode("trainable_strategy", capabilities, "walk_forward")


@pytest.mark.parametrize(
    "payload",
    [
        {"training_requirement": "required", "supported_execution_modes": ["backtest"]},
        {
            "training_requirement": "unknown",
            "supported_execution_modes": ["walk_forward"],
        },
        {"training_requirement": "required", "supported_execution_modes": None},
    ],
)
def test_inconsistent_capabilities_cannot_cross_a_response_boundary(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="Execution modes must match"):
        StrategyCapabilities.model_validate(payload)


def test_training_guidance_cannot_drop_validation_prerequisite() -> None:
    with pytest.raises(ValidationError, match="must require validation"):
        StrategyExecutionGuidance(
            strategy_id="trainable_strategy",
            capabilities=StrategyCapabilities.from_training_requirement("required"),
            required_configuration=[],
        )


def test_catalog_declarations_cover_bundled_training_requirements() -> None:
    entries = {entry.strategy_id: entry for entry in BundledStrategyCatalog().entries()}
    for strategy_id in ("quantile_volume", "xgboost_direction", "safety_first_volume"):
        capabilities = entries[strategy_id].execution_capabilities
        assert capabilities.training_requirement == "required"
        assert capabilities.supported_execution_modes == ["walk_forward"]
    for strategy_id in ("kd_crossover", "glft_market_making", "glft_ml"):
        capabilities = entries[strategy_id].execution_capabilities
        assert capabilities.training_requirement == "optional"
        assert capabilities.supported_execution_modes == [
            "backtest",
            "walk_forward",
            "optimization",
        ]


def test_discovery_metadata_does_not_change_execution_configuration() -> None:
    plain = StrategyRunConfig(id="example", class_name="ExampleStrategy")
    declared = StrategyRunConfig(
        id="example", class_name="ExampleStrategy", training_requirement="required"
    )
    assert declared.training_requirement == "required"
    assert plain.model_dump(mode="json") == declared.model_dump(mode="json")
