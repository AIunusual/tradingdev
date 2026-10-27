"""Intake refuses to invent missing market choices or coerce invalid assumptions."""

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import TypeAdapter, ValidationError

from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.research import (
    InheritResearchSpec,
    NewResearchSpec,
    ResearchSpec,
    research_payload,
)
from tradingdev.domain.research_policy import research_policy_payload


def new_inputs(**changes: Any) -> dict[str, Any]:
    return {
        "kind": "new",
        "strategy_outline": "Buy when the trend strengthens",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
        **changes,
    }


@pytest.mark.parametrize(
    "field", ["strategy_outline", "symbol", "timeframe", "start_date", "end_date"]
)
def test_new_research_requires_every_user_choice(field: str) -> None:
    payload = new_inputs()
    del payload[field]
    with pytest.raises(ValidationError):
        TypeAdapter(ResearchSpec).validate_python(payload)


@pytest.mark.parametrize("field", ["strategy_outline", "symbol", "timeframe"])
@pytest.mark.parametrize("value", [None, "", "   ", 1, False])
def test_required_text_never_uses_a_sample(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        NewResearchSpec.model_validate(new_inputs(**{field: value}))


@pytest.mark.parametrize("field", ["init_cash", "fees", "slippage", "annualization"])
@pytest.mark.parametrize("value", [None, True, False, float("nan"), float("inf"), "10"])
def test_invalid_explicit_assumptions_do_not_become_defaults(
    field: str, value: object
) -> None:
    for model, base in ((NewResearchSpec, new_inputs()), (InheritResearchSpec, {})):
        with pytest.raises(ValidationError):
            model.model_validate({**base, field: value})


@pytest.mark.parametrize("field", ["init_cash", "annualization"])
@pytest.mark.parametrize("value", [0, -1])
def test_cash_and_factor_are_positive(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        NewResearchSpec.model_validate(new_inputs(**{field: value}))


def test_zero_costs_and_explicit_timezone_are_preserved() -> None:
    spec = NewResearchSpec.model_validate(
        new_inputs(fees=0, slippage=0, start_date="2024-01-01T08:00:00+08:00")
    )
    assert spec.fees == spec.slippage == 0
    assert spec.start_date == datetime(2024, 1, 1, tzinfo=UTC)
    assert "init_cash" not in spec.model_fields_set
    assert "fees" in spec.model_fields_set


@pytest.mark.parametrize("value", [0, False, None, "", "2024-02-02", "2024-02-01"])
def test_invalid_or_reversed_period_is_rejected(value: object) -> None:
    with pytest.raises(ValidationError):
        NewResearchSpec.model_validate(new_inputs(start_date=value))


def test_inherit_omission_is_distinct_from_explicit_null() -> None:
    assert InheritResearchSpec().model_fields_set == set()
    with pytest.raises(ValidationError):
        InheritResearchSpec.model_validate({"symbol": None})


@pytest.mark.parametrize(
    "spec",
    [
        NewResearchSpec.model_validate(new_inputs(fees=0)),
        NewResearchSpec.model_validate(
            {key: value for key, value in new_inputs().items() if key != "kind"}
        ),
        InheritResearchSpec(),
        InheritResearchSpec(fees=0),
    ],
)
def test_worker_roundtrip_keeps_omitted_assumptions_omitted(
    spec: NewResearchSpec | InheritResearchSpec,
) -> None:
    adapter = TypeAdapter(ResearchSpec)
    validated = adapter.validate_python(spec)
    assert validated.model_fields_set == spec.model_fields_set
    copied = adapter.validate_json(json.dumps(research_payload(spec)))
    assert copied.model_dump() == spec.model_dump()
    for field in ("init_cash", "fees", "slippage"):
        assert (field in copied.model_fields_set) == (field in spec.model_fields_set)


def test_policy_and_backtest_cost_defaults_have_one_source() -> None:
    policy = research_policy_payload()
    defaults = policy["simulation_defaults"]
    assert isinstance(defaults, dict)
    config = BacktestConfig.model_validate(
        {
            **{
                k: v
                for k, v in new_inputs().items()
                if k not in {"kind", "strategy_outline"}
            },
            "init_cash": defaults["init_cash"],
        }
    )
    assert config.fees == defaults["fees"]
    assert config.slippage == defaults["slippage"]


def test_discovery_schema_exposes_required_choices_and_inheritance() -> None:
    schema = TypeAdapter(ResearchSpec).json_schema()
    required = schema["$defs"]["NewResearchSpec"]["required"]
    assert set(required) == {
        "kind",
        "strategy_outline",
        "symbol",
        "timeframe",
        "start_date",
        "end_date",
    }
    assert schema["discriminator"]["propertyName"] == "kind"


@pytest.mark.parametrize(
    "payload",
    [
        new_inputs(),
        new_inputs(fees=0, slippage=0),
        new_inputs(init_cash=2500, annualization=252),
        new_inputs(annualization="unavailable"),
        {"kind": "inherit_selected_revision"},
        {"kind": "inherit_selected_revision", "fees": 0},
        {"kind": "inherit_selected_revision", "symbol": "ETH/USDT"},
    ],
)
def test_schema_valid_research_choices_pass_runtime(payload: dict[str, Any]) -> None:
    adapter = TypeAdapter(ResearchSpec)
    schema = adapter.json_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(payload)
    resolved = adapter.validate_json(json.dumps(payload))
    transported = research_payload(resolved)
    validator.validate(transported)
    assert adapter.validate_json(json.dumps(transported)) == resolved


@pytest.mark.parametrize("kind", ["new", "inherit_selected_revision"])
@pytest.mark.parametrize(
    "field",
    [
        "strategy_outline",
        "symbol",
        "timeframe",
        "start_date",
        "end_date",
        "init_cash",
        "fees",
        "slippage",
        "annualization",
    ],
)
def test_schema_and_runtime_reject_explicit_null(kind: str, field: str) -> None:
    payload: dict[str, Any] = new_inputs() if kind == "new" else {"kind": kind}
    payload[field] = None
    adapter = TypeAdapter(ResearchSpec)
    assert not Draft202012Validator(adapter.json_schema()).is_valid(payload)
    with pytest.raises(ValidationError):
        adapter.validate_json(json.dumps(payload))


@pytest.mark.parametrize("payload", [{"fees": 0}, {}])
def test_schema_and_runtime_require_wire_discriminator(payload: dict[str, Any]) -> None:
    adapter = TypeAdapter(ResearchSpec)
    assert not Draft202012Validator(adapter.json_schema()).is_valid(payload)
    with pytest.raises(ValidationError, match="union_tag_not_found"):
        adapter.validate_json(json.dumps(payload))


def test_constructor_kind_default_is_added_to_valid_wire_payload() -> None:
    spec = InheritResearchSpec(fees=0)
    assert "kind" not in spec.model_fields_set
    payload = research_payload(spec)
    assert payload == {"kind": "inherit_selected_revision", "fees": 0}
    Draft202012Validator(TypeAdapter(ResearchSpec).json_schema()).validate(payload)
