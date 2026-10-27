"""Shared research policy is applied to the actual complete execution config."""

from copy import deepcopy
from typing import Any

import pytest
import yaml

from tradingdev.app.research import (
    SelectedResearchRevision,
    build_research_config,
    resolve_research_config,
)
from tradingdev.domain.backtest.schemas import BacktestRunConfig
from tradingdev.domain.research import InheritResearchSpec, NewResearchSpec


def intake(**changes: Any) -> NewResearchSpec:
    return NewResearchSpec.model_validate(
        {
            "strategy_outline": "Follow a strengthening trend",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            **changes,
        }
    )


def draft(**changes: Any) -> dict[str, Any]:
    return {
        "strategy": {"class_name": "TrendStrategy", "parameters": {"fee_rate": 0.02}},
        "data": {"source": "binance_vision", "market_type": "spot"},
        **changes,
    }


def selected(config: dict[str, Any]) -> SelectedResearchRevision:
    return SelectedResearchRevision(
        strategy_id="generated_trend",
        revision_id="a" * 32,
        config=config,
        config_hash="b" * 64,
    )


def test_builder_creates_complete_config_without_changing_input() -> None:
    raw = draft()
    before = deepcopy(raw)
    config = build_research_config(raw, intake())
    assert raw == before
    assert config["backtest"]["init_cash"] == 10000
    assert config["backtest"]["fees"] == 0.0006
    assert config["backtest"]["slippage"] == 0.0005
    assert config["backtest"]["periods_per_year"] == 365
    assert config["strategy"]["parameters"]["fee_rate"] == 0.02
    assert config["data"]["requirements"]["market"] == {
        "source": "binance_vision",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
    }
    metadata = config["research"]
    assert metadata["policy_version"] == "research-v1"
    assert metadata["field_sources"]["/backtest/init_cash"] == "project_default"
    assert metadata["field_sources"]["/backtest/symbol"] == "request"
    assert "init_cash" not in metadata
    roundtrip = BacktestRunConfig.model_validate(yaml.safe_load(yaml.safe_dump(config)))
    assert roundtrip.model_dump(mode="json") == config


def test_explicit_zero_cost_and_capital_are_saved_with_request_origin() -> None:
    config = build_research_config(draft(), intake(init_cash=2500, fees=0, slippage=0))
    assert config["backtest"]["fees"] == config["backtest"]["slippage"] == 0
    assert config["backtest"]["init_cash"] == 2500
    for field in ("init_cash", "fees", "slippage"):
        assert config["research"]["field_sources"][f"/backtest/{field}"] == "request"


def test_preparation_replaces_bundled_examples_with_new_intake() -> None:
    raw = build_research_config(draft(), intake())
    raw["backtest"].update(symbol="OLD", timeframe="4h", init_cash=999999, fees=0.8)
    config = resolve_research_config(raw, intake(symbol="ETH/USDT", timeframe="15m"))
    assert config["backtest"]["symbol"] == "ETH/USDT"
    assert config["backtest"]["timeframe"] == "15m"
    assert config["backtest"]["init_cash"] == 10000
    assert config["backtest"]["fees"] == 0.0006
    assert config["data"]["requirements"]["market"]["symbol"] == "ETH/USDT"


def test_inherit_requires_an_immutable_selected_revision() -> None:
    with pytest.raises(ValueError, match="immutable revision"):
        resolve_research_config(draft(), InheritResearchSpec())


def test_inherit_preserves_frozen_values_and_true_yaml_identity() -> None:
    previous = build_research_config(
        draft(), intake(init_cash=12500, fees=0, slippage=0.0001, annualization=360)
    )
    config = resolve_research_config(
        previous, InheritResearchSpec(), selected_revision=selected(previous)
    )
    assert config["backtest"] == previous["backtest"]
    assert config["research"]["selected_revision"]["config_hash"] == "b" * 64
    assert set(config["research"]["field_sources"].values()) == {"selected_revision"}


def test_missing_inherited_outline_does_not_use_strategy_description() -> None:
    previous = build_research_config(draft(), intake())
    del previous["research"]
    previous["strategy"]["description"] = "sample strategy description"
    with pytest.raises(ValueError, match="strategy_outline"):
        resolve_research_config(
            previous, InheritResearchSpec(), selected_revision=selected(previous)
        )


def test_changed_market_does_not_inherit_annualization() -> None:
    previous = build_research_config(draft(), intake(annualization=360))
    config = resolve_research_config(
        previous,
        InheritResearchSpec(symbol="ETH/USDT"),
        selected_revision=selected(previous),
    )
    assert config["backtest"]["periods_per_year"] == 365
    assert (
        config["research"]["field_sources"]["/backtest/periods_per_year"]
        == "project_default"
    )
    yahoo = deepcopy(previous)
    yahoo["data"] = {"source": "yahoo_finance"}
    with pytest.raises(ValueError, match="No market annualization policy"):
        resolve_research_config(
            yahoo,
            InheritResearchSpec(symbol="AAPL"),
            selected_revision=selected(previous),
        )


@pytest.mark.parametrize("factor", [252, "unavailable"])
def test_mixed_market_requires_explicit_supported_assumption(factor: object) -> None:
    config = build_research_config(
        draft(data={"source": "yahoo_finance"}),
        intake(symbol="AAPL", annualization=factor),
    )
    assert config["backtest"]["periods_per_year"] == (
        None if factor == "unavailable" else factor
    )
    assert (
        config["research"]["field_sources"]["/backtest/periods_per_year"] == "request"
    )


def test_yahoo_does_not_imply_one_stock_calendar() -> None:
    with pytest.raises(ValueError, match="positive annualization factor"):
        build_research_config(
            draft(data={"source": "yahoo_finance"}), intake(symbol="AAPL")
        )


def test_weekly_bars_do_not_become_daily_observations_when_factor_is_given() -> None:
    config = build_research_config(draft(), intake(timeframe="1w", annualization=365))
    assert "unsupported_daily_sampling" in config["research"]["annualization_reason"]


@pytest.mark.parametrize("data", [{}, {"market_type": "spot"}, {"source": None}])
def test_data_provider_must_be_selected(data: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="explicit data.source"):
        build_research_config(draft(data=data), intake())


def test_market_type_is_not_filled_from_binance_example() -> None:
    with pytest.raises(ValueError, match="explicit data.market_type"):
        build_research_config(draft(data={"source": "binance_vision"}), intake())


def test_authoring_may_repeat_equal_owned_values_but_not_conflicts() -> None:
    complete = build_research_config(draft(), intake())
    complete["backtest"]["start_date"] = "2024-01-01T08:00:00+08:00"
    assert build_research_config(complete, intake())["backtest"]["init_cash"] == 10000
    complete["backtest"]["fees"] = 0.001
    with pytest.raises(ValueError, match="backtest.fees conflicts"):
        build_research_config(complete, intake())


@pytest.mark.parametrize("value", [None, True, -0.1, float("nan")])
def test_invalid_inherited_costs_do_not_become_defaults(value: object) -> None:
    previous = build_research_config(draft(), intake())
    previous["backtest"]["fees"] = value
    with pytest.raises(ValueError):
        resolve_research_config(
            previous, InheritResearchSpec(), selected_revision=selected(previous)
        )


def test_authoring_inherit_can_omit_research_data_without_false_conflict() -> None:
    previous = build_research_config(draft(), intake())
    config = build_research_config(
        {"strategy": {"class_name": "Revised"}},
        InheritResearchSpec(symbol="ETH/USDT"),
        selected_revision=selected(previous),
    )
    assert config["data"]["requirements"]["market"]["symbol"] == "ETH/USDT"


def test_backtest_overrides_cannot_create_a_second_research_source() -> None:
    with pytest.raises(ValueError, match="put research choices in ResearchSpec"):
        resolve_research_config(draft(), intake(), backtest_overrides={"fees": 0.0006})
    config = resolve_research_config(
        draft(), intake(), backtest_overrides={"stop_loss": 0.05}
    )
    assert config["backtest"]["stop_loss"] == 0.05


def test_direct_complete_config_remains_independent_of_research_intake() -> None:
    config = build_research_config(draft(), intake())
    del config["research"]
    parsed = BacktestRunConfig.model_validate(config)
    assert "research" not in parsed.model_dump(mode="json")


@pytest.mark.parametrize("change", ["missing", "unknown"])
def test_metadata_does_not_hide_fields_from_confirmation(change: str) -> None:
    config = build_research_config(draft(), intake())
    sources = config["research"]["field_sources"]
    if change == "missing":
        del sources["/backtest/fees"]
    else:
        sources["/backtest/secret"] = "request"
    with pytest.raises(ValueError, match="exactly the research-owned fields"):
        BacktestRunConfig.model_validate(config)


def test_volume_mode_retains_its_existing_capital_semantics() -> None:
    config = build_research_config(draft(backtest={"mode": "volume"}), intake())
    assert config["backtest"]["init_cash"] is None
    assert config["strategy"]["parameters"]["fee_rate"] == 0.02
