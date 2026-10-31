"""Backtest schema tests."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from tradingdev.domain.backtest.schemas import (
    BacktestConfig,
    BacktestRunConfig,
    ParallelConfig,
)
from tradingdev.domain.execution import ExecutionManifest, ManifestError
from tradingdev.domain.strategies.execution import StrategyExecution


def _config() -> dict[str, Any]:
    return {
        "random_seed": 42,
        "strategy": {"id": "fixture", "parameters": {}},
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "init_cash": 10000.0,
        },
    }


@pytest.mark.parametrize(
    ("multiplier", "config_message"),
    [
        (0.0, r"parallel.overhead_multiplier"),
        (-1.0, r"parallel.overhead_multiplier"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (-float("inf"), "finite"),
    ],
)
def test_parallel_overhead_rejected_at_configuration_boundary(
    multiplier: float,
    config_message: str,
) -> None:
    with pytest.raises(ValidationError, match="overhead_multiplier") as error:
        ParallelConfig(overhead_multiplier=multiplier)
    assert error.value.errors()[0]["loc"] == ("overhead_multiplier",)
    config = {**_config(), "parallel": {"overhead_multiplier": multiplier}}
    # Raw nonfinite JSON is rejected by the enclosing run contract first.
    with pytest.raises(ValidationError, match=config_message):
        BacktestRunConfig.model_validate(config)


@pytest.mark.parametrize("multiplier", [0.5, 3.0, 1e308])
def test_parallel_overhead_accepts_finite_positive_values(multiplier: float) -> None:
    assert (
        ParallelConfig(overhead_multiplier=multiplier).overhead_multiplier == multiplier
    )


def test_backtest_run_config_accepts_only_root_random_seed() -> None:
    config = BacktestRunConfig.model_validate(_config())

    assert config.random_seed == 42
    assert "random_seed" not in config.backtest.model_dump()
    wrong_location = _config()
    wrong_location["backtest"]["random_seed"] = 7
    with pytest.raises(ValidationError, match=r"backtest.random_seed"):
        BacktestRunConfig.model_validate(wrong_location)


@pytest.mark.parametrize("seed", [True, "42", 42.0, -1, 2**32])
def test_run_seed_rejects_coercion_and_out_of_range_values(seed: object) -> None:
    config = _config()
    config["random_seed"] = seed
    with pytest.raises(ValidationError, match="random_seed"):
        BacktestRunConfig.model_validate(config)


@pytest.mark.parametrize(
    ("path", "unknown_field"),
    [
        ((), "random_sead"),
        (("backtest",), "random_sead"),
        (("strategy",), "paramaters"),
        (("data",), "sorce"),
        (("data", "requirements"), "feature"),
        (("data", "requirements", "market"), "timefram"),
        (("validation",), "n_split"),
        (("parallel",), "reserved_cores"),
    ],
)
def test_unknown_fixed_config_fields_are_not_silently_ignored(
    path: tuple[str, ...], unknown_field: str
) -> None:
    config = _config()
    config["data"] = {
        "requirements": {"market": {"symbol": "BTC/USDT", "timeframe": "1h"}}
    }
    config.update(validation={"mode": "automatic"}, parallel={})
    section = config
    for key in path:
        section = section[key]
    section[unknown_field] = 42
    with pytest.raises(ValidationError, match=unknown_field):
        BacktestRunConfig.model_validate(config)


def test_parameters_remain_dynamic_without_inventing_environment_defaults() -> None:
    config = _config()
    config["strategy"]["parameters"] = {"nested": {"any_parameter": [1, "x"]}}
    validated = BacktestRunConfig.model_validate(config).model_dump(mode="json")
    assert validated["strategy"] == config["strategy"]
    assert validated["data"] == {}


@pytest.mark.parametrize(
    ("section", "invalid", "message"),
    [
        ("backtest", float("nan"), "finite"),
        ("backtest", "NaN", "finite"),
        ("backtest", "1e999", "finite"),
        ("parameters", {"threshold": float("nan")}, "finite"),
        ("parameters", {"nested": [float("inf")]}, "finite"),
        ("parameters", {"nested": {1: "value"}}, "keys must be strings"),
        ("parameters", {"nested": object()}, "Unsupported JSON value"),
    ],
)
def test_raw_config_and_manifest_reject_the_same_lossy_values(
    section: str, invalid: object, message: str
) -> None:
    config = _config()
    if section == "backtest":
        config["backtest"]["fees"] = invalid
    else:
        config["strategy"]["parameters"] = invalid
    with pytest.raises(ValidationError, match=message):
        BacktestRunConfig.model_validate(config)
    with pytest.raises(ManifestError, match=message):
        ExecutionManifest.create(
            kind="backtest",
            config=config,
            strategy_execution=StrategyExecution(
                kind="generated", constructor_kwargs={}
            ),
        )


def test_raw_config_and_manifest_reject_circular_parameters() -> None:
    config = _config()
    circular: list[Any] = []
    circular.append(circular)
    config["strategy"]["parameters"]["nested"] = circular
    with pytest.raises(ValidationError, match="circular references"):
        BacktestRunConfig.model_validate(config)
    with pytest.raises(ManifestError, match="circular references"):
        ExecutionManifest.create(
            kind="backtest",
            config=config,
            strategy_execution=StrategyExecution(
                kind="generated", constructor_kwargs={}
            ),
        )


def test_resolved_config_rejects_nonfinite_model_defaults() -> None:
    class InvalidDefaultBacktest(BacktestConfig):
        fees: float = float("nan")

    class InvalidDefaultRun(BacktestRunConfig):
        backtest: InvalidDefaultBacktest

    with pytest.raises(ValidationError, match="finite"):
        InvalidDefaultRun.model_validate(_config())


def test_backtest_config_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError, match="end_date must be after start_date"):
        BacktestConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-02-01",
            end_date="2024-01-01",
            init_cash=10_000.0,
        )


def test_backtest_config_rejects_equal_start_and_end() -> None:
    with pytest.raises(ValidationError, match="end_date must be after start_date"):
        BacktestConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-01",
            init_cash=10_000.0,
        )
