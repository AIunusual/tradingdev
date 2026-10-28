"""Integrity and canonicalization of complete execution requests."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from tradingdev.domain import execution as execution_module
from tradingdev.domain.backtest.schemas import BacktestRunConfig, ParallelConfig
from tradingdev.domain.data.requirements import (
    DataRequirement,
    FeatureSpec,
    MarketDataSpec,
)
from tradingdev.domain.data.schemas import DataConfig
from tradingdev.domain.execution import (
    ExecutionManifest,
    ManifestError,
    OptimizationSpec,
)
from tradingdev.domain.performance.catalog import METRIC_CATALOG
from tradingdev.domain.strategies.execution import StrategyExecution


def _strategy_execution() -> StrategyExecution:
    return StrategyExecution(
        kind="generated", constructor_kwargs={"parameters": {"windows": [2, 10]}}
    )


def _config() -> dict[str, Any]:
    return {
        "strategy": {
            "id": "example",
            "revision_id": "revision_a",
            "parameters": {"windows": [2, 10]},
        },
        "backtest": {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-04-30",
            "init_cash": 10_000,
            "periods_per_year": 365,
        },
        "data": {
            "source": "binance_vision",
            "market_type": "futures/um",
            "raw_dir": "/resolved/raw",
            "processed_dir": "/resolved/processed",
            "requirements": {
                "market": {
                    "source": "binance_vision",
                    "symbol": "BTCUSDT",
                    "timeframe": "1h",
                },
                "features": [
                    {
                        "type": "custom",
                        "source": "fixture",
                        "column": "value",
                        "path": "/resolved/features.parquet",
                        "raw_path": None,
                    }
                ],
            },
        },
    }


def _search(**overrides: Any) -> OptimizationSpec:
    values: dict[str, Any] = {
        "param_ranges": {"slow": [20, 10], "fast": [2, 4]},
        "optimization_metric": "sharpe_ratio",
        "train_start": "2024-01-01",
        "train_end": "2024-02-29",
        "test_start": "2024-03-01",
        "test_end": "2024-04-30",
    }
    values.update(overrides)
    return OptimizationSpec.model_validate(values)


def test_manifest_roundtrip_and_independent_canonical_digest() -> None:
    manifest = ExecutionManifest.create(
        strategy_execution=_strategy_execution(), kind="backtest", config=_config()
    )
    payload = manifest.model_dump(mode="json", exclude={"manifest_hash"})
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()

    assert manifest.manifest_hash == hashlib.sha256(canonical).hexdigest()
    assert ExecutionManifest.model_validate_json(manifest.model_dump_json()) == manifest
    manifest.verify(expected_hash=manifest.manifest_hash)
    assert "job_id" not in payload
    assert "created_at" not in payload


def test_implicit_defaults_match_explicit_defaults_without_changing_input() -> None:
    original = _config()
    before = deepcopy(original)
    implicit = ExecutionManifest.create(
        strategy_execution=_strategy_execution(), kind="backtest", config=original
    )
    explicit = ExecutionManifest.create(
        strategy_execution=_strategy_execution(),
        kind="backtest",
        config=implicit.config_copy(),
    )
    nullable = _config()
    nullable.update(validation=None, random_seed=None, parallel=None)
    with_nulls = ExecutionManifest.create(
        strategy_execution=_strategy_execution(), kind="backtest", config=nullable
    )

    assert implicit.manifest_hash == explicit.manifest_hash == with_nulls.manifest_hash
    assert original == before
    assert implicit.config["strategy"] == original["strategy"]
    assert implicit.config["validation"] is None
    assert implicit.config["random_seed"] is None
    assert implicit.config["parallel"] == {
        "reserve_cores": 2,
        "safety_factor": 0.6,
        "overhead_multiplier": 3.0,
    }
    backtest = implicit.config["backtest"]
    assert isinstance(backtest, dict)
    assert backtest["fees"] == 0.0006
    assert backtest["slippage"] == 0.0005
    assert backtest["mode"] == "signal"


def test_decode_preserves_saved_values_when_runtime_schema_adds_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = ExecutionManifest.create(
        kind="backtest", config=_config(), strategy_execution=_strategy_execution()
    )
    encoded = manifest.model_dump_json()

    class FutureParallelConfig(ParallelConfig):
        reserve_cores: int = 8
        new_parallel_option: bool = True

    class FutureRunConfig(BacktestRunConfig):
        new_execution_option: str = "future-default"
        parallel: FutureParallelConfig | None = None

    monkeypatch.setattr(execution_module, "BacktestRunConfig", FutureRunConfig)
    monkeypatch.setattr(execution_module, "ParallelConfig", FutureParallelConfig)
    loaded = ExecutionManifest.model_validate_json(encoded)
    submitted_later = ExecutionManifest.create(
        kind="backtest", config=_config(), strategy_execution=_strategy_execution()
    )

    assert loaded.model_dump_json() == encoded
    assert loaded.config_copy() == manifest.config_copy()
    assert "new_execution_option" not in loaded.config
    assert loaded.config_copy()["parallel"]["reserve_cores"] == 2
    assert submitted_later.config_copy()["new_execution_option"] == "future-default"
    assert submitted_later.config_copy()["parallel"] == {
        "reserve_cores": 8,
        "safety_factor": 0.6,
        "overhead_multiplier": 3.0,
        "new_parallel_option": True,
    }
    with pytest.raises(ManifestError, match="current runtime schema"):
        loaded.config_for_execution()
    assert submitted_later.config_for_execution() == submitted_later.config_copy()


def test_decode_does_not_synthesize_missing_execution_settings() -> None:
    manifest = ExecutionManifest.create(
        kind="backtest", config=_config(), strategy_execution=_strategy_execution()
    )
    payload = manifest.model_dump(mode="json", exclude={"manifest_hash"})
    del payload["config"]["backtest"]["fees"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_hash"] = hashlib.sha256(encoded.encode()).hexdigest()

    loaded = ExecutionManifest.model_validate_json(json.dumps(payload))

    assert "fees" not in loaded.config_copy()["backtest"]
    assert loaded.manifest_hash == payload["manifest_hash"]
    with pytest.raises(ManifestError, match="current runtime schema"):
        loaded.config_for_execution()


@pytest.mark.parametrize("value", [True, 1, "0.0006"])
def test_execution_refuses_implicit_coercion_of_saved_values(value: Any) -> None:
    manifest = ExecutionManifest.create(
        kind="backtest", config=_config(), strategy_execution=_strategy_execution()
    )
    payload = manifest.model_dump(mode="json", exclude={"manifest_hash"})
    payload["config"]["backtest"]["fees"] = value
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_hash"] = hashlib.sha256(encoded.encode()).hexdigest()
    loaded = ExecutionManifest.model_validate_json(json.dumps(payload))

    with pytest.raises(ManifestError, match="current runtime schema"):
        loaded.config_for_execution()


def test_execution_config_is_detached_and_saved_values_are_unchanged() -> None:
    manifest = ExecutionManifest.create(
        kind="backtest", config=_config(), strategy_execution=_strategy_execution()
    )
    execution = manifest.config_for_execution()
    assert execution == manifest.config_copy()
    execution["strategy"]["parameters"]["windows"].append(99)

    manifest.verify()
    assert manifest.config_copy()["strategy"]["parameters"]["windows"] == [2, 10]


@pytest.mark.parametrize("section", ["data", "requirements", "market", "features"])
def test_new_data_schema_defaults_do_not_change_saved_or_executed_settings(
    monkeypatch: pytest.MonkeyPatch, section: str
) -> None:
    manifest = ExecutionManifest.create(
        kind="backtest", config=_config(), strategy_execution=_strategy_execution()
    )
    encoded = manifest.model_dump_json()

    class FutureDataConfig(DataConfig):
        new_data_option: str = "future-default"

    class FutureDataRequirement(DataRequirement):
        new_requirement_option: str = "future-default"

    class FutureMarketDataSpec(MarketDataSpec):
        new_market_option: str = "future-default"

    class FutureFeatureSpec(FeatureSpec):
        new_feature_option: str = "future-default"

    class FutureMarketRequirement(DataRequirement):
        market: FutureMarketDataSpec

    class FutureFeatureRequirement(BaseModel):
        market: MarketDataSpec
        features: list[FutureFeatureSpec]

    if section == "data":
        monkeypatch.setattr(execution_module, "DataConfig", FutureDataConfig)
    else:
        model = {
            "requirements": FutureDataRequirement,
            "market": FutureMarketRequirement,
            "features": FutureFeatureRequirement,
        }[section]
        monkeypatch.setattr(execution_module, "DataRequirement", model)
    loaded = ExecutionManifest.model_validate_json(encoded)

    assert loaded.model_dump_json() == encoded
    with pytest.raises(ManifestError, match="Saved data settings differ"):
        loaded.config_for_execution()


@pytest.mark.parametrize("missing", ["data", "requirements", "market_type", "raw_path"])
def test_execution_requires_complete_saved_data_settings(missing: str) -> None:
    config = _config()
    if missing == "data":
        del config["data"]
    elif missing == "raw_path":
        del config["data"]["requirements"]["features"][0]["raw_path"]
    else:
        del config["data"][missing]
    manifest = ExecutionManifest.create(
        kind="backtest", config=config, strategy_execution=_strategy_execution()
    )

    manifest.verify()
    with pytest.raises(ManifestError, match="data settings"):
        manifest.config_for_execution()


def test_resolved_strategy_parameters_are_independent_from_revision_declaration() -> (
    None
):
    config = _config()
    strategy = _strategy_execution()
    manifest = ExecutionManifest.create(
        kind="backtest", config=config, strategy_execution=strategy
    )
    changed_strategy = StrategyExecution(
        kind="generated", constructor_kwargs={"parameters": {"windows": [3, 10]}}
    )
    changed = ExecutionManifest.create(
        kind="backtest", config=config, strategy_execution=changed_strategy
    )
    parameters = strategy.constructor_kwargs["parameters"]
    assert isinstance(parameters, dict)
    windows = parameters["windows"]
    assert isinstance(windows, list)
    windows.append(20)

    manifest.verify()
    assert manifest.config["strategy"] == config["strategy"]
    assert manifest.config == changed.config
    assert manifest.manifest_hash != changed.manifest_hash
    assert manifest.strategy_execution.constructor_kwargs == {
        "parameters": {"windows": [2, 10]}
    }


def test_resolved_strategy_nested_mutation_breaks_manifest_integrity() -> None:
    manifest = ExecutionManifest.create(
        kind="backtest", config=_config(), strategy_execution=_strategy_execution()
    )
    manifest.strategy_execution.constructor_kwargs["new_parameter"] = 20
    with pytest.raises(ManifestError, match="hash does not match"):
        manifest.verify()


@pytest.mark.parametrize("target", ["config", "strategy", "search"])
def test_nested_non_json_mutation_cannot_hide_behind_date_canonicalization(
    target: str,
) -> None:
    manifest = ExecutionManifest.create(
        kind="optimization",
        config=_config(),
        strategy_execution=StrategyExecution(
            kind="generated", constructor_kwargs={"day": "2024-01-01"}
        ),
        optimization=_search(param_ranges={"day": ["2024-01-01"]}),
    )
    if target == "config":
        mutable: Any = manifest.config["backtest"]
        mutable["start_date"] = dt.datetime(2024, 1, 1)
    elif target == "strategy":
        mutable = manifest.strategy_execution.constructor_kwargs
        mutable["day"] = dt.date(2024, 1, 1)
    else:
        assert manifest.optimization is not None
        mutable = manifest.optimization.param_ranges["day"]
        mutable[0] = dt.date(2024, 1, 1)

    with pytest.raises(ManifestError, match="Unsupported JSON value"):
        manifest.verify()


@pytest.mark.parametrize("missing", ["schema_version", "strategy_execution"])
def test_saved_manifest_requires_version_and_resolved_strategy(missing: str) -> None:
    manifest = ExecutionManifest.create(
        kind="backtest", config=_config(), strategy_execution=_strategy_execution()
    )
    payload = manifest.model_dump(mode="json")
    del payload[missing]
    with pytest.raises(ValidationError, match=missing):
        ExecutionManifest.model_validate_json(json.dumps(payload))


def test_dates_and_dictionary_order_have_stable_hashes() -> None:
    original = _config()
    equivalent = dict(reversed(list(original.items())))
    equivalent["backtest"] = dict(reversed(list(original["backtest"].items())))
    equivalent["backtest"]["start_date"] = dt.date(2024, 1, 1)
    equivalent["backtest"]["end_date"] = dt.datetime(2024, 4, 30)
    first = ExecutionManifest.create(
        strategy_execution=_strategy_execution(), kind="backtest", config=original
    )
    second = ExecutionManifest.create(
        strategy_execution=_strategy_execution(), kind="backtest", config=equivalent
    )
    assert first.manifest_hash == second.manifest_hash


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("strategy", "revision_id", "revision_b"),
        ("strategy", "parameters", {"windows": [3, 10]}),
        ("backtest", "fees", 0.001),
        ("backtest", "slippage", 0.002),
        ("backtest", "mode", "volume"),
        ("backtest", "end_date", "2024-04-29"),
        ("data", "raw_dir", "/another/raw"),
    ],
)
def test_execution_choices_change_digest(section: str, field: str, value: Any) -> None:
    original = _config()
    changed = deepcopy(original)
    changed[section][field] = value
    assert ExecutionManifest.create(
        strategy_execution=_strategy_execution(), kind="backtest", config=original
    ).manifest_hash != (
        ExecutionManifest.create(
            strategy_execution=_strategy_execution(), kind="backtest", config=changed
        ).manifest_hash
    )


def test_top_level_seed_and_parallel_policy_change_digest() -> None:
    config = _config()
    original = ExecutionManifest.create(
        strategy_execution=_strategy_execution(), kind="backtest", config=config
    )
    config["random_seed"] = 42
    seeded = ExecutionManifest.create(
        strategy_execution=_strategy_execution(), kind="backtest", config=config
    )
    config["parallel"] = {"reserve_cores": 1}
    parallel = ExecutionManifest.create(
        strategy_execution=_strategy_execution(), kind="backtest", config=config
    )
    assert (
        len({original.manifest_hash, seeded.manifest_hash, parallel.manifest_hash}) == 3
    )


def test_manifest_owns_inputs_and_returns_independent_nested_config() -> None:
    config = _config()
    search = _search()
    manifest = ExecutionManifest.create(
        strategy_execution=_strategy_execution(),
        kind="optimization",
        config=config,
        optimization=search,
    )
    config["strategy"]["parameters"]["windows"].append(100)
    search.param_ranges["fast"].append(99)
    copied = manifest.config_copy()
    copied["strategy"]["parameters"]["windows"].append(200)

    manifest.verify()
    assert manifest.config_copy()["strategy"]["parameters"]["windows"] == [2, 10]
    assert manifest.optimization is not None
    assert manifest.optimization.param_ranges["fast"] == [2, 4]


def test_nested_mutation_and_wrong_expected_hash_are_rejected() -> None:
    manifest = ExecutionManifest.create(
        strategy_execution=_strategy_execution(), kind="backtest", config=_config()
    )
    with pytest.raises(ManifestError, match="expected hash"):
        manifest.verify("0" * 64)
    strategy = manifest.config["strategy"]
    assert isinstance(strategy, dict)
    strategy["revision_id"] = "modified"
    with pytest.raises(ManifestError, match="hash does not match"):
        manifest.verify()
    with pytest.raises(ManifestError, match="hash does not match"):
        manifest.config_copy()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_config_and_ranges_are_rejected(value: float) -> None:
    config = _config()
    config["strategy"]["parameters"]["nested"] = [value]
    with pytest.raises(ManifestError, match="finite"):
        ExecutionManifest.create(
            strategy_execution=_strategy_execution(), kind="backtest", config=config
        )
    with pytest.raises(ValidationError, match="finite"):
        _search(param_ranges={"fast": [{"nested": value}]})


def test_nonfinite_search_mutation_is_not_serialized_to_null() -> None:
    manifest = ExecutionManifest.create(
        strategy_execution=_strategy_execution(),
        kind="optimization",
        config=_config(),
        optimization=_search(param_ranges={"optional": [None]}),
    )
    assert manifest.optimization is not None
    manifest.optimization.param_ranges["optional"][0] = float("nan")
    with pytest.raises(ManifestError, match="finite"):
        manifest.verify()


@pytest.mark.parametrize("invalid", [{1: "bad"}, {"bad": object()}, {"bad": (1, 2)}])
def test_non_json_values_are_rejected_without_lossy_coercion(invalid: Any) -> None:
    config = _config()
    config["extra"] = invalid
    with pytest.raises(ManifestError):
        ExecutionManifest.create(
            strategy_execution=_strategy_execution(), kind="backtest", config=config
        )


def test_unknown_version_extra_fields_and_tampered_content_cannot_load() -> None:
    manifest = ExecutionManifest.create(
        strategy_execution=_strategy_execution(), kind="backtest", config=_config()
    )
    payload = manifest.model_dump(mode="json")
    for changes in (
        {"schema_version": 1},
        {"schema_version": 3},
        {"schema_version": 99},
        {"unexpected": True},
        {"kind": "optimization"},
    ):
        with pytest.raises(ValidationError):
            ExecutionManifest.model_validate_json(json.dumps(payload | changes))
    payload["config"]["backtest"]["fees"] = 1.0
    with pytest.raises(ValidationError, match="hash does not match"):
        ExecutionManifest.model_validate_json(json.dumps(payload))


def test_modes_require_matching_validation_and_optimization() -> None:
    config = _config()
    with pytest.raises(ManifestError, match="validation"):
        ExecutionManifest.create(
            strategy_execution=_strategy_execution(), kind="walk_forward", config=config
        )
    with pytest.raises(ManifestError, match="optimization spec"):
        ExecutionManifest.create(
            strategy_execution=_strategy_execution(), kind="optimization", config=config
        )
    with pytest.raises(ManifestError, match="optimization spec"):
        ExecutionManifest.create(
            strategy_execution=_strategy_execution(),
            kind="backtest",
            config=config,
            optimization=_search(),
        )
    config["validation"] = {"mode": "automatic", "n_splits": 3}
    walk_forward = ExecutionManifest.create(
        strategy_execution=_strategy_execution(), kind="walk_forward", config=config
    )
    assert walk_forward.config_copy()["validation"]["train_ratio"] == 0.8
    for kind in ("backtest", "optimization"):
        with pytest.raises(ManifestError, match="validation"):
            ExecutionManifest.create(
                strategy_execution=_strategy_execution(), kind=kind, config=config
            )


def test_search_order_defaults_roundtrip_and_derived_combination_count() -> None:
    search = _search()
    assert list(search.param_ranges) == ["fast", "slow"]
    assert search.param_ranges["slow"] == [20, 10]
    assert search.total_combinations == 4
    assert "total_combinations" not in search.model_dump()
    manifest = ExecutionManifest.create(
        strategy_execution=_strategy_execution(),
        kind="optimization",
        config=_config(),
        optimization=search,
    )
    assert ExecutionManifest.model_validate_json(manifest.model_dump_json()) == manifest
    assert search.direction == "maximize"
    assert search.trial_timeout_seconds == 300


@pytest.mark.parametrize(
    "field",
    [
        "direction",
        "trial_timeout_seconds",
    ],
)
def test_saved_search_cannot_omit_defaulted_policy_fields(field: str) -> None:
    manifest = ExecutionManifest.create(
        kind="optimization",
        config=_config(),
        strategy_execution=_strategy_execution(),
        optimization=_search(),
    )
    payload = manifest.model_dump(mode="json")
    del payload["optimization"][field]

    with pytest.raises(ValidationError, match="Saved optimization policy is missing"):
        ExecutionManifest.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "change",
    [
        {"param_ranges": {"slow": [10, 20], "fast": [2, 4]}},
        {"optimization_metric": "total_return"},
        {"train_end": "2024-02-28"},
        {"test_start": "2024-03-02"},
        {"trial_timeout_seconds": 301},
    ],
)
def test_search_choices_change_digest(change: dict[str, Any]) -> None:
    original = ExecutionManifest.create(
        strategy_execution=_strategy_execution(),
        kind="optimization",
        config=_config(),
        optimization=_search(),
    )
    changed = ExecutionManifest.create(
        strategy_execution=_strategy_execution(),
        kind="optimization",
        config=_config(),
        optimization=_search(**change),
    )
    assert original.manifest_hash != changed.manifest_hash


@pytest.mark.parametrize(
    "change",
    [
        {"param_ranges": {}},
        {"param_ranges": {"fast": []}},
        {"optimization_metric": "unknown"},
        {"direction": "invalid"},
        {"trial_timeout_seconds": 0},
        {"trial_timeout_seconds": True},
        {"confirmation_timeout_seconds": -1},
        {"confirmation_poll_interval": float("inf")},
        {"confirmation_poll_interval": 0},
        {"train_end": "2024-03-01"},
        {"test_end": "2024-03-01"},
        {"test_start": "2024-03-01T00:00:00"},
        {"train_start": "20240101"},
        {"unexpected": True},
    ],
)
def test_invalid_search_requests_are_rejected(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _search(**change)


@pytest.mark.parametrize(
    ("metric", "direction"),
    [
        ("max_drawdown", "minimize"),
        ("max_drawdown_amount", "minimize"),
        ("sortino_ratio", "maximize"),
        ("daily_pnl_mean", "maximize"),
    ],
)
def test_new_optimization_targets_resolve_catalog_direction(
    metric: str, direction: str
) -> None:
    manifest = ExecutionManifest.create(
        kind="optimization",
        config=_config(),
        strategy_execution=_strategy_execution(),
        optimization=_search(optimization_metric=metric),
    )
    assert manifest.schema_version == 5
    assert manifest.optimization is not None
    assert manifest.optimization.direction == direction


@pytest.mark.parametrize("frequency", ["1w", "1M", "unknown"])
@pytest.mark.parametrize(
    "metric",
    [
        "annual_return",
        "sharpe_ratio",
        "daily_max_drawdown",
        "daily_pnl_mean",
        "monthly_pnl_std",
    ],
)
def test_new_optimization_rejects_objectives_without_daily_observations(
    frequency: str, metric: str
) -> None:
    config = _config()
    config["backtest"]["timeframe"] = frequency
    with pytest.raises(ManifestError, match="requires daily or finer bars"):
        ExecutionManifest.create(
            kind="optimization",
            config=config,
            strategy_execution=_strategy_execution(),
            optimization=_search(optimization_metric=metric),
        )


@pytest.mark.parametrize("frequency", ["1h", "1d", "24h"])
def test_new_optimization_accepts_daily_observation_objectives(frequency: str) -> None:
    config = _config()
    config["backtest"]["timeframe"] = frequency
    manifest = ExecutionManifest.create(
        kind="optimization",
        config=config,
        strategy_execution=_strategy_execution(),
        optimization=_search(),
    )
    assert manifest.config_copy()["backtest"]["timeframe"] == frequency


@pytest.mark.parametrize(
    "metric", ["total_return", "total_pnl", "max_drawdown", "win_rate"]
)
def test_new_optimization_accepts_bar_objectives_on_weekly_data(metric: str) -> None:
    config = _config()
    config["backtest"].update(timeframe="1w", periods_per_year=None)
    manifest = ExecutionManifest.create(
        kind="optimization",
        config=config,
        strategy_execution=_strategy_execution(),
        optimization=_search(optimization_metric=metric),
    )
    assert manifest.optimization is not None
    assert manifest.optimization.optimization_metric == metric


@pytest.mark.parametrize(
    ("metric", "mode", "annualization", "message"),
    [
        ("sharpe_ratio", "volume", 365, "not applicable"),
        ("total_return", "volume", 365, "not applicable"),
        ("max_drawdown", "volume", 365, "not applicable"),
        ("annual_return", "signal", None, "requires periods_per_year"),
        ("sortino_ratio", "signal", None, "requires periods_per_year"),
    ],
)
def test_new_optimization_rejects_unavailable_objectives(
    metric: str, mode: str, annualization: float | None, message: str
) -> None:
    config = _config()
    config["backtest"].update(mode=mode, periods_per_year=annualization)
    if mode == "volume":
        config["backtest"].update(init_cash=None, position_size=100)
    with pytest.raises(ManifestError, match=message):
        ExecutionManifest.create(
            kind="optimization",
            config=config,
            strategy_execution=_strategy_execution(),
            optimization=_search(optimization_metric=metric),
        )


def test_new_optimization_rejects_direction_override() -> None:
    with pytest.raises(ManifestError, match="requires direction 'minimize'"):
        ExecutionManifest.create(
            kind="optimization",
            config=_config(),
            strategy_execution=_strategy_execution(),
            optimization=_search(
                optimization_metric="max_drawdown", direction="maximize"
            ),
        )


@pytest.mark.parametrize("metric", ["n_days", "open_trades", "unknown_metric"])
def test_non_objective_catalog_entries_are_not_search_targets(metric: str) -> None:
    with pytest.raises(ValidationError, match="not an optimization objective"):
        _search(optimization_metric=metric)


def test_saved_direction_is_not_reinterpreted_by_changed_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = ExecutionManifest.create(
        kind="optimization",
        config=_config(),
        strategy_execution=_strategy_execution(),
        optimization=_search(optimization_metric="max_drawdown"),
    )
    saved = manifest.model_dump_json()
    monkeypatch.setitem(
        METRIC_CATALOG,
        "max_drawdown",
        replace(METRIC_CATALOG["max_drawdown"], optimization_direction="maximize"),
    )
    loaded = ExecutionManifest.model_validate_json(saved)
    loaded.verify()
    assert loaded.optimization is not None
    assert loaded.optimization.direction == "minimize"
    assert loaded.manifest_hash == manifest.manifest_hash
    # Removing a catalog entry also cannot change the immutable saved policy.
    monkeypatch.delitem(METRIC_CATALOG, "max_drawdown")
    assert ExecutionManifest.model_validate_json(saved) == manifest
