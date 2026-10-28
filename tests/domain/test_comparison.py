"""A fixed comparison cannot vary research assumptions or hide candidate settings."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from tradingdev.domain.comparison import ComparisonCandidate, ComparisonManifest
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.strategies.execution import StrategyExecution


def candidate(index: int, period: int, **changes: Any) -> ComparisonCandidate:
    config: dict[str, Any] = {
        "strategy": {
            "id": "shared",
            "revision_id": "revision_a",
            "parameters": {"period": period},
        },
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-08",
            "init_cash": 10000,
            "fees": 0.0006,
            "slippage": 0.0005,
            "periods_per_year": 365,
        },
        "data": {"source": "binance_vision", "market_type": "futures/um"},
        "random_seed": 42,
    }
    for section, value in changes.items():
        if isinstance(value, dict):
            config.setdefault(section, {}).update(value)
        else:
            config[section] = value
    manifest = ExecutionManifest.create(
        kind="backtest",
        config=config,
        strategy_execution=StrategyExecution(
            kind="generated", constructor_kwargs={"period": period}
        ),
    )
    return ComparisonCandidate(
        candidate_id=f"candidate_{index}",
        label=f"Window {period}",
        rationale="An explicitly proposed fixed setting, not a claimed optimum.",
        manifest=manifest,
    )


def test_comparison_hash_binds_complete_ordered_candidates_and_owns_input() -> None:
    first, second = candidate(1, 10), candidate(2, 20)
    bundle = ComparisonManifest.create([first, second])
    payload = bundle.model_dump(mode="json", exclude={"manifest_hash"})
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert bundle.manifest_hash == hashlib.sha256(canonical).hexdigest()
    assert ComparisonManifest.model_validate_json(bundle.model_dump_json()) == bundle
    first.manifest.strategy_execution.constructor_kwargs["period"] = 99
    bundle.verify()
    selected = bundle.candidate("candidate_1")
    assert selected.manifest.strategy_execution.constructor_kwargs == {"period": 10}
    selected.manifest.config["random_seed"] = 7
    bundle.verify()
    shared = bundle.common_config()
    assert "parameters" not in shared["strategy"]
    shared["backtest"]["fees"] = 0
    assert bundle.common_config()["backtest"]["fees"] == 0.0006


@pytest.mark.parametrize(
    "changes",
    [
        {"strategy": {"revision_id": "revision_b"}},
        {"strategy": {"id": "other"}},
        {"backtest": {"fees": 0}},
        {"backtest": {"symbol": "ETH/USDT"}},
        {"backtest": {"timeframe": "4h"}},
        {"backtest": {"end_date": "2024-01-09"}},
        {"data": {"market_type": "spot"}},
        {"random_seed": 43},
        {"parallel": {"reserve_cores": 1}},
    ],
)
def test_candidates_cannot_change_shared_research(changes: dict[str, Any]) -> None:
    changed = candidate(2, 20, **changes)
    with pytest.raises(ValueError, match="must share"):
        ComparisonManifest.create([candidate(1, 10), changed])


def test_fixed_candidates_are_bounded_distinct_and_ordered() -> None:
    with pytest.raises(ValueError):
        ComparisonManifest.create([])
    with pytest.raises(ValueError, match="duplicate effective"):
        ComparisonManifest.create([candidate(1, 10), candidate(2, 10)])
    with pytest.raises(ValueError, match="must follow"):
        ComparisonManifest.create([candidate(2, 10), candidate(1, 20)])
    duplicate_label = candidate(2, 20).model_copy(update={"label": "Window 10"})
    with pytest.raises(ValueError, match="labels must be distinct"):
        ComparisonManifest.create([candidate(1, 10), duplicate_label])
    eight = ComparisonManifest.create([candidate(i, i) for i in range(1, 9)])
    assert len(eight.candidates) == 8
    with pytest.raises(ValueError):
        ComparisonManifest.create([*eight.candidates, candidate(1, 9)])
    assert len(ComparisonManifest.create([candidate(1, 10)]).candidates) == 1


def test_mutation_unknown_member_and_wrong_expected_hash_fail() -> None:
    bundle = ComparisonManifest.create([candidate(1, 10), candidate(2, 20)])
    with pytest.raises(ValueError, match="expected hash"):
        bundle.verify(expected_hash="0" * 64)
    with pytest.raises(ValueError, match="Unknown comparison candidate"):
        bundle.candidate("candidate_3")
    bundle.candidates[1].manifest.config["random_seed"] = 43
    with pytest.raises(ValueError, match="hash"):
        bundle.verify()


def test_rationale_and_labels_are_frozen_before_results() -> None:
    original = ComparisonManifest.create([candidate(1, 10)])
    changed = original.model_copy(
        update={
            "candidates": (
                original.candidates[0].model_copy(
                    update={"rationale": "Selected after results"}
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="hash"):
        changed.verify()


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_schema_version_is_not_coerced(version: object) -> None:
    payload = ComparisonManifest.create([candidate(1, 10)]).model_dump(mode="json")
    payload["schema_version"] = version
    with pytest.raises(ValueError, match="must be an integer"):
        ComparisonManifest.model_validate(payload)


@pytest.mark.parametrize("strategy", [None, {}, "missing"])
def test_damaged_historical_strategy_shape_is_rejected(strategy: object) -> None:
    member = candidate(1, 10)
    payload = member.manifest.model_dump(mode="json", exclude={"manifest_hash"})
    payload["config"]["strategy"] = strategy
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    malformed = ExecutionManifest.model_validate(
        {**payload, "manifest_hash": hashlib.sha256(encoded).hexdigest()}
    )
    with pytest.raises(ValueError, match="requires a strategy identity"):
        ComparisonManifest.create([member.model_copy(update={"manifest": malformed})])


def test_comparison_cannot_hide_walk_forward_execution() -> None:
    selected = candidate(1, 10)
    config = selected.manifest.config_copy()
    config["validation"] = {"mode": "automatic", "n_splits": 2}
    walk_forward = ExecutionManifest.create(
        kind="walk_forward",
        config=config,
        strategy_execution=selected.manifest.strategy_execution,
    )
    with pytest.raises(ValueError, match="ordinary backtest"):
        ComparisonManifest.create(
            [selected.model_copy(update={"manifest": walk_forward})]
        )
