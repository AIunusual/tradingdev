"""Bind saved performance scopes to approved constructor and search provenance."""

from __future__ import annotations

import itertools
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal

import pandas as pd

from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.execution import ManifestError
from tradingdev.domain.optimization.grid_search import (
    finite_metric_value,
    is_better_metric,
    parameter_grid,
)
from tradingdev.domain.parameter_values import parameter_leaves
from tradingdev.domain.performance.artifacts import OptimizationSelection
from tradingdev.domain.research_evidence_values import canonical
from tradingdev.domain.strategies.execution import merge_strategy_parameters
from tradingdev.domain.validation.config import (
    ExplicitWalkForwardConfig,
    parse_walk_forward_config,
)
from tradingdev.domain.validation.date_windows import inclusive_day_window

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tradingdev.domain.approved_research import ApprovedResearch
    from tradingdev.domain.execution import ExecutionManifest
    from tradingdev.domain.performance.artifacts import (
        ObservationsBundle,
        PerformanceBundle,
        PerformanceScope,
        ScopeObservations,
    )
    from tradingdev.domain.research_evidence import BoundParameterProvenance


def effective_scope_parameters(
    manifest: ExecutionManifest, scope: PerformanceScope
) -> dict[str, JsonValue]:
    """Use the same recursive overlay as execution for captured trial overrides."""
    raw = manifest.strategy_execution.constructor_kwargs
    base = (
        raw.get("config", {}) if manifest.strategy_execution.kind == "bundled" else raw
    )
    if not isinstance(base, dict):
        raise ManifestError("Saved constructor parameters are not an object")
    return merge_strategy_parameters(
        base, scope.parameters if manifest.optimization else {}
    )


def verify_research_scopes(
    manifest: ExecutionManifest,
    performance: PerformanceBundle,
    observations: ObservationsBundle,
) -> OptimizationSelection | None:
    """Validate identities before attributing any saved result to approved research."""
    if performance.manifest_hash != manifest.manifest_hash:
        raise ManifestError("Research performance differs from its execution manifest")
    if manifest.optimization is not None:
        selection = verify_optimization_selection(manifest, performance)
        if performance.scopes.keys() != observations.scopes.keys():
            raise ManifestError(
                "Research optimization does not contain its complete grid"
            )
        train_bounds = optimization_date_bounds(
            optimization_backtest(manifest, "train")
        )
        test_bounds = optimization_date_bounds(optimization_backtest(manifest, "test"))
        for scope_id, observed in observations.scopes.items():
            _verify_optimization_observation_range(
                observed, *(test_bounds if scope_id == "test" else train_bounds)
            )
        return selection
    if performance.optimization_selection is not None:
        raise ManifestError("Search selection requires an optimization manifest")
    config = manifest.config_copy()
    if manifest.kind == "walk_forward":
        _verify_walk_forward(manifest, performance, observations)
    if manifest.kind == "backtest" and (
        list(performance.scopes) != ["full"]
        or performance.default_scope != "full"
        or performance.selected_train_scope is not None
    ):
        raise ManifestError("Research backtest requires one full scope")
    for scope_id, scope in performance.scopes.items():
        if scope.kind == "fold_summary":
            if manifest.kind != "walk_forward":
                raise ManifestError("Unexpected research summary scope")
            continue
        expected_id = (
            "full"
            if manifest.kind == "backtest"
            else f"fold/{scope.fold_index}/{scope.split}"
        )
        if (
            scope_id != expected_id
            or scope.trial_index is not None
            or scope.mode != config["backtest"]["mode"]
            or (
                manifest.kind == "backtest"
                and (scope.split != "full" or scope.fold_index is not None)
            )
            or (
                manifest.kind == "walk_forward"
                and (scope.split not in {"train", "test"} or scope.fold_index is None)
            )
            or canonical(scope.parameters)
            != canonical(effective_scope_parameters(manifest, scope))
            or canonical(scope.metadata.get("execution_context"))
            != canonical(
                {**config["backtest"], "random_seed": config.get("random_seed")}
            )
        ):
            raise ManifestError("Research scope differs from its approved execution")
        _verify_dates(
            observations,
            scope_id,
            config["backtest"]["start_date"],
            config["backtest"]["end_date"],
        )
    return None


def _saved_fold_layout(
    raw: object,
) -> tuple[int, dict[str, tuple[datetime, datetime]]]:
    """Read frozen fold identity; legacy records are never new execution inputs.

    Old manifests contain the normalized flat configuration. Preserve its mode
    precedence (purged ignores dates), without updating saved values or hashes.
    """
    if not isinstance(raw, dict):
        raise ManifestError("Saved walk-forward configuration is not an object")
    if "mode" in raw:
        config = parse_walk_forward_config(raw)
    else:
        dates = ("train_start", "train_end", "test_start", "test_end")
        expected = set(dates) | {
            "n_splits",
            "train_ratio",
            "expanding",
            "target_metric",
            "label_horizon",
            "embargo_bars",
            "purged_split",
        }
        if set(raw) != expected or type(raw["purged_split"]) is not bool:
            raise ManifestError("Saved walk-forward configuration is incomplete")
        if not raw["purged_split"] and raw["train_start"] is not None:
            config = ExplicitWalkForwardConfig.model_validate(
                {"mode": "explicit", **{key: raw[key] for key in dates}}
            )
        else:
            count = raw["n_splits"]
            if type(count) is not int or count < 1:
                raise ManifestError("Saved walk-forward fold count is invalid")
            return count, {}
    if isinstance(config, ExplicitWalkForwardConfig):
        return 1, {
            "train": (config.train_start, config.train_end),
            "test": (config.test_start, config.test_end),
        }
    return config.n_splits, {}


def _verify_walk_forward(
    manifest: ExecutionManifest,
    performance: PerformanceBundle,
    observations: ObservationsBundle,
) -> None:
    """Bind fold membership and explicit windows without reconstructing market data."""
    config = manifest.config_copy()
    count, explicit = _saved_fold_layout(config["validation"])
    expected = {
        f"fold/{index}/{split}" for index in range(count) for split in ("train", "test")
    }
    if (
        count < 1
        or set(performance.scopes) != expected | {"test_summary"}
        or set(observations.scopes) != expected
        or performance.default_scope != "test_summary"
        or performance.selected_train_scope is not None
    ):
        raise ManifestError("Research walk-forward requires its complete fold set")
    summary = performance.scopes["test_summary"]
    if (
        summary.kind != "fold_summary"
        or summary.split != "test"
        or summary.fold_index is not None
        or summary.trial_index is not None
        or summary.mode != config["backtest"]["mode"]
        or summary.metadata.get("n_folds") != count
        or canonical(summary.parameters)
        != canonical(effective_scope_parameters(manifest, summary))
    ):
        raise ManifestError("Research walk-forward summary differs from its folds")
    previous_test_end: pd.Timestamp | None = None
    for index in range(count):
        train = observations.scopes[f"fold/{index}/train"].timestamps
        test = observations.scopes[f"fold/{index}/test"].timestamps
        if not train or not test:
            raise ManifestError("Research walk-forward has no recorded timestamps")
        if pd.Timestamp(train[-1]) >= pd.Timestamp(test[0]) or (
            previous_test_end is not None and previous_test_end >= pd.Timestamp(test[0])
        ):
            raise ManifestError("Research walk-forward windows overlap or are reversed")
        previous_test_end = pd.Timestamp(test[-1])
        for split in ("train", "test"):
            scope_id = f"fold/{index}/{split}"
            if performance.scopes[scope_id].kind != "backtest":
                raise ManifestError("Research walk-forward fold is not a scalar scope")
            if explicit:
                start, end = explicit[split]
                lower, upper = inclusive_day_window(start, end)
                _verify_dates(
                    observations,
                    scope_id,
                    lower.isoformat(),
                    (upper - timedelta(microseconds=1)).isoformat(),
                )


def verify_optimization_selection(
    manifest: ExecutionManifest,
    performance: PerformanceBundle,
) -> OptimizationSelection | None:
    """Verify the metrics and settings behind a published training search."""
    search = manifest.optimization
    if search is None or performance.manifest_hash != manifest.manifest_hash:
        raise ManifestError("Optimization performance differs from its manifest")
    if len(performance.scopes) != search.total_combinations + 1:
        raise ManifestError("Research optimization does not contain its complete grid")
    combos = parameter_grid(search.param_ranges)
    expected_ids = [f"trial/{index}/train" for index in range(len(combos))] + ["test"]
    if (
        set(performance.scopes) != set(expected_ids)
        or performance.default_scope != "test"
    ):
        raise ManifestError("Research optimization does not contain its complete grid")
    selected = performance.selected_train_scope
    if selected not in expected_ids[:-1]:
        raise ManifestError("Research optimization has no selected training scope")
    winner: str | None = None
    winner_value: float | None = None
    objective_available = 0
    missing_objective = False
    train_config = optimization_backtest(manifest, "train")
    train_context = _optimization_scope_context(manifest, train_config)
    for index, parameters in enumerate(combos):
        scope_id = f"trial/{index}/train"
        scope = performance.scopes[scope_id]
        if scope.trial_index != index or canonical(scope.parameters) != canonical(
            parameters
        ):
            raise ManifestError("Research trial differs from its fixed grid position")
        _verify_optimization_scope(scope, "train", train_config, train_context)
        effective_scope_parameters(manifest, scope)
        missing_objective |= search.optimization_metric not in scope.values
        value = finite_metric_value(scope.values.get(search.optimization_metric))
        objective_available += value is not None
        if is_better_metric(value, winner_value, search.direction):
            winner, winner_value = scope_id, value
    if winner is None or selected != winner:
        raise ManifestError(
            "Research selected training scope differs from the saved objective"
        )
    test = performance.scopes["test"]
    if test.trial_index is not None or canonical(test.parameters) != canonical(
        performance.scopes[selected].parameters
    ):
        raise ManifestError(
            "Research test parameters differ from the selected training scope"
        )
    test_config = optimization_backtest(manifest, "test")
    _verify_optimization_scope(
        test, "test", test_config, _optimization_scope_context(manifest, test_config)
    )
    effective_scope_parameters(manifest, test)
    if missing_objective:
        if performance.optimization_selection is not None:
            raise ManifestError("Research trial lacks its recorded objective")
        return None
    selection = OptimizationSelection(
        selected_train_scope=winner,
        objective=search.optimization_metric,
        direction=search.direction,
        planned=len(combos),
        raw_attempted=len(combos),
        completed=len(combos),
        objective_available=objective_available,
        objective_unavailable=len(combos) - objective_available,
    )
    if (
        performance.optimization_selection is not None
        and performance.optimization_selection != selection
    ):
        raise ManifestError("Optimization selection differs from the verified grid")
    return selection


def verify_optimization_observations(
    manifest: ExecutionManifest,
    scope: PerformanceScope,
    observations: ScopeObservations,
    *,
    split: Literal["train", "test"],
) -> None:
    """Verify one observation with the same rules for complete and partial searches."""
    expected = optimization_backtest(manifest, split)
    _verify_optimization_scope(
        scope, split, expected, _optimization_scope_context(manifest, expected)
    )
    effective_scope_parameters(manifest, scope)
    _verify_optimization_observation_range(
        observations, *optimization_date_bounds(expected)
    )


def _optimization_scope_context(
    manifest: ExecutionManifest, config: BacktestConfig
) -> bytes:
    return canonical(
        {
            **config.model_dump(mode="json"),
            "random_seed": manifest.config.get("random_seed"),
        }
    )


def _verify_optimization_scope(
    scope: PerformanceScope,
    split: Literal["train", "test"],
    expected: BacktestConfig,
    expected_context: bytes,
) -> None:
    if (
        scope.kind != "backtest"
        or scope.fold_index is not None
        or scope.split != split
        or scope.mode != expected.mode
        or canonical(scope.metadata.get("execution_context")) != expected_context
    ):
        raise ManifestError(
            "Research optimization scope differs from its execution range"
        )
    settings = scope.metadata.get("settings")
    expected_settings = {
        "frequency": expected.timeframe,
        "periods_per_year": expected.periods_per_year,
        "risk_free_rate": expected.risk_free_rate,
        "required_return": expected.required_return,
    }
    if isinstance(settings, dict) and any(
        key in settings and (isinstance(settings[key], bool) or settings[key] != value)
        for key, value in expected_settings.items()
    ):
        raise ManifestError(
            "Research optimization metric settings differ from its manifest"
        )


def _verify_optimization_observation_range(
    observations: ScopeObservations, start: pd.Timestamp, end: pd.Timestamp
) -> None:
    if (
        not observations.timestamps
        or pd.to_datetime(observations.timestamps[0], utc=True) < start
        or pd.to_datetime(observations.timestamps[-1], utc=True) >= end
    ):
        raise ManifestError(
            "Research observations lie outside the approved scope range"
        )


def optimization_backtest(
    manifest: ExecutionManifest, split: Literal["train", "test"]
) -> BacktestConfig:
    """Resolve the frozen backtest configuration for an optimization split."""
    search = manifest.optimization
    if search is None:
        raise ManifestError("Missing optimization policy")
    start = search.test_start if split == "test" else search.train_start
    end = search.test_end if split == "test" else search.train_end
    return BacktestConfig.model_validate(
        {
            **manifest.config_copy()["backtest"],
            "start_date": start.isoformat(),
            "end_date": f"{end.isoformat()}T23:59:59.999999",
        }
    )


def optimization_date_bounds(
    config: BacktestConfig,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The approved dates are complete UTC calendar days."""
    return (
        pd.Timestamp(config.start_date, tz="UTC"),
        pd.Timestamp(config.end_date.date(), tz="UTC") + pd.Timedelta(days=1),
    )


def _verify_dates(
    observations: ObservationsBundle, scope_id: str, start: str, end: str
) -> None:
    scope = observations.scopes.get(scope_id)
    if scope is None:
        raise ManifestError("Research scope has no recorded timestamps")
    _verify_observation_dates(scope, start, end)


def _verify_observation_dates(scope: ScopeObservations, start: str, end: str) -> None:
    if not scope.timestamps:
        raise ManifestError("Research scope has no recorded timestamps")
    first = pd.Timestamp(start)
    last = pd.Timestamp(end)
    first = (
        first.tz_localize("UTC") if first.tzinfo is None else first.tz_convert("UTC")
    )
    last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
    if (
        pd.Timestamp(scope.timestamps[0]) < first
        or pd.Timestamp(scope.timestamps[-1]) > last
    ):
        raise ManifestError(
            "Research observations lie outside the approved scope range"
        )


def scope_parameter_sources(
    evidence: ApprovedResearch,
    manifest: ExecutionManifest,
    performance: PerformanceBundle,
    scope_id: str,
) -> tuple[BoundParameterProvenance, ...]:
    """Map Cartesian positions and preserve origins of duplicate candidate values."""
    search = manifest.optimization
    if search is None:
        targets = {
            target.target_id
            for target in evidence.research.targets
            if target.manifest_hash == manifest.manifest_hash
        }
        return tuple(
            item for item in evidence.research.parameters if item.target_id in targets
        )
    selected_id = performance.selected_train_scope if scope_id == "test" else scope_id
    match = re.fullmatch(r"trial/(\d+)/train", selected_id or "")
    if match is None:
        raise ManifestError("Research optimization scope has no fixed trial position")
    index = int(match[1])
    positions = itertools.islice(
        itertools.product(
            *(range(1, len(values) + 1) for values in search.param_ranges.values())
        ),
        index,
        index + 1,
    )
    position = next(positions, None)
    if position is None:
        raise ManifestError("Research optimization trial is outside its approved grid")
    target_ids = {
        f"search_{group}_{option}" for group, option in enumerate(position, 1)
    }
    prefix = "/config" if manifest.strategy_execution.kind == "bundled" else ""
    replaced = tuple(
        prefix + "/" + name.replace("~", "~0").replace("/", "~1")
        for name in search.param_ranges
    )
    sources = tuple(
        item
        for item in evidence.research.parameters
        if item.target_id in target_ids
        or (
            item.target_id == "base"
            and not any(
                item.parameter_path == path
                or item.parameter_path.startswith(path + "/")
                for path in replaced
            )
        )
    )
    effective = effective_scope_parameters(manifest, performance.scopes[scope_id])
    raw = manifest.strategy_execution.constructor_kwargs
    constructor = {**raw, "config": effective} if prefix else effective
    expected = parameter_leaves(constructor)
    if canonical({item.parameter_path: item.value for item in sources}) != canonical(
        expected
    ):
        raise ManifestError(
            "Research scope parameter sources differ from effective values"
        )
    return sources
