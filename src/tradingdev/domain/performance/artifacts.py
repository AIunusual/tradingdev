"""Versioned, JSON-native performance scopes and original observations."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Literal, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

# Pydantic resolves these nested schemas at runtime.
from tradingdev.domain.backtest.execution_records import (  # noqa: TC001
    AccountState,
    ExecutionRecord,
)
from tradingdev.domain.backtest.metrics import calculate_passive_metrics
from tradingdev.domain.benchmark import (
    BenchmarkEvidence,
    aligned_closes,
    buy_and_hold,
)
from tradingdev.domain.performance.catalog import METRIC_CATALOG
from tradingdev.domain.research_policy import DEFAULT_FEES, DEFAULT_SLIPPAGE
from tradingdev.shared.utils.json_values import (
    normalize_json_object,
    normalize_json_value,
)

if TYPE_CHECKING:
    from tradingdev.domain.backtest.pipeline_result import PipelineResult
    from tradingdev.domain.backtest.result import BacktestResult


type MetricScalar = int | float | None


class ArtifactModel(BaseModel):
    """Validate stored structure without coercing booleans or accepting NaN."""

    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )


class FoldStats(ArtifactModel):
    """Descriptive finite test-fold observations, never a pooled return metric."""

    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    valid_count: int = Field(ge=0)


class MetricDefinitionSnapshot(ArtifactModel):
    """The exact metric definition recorded by the result producer."""

    id: str
    unit: str
    category: str
    provider: str
    description: str
    summary: bool
    optimization_direction: Literal["maximize", "minimize"] | None
    modes: tuple[str, ...]
    requires_annualization: bool


class BenchmarkDefinitionSnapshot(ArtifactModel):
    """The fixed baseline identity recorded by the result producer."""

    id: str
    asset_class: str
    symbol: str
    label: str
    weighting: str
    rationale: str


class BenchmarkCoverage(ArtifactModel):
    """How much of one result scope the baseline actually observed."""

    scope_bars: int = Field(ge=0)
    aligned_bars: int = Field(ge=0)
    missing_bars: int = Field(ge=0)
    complete: bool
    start: str | None = None
    end: str | None = None

    @model_validator(mode="after")
    def consistent_counts(self) -> Self:
        if self.aligned_bars + self.missing_bars != self.scope_bars:
            raise ValueError("Benchmark coverage counts must add up to the scope bars")
        if self.complete != (self.missing_bars == 0):
            raise ValueError("Benchmark coverage flag must match its missing bars")
        return self


class BenchmarkScope(ArtifactModel):
    """One result scope's baseline values, series and acquisition evidence.

    A scope without a defined baseline or without usable observations keeps an
    explicit status and reason. It never becomes an empty equity curve or a
    zero return, and the strategy's own values are unaffected.
    """

    status: Literal["available", "unavailable", "undefined"]
    asset_class: str | None = None
    definition: BenchmarkDefinitionSnapshot | None = None
    reason: str | None = None
    detail: str | None = None
    source: str | None = None
    market_type: str | None = None
    dataset_id: str | None = None
    reused_run_frame: bool | None = None
    init_cash: float | None = None
    capital_basis: Literal["run_init_cash", "normalized_unit"] | None = None
    fees: float | None = None
    slippage: float | None = None
    coverage: BenchmarkCoverage | None = None
    timestamps: list[str] | None = None
    equity_curve: list[float | None] | None = None
    returns: list[float | None] | None = None
    values: dict[str, MetricScalar] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def status_matches_evidence(self) -> Self:
        """An available baseline needs a series; an absent one must not fake it."""
        if self.status == "available":
            if self.definition is None or self.timestamps is None:
                raise ValueError("Available benchmark requires a definition and series")
            if self.equity_curve is None or len(self.equity_curve) != len(
                self.timestamps
            ):
                raise ValueError("Benchmark equity must align with its timestamps")
            if self.returns is not None and len(self.returns) != len(self.timestamps):
                raise ValueError("Benchmark returns must align with its timestamps")
            if self.reason is not None or self.capital_basis is None:
                raise ValueError("Available benchmark cannot carry an absent reason")
        elif self.values or self.equity_curve is not None:
            raise ValueError("An absent benchmark cannot record values or a series")
        if self.status != "available" and not self.reason:
            raise ValueError("An absent benchmark requires an explicit reason")
        return self


class PerformanceScope(ArtifactModel):
    """Complete values and provenance for one split or fold summary."""

    kind: Literal["backtest", "fold_summary"] = "backtest"
    mode: Literal["signal", "volume"]
    values: dict[str, MetricScalar | FoldStats]
    metadata: dict[str, JsonValue]
    split: Literal["full", "train", "test"] | None = "full"
    fold_index: int | None = Field(default=None, ge=0)
    trial_index: int | None = Field(default=None, ge=0)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def value_shape(self) -> Self:
        """Keep scalar results separate from descriptive fold statistics."""
        if self.kind == "backtest" and any(
            isinstance(value, FoldStats) for value in self.values.values()
        ):
            raise ValueError("Backtest scopes require scalar metrics")
        if self.kind == "fold_summary" and any(
            not isinstance(value, FoldStats) for value in self.values.values()
        ):
            raise ValueError("Fold summary scopes require FoldStats values")
        return self


class PerformanceBundle(ArtifactModel):
    """Complete scoped metrics, definitions and their execution identity."""

    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    manifest_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    default_scope: str
    scopes: dict[str, PerformanceScope]
    definitions: dict[str, MetricDefinitionSnapshot]
    selected_train_scope: str | None = None
    benchmarks: dict[str, BenchmarkScope] = Field(default_factory=dict)

    @model_validator(mode="after")
    def consistent_scopes(self) -> Self:
        """Every advertised scope and definition must resolve without guessing."""
        if self.default_scope not in self.scopes:
            raise ValueError("Default performance scope does not exist")
        if self.selected_train_scope is not None:
            selected = self.scopes.get(self.selected_train_scope)
            if selected is None or selected.split != "train":
                raise ValueError("Selected train scope must reference a train result")
        for key, definition in self.definitions.items():
            if key != definition.id:
                raise ValueError("Metric definition ID differs from its mapping key")
        for scope in self.scopes.values():
            if missing := scope.values.keys() - self.definitions.keys():
                raise ValueError(
                    f"Missing metric definitions: {', '.join(sorted(missing))}"
                )
        scalar_scopes = {
            key for key, scope in self.scopes.items() if scope.kind == "backtest"
        }
        if unknown := self.benchmarks.keys() - scalar_scopes:
            raise ValueError(
                "Benchmarks must reference scalar result scopes: "
                + ", ".join(sorted(unknown))
            )
        for benchmark in self.benchmarks.values():
            if missing := benchmark.values.keys() - self.definitions.keys():
                unknown_metrics = ", ".join(sorted(missing))
                raise ValueError(
                    f"Missing benchmark metric definitions: {unknown_metrics}"
                )
        return self


class ScopeObservations(ArtifactModel):
    """Aligned original bars and trade records, before presentation selection."""

    init_cash: float | None
    equity_curve: list[float | None]
    returns: list[float | None] | None
    timestamps: list[str] | None
    trades: list[dict[str, JsonValue]]
    execution_records: list[ExecutionRecord] | None = None
    account_history: list[AccountState] | None = None

    @model_validator(mode="after")
    def aligned_observations(self) -> Self:
        """Reject shifted series and references outside the stored bar sequence."""
        count = len(self.equity_curve)
        if self.returns is not None and len(self.returns) != count:
            raise ValueError("Returns and equity observations have different lengths")
        if self.timestamps is not None:
            if len(self.timestamps) != count:
                raise ValueError(
                    "Timestamps and equity observations have different lengths"
                )
            parsed = pd.to_datetime(self.timestamps, utc=True, errors="raise")
            if (
                parsed.hasnans
                or not parsed.is_monotonic_increasing
                or parsed.has_duplicates
            ):
                raise ValueError(
                    "Observation timestamps must be valid and strictly ordered"
                )
        for trade in self.trades:
            for key in ("entry_idx", "exit_idx"):
                if key in trade:
                    index = trade[key]
                    if (
                        isinstance(index, bool)
                        or not isinstance(index, int)
                        or not 0 <= index < count
                    ):
                        raise ValueError(
                            f"Trade {key} lies outside the observation sequence"
                        )
        if self.execution_records is not None:
            previous_id = previous_bar = previous_order = -1
            for record in self.execution_records:
                self._aligned_record(record.bar_index, record.timestamp, count)
                if (
                    record.execution_id <= previous_id
                    or record.bar_index < previous_bar
                ):
                    raise ValueError("Execution records must have ordered unique IDs")
                previous_id, previous_bar = record.execution_id, record.bar_index
                if record.order_id is not None:
                    if record.order_id <= previous_order:
                        raise ValueError("Filled order IDs must be unique and ordered")
                    previous_order = record.order_id
        if self.account_history is not None:
            if len(self.account_history) != count:
                raise ValueError(
                    "Account and equity observations have different lengths"
                )
            for index, state in enumerate(self.account_history):
                self._aligned_record(state.bar_index, state.timestamp, count)
                if state.bar_index != index:
                    raise ValueError("Account states must follow the bar sequence")
                equity = self.equity_curve[index]
                if equity is None or not math.isclose(
                    state.equity, equity, rel_tol=1e-10, abs_tol=1e-8
                ):
                    raise ValueError("Account equity differs from saved equity curve")
        return self

    def _aligned_record(self, index: int, timestamp: str | None, count: int) -> None:
        if not 0 <= index < count:
            raise ValueError("Execution/account index lies outside the bar sequence")
        expected = self.timestamps[index] if self.timestamps is not None else None
        if timestamp != expected:
            raise ValueError("Execution/account timestamp differs from its bar")


class ObservationsBundle(ArtifactModel):
    """Original observations for the scalar performance scopes."""

    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    manifest_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    scopes: dict[str, ScopeObservations]


class PerformanceArtifacts(ArtifactModel):
    """One validated pair of performance and observations JSON artifacts."""

    performance: PerformanceBundle
    observations: ObservationsBundle

    @model_validator(mode="after")
    def matching_identity_and_scopes(self) -> Self:
        """The two files describe the same run and all scalar scopes exactly once."""
        if (
            self.performance.run_id != self.observations.run_id
            or self.performance.manifest_hash != self.observations.manifest_hash
        ):
            raise ValueError("Performance and observations execution identities differ")
        expected = {
            key
            for key, scope in self.performance.scopes.items()
            if scope.kind == "backtest"
        }
        if expected != self.observations.scopes.keys():
            raise ValueError("Performance and observations scopes differ")
        return self


def _timestamps(result: BacktestResult) -> list[str] | None:
    if result.timestamps is None:
        return None
    values = result.timestamps
    if values.dtype.kind in {"i", "u", "f", "b"}:
        raise ValueError("Numeric observation indices are not timestamps")
    converted = pd.to_datetime(values, utc=True, errors="raise")
    if converted.hasnans:
        raise ValueError("Missing observation timestamp")
    return [timestamp.isoformat() for timestamp in converted]


def scope_from_backtest(
    result: BacktestResult,
    *,
    split: Literal["full", "train", "test"] = "full",
    fold_index: int | None = None,
    trial_index: int | None = None,
    parameters: dict[str, Any] | None = None,
) -> tuple[PerformanceScope, ScopeObservations]:
    """Copy calculated values and observations without recalculating metrics."""
    observations = ScopeObservations.model_validate(
        {
            "init_cash": result.init_cash,
            "equity_curve": normalize_json_value(result.equity_curve.tolist()),
            "returns": normalize_json_value(result.returns.tolist())
            if result.returns is not None
            else None,
            "timestamps": _timestamps(result),
            "trades": [normalize_json_object(trade) for trade in result.trades],
            "execution_records": result.execution_records,
            "account_history": result.account_history,
        }
    )
    metadata = normalize_json_object(result.metric_metadata)
    metadata["observations"] = {
        "bar_count": len(observations.equity_curve),
        "trade_count": len(observations.trades),
        "start_date": observations.timestamps[0] if observations.timestamps else None,
        "end_date": observations.timestamps[-1] if observations.timestamps else None,
        "execution_records_availability": _execution_availability(
            result.mode, observations.execution_records
        ),
        "account_history_availability": _execution_availability(
            result.mode, observations.account_history
        ),
        "accounting": "vectorbt_generic" if result.mode == "signal" else None,
        "execution_equity_mark": "requested_price_before_slippage",
        "account_history_mark": "bar_close",
        "execution_timestamp_semantics": "bar_timestamp_not_exact_intrabar_fill",
    }
    scope = PerformanceScope.model_validate(
        {
            "mode": result.mode,
            "values": normalize_json_object(result.metrics),
            "metadata": metadata,
            "split": split,
            "fold_index": fold_index,
            "trial_index": trial_index,
            "parameters": normalize_json_object(parameters or {}),
        }
    )
    return scope, observations


def _execution_availability(mode: str, values: object) -> str:
    if values is not None:
        return "available"
    if mode == "volume":
        return "unsupported_volume_accounting"
    return "not_recorded"


def build_artifacts(
    run_id: str,
    manifest_hash: str | None,
    default_scope: str,
    scopes: dict[str, PerformanceScope],
    observations: dict[str, ScopeObservations],
    *,
    selected_train_scope: str | None = None,
    benchmarks: dict[str, BenchmarkScope] | None = None,
) -> PerformanceArtifacts:
    """Freeze the producer's definitions and complete scope/observation mapping."""
    frozen_benchmarks = {} if benchmarks is None else benchmarks
    keys = {key for scope in scopes.values() for key in scope.values}
    keys |= {
        key for benchmark in frozen_benchmarks.values() for key in benchmark.values
    }
    definitions = {
        key: MetricDefinitionSnapshot.model_validate(asdict(METRIC_CATALOG[key]))
        for key in sorted(keys)
        if key in METRIC_CATALOG
    }
    return PerformanceArtifacts(
        performance=PerformanceBundle(
            run_id=run_id,
            manifest_hash=manifest_hash,
            default_scope=default_scope,
            scopes=scopes,
            definitions=definitions,
            selected_train_scope=selected_train_scope,
            benchmarks=frozen_benchmarks,
        ),
        observations=ObservationsBundle(
            run_id=run_id, manifest_hash=manifest_hash, scopes=observations
        ),
    )


def validate_projection(
    artifacts: PerformanceArtifacts, projection: dict[str, Any]
) -> None:
    """Reject a numeric result projection that disagrees with saved scope values."""
    bundle = artifacts.performance
    default = bundle.scopes[bundle.default_scope]
    values = default.model_dump(mode="json")["values"]
    if bundle.selected_train_scope is not None:
        train = bundle.scopes[bundle.selected_train_scope]
        train_values = train.model_dump(mode="json")["values"]
        if (
            projection.get("train_metrics") != train_values
            or projection.get("test_metrics") != values
        ):
            raise ValueError(
                "Optimization projection differs from saved train/test metrics"
            )
        if projection.get("best_params") != train.parameters:
            raise ValueError("Optimization projection differs from selected parameters")
        metric = projection.get("optimization_metric")
        if (
            not isinstance(metric, str)
            or metric not in train_values
            or metric not in values
        ):
            raise ValueError("Optimization projection has no valid target metric")
        for key, expected in (
            ("best_train_metric_value", train_values[metric]),
            ("best_oos_metric_value", values[metric]),
        ):
            if projection.get(key) != expected:
                raise ValueError(f"Optimization projection disagrees with {key}")
    elif default.kind == "fold_summary":
        expected = {"n_folds": default.metadata.get("n_folds"), **values}
        if projection != expected:
            raise ValueError("Fold projection differs from saved summary metrics")
    elif projection != values:
        raise ValueError("Result projection differs from saved performance metrics")


def _execution_scope(
    result: BacktestResult,
    config: dict[str, Any],
    *,
    split: Literal["full", "train", "test"],
    fold_index: int | None = None,
    parameters: dict[str, Any] | None = None,
    strategy_parameters: dict[str, Any] | None = None,
) -> tuple[PerformanceScope, ScopeObservations]:
    scope, observations = scope_from_backtest(
        result, split=split, fold_index=fold_index, parameters=parameters
    )
    metadata = dict(scope.metadata)
    if strategy_parameters is not None:
        metadata["strategy_parameters"] = normalize_json_object(strategy_parameters)
    metadata["execution_context"] = normalize_json_object(
        {**config.get("backtest", {}), "random_seed": config.get("random_seed")}
    )
    return scope.model_copy(update={"metadata": metadata}), observations


def _pipeline_parameters(pipeline: PipelineResult) -> dict[str, Any]:
    """Read initial parameters from the executed constructor, including defaults."""
    manifest = pipeline.execution_manifest
    if manifest is None:
        return normalize_json_object(
            pipeline.config_snapshot.get("strategy", {}).get("parameters", {})
        )
    execution = manifest.strategy_execution
    parameters = (
        execution.constructor_kwargs.get("config", {})
        if execution.kind == "bundled"
        else execution.constructor_kwargs
    )
    if not isinstance(parameters, dict):
        raise ValueError("Recorded strategy parameters must be an object")
    return normalize_json_object(parameters)


def _benchmark_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Read the cost and annualization settings the strategy itself executed with."""
    raw = config.get("backtest")
    values = raw if isinstance(raw, dict) else {}
    return {
        "fees": float(values.get("fees", DEFAULT_FEES)),
        "slippage": float(values.get("slippage", DEFAULT_SLIPPAGE)),
        "periods_per_year": values.get("periods_per_year"),
        "risk_free_rate": float(values.get("risk_free_rate", 0.0)),
        "required_return": float(values.get("required_return", 0.0)),
        "frequency": str(values.get("timeframe", "1h")),
    }


def _benchmark_identity(evidence: BenchmarkEvidence) -> dict[str, Any]:
    """Fields shared by every baseline status, carrying no values or series."""
    definition = evidence.definition
    return {
        "asset_class": evidence.asset_class,
        "definition": None
        if definition is None
        else {
            "id": definition.id,
            "asset_class": definition.asset_class,
            "symbol": definition.symbol,
            "label": definition.label,
            "weighting": definition.weighting,
            "rationale": definition.rationale,
        },
        "source": evidence.source,
        "market_type": evidence.market_type,
        "dataset_id": evidence.dataset_id,
        "reused_run_frame": evidence.reused_run_frame,
    }


def _absent_benchmark(
    evidence: BenchmarkEvidence,
    status: Literal["unavailable", "undefined"],
    reason: str,
    *,
    detail: str | None = None,
    coverage: dict[str, Any] | None = None,
) -> BenchmarkScope:
    """Record why a scope has no baseline instead of inventing a flat series."""
    return BenchmarkScope.model_validate(
        {
            "status": status,
            **_benchmark_identity(evidence),
            "reason": reason,
            "detail": evidence.detail if detail is None else detail,
            "coverage": coverage,
        }
    )


def _benchmark_scope(
    evidence: BenchmarkEvidence,
    *,
    timestamps: list[str] | None,
    init_cash: float | None,
    settings: dict[str, Any],
) -> BenchmarkScope:
    """Align one scope's bars to the baseline and price it with the run's costs."""
    if evidence.status == "undefined":
        return _absent_benchmark(
            evidence, "undefined", evidence.reason or "undefined_asset_class"
        )
    if evidence.status == "unavailable":
        return _absent_benchmark(
            evidence, "unavailable", evidence.reason or "benchmark_data_unavailable"
        )
    if timestamps is None:
        return _absent_benchmark(evidence, "unavailable", "missing_timestamps")
    index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    aligned_index, closes, missing = aligned_closes(evidence, index)
    coverage = {
        "scope_bars": len(index),
        "aligned_bars": len(aligned_index),
        "missing_bars": missing,
        "complete": missing == 0,
        "start": aligned_index[0].isoformat() if len(aligned_index) else None,
        "end": aligned_index[-1].isoformat() if len(aligned_index) else None,
    }
    if not len(closes):
        return _absent_benchmark(
            evidence,
            "unavailable",
            "no_shared_observations",
            coverage=coverage,
        )
    capital = 1.0 if init_cash is None else init_cash
    basis = "normalized_unit" if init_cash is None else "run_init_cash"
    try:
        hold = buy_and_hold(
            closes, capital, fees=settings["fees"], slippage=settings["slippage"]
        )
    except ValueError as exc:
        return _absent_benchmark(
            evidence,
            "unavailable",
            "invalid_benchmark_cost_settings",
            detail=str(exc),
            coverage=coverage,
        )
    try:
        analysis = calculate_passive_metrics(
            hold.equity,
            init_cash=capital,
            total_fees=hold.entry_fee,
            total_volume=hold.entry_notional,
            timestamps=aligned_index,
            periods_per_year=settings["periods_per_year"],
            risk_free_rate=settings["risk_free_rate"],
            required_return=settings["required_return"],
            frequency=settings["frequency"],
        )
    except ValueError as exc:
        # Supplementary evidence must never abort publishing the run itself.
        return _absent_benchmark(
            evidence,
            "unavailable",
            "benchmark_metrics_failed",
            detail=str(exc),
            coverage=coverage,
        )
    metadata = normalize_json_object(analysis.metadata)
    metadata["benchmark"] = normalize_json_object(
        {
            "weighting": "buy_and_hold",
            "capital_basis": basis,
            "entry_price": hold.entry_price,
            "quantity": hold.quantity,
            "entry_fee": hold.entry_fee,
            "entry_notional": hold.entry_notional,
            "exit": "not_liquidated_marked_at_last_close",
            "bar_alignment": "shared_timestamps_only",
        }
    )
    return BenchmarkScope.model_validate(
        {
            "status": "available",
            **_benchmark_identity(evidence),
            "init_cash": capital,
            "capital_basis": basis,
            "fees": settings["fees"],
            "slippage": settings["slippage"],
            "coverage": coverage,
            "timestamps": [stamp.isoformat() for stamp in aligned_index],
            "equity_curve": normalize_json_value(hold.equity.tolist()),
            "returns": normalize_json_value(analysis.returns.tolist())
            if analysis.returns is not None
            else None,
            "values": normalize_json_object(analysis.metrics),
            "metadata": metadata,
        }
    )


def bundles_from_pipeline(
    run_id: str,
    pipeline: PipelineResult,
    projection: dict[str, Any],
) -> PerformanceArtifacts:
    """Create simple or train/test fold scopes from an executed pipeline."""
    scopes: dict[str, PerformanceScope] = {}
    observations: dict[str, ScopeObservations] = {}
    parameters = _pipeline_parameters(pipeline)
    manifest_hash = (
        pipeline.execution_manifest.manifest_hash
        if pipeline.execution_manifest is not None
        else None
    )
    if pipeline.mode == "simple":
        if pipeline.backtest_result is None:
            raise ValueError("Simple pipeline has no backtest observations")
        scopes["full"], observations["full"] = _execution_scope(
            pipeline.backtest_result,
            pipeline.config_snapshot,
            split="full",
            parameters=parameters,
        )
        if scopes["full"].values != normalize_json_object(projection):
            raise ValueError("Saved metrics differ from executed backtest metrics")
        default_scope = "full"
    elif pipeline.mode == "walk_forward":
        if not pipeline.fold_results:
            raise ValueError("Walk-forward pipeline has no fold observations")
        for fold in pipeline.fold_results:
            splits: tuple[
                tuple[Literal["train", "test"], BacktestResult | None], ...
            ] = (
                ("train", fold.train_backtest),
                ("test", fold.test_backtest),
            )
            for split, result in splits:
                if result is None:
                    raise ValueError("Walk-forward fold has no backtest observations")
                scope_id = f"fold/{fold.fold_index}/{split}"
                if scope_id in scopes:
                    raise ValueError("Duplicate walk-forward fold index")
                scopes[scope_id], observations[scope_id] = _execution_scope(
                    result,
                    pipeline.config_snapshot,
                    split=split,
                    fold_index=fold.fold_index,
                    parameters=parameters,
                    strategy_parameters=fold.strategy_params,
                )
        test_scopes = {
            key: scope for key, scope in scopes.items() if scope.split == "test"
        }
        modes = {scope.mode for scope in test_scopes.values()}
        if len(modes) != 1:
            raise ValueError("Walk-forward folds use different backtest modes")
        metadata: dict[str, JsonValue] = {
            "aggregation": "fold_descriptive",
            "n_folds": len(pipeline.fold_results),
            "fold_metadata": {
                key: scope.metadata for key, scope in test_scopes.items()
            },
        }
        for key in ("settings", "providers", "execution_context"):
            values = [scope.metadata.get(key) for scope in test_scopes.values()]
            if all(value == values[0] for value in values):
                metadata[key] = values[0]
            else:
                metadata[f"{key}_variants"] = values
        if projection.get("n_folds") != len(pipeline.fold_results):
            raise ValueError("Saved fold count differs from executed folds")
        scopes["test_summary"] = PerformanceScope.model_validate(
            {
                "kind": "fold_summary",
                "parameters": parameters,
                "mode": modes.pop(),
                "values": {
                    key: value
                    for key, value in normalize_json_object(projection).items()
                    if key != "n_folds"
                },
                "metadata": metadata,
                "split": "test",
            }
        )
        default_scope = "test_summary"
    else:
        raise ValueError(f"Unsupported pipeline mode: {pipeline.mode}")
    benchmarks: dict[str, BenchmarkScope] = {}
    evidence = pipeline.benchmark
    if evidence is not None:
        settings = _benchmark_settings(pipeline.config_snapshot)
        for scope_id, scope in scopes.items():
            if scope.kind != "backtest":
                continue
            observation = observations[scope_id]
            benchmarks[scope_id] = _benchmark_scope(
                evidence,
                timestamps=observation.timestamps,
                init_cash=observation.init_cash,
                settings=settings,
            )
    return build_artifacts(
        run_id,
        manifest_hash,
        default_scope,
        scopes,
        observations,
        benchmarks=benchmarks,
    )
