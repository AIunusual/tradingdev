"""Bind explicit research intake to the existing offline execution capture."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.contracts.execution_requests import (
    BacktestInput,
    ComparisonInput,
    OptimizationInput,
    ResearchExecutionInput,
    WalkForwardInput,
)
from tradingdev.app.job_config import bind_strategy_revision, validate_optimization_grid
from tradingdev.app.prepared_execution import PreparedComparison, PreparedExecution
from tradingdev.app.research import SelectedResearchRevision, resolve_research_config
from tradingdev.domain.backtest.schemas import BacktestRunConfig
from tradingdev.domain.comparison import ComparisonCandidate, ComparisonManifest
from tradingdev.domain.execution import ExecutionManifest, ManifestError
from tradingdev.domain.strategies.capabilities import ensure_execution_mode
from tradingdev.domain.strategies.contract import (
    DRY_RUN_FIXTURE_ROWS,
    VALIDATE_FIXTURE_ROWS,
    SignalContractChecker,
)
from tradingdev.domain.strategies.schemas import StrategyMetadata
from tradingdev.shared.utils.config import load_config

if TYPE_CHECKING:
    from tradingdev.app.data_service import DataService
    from tradingdev.app.strategy_service import StrategyService
    from tradingdev.domain.execution import ExecutionKind, OptimizationSpec
    from tradingdev.domain.strategies.loader import StrategyLoader


class ResearchPreparationService:
    """Resolve intake once and capture that exact effective config and provenance."""

    def __init__(
        self,
        *,
        strategy_service: StrategyService,
        strategy_loader: StrategyLoader,
        data_service: DataService,
    ) -> None:
        self._strategies = strategy_service
        self._loader = strategy_loader
        self._data = data_service
        self._backtests = BacktestService(
            strategy_gate=strategy_service,
            strategy_loader=strategy_loader,
            data_service=data_service,
        )

    def prepare(
        self, kind: ExecutionKind, request: BacktestInput | OptimizationInput
    ) -> PreparedExecution:
        effective, config_path = self._research_context(kind, request)
        optimization = None
        if kind == "optimization":
            if not isinstance(request, OptimizationInput):
                raise ValueError("Optimization requires its search specification")
            optimization = request.optimization_spec()
            if effective.get("validation") is not None:
                raise ValueError(
                    "Optimization config must not contain validation settings; "
                    "use a config without walk-forward validation."
                )
            self._validate_optimization_period(effective, optimization)
        checked = self._backtests.capture_execution(
            effective,
            kind=kind,
            parameters=request.parameters,
            optimization=optimization,
        )
        if optimization is not None:
            validate_optimization_grid(self._loader, checked.manifest, optimization)
        return PreparedExecution(
            checked.manifest, config_path, checked.synthetic_fixture_rows
        )

    def prepare_comparison(self, request: ComparisonInput) -> PreparedComparison:
        """Resolve research and constructor defaults once, then check every proposal."""
        effective, config_path = self._research_context("backtest", request)
        run_config = BacktestRunConfig.model_validate(effective)
        if run_config.is_walk_forward:
            raise ManifestError(
                "Fixed comparisons require ordinary backtest settings without "
                "walk-forward validation"
            )
        effective["data"] = self._data.execution_config(effective, run_config.backtest)
        execution = self._loader.resolve_execution(effective["strategy"])
        candidates: list[ComparisonCandidate] = []
        evidence: dict[str, tuple[int, ...]] = {}
        checker = SignalContractChecker(self._loader)
        for index, proposal in enumerate(request.candidates, start=1):
            config = deepcopy(effective)
            self._backtests.prepare_strategy(config)
            selected = self._loader.override_execution(
                config["strategy"], execution, proposal.parameters
            )
            config["strategy"]["parameters"] = deepcopy(
                selected.constructor_kwargs["config"]
                if selected.kind == "bundled"
                else selected.constructor_kwargs
            )
            manifest = ExecutionManifest.create(
                kind="backtest", config=config, strategy_execution=selected
            )
            candidate_id = f"candidate_{index}"
            checked_rows: list[int] = []
            for fixture_rows in (VALIDATE_FIXTURE_ROWS, DRY_RUN_FIXTURE_ROWS):
                result = checker.check_config(
                    manifest.config_for_execution(),
                    fixture_rows=fixture_rows,
                    execution=manifest.strategy_execution,
                )
                errors = [
                    item for item in result["diagnostics"] if item.level == "error"
                ]
                if errors:
                    detail = "; ".join(
                        f"{item.code}: {item.message}" for item in errors
                    )
                    raise ManifestError(
                        f"{candidate_id} strategy execution contract failed "
                        f"({fixture_rows} rows): {detail}"
                    )
                checked_rows.append(fixture_rows)
            evidence[candidate_id] = tuple(checked_rows)
            candidates.append(
                ComparisonCandidate(
                    candidate_id=candidate_id,
                    label=proposal.label,
                    rationale=proposal.rationale,
                    manifest=manifest,
                )
            )
        return PreparedComparison(
            ComparisonManifest.create(candidates), config_path, evidence
        )

    def _research_context(
        self, kind: ExecutionKind, request: ResearchExecutionInput
    ) -> tuple[dict[str, Any], Path]:
        """Resolve one pinned base revision and research choice without execution."""
        spec = self._strategies.resolve_executable(
            request.strategy_id, request.revision_id
        )
        ensure_execution_mode(spec.strategy_id, spec.execution_capabilities, kind)
        config_path = Path(spec.config_path)
        raw_config = load_config(config_path)
        if not isinstance(raw_config, dict):
            raise ValueError("YAML config must be a mapping")
        selected = (
            SelectedResearchRevision(
                strategy_id=spec.strategy_id,
                revision_id=spec.revision_id,
                config=deepcopy(raw_config),
                config_hash=(
                    spec.metadata.config_hash
                    if isinstance(spec.metadata, StrategyMetadata)
                    else None
                ),
            )
            if spec.kind == "generated" and spec.revision_id is not None
            else None
        )
        # Bundled walk-forward examples may provide validation in a separate file.
        # Their sample market/date values still never replace the research request.
        if kind == "walk_forward" and raw_config.get("validation") is None:
            fallback = config_path.with_name("walkforward_config.yaml")
            if spec.kind == "bundled" and fallback.exists():
                candidate = load_config(fallback)
                strategy = (
                    candidate.get("strategy") if isinstance(candidate, dict) else None
                )
                if (
                    isinstance(candidate, dict)
                    and isinstance(strategy, dict)
                    and strategy.get("id") == spec.strategy_id
                    and candidate.get("validation") is not None
                ):
                    config_path, raw_config = fallback, candidate
        bind_strategy_revision(raw_config, spec)
        if (
            isinstance(request, WalkForwardInput)
            and request.validation_config is not None
        ):
            raw_config["validation"] = request.validation_config.model_dump(mode="json")
        effective = resolve_research_config(
            raw_config,
            request.research,
            selected_revision=selected,
            backtest_overrides=request.backtest_overrides,
        )
        return effective, config_path

    @staticmethod
    def _validate_optimization_period(
        config: dict[str, object], optimization: OptimizationSpec
    ) -> None:
        backtest = BacktestRunConfig.model_validate(config).backtest
        start = (
            backtest.start_date.replace(tzinfo=UTC)
            if backtest.start_date.tzinfo is None
            else backtest.start_date.astimezone(UTC)
        )
        end = (
            backtest.end_date.replace(tzinfo=UTC)
            if backtest.end_date.tzinfo is None
            else backtest.end_date.astimezone(UTC)
        )
        expected_start = datetime.combine(
            optimization.train_start, time.min, tzinfo=UTC
        )
        expected_end = datetime.combine(optimization.test_end, time.max, tzinfo=UTC)
        if (start, end) != (expected_start, expected_end):
            raise ManifestError(
                "Optimization uses inclusive UTC calendar days. Research start_date "
                "must equal train_start at 00:00:00 UTC and research end_date must "
                "equal test_end at 23:59:59.999999 UTC; explicitly revise the research "
                "period or search windows instead of changing the requested period."
            )
