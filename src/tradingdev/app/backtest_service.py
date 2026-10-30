"""Application service for backtest execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal, Protocol

from tradingdev.app.benchmark_service import BenchmarkService
from tradingdev.app.data_service import DataService, LoadedDataset
from tradingdev.app.execution_dataset import ExecutionDataset
from tradingdev.app.job_config import bind_strategy_revision
from tradingdev.app.strategy_service import (
    StrategyNotExecutableError,
    StrategyService,
)
from tradingdev.domain.backtest.engines import create_backtest_engine
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.schemas import (
    BacktestConfig,
    BacktestRunConfig,
    ParallelConfig,
)
from tradingdev.domain.execution import (
    ExecutionManifest,
    ManifestError,
    OptimizationSpec,
)
from tradingdev.domain.randomness import execution_randomness
from tradingdev.domain.research import research_datetime
from tradingdev.domain.strategies.capabilities import ensure_execution_mode
from tradingdev.domain.strategies.contract import (
    DRY_RUN_FIXTURE_ROWS,
    VALIDATE_FIXTURE_ROWS,
    SignalContractChecker,
)
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.strategies.signals import generate_checked_signals
from tradingdev.domain.validation.config import (
    ExplicitWalkForwardConfig,
    WalkForwardConfig,
    parse_walk_forward_config,
)
from tradingdev.domain.validation.date_windows import inclusive_day_window
from tradingdev.domain.validation.report import summarize_results
from tradingdev.domain.validation.walk_forward import WalkForwardValidator
from tradingdev.shared.utils.config import load_config
from tradingdev.shared.utils.json_values import normalize_json_object

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
    from tradingdev.domain.benchmark import BenchmarkEvidence
    from tradingdev.domain.strategies.execution import StrategyExecution
    from tradingdev.domain.strategies.schemas import StrategySpec


class StrategyExecutionGate(Protocol):
    """Narrow interface for the strategy lifecycle execution gate."""

    def resolve_executable(
        self, strategy_id: str, revision_id: str | None = None
    ) -> StrategySpec:
        """Return the spec for an executable strategy or raise."""
        ...


@dataclass(frozen=True)
class CheckedExecution:
    """A captured manifest and the synthetic inputs actually checked for it."""

    manifest: ExecutionManifest
    synthetic_fixture_rows: tuple[int, ...]


@dataclass(frozen=True)
class BacktestRun:
    """Result and metadata from a backtest service run."""

    mode: str
    pipeline: PipelineResult
    metrics: dict[str, Any]
    processed_path: Path
    dataset_id: str


class BacktestService:
    """Run simple and walk-forward backtests through one service path."""

    def __init__(
        self,
        *,
        data_service: DataService | None = None,
        strategy_loader: StrategyLoader | None = None,
        strategy_gate: StrategyExecutionGate | None = None,
        benchmark_service: BenchmarkService | None = None,
    ) -> None:
        self._data_service = data_service or DataService()
        self._strategy_loader = strategy_loader or StrategyLoader()
        self._strategy_gate = strategy_gate or StrategyService()
        self._benchmark_service = benchmark_service or BenchmarkService(
            self._data_service
        )

    def run_config(
        self, config_path: Path, *, walk_forward: bool = False
    ) -> BacktestRun:
        """Run a YAML config as simple backtest or walk-forward validation."""
        raw_config: dict[str, Any] = load_config(config_path)
        return self.run_raw_config(raw_config, walk_forward=walk_forward)

    def run_raw_config(
        self,
        raw_config: dict[str, Any],
        *,
        walk_forward: bool = False,
    ) -> BacktestRun:
        """Run an already parsed config.

        Raises:
            StrategyNotExecutableError: If the strategy has not reached
                runnable or promoted status, or the config's source_path does
                not match the registered strategy source.
        """
        manifest = self.prepare_execution(
            raw_config, kind="walk_forward" if walk_forward else "backtest"
        )
        return self.run_manifest(manifest)

    def prepare_execution(
        self,
        raw_config: dict[str, Any],
        *,
        kind: Literal["backtest", "walk_forward", "optimization"],
        optimization: OptimizationSpec | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> ExecutionManifest:
        """Bind the strategy and resolve application defaults before execution."""
        return self.capture_execution(
            raw_config, kind=kind, optimization=optimization, parameters=parameters
        ).manifest

    def capture_execution(
        self,
        raw_config: dict[str, Any],
        *,
        kind: Literal["backtest", "walk_forward", "optimization"],
        optimization: OptimizationSpec | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> CheckedExecution:
        """Return the same preparation with evidence from completed contract checks."""
        config = deepcopy(raw_config)
        run_config = BacktestRunConfig.model_validate(config)
        spec = self.prepare_strategy(config)
        ensure_execution_mode(spec.strategy_id, spec.execution_capabilities, kind)
        if run_config.is_walk_forward and kind == "backtest":
            msg = "Config contains validation settings; use prepare_walk_forward."
            raise ValueError(msg)
        if kind == "walk_forward" and not run_config.is_walk_forward:
            msg = "Config has no validation section for walk-forward."
            raise ValueError(msg)
        if kind == "walk_forward":
            assert run_config.validation is not None
            self._check_validation_settings(run_config.validation, run_config.backtest)
        config["data"] = self._data_service.execution_config(
            config, run_config.backtest
        )
        try:
            strategy_execution = self._strategy_loader.resolve_execution(
                config["strategy"]
            )
        except Exception as exc:
            raise ManifestError(f"Invalid strategy execution settings: {exc}") from exc
        if parameters is not None:
            strategy_execution = self._strategy_loader.override_execution(
                config["strategy"], strategy_execution, parameters
            )
            config["strategy"]["parameters"] = (
                strategy_execution.constructor_kwargs["config"]
                if strategy_execution.kind == "bundled"
                else strategy_execution.constructor_kwargs
            )
        manifest = ExecutionManifest.create(
            kind=kind,
            config=config,
            optimization=optimization,
            strategy_execution=strategy_execution,
        )
        fixture_rows = self.check_execution(
            manifest.config_for_execution(), strategy_execution
        )
        return CheckedExecution(manifest, fixture_rows)

    @staticmethod
    def _check_validation_settings(
        config: WalkForwardConfig, backtest: BacktestConfig
    ) -> None:
        """Check active split settings without claiming data-dependent fold coverage."""
        if isinstance(config, ExplicitWalkForwardConfig):
            train_lower, _ = inclusive_day_window(config.train_start, config.train_end)
            _, test_upper = inclusive_day_window(config.test_start, config.test_end)
            if train_lower < research_datetime(
                backtest.start_date
            ) or test_upper - timedelta(microseconds=1) > research_datetime(
                backtest.end_date
            ):
                raise ValueError(
                    "Active walk-forward validation windows must fit inside the "
                    "research period, including each complete UTC calendar day. "
                    "Use prepare_walk_forward validation_config to replace the "
                    "entire validation configuration with automatic folds or valid "
                    "explicit windows; keep the requested research period."
                )

    def check_execution(
        self,
        raw_config: dict[str, Any],
        execution: StrategyExecution,
        *,
        parameter_overrides: dict[str, Any] | None = None,
    ) -> tuple[int, ...]:
        """Check generated experiment settings at both fixture depths, read-only."""
        if execution.kind != "generated":
            return ()
        checker = SignalContractChecker(self._strategy_loader)
        checked_rows: list[int] = []
        for fixture_rows in (VALIDATE_FIXTURE_ROWS, DRY_RUN_FIXTURE_ROWS):
            checked = checker.check_config(
                raw_config,
                fixture_rows=fixture_rows,
                execution=execution,
                parameter_overrides=parameter_overrides,
            )
            errors = [item for item in checked["diagnostics"] if item.level == "error"]
            if errors:
                detail = "; ".join(f"{item.code}: {item.message}" for item in errors)
                raise ManifestError(
                    f"Strategy execution contract failed ({fixture_rows} rows): "
                    f"{detail}"
                )
            checked_rows.append(fixture_rows)
        return tuple(checked_rows)

    @staticmethod
    def _execution_config(manifest: ExecutionManifest) -> dict[str, Any]:
        """Verify a supported execution before checking or loading its inputs."""
        manifest.verify()
        if manifest.kind not in {"backtest", "walk_forward"}:
            msg = "BacktestService cannot execute an optimization manifest"
            raise ValueError(msg)
        return manifest.config_for_execution()

    def load_execution_dataset(self, manifest: ExecutionManifest) -> ExecutionDataset:
        """Check the execution, then capture its complete merged inputs once.

        Callers control the approval boundary. This method accesses real data;
        preparation and synthetic contract checks do not call it implicitly.
        """
        raw_config = self._execution_config(manifest)
        with execution_randomness(raw_config["random_seed"]):
            self._check_manifest_strategy(manifest, raw_config)
            dataset = self._data_service.load(
                raw_config, BacktestConfig(**raw_config["backtest"])
            )
            return ExecutionDataset.capture(manifest, dataset)

    def _check_manifest_strategy(
        self, manifest: ExecutionManifest, raw_config: dict[str, Any]
    ) -> None:
        """Recheck identity and the effective strategy contract before real data."""
        spec = self.prepare_strategy(raw_config)
        ensure_execution_mode(
            spec.strategy_id, spec.execution_capabilities, manifest.kind
        )
        self.check_execution(raw_config, manifest.strategy_execution)

    def run_manifest(
        self,
        manifest: ExecutionManifest,
        *,
        loaded_dataset: ExecutionDataset | None = None,
        benchmark: BenchmarkEvidence | None = None,
    ) -> BacktestRun:
        """Execute with fresh state and optionally reuse immutable captured inputs."""
        raw_config = self._execution_config(manifest)
        with execution_randomness(raw_config["random_seed"]):
            return self._execute(manifest, raw_config, loaded_dataset, benchmark)

    def prepare_benchmark(
        self, frame: pd.DataFrame, manifest: ExecutionManifest
    ) -> BenchmarkEvidence:
        """Capture the asset class's baseline once for runs sharing one dataset."""
        raw_config = self._execution_config(manifest)
        return self._benchmark_service.evidence(
            frame, raw_config, BacktestConfig(**raw_config["backtest"])
        )

    def _execute(
        self,
        manifest: ExecutionManifest,
        raw_config: dict[str, Any],
        loaded_dataset: ExecutionDataset | None,
        benchmark: BenchmarkEvidence | None = None,
    ) -> BacktestRun:
        """Construct and execute the strategy within its run's random context."""
        self._check_manifest_strategy(manifest, raw_config)
        bt_cfg = BacktestConfig(**raw_config["backtest"])
        parallel_cfg = ParallelConfig(**raw_config.get("parallel", {}))
        dataset = (
            self._data_service.load(raw_config, bt_cfg)
            if loaded_dataset is None
            else LoadedDataset(
                frame=loaded_dataset.frame_for(manifest),
                processed_path=loaded_dataset.processed_path,
                dataset_id=loaded_dataset.dataset_id,
            )
        )
        engine = self.create_engine(bt_cfg)
        strategy = self._strategy_loader.create_from_execution(
            raw_config["strategy"], manifest.strategy_execution, engine, parallel_cfg
        )

        if manifest.kind == "walk_forward":
            wf_cfg = parse_walk_forward_config(raw_config["validation"])
            validator = WalkForwardValidator(config=wf_cfg, engine=engine)
            folds = validator.validate(strategy, dataset.frame)
            pipeline = PipelineResult(
                mode="walk_forward",
                fold_results=folds,
                config_snapshot=raw_config,
                execution_manifest=manifest,
                benchmark=self._benchmark(dataset.frame, raw_config, bt_cfg, benchmark),
            )
            return BacktestRun(
                mode="walk_forward",
                pipeline=pipeline,
                metrics=summarize_results(folds),
                processed_path=dataset.processed_path,
                dataset_id=dataset.dataset_id,
            )

        signals = generate_checked_signals(strategy, dataset.frame)
        result = engine.run(signals)
        result.metric_metadata["strategy_parameters"] = deepcopy(
            strategy.get_parameters()
        )
        rolling_trace = strategy.rolling_trace()
        if rolling_trace is not None:
            # Provenance, not a parameter: which window trained the model that
            # scored each roll. Absent whenever the rolling switch is off.
            result.metric_metadata["rolling_trace"] = deepcopy(rolling_trace)
        pipeline = PipelineResult(
            mode="simple",
            backtest_result=result,
            config_snapshot=raw_config,
            execution_manifest=manifest,
            benchmark=self._benchmark(dataset.frame, raw_config, bt_cfg, benchmark),
        )
        return BacktestRun(
            mode="simple",
            pipeline=pipeline,
            metrics=self.serialize_metrics(result.metrics),
            processed_path=dataset.processed_path,
            dataset_id=dataset.dataset_id,
        )

    def _benchmark(
        self,
        frame: pd.DataFrame,
        raw_config: dict[str, Any],
        backtest_config: BacktestConfig,
        captured: BenchmarkEvidence | None,
    ) -> BenchmarkEvidence:
        """Reuse a baseline captured for a shared dataset instead of reloading it."""
        if captured is not None:
            return captured
        return self._benchmark_service.evidence(frame, raw_config, backtest_config)

    def prepare_strategy(
        self,
        raw_config: dict[str, Any],
    ) -> StrategySpec:
        """Validate a pinned strategy before backtest or optimization loading."""
        strategy_cfg = raw_config.get("strategy")
        if not isinstance(strategy_cfg, dict):
            msg = "strategy config must be a mapping"
            raise ValueError(msg)
        strategy_id = strategy_cfg.get("id")
        if not isinstance(strategy_id, str) or not strategy_id:
            msg = "strategy.id is required"
            raise ValueError(msg)
        revision_id = strategy_cfg.get("revision_id")
        if revision_id is not None and not isinstance(revision_id, str):
            msg = "strategy.revision_id must be a string"
            raise StrategyNotExecutableError(msg)
        spec = self._strategy_gate.resolve_executable(strategy_id, revision_id)
        bind_strategy_revision(raw_config, spec)
        return spec

    def create_engine(self, config: BacktestConfig) -> BaseBacktestEngine:
        """Create a backtest engine from config."""
        return create_backtest_engine(config)

    def serialize_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Preserve all computed metrics while normalizing JSON values."""
        return normalize_json_object(metrics)
