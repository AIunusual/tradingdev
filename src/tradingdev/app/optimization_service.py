"""Application service for parameter optimization jobs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError

from tradingdev.adapters.execution.process_runner import ProcessRunner
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService
from tradingdev.app.execution_submission import (
    ExecutionSubmissionService,
)
from tradingdev.app.job_config import (
    apply_backtest_overrides,
    apply_run_overrides,
    bind_strategy_revision,
    validate_optimization_grid,
)
from tradingdev.app.job_store import JobStore, get_default_job_store
from tradingdev.app.prepared_execution import PreparedExecution
from tradingdev.app.strategy_service import (
    StrategyNotExecutableError,
    StrategyService,
)
from tradingdev.domain.execution import OptimizationSpec
from tradingdev.domain.strategies.capabilities import (
    StrategyExecutionModeError,
    ensure_execution_mode,
)
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.shared.utils.config import load_config

if TYPE_CHECKING:
    from tradingdev.domain.strategies.schemas import StrategySpec


class OptimizationService:
    """Create background optimization jobs."""

    def __init__(
        self,
        *,
        strategy_service: StrategyService | None = None,
        job_store: JobStore | None = None,
        process_runner: ProcessRunner | None = None,
        strategy_loader: StrategyLoader | None = None,
        data_service: DataService | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._job_store = job_store or get_default_job_store()
        self._strategy_service = strategy_service or StrategyService(
            self._job_store.workspace
        )
        self._process_runner = process_runner or ProcessRunner(
            project_root, workspace=self._job_store.workspace
        )
        self._strategy_loader = strategy_loader or StrategyLoader(
            workspace_root=self._job_store.workspace.root
        )
        self._data_service = data_service or DataService(self._job_store.workspace)
        self._submission = ExecutionSubmissionService(
            self._job_store, self._process_runner
        )

    def start_optimization(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        param_ranges: dict[str, list[Any]],
        optimization_metric: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        revision_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        backtest_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare and submit a parameter optimization worker."""
        prepared = self.prepare_optimization(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            param_ranges=param_ranges,
            optimization_metric=optimization_metric,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            revision_id=revision_id,
            parameters=parameters,
            backtest_overrides=backtest_overrides,
        )
        if isinstance(prepared, dict):
            return prepared
        optimization = prepared.manifest.optimization
        assert optimization is not None
        submitted = self._submission.submit(prepared)
        total_combinations = optimization.total_combinations
        return {
            **submitted,
            "optimization_metric": optimization.optimization_metric,
            "direction": optimization.direction,
            "message": (
                f"Optimization started. {total_combinations} parameter combinations. "
                "Use get_job_status() to follow the search."
            ),
            "total_combinations": total_combinations,
        }

    def prepare_optimization(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        param_ranges: dict[str, list[Any]],
        optimization_metric: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        revision_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        backtest_overrides: dict[str, Any] | None = None,
    ) -> PreparedExecution | dict[str, Any]:
        """Capture a validated search without creating a job or launching a worker."""
        try:
            spec, error = self._resolve_strategy_config(strategy_id, revision_id)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            return {
                "job_id": "",
                "message": str(exc),
                "total_combinations": 0,
                "code": "invalid_optimization_request",
            }
        if spec is None:
            return {
                "job_id": "",
                "message": error,
                "total_combinations": 0,
                "code": "strategy_not_executable",
            }
        try:
            optimization = OptimizationSpec.model_validate(
                {
                    "param_ranges": param_ranges,
                    "optimization_metric": optimization_metric,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )
        except ValidationError as exc:
            return {
                "job_id": "",
                "message": str(exc),
                "total_combinations": 0,
                "code": "invalid_optimization_request",
            }

        config_path = Path(spec.config_path)
        try:
            raw_config = load_config(config_path)
            if not isinstance(raw_config, dict):
                raise ValueError("YAML config must be a mapping")
            bind_strategy_revision(raw_config, spec)
            ensure_execution_mode(
                spec.strategy_id, spec.execution_capabilities, "optimization"
            )
            if raw_config.get("validation") is not None:
                raise ValueError(
                    "Optimization config must not contain validation settings; "
                    "use a config without walk-forward validation."
                )
            effective_config = apply_run_overrides(
                raw_config,
                symbol=symbol,
                timeframe=timeframe,
                start_date=train_start,
                # Optimization bounds are calendar days, including the final day.
                end_date=f"{test_end}T23:59:59.999999",
            )
            apply_backtest_overrides(effective_config, backtest_overrides)
            checked = BacktestService(
                strategy_gate=self._strategy_service,
                strategy_loader=self._strategy_loader,
                data_service=self._data_service,
            ).capture_execution(
                effective_config,
                kind="optimization",
                optimization=optimization,
                parameters=parameters,
            )
            manifest = checked.manifest
            validate_optimization_grid(self._strategy_loader, manifest, optimization)
        except StrategyExecutionModeError as exc:
            return {
                "job_id": "",
                "message": str(exc),
                "total_combinations": 0,
                "code": exc.code,
                "execution_guidance": exc.guidance.model_dump(mode="json"),
            }
        except (
            OSError,
            TypeError,
            ValueError,
            yaml.YAMLError,
            StrategyNotExecutableError,
        ) as exc:
            return {
                "job_id": "",
                "message": str(exc),
                "total_combinations": 0,
                "code": "invalid_optimization_request",
            }
        return PreparedExecution(manifest, config_path, checked.synthetic_fixture_rows)

    def _resolve_strategy_config(
        self, strategy_id: str, revision_id: str | None
    ) -> tuple[StrategySpec | None, str]:
        try:
            spec = self._strategy_service.resolve_executable(strategy_id, revision_id)
        except StrategyNotExecutableError as exc:
            return None, str(exc)
        path = Path(spec.config_path)
        return (spec, "") if path.exists() else (None, f"Config not found: {path}")
