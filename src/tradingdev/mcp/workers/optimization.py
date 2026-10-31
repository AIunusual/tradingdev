"""Subprocess worker that executes a parameter optimization job.

Start jobs through prepare_optimization and submit_execution_decision after review.
The submission service uses
ProcessRunner to launch a supervisor, which starts this worker and supplies
TRADINGDEV_WORKER_IDENTITY. Running this module directly is not supported.

The worker verifies the job's immutable execution manifest (populated by
the approved execution plan) and uses it for every phase:

1. Downloads / loads OHLCV data
2. Runs a single trial combo with a 5-minute timeout
3. Reports estimated total time for the already approved search
4. Runs all remaining combos in parallel batches
5. Selects the best params and runs an out-of-sample test
6. Persists results to job_store
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from joblib import Parallel, delayed

from tradingdev.adapters.execution.process_runner import WorkerHandle
from tradingdev.app import job_store
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService
from tradingdev.app.execution_submission import verify_submission_research
from tradingdev.domain.backtest.schemas import BacktestConfig, ParallelConfig
from tradingdev.domain.execution import ExecutionManifest, ManifestError
from tradingdev.domain.optimization.grid_search import (
    GridSearchResult,
    best_result,
    finite_metric_value,
    parameter_grid,
)
from tradingdev.domain.performance.artifacts import build_artifacts, scope_from_backtest
from tradingdev.domain.randomness import execution_randomness
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.strategies.signals import generate_checked_signals
from tradingdev.shared.utils.logger import setup_logger
from tradingdev.shared.utils.parallel import estimate_n_jobs

if TYPE_CHECKING:
    import pandas as pd

    from tradingdev.domain.backtest.result import BacktestResult
    from tradingdev.domain.performance.artifacts import PerformanceArtifacts
    from tradingdev.domain.strategies.execution import StrategyExecution

logger = setup_logger(__name__)


@dataclass(frozen=True)
class ComboEvaluation:
    """One complete trial, retaining raw observations and metric provenance."""

    parameters: dict[str, Any]
    target_metric: str
    result: BacktestResult

    @property
    def target_value(self) -> float | None:
        """Derive the objective from the same values persisted for this trial."""
        return finite_metric_value(self.result.metrics.get(self.target_metric))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _fail(job_id: str, error: str) -> None:
    """Mark a job as failed and log the error."""
    logger.error("Job %s failed: %s", job_id, error)
    job_store.update_job(
        job_id,
        status="failed",
        error=error,
        ended_at=_now_iso(),
    )


class _TrialTimeoutError(BaseException):
    """Interrupt the trial without being translated into strategy diagnostics.

    This is worker control flow, like cancellation, rather than a strategy error.
    The trial boundary catches it explicitly and records estimation_timeout.
    """


def _trial_timeout_handler(signum: int, frame: Any) -> None:
    raise _TrialTimeoutError("Trial run exceeded the manifest timeout")


# ---------------------------------------------------------------------------
# Module-level evaluation function (must be picklable for joblib)
# ---------------------------------------------------------------------------
def _evaluate_combo(
    strategy_cfg: dict[str, Any],
    strategy_execution: StrategyExecution,
    bt_cfg_dict: dict[str, Any],
    df: pd.DataFrame,
    param_dict: dict[str, Any],
    metric_name: str,
    parallel_cfg_dict: dict[str, Any],
    random_seed: int | None,
) -> ComboEvaluation:
    """Evaluate a single parameter combination.

    This function is called by joblib workers.  It re-loads the strategy
    class from source to avoid pickle issues with dynamically loaded modules.

    Args:
        strategy_cfg: Strategy config section.
        strategy_execution: Captured constructor arguments, including defaults.
        bt_cfg_dict: BacktestConfig fields as a plain dict.
        df: Training DataFrame transferred losslessly by joblib.
        param_dict: Parameter combination to evaluate.
        metric_name: Target metric to extract.
        parallel_cfg_dict: Fixed parallel resource policy from the manifest.
        random_seed: Fixed run seed, re-created independently in each worker.

    Returns:
        Complete backtest result, parameters, and objective identifier.
    """
    return _run_single_combo(
        strategy_cfg,
        strategy_execution,
        BacktestConfig(**bt_cfg_dict),
        df,
        param_dict,
        metric_name,
        ParallelConfig(**parallel_cfg_dict),
        random_seed,
    )


def _run_single_combo(
    strategy_cfg: dict[str, Any],
    strategy_execution: StrategyExecution,
    bt_cfg: BacktestConfig,
    df: pd.DataFrame,
    param_dict: dict[str, Any],
    metric_name: str,
    parallel_cfg: ParallelConfig,
    random_seed: int | None,
) -> ComboEvaluation:
    """Run a single combo in the main process (for trial run)."""
    with execution_randomness(random_seed):
        service = BacktestService()
        engine = service.create_engine(bt_cfg)
        service.prepare_strategy({"strategy": strategy_cfg})
        service.check_execution(
            {
                "strategy": strategy_cfg,
                "backtest": bt_cfg.model_dump(mode="json"),
                "parallel": parallel_cfg.model_dump(mode="json"),
                "random_seed": random_seed,
            },
            strategy_execution,
            parameter_overrides=param_dict,
        )
        strategy = StrategyLoader().create_from_execution(
            strategy_cfg,
            strategy_execution,
            engine,
            parallel_cfg,
            parameter_overrides=param_dict,
        )

        signals_df = generate_checked_signals(strategy, df)
        result = engine.run(signals_df)

        result.metric_metadata["execution_context"] = {
            **bt_cfg.model_dump(mode="json"),
            "random_seed": random_seed,
        }
        result.metric_metadata["strategy_parameters"] = deepcopy(
            strategy.get_parameters()
        )
        rolling_trace = strategy.rolling_trace()
        if rolling_trace is not None:
            result.metric_metadata["rolling_trace"] = deepcopy(rolling_trace)

    return ComboEvaluation(deepcopy(param_dict), metric_name, result)


def _performance_artifacts(
    job_id: str,
    manifest_hash: str,
    evaluations: list[ComboEvaluation],
    selected_index: int,
    oos: ComboEvaluation,
) -> PerformanceArtifacts:
    """Keep every trial once; reference the selected training scope."""
    scopes = {}
    observations = {}
    for index, evaluation in enumerate(evaluations):
        scope, raw = scope_from_backtest(
            evaluation.result,
            split="train",
            trial_index=index,
            parameters=evaluation.parameters,
        )
        scope_id = f"trial/{index}/train"
        scopes[scope_id] = scope
        observations[scope_id] = raw
    scope, raw = scope_from_backtest(
        oos.result, split="test", parameters=oos.parameters
    )
    scopes["test"] = scope
    observations["test"] = raw
    return build_artifacts(
        run_id=job_id,
        manifest_hash=manifest_hash,
        default_scope="test",
        scopes=scopes,
        observations=observations,
        selected_train_scope=f"trial/{selected_index}/train",
    )


# ---------------------------------------------------------------------------
# Main worker logic
# ---------------------------------------------------------------------------
def _run_optimization(job_id: str) -> None:  # noqa: C901, PLR0912, PLR0915
    logger.info("Optimization worker started: job=%s", job_id)

    job = job_store.get_job(job_id)
    if job is None:
        logger.error("Job %s not found in store", job_id)
        return

    # --- Phase 1: mark running & preserve the supervisor control identity ---
    handle = WorkerHandle.from_environment()
    job_store.update_job(
        job_id,
        status="downloading_data",
        **handle.job_fields(),
    )

    # --- Phase 2: verify and load the sole execution specification ---
    try:
        manifest = job_store.load_manifest(job_id)
        if not isinstance(manifest, ExecutionManifest):
            raise ManifestError("Comparison requires the comparison worker")
        optimization = manifest.optimization
        if manifest.kind != "optimization" or optimization is None:
            msg = "Job requires an optimization execution manifest"
            raise ValueError(msg)
        verify_submission_research(job_store.get_default_job_store(), job_id, manifest)
        raw_config = manifest.config_for_execution()
        BacktestService().prepare_strategy(raw_config)
        bt_cfg = BacktestConfig(**raw_config["backtest"])
        parallel_cfg = ParallelConfig(**raw_config["parallel"])
        random_seed = raw_config["random_seed"]
    except Exception as exc:
        _fail(job_id, f"Execution manifest error: {exc}")
        return

    param_ranges = optimization.param_ranges
    optimization_metric = optimization.optimization_metric
    train_start = optimization.train_start.isoformat()
    train_end = optimization.train_end.isoformat()
    test_start = optimization.test_start.isoformat()
    test_end = optimization.test_end.isoformat()

    # --- Phase 3: load / download data ---
    try:
        dataset = DataService().load(raw_config, bt_cfg)
        full_df = dataset.frame
        job_store.update_job(
            job_id, data_downloaded=True, dataset_id=dataset.dataset_id
        )
        logger.info("Data loaded: %d rows", len(full_df))
    except Exception as exc:
        _fail(job_id, f"Data error: {exc}")
        return

    # Split into train / test
    import pandas as pd

    train_df = full_df[
        (full_df["timestamp"] >= pd.Timestamp(train_start, tz="UTC"))
        & (
            full_df["timestamp"]
            < pd.Timestamp(train_end, tz="UTC") + pd.Timedelta(days=1)
        )
    ].copy()
    test_df = full_df[
        (full_df["timestamp"] >= pd.Timestamp(test_start, tz="UTC"))
        & (
            full_df["timestamp"]
            < pd.Timestamp(test_end, tz="UTC") + pd.Timedelta(days=1)
        )
    ].copy()

    if train_df.empty:
        _fail(job_id, f"No training data found for {train_start} ~ {train_end}")
        return
    if test_df.empty:
        _fail(job_id, f"No test data found for {test_start} ~ {test_end}")
        return

    logger.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    # --- Phase 4: use the captured constructor inputs for every combination ---
    strategy_cfg: dict[str, Any] = raw_config["strategy"]
    strategy_execution = manifest.strategy_execution
    # Build all combinations through the domain grid-search helper.
    all_combos = parameter_grid(param_ranges)
    total_combinations = len(all_combos)
    logger.info("Total combinations: %d", total_combinations)

    # --- Phase 5: trial run with timeout ---
    job_store.update_job(
        job_id,
        status="estimating",
        total_combinations=total_combinations,
    )

    # Use BacktestConfig with training period for evaluation
    train_bt_raw = dict(raw_config["backtest"])
    train_bt_raw["start_date"] = train_start
    train_bt_raw["end_date"] = f"{train_end}T23:59:59.999999"
    train_bt_cfg = BacktestConfig(**train_bt_raw)

    first_combo = all_combos[0]
    trial_result: ComboEvaluation | None = None

    # Set up SIGALRM timeout
    old_handler = signal.signal(signal.SIGALRM, _trial_timeout_handler)
    signal.alarm(optimization.trial_timeout_seconds)
    try:
        t0 = time.monotonic()
        trial_result = _run_single_combo(
            strategy_cfg,
            strategy_execution,
            train_bt_cfg,
            train_df,
            first_combo,
            optimization_metric,
            parallel_cfg,
            random_seed,
        )
        time_per_combo = time.monotonic() - t0
    except _TrialTimeoutError:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        job_store.update_job(
            job_id,
            status="estimation_timeout",
            error=(
                f"Trial run exceeded {optimization.trial_timeout_seconds}s timeout. "
                "This strategy is too slow for parameter optimization."
            ),
            ended_at=_now_iso(),
        )
        logger.warning("Job %s: trial run timed out", job_id)
        return
    except Exception as exc:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        _fail(job_id, f"Trial run error: {exc}")
        return
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    # Estimate total time
    n_jobs = min(
        estimate_n_jobs(train_df, **parallel_cfg.model_dump()),
        max(total_combinations - 1, 1),
    )
    # Remaining combos after trial (first already done)
    remaining_count = total_combinations - 1
    estimated_total_seconds = round(
        time_per_combo + (time_per_combo * remaining_count / max(n_jobs, 1)),
        1,
    )

    logger.info(
        "Trial: %.2fs/combo, %d combos, %d workers → est %.1fs total",
        time_per_combo,
        total_combinations,
        n_jobs,
        estimated_total_seconds,
    )

    job_store.update_job(
        job_id,
        status="optimizing",
        time_per_combo=round(time_per_combo, 2),
        estimated_total_seconds=estimated_total_seconds,
        n_parallel_workers=n_jobs,
    )

    # --- Phase 7: run all combos in parallel batches ---
    job_store.update_job(
        job_id,
        status="optimizing",
        completed=1,
        total_combinations=total_combinations,
    )

    # Collect all results (including trial)
    all_results: list[ComboEvaluation] = []
    if trial_result is not None:
        all_results.append(trial_result)

    remaining_combos = all_combos[1:]
    optimization_start = time.monotonic()

    batch_size = max(n_jobs * 2, 10)
    for batch_start in range(0, len(remaining_combos), batch_size):
        batch = remaining_combos[batch_start : batch_start + batch_size]

        try:
            batch_results: list[ComboEvaluation] = Parallel(n_jobs=n_jobs)(
                delayed(_evaluate_combo)(
                    strategy_cfg,
                    strategy_execution,
                    train_bt_cfg.model_dump(),
                    train_df,
                    combo,
                    optimization_metric,
                    parallel_cfg.model_dump(),
                    random_seed,
                )
                for combo in batch
            )
        except Exception as exc:
            _fail(job_id, f"Optimization error at batch {batch_start}: {exc}")
            return

        all_results.extend(batch_results)

        # Update progress
        completed = 1 + batch_start + len(batch)
        elapsed = time.monotonic() - optimization_start
        rate = completed / elapsed if elapsed > 0 else 1.0
        remaining_seconds = (total_combinations - completed) / rate

        job_store.update_job(
            job_id,
            completed=completed,
            estimated_remaining_seconds=round(remaining_seconds, 1),
        )

    logger.info("All %d combos evaluated", total_combinations)

    # --- Phase 8: select best params ---
    candidates = [
        GridSearchResult(
            params=evaluation.parameters,
            metric_value=evaluation.target_value,
            metrics=evaluation.result.metrics,
        )
        for evaluation in all_results
    ]
    try:
        best = best_result(candidates, direction=optimization.direction)
    except ValueError as exc:
        _fail(job_id, f"Optimization selection error: {exc}")
        return
    best_params = best.params
    selected_index = next(
        index for index, candidate in enumerate(candidates) if candidate is best
    )
    best_metric_value = best.metric_value
    best_train_metrics = best.metrics
    logger.info(
        "Best params: %s (%s=%.4f)",
        json.dumps(best_params),
        optimization_metric,
        best_metric_value,
    )

    # --- Phase 9: out-of-sample test ---
    job_store.update_job(job_id, status="testing_oos")

    test_bt_raw = dict(raw_config["backtest"])
    test_bt_raw["start_date"] = test_start
    test_bt_raw["end_date"] = f"{test_end}T23:59:59.999999"
    test_bt_cfg = BacktestConfig(**test_bt_raw)

    try:
        oos = _run_single_combo(
            strategy_cfg,
            strategy_execution,
            test_bt_cfg,
            test_df,
            best_params,
            optimization_metric,
            parallel_cfg,
            random_seed,
        )
    except Exception as exc:
        _fail(job_id, f"Out-of-sample test error: {exc}")
        return

    logger.info("OOS result: %s=%s", optimization_metric, oos.target_value)

    # --- Phase 10: persist results ---
    try:
        optimization_result: dict[str, Any] = {
            "best_params": best_params,
            "optimization_metric": optimization_metric,
            "direction": optimization.direction,
            "best_train_metric_value": best_metric_value,
            "best_oos_metric_value": oos.target_value,
            "train_metrics": best_train_metrics,
            "test_metrics": oos.result.metrics,
            "total_combinations": total_combinations,
            "time_per_combo": round(time_per_combo, 2),
            "n_parallel_workers": n_jobs,
        }
        result_path = job_store.save_result(
            job_id,
            optimization_result,
            config_snapshot=raw_config,
            execution_manifest=manifest,
            performance=_performance_artifacts(
                job_id, manifest.manifest_hash, all_results, selected_index, oos
            ),
        )
        job_store.update_job(
            job_id,
            status="done",
            ended_at=_now_iso(),
            result_path=str(result_path),
            best_params=best_params,
        )
        logger.info("Job %s done → %s", job_id, result_path)
    except Exception as exc:
        _fail(job_id, f"Result save error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCP optimization worker (run as subprocess)"
    )
    parser.add_argument("job_id", help="Job ID assigned by MCP server")
    args = parser.parse_args()
    _run_optimization(args.job_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
