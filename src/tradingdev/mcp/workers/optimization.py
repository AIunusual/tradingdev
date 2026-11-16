"""Subprocess worker that executes a parameter optimization job.

Start jobs through prepare_optimization and submit_execution_decision after review.
The submission service uses
ProcessRunner to launch a supervisor, which starts this worker and supplies
TRADINGDEV_WORKER_IDENTITY. Running this module directly is not supported.

Each candidate commits its attempt and outcome against the approved grid.
A complete grid with a finite winner can proceed to the separate test period.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import multiprocessing
import signal
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.execution.process_runner import WorkerHandle
from tradingdev.adapters.execution.strategy_factory import StrategyFactory
from tradingdev.adapters.storage.optimization_trials import (
    TEST_POSITION,
    OptimizationLedger,
    TrialSnapshot,
)
from tradingdev.app import job_store
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService
from tradingdev.app.execution_submission import verify_submission_research
from tradingdev.domain.backtest.schemas import BacktestConfig, ParallelConfig
from tradingdev.domain.execution import ExecutionManifest, ManifestError
from tradingdev.domain.optimization.grid_search import finite_metric_value
from tradingdev.domain.optimization.trials import TrialEvidence
from tradingdev.domain.performance.artifacts import build_artifacts, scope_from_backtest
from tradingdev.domain.randomness import execution_randomness
from tradingdev.domain.research_scope import (
    optimization_backtest,
    optimization_date_bounds,
)
from tradingdev.domain.runtime import IsolatedRuntimeError
from tradingdev.domain.strategies.signals import (
    SignalContractError,
    StrategyCallbackError,
    generate_checked_signals,
    strategy_callback,
)
from tradingdev.shared.utils.logger import setup_logger
from tradingdev.shared.utils.parallel import estimate_n_jobs

if TYPE_CHECKING:
    from concurrent.futures import Future
    from types import FrameType

    import pandas as pd

    from tradingdev.domain.backtest.result import BacktestResult
    from tradingdev.domain.strategies.execution import StrategyExecution

logger = setup_logger(__name__)

# Absorb process startup and runtime cleanup after the worker alarm.
TRIAL_PROCESS_GRACE_SECONDS = 60


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
    The trial boundary catches it explicitly and records a candidate timeout.
    """


def _trial_timeout_handler(signum: int, frame: FrameType | None) -> None:
    raise _TrialTimeoutError("Trial run exceeded the manifest timeout")


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
            raise_execution_errors=True,
        )
        bundled = strategy_execution.kind == "bundled"
        with StrategyFactory().open_from_execution(
            strategy_cfg,
            strategy_execution,
            engine,
            parallel_cfg,
            parameter_overrides=param_dict,
            classify_callback_error=bundled,
        ) as strategy:
            signals_df = generate_checked_signals(
                strategy, df, classify_callback_error=bundled
            )
            if bundled:
                with strategy_callback():
                    raw_parameters = strategy.get_parameters()
                    raw_trace = strategy.rolling_trace()
            else:
                raw_parameters = strategy.get_parameters()
                raw_trace = strategy.rolling_trace()
            parameters = deepcopy(raw_parameters)
            rolling_trace = deepcopy(raw_trace)
        result = engine.run(signals_df)
        result.metric_metadata["execution_context"] = {
            **bt_cfg.model_dump(mode="json"),
            "random_seed": random_seed,
        }
        result.metric_metadata["strategy_parameters"] = parameters
        if rolling_trace is not None:
            result.metric_metadata["rolling_trace"] = rolling_trace

    return ComboEvaluation(deepcopy(param_dict), metric_name, result)


@dataclass(frozen=True)
class TrialRunner:
    """Run one manifest-bound candidate and commit its outcome in its own process."""

    ledger: OptimizationLedger
    strategy_config: dict[str, Any]
    execution: StrategyExecution
    config: BacktestConfig
    data: pd.DataFrame
    parallel: ParallelConfig
    seed: int | None

    def run(self, index: int, *, test: bool = False) -> float | None:
        position = TEST_POSITION if test else index
        selected_index = index if test else None
        self.ledger.start(position, selected_index=selected_index)
        params = self.ledger.grid[index]
        start = time.monotonic()
        try:
            evaluation = self._evaluate(params)
        except _TrialTimeoutError as error:
            self.ledger.finish(
                position,
                selected_index=selected_index,
                reason=str(error),
                timed_out=True,
            )
            return None
        except (StrategyCallbackError, SignalContractError) as error:
            self._fail_candidate(position, selected_index, error)
            return None
        except IsolatedRuntimeError as error:
            if not error.cleanup_verified or not (
                error.strategy_callback_failed or error.code == "runtime_timeout"
            ):
                raise
            self.ledger.finish(
                position,
                selected_index=selected_index,
                reason=str(error)[:1024] or error.code,
                timed_out=error.code == "runtime_timeout",
            )
            return None

        scope, observations = scope_from_backtest(
            evaluation.result,
            split="test" if test else "train",
            trial_index=None if test else index,
            parameters=evaluation.parameters,
        )
        self.ledger.finish(
            position,
            selected_index=selected_index,
            evidence=TrialEvidence.capture(scope, observations),
        )
        return time.monotonic() - start

    def _fail_candidate(
        self, position: int, selected_index: int | None, error: Exception
    ) -> None:
        self.ledger.finish(
            position,
            selected_index=selected_index,
            reason=f"{type(error).__name__}: {error}"[:1024],
        )

    def _evaluate(self, params: dict[str, Any]) -> ComboEvaluation:
        timeout = self.ledger.search.trial_timeout_seconds
        old_handler = signal.signal(signal.SIGALRM, _trial_timeout_handler)
        try:
            signal.alarm(timeout)
            return _run_single_combo(
                self.strategy_config,
                self.execution,
                self.config,
                self.data,
                params,
                self.ledger.search.optimization_metric,
                self.parallel,
                self.seed,
            )
        except _TrialTimeoutError as error:
            raise _TrialTimeoutError(f"Trial exceeded {timeout}s timeout") from error
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


_pool_runner: TrialRunner | None = None


def _init_pool(runner: TrialRunner) -> None:
    global _pool_runner  # noqa: PLW0603
    _pool_runner = runner


def _run_in_pool(index: int, test: bool) -> float | None:
    assert _pool_runner is not None
    try:
        return _pool_runner.run(index, test=test)
    except Exception as error:
        # Infrastructure exceptions may not be picklable. Preserve their
        # diagnostics across the pool and let the owning worker stop the job.
        if isinstance(error, IsolatedRuntimeError):
            detail = f"{error.code}: {error}"
            if error.control_path:
                detail += f" (control path: {error.control_path})"
        else:
            detail = f"{type(error).__name__}: {error}"
        raise RuntimeError(detail) from error


def _run_trials(
    runner: TrialRunner, indices: range, n_jobs: int, *, test: bool = False
) -> dict[int, float | None]:
    """Run bounded work units. A hung process is an infrastructure failure."""
    if not indices:
        return {}
    remaining = iter(indices)
    limit = runner.ledger.search.trial_timeout_seconds + TRIAL_PROCESS_GRACE_SECONDS
    durations: dict[int, float | None] = {}
    # Even first-trial and sequential execution need a supervising process:
    # Python cannot deliver its alarm while certain native calls are blocked.
    with ProcessPoolExecutor(
        max_workers=n_jobs,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_pool,
        initargs=(runner,),
    ) as pool:

        def submit(count: int) -> dict[Future[float | None], tuple[int, float]]:
            dispatched = {}
            for index in itertools.islice(remaining, count):
                deadline = time.monotonic() + limit
                dispatched[pool.submit(_run_in_pool, index, test)] = (index, deadline)
            return dispatched

        try:
            active = submit(n_jobs)
            while active:
                oldest_index, deadline = min(
                    active.values(), key=lambda trial: trial[1]
                )
                ready, _ = wait(
                    active,
                    timeout=max(0.0, deadline - time.monotonic()),
                    return_when=FIRST_COMPLETED,
                )
                for future in ready:
                    index, _ = active.pop(future)
                    durations[index] = future.result()
                now = time.monotonic()
                expired = (
                    oldest_index
                    if not ready
                    else next(
                        (
                            index
                            for index, deadline in active.values()
                            if now >= deadline
                        ),
                        None,
                    )
                )
                if expired is not None:
                    stage = "Held-out test" if test else f"Trial {expired}"
                    raise TimeoutError(
                        f"{stage} process exceeded its timeout and cleanup allowance"
                    )
                active.update(submit(len(ready)))
        except BaseException:
            # The job supervisor also checks the entire host process group and
            # recovers container leases left by workers that could not unwind.
            pool.kill_workers()
            raise
    return durations


def _save_complete(
    ledger: OptimizationLedger,
    snapshot: TrialSnapshot,
    raw_config: dict[str, Any],
    *,
    time_per_combo: float | None,
    n_jobs: int,
) -> None:
    index = snapshot.leading_index
    if (
        index is None
        or not snapshot.search_complete
        or snapshot.test.state != "completed"
    ):
        raise ManifestError("Complete optimization requires the full grid and test")
    evidence = {
        f"trial/{trial.position}/train": ledger.read_evidence(trial)
        for trial in snapshot.trials
    }
    evidence["test"] = ledger.read_evidence(snapshot.test)
    selected_scope = f"trial/{index}/train"
    train = evidence[selected_scope].scope
    test = evidence["test"].scope
    metric = ledger.search.optimization_metric
    result = {
        "best_params": train.parameters,
        "optimization_metric": metric,
        "direction": ledger.search.direction,
        "best_train_metric_value": train.values[metric],
        "best_oos_metric_value": test.values[metric],
        "train_metrics": train.values,
        "test_metrics": test.values,
        "total_combinations": len(snapshot.trials),
        "time_per_combo": time_per_combo,
        "n_parallel_workers": n_jobs,
    }
    result_path = job_store.save_result(
        ledger.job_id,
        result,
        config_snapshot=raw_config,
        execution_manifest=ledger.manifest,
        performance=build_artifacts(
            ledger.job_id,
            ledger.manifest.manifest_hash,
            "test",
            {key: item.scope for key, item in evidence.items()},
            {key: item.observations for key, item in evidence.items()},
            selected_train_scope=selected_scope,
        ),
    )
    job_store.update_job(
        ledger.job_id,
        status="done",
        result_path=str(result_path),
        best_params=train.parameters,
    )


def _split(frame: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    start, end = optimization_date_bounds(config)
    selected = frame[frame["timestamp"].between(start, end, inclusive="left")].copy()
    if selected.empty:
        raise ValueError(f"No data found for {start} to {end}")
    return selected


def _snapshot(ledger: OptimizationLedger) -> TrialSnapshot:
    snapshot = ledger.snapshot()
    if snapshot is None:
        raise ManifestError("Optimization ledger is missing")
    return snapshot


def _run_optimization(job_id: str) -> None:
    store = job_store.get_default_job_store()
    if store.get_job(job_id) is None:
        logger.error("Job %s not found in store", job_id)
        return
    stage = "Execution manifest"
    try:
        job_store.update_job(
            job_id,
            status="downloading_data",
            **WorkerHandle.from_environment().job_fields(),
        )
        manifest = store.load_manifest(job_id)
        if (
            not isinstance(manifest, ExecutionManifest)
            or manifest.kind != "optimization"
        ):
            raise ManifestError("Job requires an optimization execution manifest")
        verify_submission_research(store, job_id, manifest)
        raw_config = manifest.config_for_execution()
        BacktestService().prepare_strategy(raw_config)
        parallel = ParallelConfig(**raw_config["parallel"])
        train_config = optimization_backtest(manifest, "train")
        test_config = optimization_backtest(manifest, "test")

        stage = "Ledger initialization"
        OptimizationLedger(store.store, job_id, manifest, None).initialize()
        stage = "Data"
        dataset = DataService().load(
            raw_config, BacktestConfig(**raw_config["backtest"])
        )
        job_store.update_job(
            job_id, data_downloaded=True, dataset_id=dataset.dataset_id
        )
        ledger = OptimizationLedger(store.store, job_id, manifest, dataset.dataset_id)
        train = _split(dataset.frame, train_config)
        test = _split(dataset.frame, test_config)
        runner = TrialRunner(
            ledger,
            raw_config["strategy"],
            manifest.strategy_execution,
            train_config,
            train,
            parallel,
            raw_config["random_seed"],
        )
        stage = "Estimation trial"
        job_store.update_job(
            job_id, status="estimating", total_combinations=len(ledger.grid)
        )
        first_seconds = _run_trials(runner, range(1), 1)[0]
        n_jobs = min(
            estimate_n_jobs(train, **parallel.model_dump()),
            max(len(ledger.grid) - 1, 1),
        )
        time_per_combo = round(first_seconds, 2) if first_seconds is not None else None
        estimate = (
            round(first_seconds * (1 + (len(ledger.grid) - 1) / n_jobs), 1)
            if first_seconds is not None
            else None
        )
        job_store.update_job(
            job_id,
            status="optimizing",
            time_per_combo=time_per_combo,
            estimated_total_seconds=estimate,
            n_parallel_workers=n_jobs,
        )

        stage = "Training trials"
        _run_trials(runner, range(1, len(ledger.grid)), n_jobs)
        snapshot = _snapshot(ledger)
        if not snapshot.search_complete or snapshot.leading_index is None:
            message = (
                "Some approved combinations did not produce complete observations."
                if not snapshot.search_complete
                else "No parameter combination has a finite objective value."
            )
            job_store.update_job(job_id, status="partial", error=message)
            return

        stage = "Test period"
        job_store.update_job(job_id, status="testing_oos")
        _run_trials(
            replace(runner, config=test_config, data=test),
            range(snapshot.leading_index, snapshot.leading_index + 1),
            1,
            test=True,
        )
        snapshot = _snapshot(ledger)
        if snapshot.test.state != "completed":
            job_store.update_job(job_id, status="partial", error=snapshot.test.reason)
            return

        stage = "Result save"
        _save_complete(
            ledger, snapshot, raw_config, time_per_combo=time_per_combo, n_jobs=n_jobs
        )
    except Exception as exc:
        _fail(job_id, f"{stage} error: {exc}")


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
