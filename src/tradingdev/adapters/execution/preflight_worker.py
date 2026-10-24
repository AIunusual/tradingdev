"""Run preparation and a bounded historical sample under a worker supervisor."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService, LoadedDataset
from tradingdev.app.execution_submission import PreparedExecution
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.optimization_service import OptimizationService
from tradingdev.app.preflight_service import PreflightError
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.schemas import (
    BacktestConfig,
    ParallelConfig,
    WalkForwardConfig,
)
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.optimization.grid_search import finite_metric_value
from tradingdev.domain.performance.artifacts import (
    PerformanceArtifacts,
    build_artifacts,
    bundles_from_pipeline,
    scope_from_backtest,
    validate_projection,
)
from tradingdev.domain.preflight import (
    PreflightCheck,
    PreflightPayload,
    PreflightReceipt,
    PreflightRequest,
    PreflightWindow,
)
from tradingdev.domain.randomness import execution_randomness
from tradingdev.domain.strategies.capabilities import (
    StrategyExecutionGuidance,
    StrategyExecutionModeError,
)
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.validation.date_windows import inclusive_day_window
from tradingdev.domain.validation.purged_splitter import PurgedDataSplitter
from tradingdev.domain.validation.report import summarize_results
from tradingdev.domain.validation.splitter import DataSplitter
from tradingdev.domain.validation.walk_forward import WalkForwardValidator
from tradingdev.shared.utils.json_values import normalize_json_object

if TYPE_CHECKING:
    from tradingdev.domain.backtest.result import BacktestResult
    from tradingdev.domain.strategies.execution import StrategyExecution


class _FixedDatasetService(DataService):
    def __init__(self, dataset: LoadedDataset) -> None:
        self._dataset = dataset

    def load(
        self, raw_config: dict[str, Any], backtest_config: BacktestConfig
    ) -> LoadedDataset:
        return self._dataset


def _sample_manifest(
    original: ExecutionManifest,
    frame: pd.DataFrame,
    *,
    execution: StrategyExecution | None = None,
    walk_forward: bool = False,
) -> ExecutionManifest:
    config = original.config_copy()
    if not walk_forward:
        config.pop("validation", None)
    config["backtest"]["start_date"] = frame["timestamp"].min().isoformat()
    config["backtest"]["end_date"] = frame["timestamp"].max().isoformat()
    selected = execution or original.strategy_execution
    if execution is not None:
        config["strategy"]["parameters"] = (
            selected.constructor_kwargs["config"]
            if selected.kind == "bundled"
            else selected.constructor_kwargs
        )
    return ExecutionManifest.create(
        kind="walk_forward" if walk_forward else "backtest",
        config=config,
        strategy_execution=selected,
    )


def _window(
    role: Literal["full", "train", "test"], frame: pd.DataFrame, minimum: int
) -> PreflightWindow:
    if len(frame) < max(minimum, 2):
        raise PreflightError(
            "insufficient_sample_history",
            f"The {role} sample has {len(frame)} bars; "
            f"it requires at least {max(minimum, 2)}.",
        )
    return PreflightWindow(
        role=role,
        start=frame["timestamp"].iloc[0].isoformat(),
        end=frame["timestamp"].iloc[-1].isoformat(),
        rows=len(frame),
    )


def _serialize(
    directory: Path, index: int, pipeline: PipelineResult, metrics: dict[str, Any]
) -> None:
    artifacts = bundles_from_pipeline(f"preflight_{index}", pipeline, metrics)
    encoded = artifacts.model_dump_json()
    PerformanceArtifacts.model_validate_json(encoded)
    (directory / f"performance-{index}.json").write_text(encoded, encoding="utf-8")
    (directory / f"pipeline-{index}.pkl").write_bytes(pickle.dumps(pipeline))
    (directory / f"metrics-{index}.json").write_text(
        json.dumps(metrics, allow_nan=False),
        encoding="utf-8",
    )


def execute_preflight(
    request: PreflightRequest,
    *,
    source_workspace: WorkspacePaths,
    directory: Path,
) -> PreflightPayload:
    """Keep every import, constructor, data fetch and engine call inside the child."""
    started = time.monotonic()
    temporary_workspace = WorkspacePaths(directory / "runtime")
    store = JobStore(workspace=temporary_workspace)
    strategies = StrategyService(source_workspace, store=store.store)
    loader = StrategyLoader(workspace_root=source_workspace.root)
    data = DataService(source_workspace)
    kwargs: dict[str, Any] = dict(request.arguments)
    if request.kind == "optimization":
        prepared = OptimizationService(
            strategy_service=strategies,
            strategy_loader=loader,
            data_service=data,
            job_store=store,
        ).prepare_optimization(**kwargs)
    else:
        jobs = JobService(
            strategy_service=strategies,
            strategy_loader=loader,
            data_service=data,
            job_store=store,
        )
        prepare = (
            jobs.prepare_walk_forward
            if request.kind == "walk_forward"
            else jobs.prepare_backtest
        )
        prepared = prepare(**kwargs)
    if not isinstance(prepared, PreparedExecution):
        guidance = prepared.get("execution_guidance")
        raise PreflightError(
            str(prepared.get("code", "invalid_execution_request")),
            str(prepared.get("message", "Preparation was rejected")),
            execution_guidance=(
                None
                if guidance is None
                else StrategyExecutionGuidance.model_validate(guidance)
            ),
        )
    original = prepared.manifest
    expected_hash = original.manifest_hash
    config = original.config_for_execution()
    bt = BacktestConfig.model_validate(config["backtest"])
    windows: list[PreflightWindow] = []
    results: list[BacktestResult] = []
    checks: list[PreflightCheck] = [
        "configuration",
        "signals",
        "engine",
        "serialization",
    ]
    if original.strategy_execution.kind == "generated":
        checks.append("signal_contract")
    warnings = ["partial_historical_sample"]
    sample_dir = directory / "sample"
    sample_dir.mkdir()
    coverage: dict[str, Any] = {}

    def load(role: str, begin: object, finish: object, limit: int) -> LoadedDataset:
        sampled = bt.model_dump(mode="python")

        def utc(value: object) -> pd.Timestamp:
            timestamp = pd.Timestamp(str(value))
            return (
                timestamp.tz_localize("UTC")
                if timestamp.tzinfo is None
                else timestamp.tz_convert("UTC")
            )

        sampled.update(
            start_date=max(utc(begin), utc(bt.start_date)).to_pydatetime(),
            end_date=min(utc(finish), utc(bt.end_date)).to_pydatetime(),
        )
        return data.load_sample(
            config,
            BacktestConfig.model_validate(sampled),
            max_rows=limit,
            output_dir=sample_dir / role,
        )

    def run_simple(
        dataset: LoadedDataset, execution: StrategyExecution | None = None
    ) -> None:
        sample = _sample_manifest(original, dataset.frame, execution=execution)
        run = BacktestService(
            data_service=_FixedDatasetService(dataset),
            strategy_gate=strategies,
            strategy_loader=loader,
        ).run_manifest(sample)
        _serialize(sample_dir, len(results), run.pipeline, run.metrics)
        assert run.pipeline.backtest_result is not None
        results.append(run.pipeline.backtest_result)

    if request.kind == "backtest":
        dataset = load("full", bt.start_date, bt.end_date, request.sample_bars)
        windows.append(_window("full", dataset.frame, request.minimum_history_bars))
        run_simple(dataset)
    elif request.kind == "optimization":
        optimization = original.optimization
        assert optimization is not None
        if request.sample_bars // 2 < request.minimum_history_bars:
            raise PreflightError(
                "insufficient_sample_budget",
                "Training and test samples each need the declared history.",
            )
        parameters = {
            key: values[0] for key, values in optimization.param_ranges.items()
        }
        selected = loader.override_execution(
            config["strategy"], original.strategy_execution, parameters
        )
        for role, begin, finish, limit in (
            (
                "train",
                optimization.train_start,
                optimization.train_end,
                request.sample_bars // 2,
            ),
            (
                "test",
                optimization.test_start,
                optimization.test_end,
                request.sample_bars - request.sample_bars // 2,
            ),
        ):
            end = (
                pd.Timestamp(finish)
                + pd.Timedelta(days=1)
                - pd.Timedelta(microseconds=1)
            )
            dataset = load(role, begin, end.to_pydatetime(), limit)
            windows.append(
                _window(
                    cast("Literal['train', 'test']", role),
                    dataset.frame,
                    request.minimum_history_bars,
                )
            )
            run_simple(dataset, selected)
        checks.append("candidate_binding")
        train_scope, train_observations = scope_from_backtest(
            results[0], split="train", trial_index=0, parameters=parameters
        )
        test_scope, test_observations = scope_from_backtest(
            results[1], split="test", parameters=parameters
        )
        artifacts = build_artifacts(
            run_id="preflight_search",
            manifest_hash=expected_hash,
            default_scope="test",
            scopes={"trial/0/train": train_scope, "test": test_scope},
            observations={
                "trial/0/train": train_observations,
                "test": test_observations,
            },
            selected_train_scope="trial/0/train",
        )
        train_metric = finite_metric_value(
            results[0].metrics.get(optimization.optimization_metric)
        )
        if train_metric is None:
            raise PreflightError(
                "unrankable_sample_objective",
                "The sample cannot evaluate the optimization objective; "
                "enlarge the sample or revise the settings.",
            )
        projection = normalize_json_object(
            {
                "best_params": parameters,
                "optimization_metric": optimization.optimization_metric,
                "train_metrics": results[0].metrics,
                "test_metrics": results[1].metrics,
                "best_train_metric_value": train_metric,
                "best_oos_metric_value": finite_metric_value(
                    results[1].metrics.get(optimization.optimization_metric)
                ),
            }
        )
        validate_projection(artifacts, projection)
        encoded = artifacts.model_dump_json()
        PerformanceArtifacts.model_validate_json(encoded)
        (sample_dir / "optimization-performance.json").write_text(
            encoded, encoding="utf-8"
        )
        coverage.update(
            tested_candidates=1, total_candidates=optimization.total_combinations
        )
        if optimization.total_combinations > 1:
            warnings.append("single_candidate_only")
    else:
        wf = WalkForwardConfig.model_validate(config["validation"])
        explicit = not wf.purged_split and all(
            value is not None
            for value in (wf.train_start, wf.train_end, wf.test_start, wf.test_end)
        )
        if explicit:
            frames = []
            for wf_role, wf_begin, wf_finish, wf_limit in (
                ("train", wf.train_start, wf.train_end, request.sample_bars // 2),
                (
                    "test",
                    wf.test_start,
                    wf.test_end,
                    request.sample_bars - request.sample_bars // 2,
                ),
            ):
                assert wf_begin is not None and wf_finish is not None
                lower, upper = inclusive_day_window(wf_begin, wf_finish)
                frames.append(
                    load(
                        wf_role,
                        lower,
                        upper - timedelta(microseconds=1),
                        wf_limit,
                    ).frame
                )
            train, test = frames
            total_folds = 1
        else:
            dataset = load("fold", bt.start_date, bt.end_date, request.sample_bars)
            splitter = PurgedDataSplitter(wf) if wf.purged_split else DataSplitter(wf)
            splits = splitter.split(dataset.frame)
            if not splits:
                raise PreflightError(
                    "insufficient_sample_history",
                    "The sample produces no walk-forward fold.",
                )
            train, test = splits[0]
            total_folds = wf.n_splits
        windows.extend(
            (
                _window("train", train, request.minimum_history_bars),
                _window("test", test, request.minimum_history_bars),
            )
        )
        sample = _sample_manifest(original, pd.concat([train, test]), walk_forward=True)
        sample_config = sample.config_for_execution()
        service = BacktestService(strategy_gate=strategies, strategy_loader=loader)
        with execution_randomness(sample_config["random_seed"]):
            engine = service.create_engine(
                BacktestConfig.model_validate(sample_config["backtest"])
            )
            strategy = loader.create_from_execution(
                sample_config["strategy"],
                sample.strategy_execution,
                engine,
                ParallelConfig.model_validate(sample_config["parallel"]),
            )
            fold = WalkForwardValidator(wf, engine).run_fold(0, strategy, train, test)
        assert fold.train_backtest is not None and fold.test_backtest is not None
        results.extend((fold.train_backtest, fold.test_backtest))
        pipeline = PipelineResult(
            mode="walk_forward",
            fold_results=[fold],
            config_snapshot=sample_config,
            execution_manifest=sample,
        )
        _serialize(sample_dir, 0, pipeline, summarize_results([fold]))
        checks.append("fit")
        coverage.update(tested_fold_count=1, total_fold_count=total_folds)
        if total_folds > 1:
            warnings.append("single_fold_only")
    original.verify(expected_hash=expected_hash)
    trade_count = sum(len(result.trades) for result in results)
    if trade_count == 0:
        warnings.append("no_trades_observed")
    record_count = (
        sum(len(result.execution_records or []) for result in results)
        if all(result.execution_records is not None for result in results)
        else None
    )
    receipt = PreflightReceipt(
        manifest_hash=expected_hash,
        elapsed_seconds=time.monotonic() - started,
        sample_bars_requested=request.sample_bars,
        sample_bars_used=sum(window.rows for window in windows),
        minimum_history_bars=request.minimum_history_bars,
        data_source=config["data"]["requirements"]["market"]["source"],
        windows=windows,
        checked_paths=checks,
        trade_count=trade_count,
        execution_record_count=record_count,
        trading_path_exercised=trade_count > 0,
        warnings=warnings,
        **coverage,
    )
    return PreflightPayload(
        manifest=original,
        original_config_path=str(prepared.original_config_path),
        receipt=receipt,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_path", type=Path)
    parser.add_argument("result_path", type=Path)
    arguments = parser.parse_args()
    try:
        envelope = json.loads(arguments.request_path.read_text(encoding="utf-8"))
        request = PreflightRequest.model_validate(envelope["request"])
        result = execute_preflight(
            request,
            source_workspace=WorkspacePaths(Path(envelope["source_workspace"])),
            directory=arguments.request_path.parent,
        )
        response: dict[str, Any] = {
            "success": True,
            "result": result.model_dump(mode="json"),
        }
    except Exception as error:
        response = {
            "success": False,
            "code": getattr(error, "code", "preflight_failed"),
            "message": str(error),
        }
        guidance = (
            error.execution_guidance
            if isinstance(error, PreflightError)
            else error.guidance
            if isinstance(error, StrategyExecutionModeError)
            else None
        )
        if guidance is not None:
            response["execution_guidance"] = guidance.model_dump(mode="json")
    arguments.result_path.write_text(
        json.dumps(response, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
