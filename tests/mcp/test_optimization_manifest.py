"""Optimization workers execute only the accepted, verified specification."""

from __future__ import annotations

import json
import os
import pickle
import signal
import time
from concurrent.futures import Future
from concurrent.futures import wait as wait_for_pool
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pytest
from tests.integration.mcp_harness import temporary_mcp_workspace
from tests.strategy_fixtures import GENERATED_PROGRAM

from tradingdev.adapters.execution.process_runner import (
    ProcessRunner,
    WorkerHandle,
    request_worker_stop,
)
from tradingdev.adapters.execution.strategy_factory import StrategyFactory
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.optimization_trials import OptimizationLedger
from tradingdev.adapters.storage.performance import PerformanceArtifactError
from tradingdev.app import job_store
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.contracts.jobs import JobStatus
from tradingdev.app.data_service import DataService, LoadedDataset
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.run_service import RunService
from tradingdev.domain.backtest.result import BacktestResult
from tradingdev.domain.backtest.schemas import ParallelConfig
from tradingdev.domain.execution import ExecutionManifest, OptimizationSpec
from tradingdev.domain.research_scope import optimization_backtest
from tradingdev.domain.runtime import IsolatedRuntimeError
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.bundled.kd_strategy.config import KDStrategyConfig
from tradingdev.domain.strategies.bundled.kd_strategy.strategy import KDStrategy
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.domain.strategies.schemas import StrategySpec, StrategyStatus
from tradingdev.domain.strategies.signals import StrategyCallbackError
from tradingdev.mcp.workers import optimization
from tradingdev.mcp.workers.optimization import ComboEvaluation, _run_optimization

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
    from tradingdev.domain.backtest.schemas import BacktestConfig


def _prepared_strategy(
    _service: BacktestService, raw_config: dict[str, Any]
) -> StrategySpec:
    """Substitute source verification while retaining its typed return contract."""
    return StrategySpec(
        strategy_id=str(raw_config["strategy"]["id"]),
        class_name="FixtureStrategy",
        source_path="fixture.py",
        config_path="fixture.yaml",
        status=StrategyStatus.RUNNABLE,
    )


def _evaluation(
    params: dict[str, Any],
    metric: str,
    value: float | None,
    frame: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> ComboEvaluation:
    """Distinct trial observations make accidental winner-only storage visible."""
    marker = float(params.get("direction", 1))
    equity = 10000.0 + np.arange(len(frame), dtype=np.float64) * marker
    result = BacktestResult(
        metrics={metric: value, "daily_pnl_median": marker * 17.0},
        equity_curve=equity,
        returns=np.zeros(len(frame), dtype=np.float64),
        timestamps=frame["timestamp"].to_numpy() if "timestamp" in frame else None,
        init_cash=10000.0,
        trades=[
            {
                "entry_idx": 0,
                "exit_idx": len(frame) - 1,
                "net_pnl": marker,
                "status": "closed",
                "entry_fees": 0.1,
                "exit_fees": 0.2,
            }
        ],
        metric_metadata={
            "schema_version": 1,
            "mode": "signal",
            "providers": {"fixture": "1.0"},
            "settings": {"trade_scope": "closed", "periods_per_year": 365},
            "unavailable": {metric: "insufficient_data"} if value is None else {},
        },
    )
    if config is not None:
        result.metric_metadata["execution_context"] = {
            **config.model_dump(mode="json"),
            "random_seed": 42,
        }
        result.metric_metadata["strategy_parameters"] = {**params, "warmup": 2}
    return ComboEvaluation(params, metric, result)


def _queued_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    metric: str = "total_return",
    trial_timeout_seconds: int = 300,
    directions: list[int] | None = None,
) -> tuple[JobStore, ExecutionManifest]:
    from tradingdev.domain.backtest.schemas import BacktestConfig

    workspace = WorkspacePaths(tmp_path / "workspace")
    store = JobStore(workspace=workspace)
    monkeypatch.setenv("TRADINGDEV_WORKSPACE", str(workspace.root))
    monkeypatch.setenv(
        "TRADINGDEV_WORKER_IDENTITY",
        json.dumps(WorkerHandle(4321, 100.0, "a" * 32).job_fields()),
    )
    config: dict[str, Any] = {
        "random_seed": 42,
        "strategy": {"id": "fixture"},
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-07T23:59:59.999999",
            "init_cash": 10000,
        },
    }
    config["data"] = DataService(workspace).execution_config(
        config, BacktestConfig(**config["backtest"])
    )
    manifest = ExecutionManifest.create(
        kind="optimization",
        strategy_execution=StrategyExecution(
            kind="generated",
            program=GENERATED_PROGRAM,
            constructor_kwargs={"direction": -1},
        ),
        config=config,
        optimization=OptimizationSpec.model_validate(
            {
                "param_ranges": {
                    "direction": [-1, 1] if directions is None else directions
                },
                "optimization_metric": metric,
                "train_start": "2024-01-01",
                "train_end": "2024-01-03",
                "test_start": "2024-01-04",
                "test_end": "2024-01-07",
                "trial_timeout_seconds": trial_timeout_seconds,
            }
        ),
    )
    store.create_job(job_id="fixture", job_type="optimization", manifest=manifest)

    def run_inline(
        runner: optimization.TrialRunner,
        indices: range,
        n_jobs: int,
        *,
        test: bool = False,
    ) -> dict[int, float | None]:
        return {index: runner.run(index, test=test) for index in indices}

    # Direct worker cases substitute external inputs in this process. The
    # supervised cases below import a fresh module and use the actual pool.
    monkeypatch.setattr(optimization, "_run_trials", run_inline)
    return store, manifest


@pytest.mark.parametrize("corruption", ["missing", "search", "kind"])
def test_invalid_optimization_manifest_fails_before_data_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    store, _ = _queued_job(tmp_path, monkeypatch)
    path = store.workspace.runs / "fixture" / "manifest.json"
    if corruption == "missing":
        path.unlink()
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if corruption == "search":
            payload["optimization"]["param_ranges"] = {"direction": [99]}
        else:
            payload["kind"] = "backtest"
        path.write_text(json.dumps(payload), encoding="utf-8")
    data_calls: list[dict[str, Any]] = []

    def unexpected_data_load(
        self: DataService, config: dict[str, Any], bt: BacktestConfig
    ) -> LoadedDataset:
        data_calls.append(config)
        raise AssertionError("Invalid manifest reached data loading")

    monkeypatch.setattr(DataService, "load", unexpected_data_load)
    _run_optimization("fixture")

    job = store.get_job("fixture")
    assert job is not None
    assert job["status"] == "failed"
    assert "Execution manifest error" in job["error"]
    assert data_calls == []
    assert not (path.parent / "result.json").exists()


def test_optimization_worker_ignores_mutable_config_and_search_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, manifest = _queued_job(tmp_path, monkeypatch)
    (store.workspace.runs / "fixture" / "config.yaml").write_text(
        "invalid: config projection changed after submission\n", encoding="utf-8"
    )
    store.update_job(
        "fixture",
        param_ranges={"direction": [99]},
        optimization_metric="profit_factor",
    )
    captured: list[dict[str, Any]] = []
    evaluations: list[tuple[dict[str, Any], str, str, str, dict[str, Any]]] = []

    def capture_data_load(
        self: DataService, config: dict[str, Any], bt: BacktestConfig
    ) -> LoadedDataset:
        captured.append(config)
        return LoadedDataset(
            frame=pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2024-01-01", periods=7, freq="D", tz="UTC"
                    )
                }
            ),
            processed_path=tmp_path / "unused.parquet",
            dataset_id="fixture-data",
        )

    def evaluate(
        strategy_cfg: dict[str, Any],
        strategy_execution: StrategyExecution,
        bt_cfg: BacktestConfig,
        frame: pd.DataFrame,
        params: dict[str, Any],
        metric: str,
        parallel_cfg: ParallelConfig,
        random_seed: int | None,
    ) -> ComboEvaluation:
        assert strategy_execution == manifest.strategy_execution
        assert random_seed == manifest.config["random_seed"]
        evaluations.append(
            (
                params,
                metric,
                bt_cfg.start_date.date().isoformat(),
                bt_cfg.end_date.date().isoformat(),
                parallel_cfg.model_dump(),
            )
        )
        return _evaluation(params, metric, 1.0, frame, bt_cfg)

    # Stub only external data/strategy evaluation. The real worker still loads
    # the manifest, expands the grid, selects its winner, performs OOS and saves.
    monkeypatch.setattr(BacktestService, "prepare_strategy", _prepared_strategy)
    monkeypatch.setattr(DataService, "load", capture_data_load)
    monkeypatch.setattr(optimization, "_run_single_combo", evaluate)
    monkeypatch.setattr(optimization, "estimate_n_jobs", lambda *args, **kwargs: 1)
    previous_handler = signal.getsignal(signal.SIGALRM)
    _run_optimization("fixture")

    assert signal.getsignal(signal.SIGALRM) == previous_handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert captured == [manifest.config_copy()]
    assert [item[:4] for item in evaluations] == [
        ({"direction": -1}, "total_return", "2024-01-01", "2024-01-03"),
        ({"direction": 1}, "total_return", "2024-01-01", "2024-01-03"),
        ({"direction": -1}, "total_return", "2024-01-04", "2024-01-07"),
    ]
    assert all(item[4] == manifest.config["parallel"] for item in evaluations)
    job = store.get_job("fixture")
    assert job is not None
    assert job["status"] == "done"
    assert job["best_params"] == {"direction": -1}
    assert store.load_manifest("fixture") == manifest


def test_optimization_evaluations_keep_unsearched_strategy_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tradingdev.domain.backtest.schemas import BacktestConfig

    strategy_config = {"id": "kd_crossover", "parameters": {"oversold": 15.0}}
    loader = StrategyFactory()
    execution = loader.resolve_execution(strategy_config)
    captured: list[dict[str, Any]] = []

    class ChangedDefaultConfig(KDStrategyConfig):
        k_period: int = 21

    monkeypatch.setattr(
        StrategyFactory,
        "_bundled_config_model",
        lambda *args, **kwargs: ChangedDefaultConfig,
    )

    def capture_signals(self: KDStrategy, frame: pd.DataFrame) -> pd.DataFrame:
        captured.append(self.get_parameters())
        return frame.assign(signal=0)

    observed_result = _evaluation(
        {}, "total_return", 1.0, pd.DataFrame({"close": [1.0, 2.0]})
    ).result
    engine = cast(
        "BaseBacktestEngine",
        SimpleNamespace(run=lambda frame: observed_result),
    )
    monkeypatch.setattr(BacktestService, "prepare_strategy", _prepared_strategy)
    monkeypatch.setattr(BacktestService, "create_engine", lambda *args: engine)
    monkeypatch.setattr(KDStrategy, "generate_signals", capture_signals)
    config = BacktestConfig(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-07",
        init_cash=10000,
    )
    frame = pd.DataFrame({"close": [1.0, 2.0]})
    parallel = ParallelConfig()
    evaluation = optimization._run_single_combo(
        strategy_config,
        execution,
        config,
        frame,
        {"d_period": 5},
        "total_return",
        parallel,
        42,
    )
    optimization._run_single_combo(
        strategy_config,
        execution,
        config,
        frame,
        {"d_period": 7},
        "total_return",
        parallel,
        42,
    )

    assert [parameters["k_period"] for parameters in captured] == [14, 14]
    assert [parameters["d_period"] for parameters in captured] == [5, 7]
    assert [parameters["oversold"] for parameters in captured] == [15.0, 15.0]
    assert evaluation.result is observed_result
    assert evaluation.result.metric_metadata["execution_context"] == {
        **config.model_dump(mode="json"),
        "random_seed": 42,
    }
    assert evaluation.result.metric_metadata["strategy_parameters"]["oversold"] == 15.0
    restored = pickle.loads(pickle.dumps(evaluation))
    assert restored.parameters == evaluation.parameters
    assert restored.result.metrics == observed_result.metrics
    np.testing.assert_array_equal(
        restored.result.equity_curve, observed_result.equity_curve
    )


def test_optimization_preserves_frame_values_types_and_index_across_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _ = _queued_job(tmp_path, monkeypatch)
    timestamps = pd.date_range("2024-01-01", periods=7, freq="D", tz="UTC")
    market_times = (timestamps + pd.Timedelta(nanoseconds=123)).where(
        timestamps != timestamps[0], timestamps
    )
    market_times = market_times.where(
        timestamps != timestamps[2],
        timestamps[3] - pd.Timedelta(nanoseconds=1),
    )
    frame = pd.DataFrame(
        {
            "timestamp": market_times,
            "close": np.arange(1, 8, dtype=float) * 1.23456789012345e-11,
            "feature": np.full(7, np.nextafter(1.0, 2.0)),
            "float32_feature": np.arange(7, dtype=np.float32),
            "nullable_feature": pd.array([1, None, 3, 4, 5, 6, 7], dtype="Int64"),
            "flag": pd.array(
                [True, False, None, True, False, True, False], dtype="boolean"
            ),
        },
        index=pd.DatetimeIndex(timestamps.tz_convert("Asia/Taipei"), name="bar_time"),
    )
    received: list[pd.DataFrame] = []

    def evaluate(
        strategy_cfg: dict[str, Any],
        strategy_execution: StrategyExecution,
        bt_cfg: BacktestConfig,
        data: pd.DataFrame,
        params: dict[str, Any],
        metric: str,
        parallel_cfg: ParallelConfig,
        random_seed: int | None,
    ) -> ComboEvaluation:
        received.append(data.copy(deep=True))
        return _evaluation(params, metric, 1.0, data, bt_cfg)

    monkeypatch.setattr(BacktestService, "prepare_strategy", _prepared_strategy)
    monkeypatch.setattr(
        DataService,
        "load",
        lambda *args: LoadedDataset(
            frame=frame,
            processed_path=tmp_path / "unused.parquet",
            dataset_id="precise-frame",
        ),
    )
    monkeypatch.setattr(optimization, "_run_single_combo", evaluate)
    monkeypatch.setattr(optimization, "estimate_n_jobs", lambda *args, **kwargs: 1)
    _run_optimization("fixture")
    job = store.get_job("fixture")
    assert job is not None and job["status"] == "done", job
    assert len(received) == 3
    for actual, expected in zip(
        received, [frame.iloc[:3], frame.iloc[:3], frame.iloc[3:]], strict=True
    ):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def _run_drawdown_optimization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    values: list[float | None],
) -> tuple[JobStore, ExecutionManifest, list[dict[str, Any]]]:
    store, manifest = _queued_job(tmp_path, monkeypatch, metric="max_drawdown")

    def load(
        self: DataService, config: dict[str, Any], bt: BacktestConfig
    ) -> LoadedDataset:
        return LoadedDataset(
            frame=pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2024-01-01", periods=7, freq="D", tz="UTC"
                    )
                }
            ),
            processed_path=tmp_path / "unused.parquet",
            dataset_id="fixture-data",
        )

    oos_calls: list[dict[str, Any]] = []

    def evaluate(
        strategy_cfg: dict[str, Any],
        strategy_execution: StrategyExecution,
        bt_cfg: BacktestConfig,
        frame: pd.DataFrame,
        params: dict[str, Any],
        metric: str,
        parallel_cfg: ParallelConfig,
        random_seed: int | None,
    ) -> ComboEvaluation:
        if bt_cfg.start_date.day == 4:
            oos_calls.append(params)
            # A valid training winner may have an unavailable OOS metric.
            return _evaluation(params, metric, None, frame, bt_cfg)
        value = values[0 if params["direction"] == -1 else 1]
        return _evaluation(params, metric, value, frame, bt_cfg)

    monkeypatch.setattr(BacktestService, "prepare_strategy", _prepared_strategy)
    monkeypatch.setattr(DataService, "load", load)
    monkeypatch.setattr(optimization, "_run_single_combo", evaluate)
    monkeypatch.setattr(optimization, "estimate_n_jobs", lambda *args, **kwargs: 1)
    _run_optimization("fixture")
    return store, manifest, oos_calls


@pytest.mark.parametrize(
    ("values", "expected"),
    [([0.5, 0.1], 1), ([None, 0.2], 1), ([0.3, None], -1), ([None, None], None)],
)
def test_worker_minimizes_drawdown_and_rejects_unrankable_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    values: list[float | None],
    expected: int | None,
) -> None:
    store, manifest, oos_calls = _run_drawdown_optimization(
        tmp_path, monkeypatch, values
    )
    assert manifest.optimization is not None
    assert manifest.optimization.direction == "minimize"
    job = store.get_job("fixture")
    assert job is not None
    if expected is None:
        assert job["status"] == "partial"
        progress = (
            RunService(workspace=store.workspace, store=store.store)
            .get_optimization_trials("fixture")
            .progress
        )
        assert (
            progress.completed == 2
            and progress.rankable == 0
            and progress.search_complete
        )
        assert (
            progress.test.state == "not_attempted"
            and progress.leading_trial_index is None
        )
        assert "No parameter combination has a finite objective value" in job["error"]
        assert oos_calls == []
        assert not (store.workspace.runs / "fixture" / "result.json").exists()
    else:
        assert job["status"] == "done"
        assert job["best_params"] == {"direction": expected}
        assert oos_calls == [{"direction": expected}]
        result = store.load_result(str(job["result_path"]))
        assert result is not None
        assert result["direction"] == "minimize"
        assert result["best_oos_metric_value"] is None
        run_dir = store.workspace.runs / "fixture"
        performance_text = (run_dir / "performance.json").read_text()
        observations_text = (run_dir / "observations.json").read_text()
        assert "NaN" not in performance_text + observations_text
        assert "Infinity" not in performance_text + observations_text
        performance = json.loads(performance_text)
        observations = json.loads(observations_text)
        expected_scopes = {"trial/0/train", "trial/1/train", "test"}
        assert performance["scopes"].keys() == expected_scopes
        assert observations["scopes"].keys() == expected_scopes
        assert performance["default_scope"] == "test"
        selected = "trial/0/train" if expected == -1 else "trial/1/train"
        assert performance["selected_train_scope"] == selected
        assert (
            performance["manifest_hash"]
            == observations["manifest_hash"]
            == manifest.manifest_hash
        )
        assert (
            performance["definitions"]["max_drawdown"]["optimization_direction"]
            == "minimize"
        )
        for index, direction in enumerate([-1, 1]):
            key = f"trial/{index}/train"
            scope = performance["scopes"][key]
            assert scope["parameters"] == {"direction": direction}
            assert scope["trial_index"] == index
            assert scope["values"]["max_drawdown"] == values[index]
            assert scope["values"]["daily_pnl_median"] == direction * 17.0
            assert scope["metadata"]["providers"] == {"fixture": "1.0"}
            assert (
                scope["metadata"]["execution_context"]["end_date"]
                == "2024-01-03T23:59:59.999999"
            )
            assert scope["metadata"]["strategy_parameters"] == {
                "direction": direction,
                "warmup": 2,
            }
            raw = observations["scopes"][key]
            assert raw["equity_curve"] == [
                10000.0,
                10000.0 + direction,
                10000.0 + direction * 2,
            ]
            assert raw["returns"] == [0.0, 0.0, 0.0]
            assert len(raw["timestamps"]) == 3
            assert raw["trades"][0]["net_pnl"] == direction
            assert raw["trades"][0]["entry_fees"] == 0.1
        oos_scope = performance["scopes"]["test"]
        assert oos_scope["values"]["max_drawdown"] is None
        assert (
            oos_scope["metadata"]["unavailable"]["max_drawdown"] == "insufficient_data"
        )
        assert len(observations["scopes"]["test"]["equity_curve"]) == 4


def test_done_status_retains_ledger_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _, _ = _run_drawdown_optimization(tmp_path, monkeypatch, [0.5, 0.1])
    jobs = JobService(job_store=store)
    runs = RunService(workspace=store.workspace, store=store.store)
    before = JobStatus.model_validate(jobs.get_job_status("fixture"))
    assert before.status == "done" and before.details_available
    assert before.detail_error is None
    assert before.optimization is not None and before.optimization.completed == 2

    with closing(store.store.connect()) as connection, connection:
        changed = connection.execute(
            "update optimization_trials set evidence = ? "
            "where job_id = ? and position = ?",
            (b"corrupted observations", "fixture", 1),
        )
        assert changed.rowcount == 1

    with pytest.raises(PerformanceArtifactError) as error:
        runs.get_optimization_trials("fixture")
    assert error.value.code == "optimization_evidence_invalid"
    assert runs.get_run("fixture")["run"]["detail_error"] is None

    status = JobStatus.model_validate(jobs.get_job_status("fixture"))
    assert status.status == "done" and status.run_id == "fixture"
    assert status.details_available
    assert status.metrics == before.metrics
    assert status.train_metrics == before.train_metrics
    assert status.test_metrics == before.test_metrics
    assert status.optimization is None
    assert status.detail_error is not None
    assert status.detail_error.code == "optimization_evidence_invalid"


@pytest.mark.parametrize(
    ("failure", "status", "states", "test_state"),
    [
        ("trial", "partial", ["failed", "completed"], "not_attempted"),
        ("timeout", "partial", ["timed_out", "completed"], "not_attempted"),
        ("batch", "failed", ["completed", "interrupted"], "not_attempted"),
        ("oos", "partial", ["completed", "completed"], "failed"),
        ("artifacts", "failed", ["completed", "completed"], "interrupted"),
        ("storage", "failed", ["completed", "completed"], "completed"),
    ],
)
def test_optimization_failure_preserves_only_committed_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    status: str,
    states: list[str],
    test_state: str,
) -> None:
    store, manifest = _queued_job(tmp_path, monkeypatch)

    def load(
        self: DataService, config: dict[str, Any], bt: BacktestConfig
    ) -> LoadedDataset:
        return LoadedDataset(
            frame=pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2024-01-01", periods=7, freq="D", tz="UTC"
                    )
                }
            ),
            processed_path=tmp_path / "unused.parquet",
            dataset_id="fixture-data",
        )

    calls = 0

    def evaluate(
        strategy_cfg: dict[str, Any],
        strategy_execution: StrategyExecution,
        bt_cfg: BacktestConfig,
        frame: pd.DataFrame,
        params: dict[str, Any],
        metric: str,
        parallel_cfg: ParallelConfig,
        random_seed: int | None,
    ) -> ComboEvaluation:
        nonlocal calls
        calls += 1
        if failure == "timeout" and calls == 1:
            raise optimization._TrialTimeoutError("fixture timeout")
        if (failure, calls) in {("trial", 1), ("oos", 3)}:
            raise StrategyCallbackError(f"fixture {failure} failure")
        if failure == "batch" and calls == 2:
            raise RuntimeError("fixture infrastructure failure")
        evaluation = _evaluation(params, metric, 1.0, frame, bt_cfg)
        if failure == "artifacts" and calls == 3:
            # A corrupt raw series must be rejected before creating either file.
            evaluation.result.returns = np.array([0.0])
        return evaluation

    monkeypatch.setattr(BacktestService, "prepare_strategy", _prepared_strategy)
    monkeypatch.setattr(DataService, "load", load)
    monkeypatch.setattr(optimization, "_run_single_combo", evaluate)
    monkeypatch.setattr(optimization, "estimate_n_jobs", lambda *args, **kwargs: 1)
    if failure == "storage":

        def cannot_save(*args: Any, **kwargs: Any) -> Path:
            raise OSError("fixture result filesystem unavailable")

        monkeypatch.setattr(job_store, "save_result", cannot_save)
    previous_handler = signal.getsignal(signal.SIGALRM)
    _run_optimization("fixture")

    assert signal.getsignal(signal.SIGALRM) == previous_handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    job = store.get_job("fixture")
    assert job is not None
    assert job["status"] == status
    service = RunService(workspace=store.workspace, store=store.store)
    evidence = service.get_optimization_trials("fixture")
    assert [trial.state for trial in evidence.trials.items] == states
    assert evidence.progress.test.state == test_state
    assert evidence.progress.manifest_hash == manifest.manifest_hash
    assert evidence.progress.dataset_id == "fixture-data"
    assert evidence.progress.completed == states.count("completed")
    assert evidence.progress.estimation == (
        "unavailable" if failure in {"trial", "timeout"} else "available"
    )
    index = states.index("completed")
    assert service.get_run_metrics("fixture")["code"] == "explicit_scope_required"
    metrics = service.get_run_metrics("fixture", scope=f"trial/{index}/train")
    assert metrics["metrics"]["total_return"] == 1.0
    assert metrics["metrics"]["daily_pnl_median"] == [-17.0, 17.0][index]
    assert metrics["parameters"] == {"direction": [-1, 1][index]}
    assert metrics["manifest_hash"] == manifest.manifest_hash
    assert metrics["result_completeness"] == "partial"
    assert metrics["default_scope"] is None
    test_metrics = service.get_run_metrics("fixture", scope="test")
    if failure == "storage":
        assert test_metrics["metrics"]["total_return"] == 1.0
        assert test_metrics["default_scope"] is None
        assert test_metrics["result_completeness"] == "partial"
    else:
        assert test_metrics["code"] == "optimization_scope_unavailable"
    assert store.list_runs() == []
    run_dir = store.workspace.runs / "fixture"
    assert (run_dir / "manifest.json").is_file()
    assert not (run_dir / "result.json").exists()
    assert not (run_dir / "performance.json").exists()
    assert not (run_dir / "observations.json").exists()
    assert not list(run_dir.glob("*.tmp"))

    if failure == "batch":
        with closing(store.store.connect()) as connection, connection:
            connection.execute(
                "update optimization_trials set evidence = ? "
                "where job_id = ? and position = ?",
                (b"corrupted observations", "fixture", index),
            )
        assert service.get_run_metrics("fixture", scope=f"trial/{index}/train")[
            "code"
        ] == ("optimization_evidence_invalid")


def test_data_failure_leaves_all_approved_trials_unattempted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, manifest = _queued_job(tmp_path, monkeypatch)

    def cannot_load(*args: Any) -> LoadedDataset:
        raise OSError("fixture corrupt market data")

    monkeypatch.setattr(BacktestService, "prepare_strategy", _prepared_strategy)
    monkeypatch.setattr(DataService, "load", cannot_load)
    _run_optimization("fixture")

    service = RunService(workspace=store.workspace, store=store.store)
    result = service.get_optimization_trials("fixture")
    assert result.progress.stop_reason == "Data error: fixture corrupt market data"
    assert result.progress.dataset_id is None
    assert result.progress.manifest_hash == manifest.manifest_hash
    assert result.progress.test.state == "not_attempted"
    assert result.progress.attempted == result.progress.completed == 0
    assert [trial.state for trial in result.trials.items] == ["not_attempted"] * 2


@pytest.mark.parametrize("phase", ["constructor", "generate_signals"])
def test_real_trial_alarm_during_contract_execution_retains_timeout_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """Exercise the host trial alarm with a substituted strategy boundary."""
    entered: list[str] = []

    class SlowStrategy(BaseStrategy):
        def __init__(
            self, direction: int, backtest_engine: BaseBacktestEngine | None = None
        ) -> None:
            # Contract fixtures inject None; reaching a real-engine execution
            # would mean this test missed the regression's actual failure site.
            assert backtest_engine is None
            self.direction = direction
            if phase == "constructor":
                entered.append("constructor")
                time.sleep(10)

        def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
            assert len(df) == 80
            entered.append("generate_signals")
            time.sleep(10)
            result = df.copy()
            result["signal"] = 0
            return result

        def get_parameters(self) -> dict[str, Any]:
            return {"direction": self.direction}

    previous_handler = signal.getsignal(signal.SIGALRM)
    with TemporaryDirectory(prefix="trial-timeout-", dir=tmp_path) as temporary:
        root = Path(temporary)
        store, _ = _queued_job(root, monkeypatch, trial_timeout_seconds=1)

        def load(
            self: DataService, config: dict[str, Any], bt: BacktestConfig
        ) -> LoadedDataset:
            return LoadedDataset(
                frame=pd.DataFrame(
                    {
                        "timestamp": pd.date_range(
                            "2024-01-01", periods=7, freq="D", tz="UTC"
                        )
                    }
                ),
                processed_path=root / "unused.parquet",
                dataset_id="fixture-data",
            )

        @contextmanager
        def open_strategy(
            self: StrategyFactory,
            config: dict[str, Any],
            execution: StrategyExecution,
            backtest_engine: BaseBacktestEngine | None,
            parallel_config: ParallelConfig | None = None,
            *,
            classify_callback_error: bool = False,
        ) -> Iterator[BaseStrategy]:
            direction = execution.constructor_kwargs["direction"]
            assert isinstance(direction, int)
            yield SlowStrategy(direction)

        # The fake context and data replace strategy/container and data inputs.
        # The host trial, contract fixture, alarm and SQLite are real.
        monkeypatch.setattr(BacktestService, "prepare_strategy", _prepared_strategy)
        monkeypatch.setattr(DataService, "load", load)
        monkeypatch.setattr(StrategyFactory, "open_from_execution", open_strategy)
        monkeypatch.setattr(
            StrategyFactory,
            "override_execution",
            lambda self, config, execution, overrides: StrategyExecution(
                kind=execution.kind,
                program=execution.program,
                constructor_kwargs={**execution.constructor_kwargs, **overrides},
            ),
        )
        try:
            _run_optimization("fixture")
            assert entered == [phase, phase]
            assert signal.getsignal(signal.SIGALRM) == previous_handler
            assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
            job = store.get_job("fixture")
            assert job is not None
            assert job["status"] == "partial", job
            results = RunService(workspace=store.workspace, store=store.store)
            trials = results.get_optimization_trials("fixture").trials.items
            assert [item.state for item in trials] == ["timed_out", "timed_out"]
            assert all("exceeded 1s timeout" in (item.reason or "") for item in trials)
            assert job["ended_at"] is not None
            assert store.list_runs() == []
            run_dir = store.workspace.runs / "fixture"
            assert {path.name for path in run_dir.iterdir()} == {
                ".submission.lock",
                "manifest.json",
                "config.yaml",
            }
            assert not list(root.rglob(".pending-*"))
        finally:
            # Restore process signal state even when a regression fails an assert.
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
    assert not root.exists()


class ProcessTrialRunner(optimization.TrialRunner):
    """Keep the real worker timer and ledger around a deterministic trial fixture."""

    def _evaluate(self, params: dict[str, Any]) -> ComboEvaluation:
        def evaluate(
            strategy: dict[str, Any],
            execution: StrategyExecution,
            config: BacktestConfig,
            frame: pd.DataFrame,
            candidate: dict[str, Any],
            metric: str,
            parallel: ParallelConfig,
            seed: int | None,
        ) -> ComboEvaluation:
            if candidate["direction"] == 1:
                if strategy["scenario"] == "crash":
                    os._exit(23)
                if strategy["scenario"] == "runtime":
                    raise IsolatedRuntimeError(
                        "runtime_exit",
                        "fixture daemon failure",
                        control_path="fixture-control",
                    )
                if strategy["scenario"] == "unresponsive":
                    signal.signal(signal.SIGALRM, signal.SIG_IGN)
                time.sleep(30)
            if candidate["direction"] == 2 and strategy["scenario"] != "timeout":
                time.sleep(30)
            return _evaluation(
                candidate, metric, float(candidate["direction"]), frame, config
            )

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(optimization, "_run_single_combo", evaluate)
            return super()._evaluate(params)


def _process_fixture(scenario: str) -> None:
    store = JobStore()
    manifest = store.load_manifest("fixture")
    assert isinstance(manifest, ExecutionManifest)
    ledger = OptimizationLedger(store.store, "fixture", manifest, "process-data")
    ledger.initialize()
    store.update_job("fixture", status="optimizing", dataset_id="process-data")
    runner = ProcessTrialRunner(
        ledger,
        {"scenario": scenario},
        manifest.strategy_execution,
        optimization_backtest(manifest, "train"),
        pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, tz="UTC")}),
        ParallelConfig(),
        42,
    )
    runner.run(0)

    def shorten_watchdog(
        futures: dict[Future[float | None], tuple[int, float]],
        *,
        timeout: float,
        return_when: str,
    ) -> Any:
        # Keep startup time independent of the test watchdog on slower hosts.
        # Once the process has entered the unresponsive trial, wait on the
        # real future with a short deadline to exercise the supervisor path.
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            snapshot = ledger.snapshot()
            if snapshot is not None and snapshot.trials[1].state == "running":
                return wait_for_pool(
                    futures, timeout=min(timeout, 0.5), return_when=return_when
                )
            time.sleep(0.05)
        raise RuntimeError("Unresponsive trial never reached its callback")

    with pytest.MonkeyPatch.context() as patch:
        if scenario == "unresponsive":
            patch.setattr(optimization, "wait", shorten_watchdog)
        try:
            optimization._run_trials(
                runner,
                range(1, len(ledger.grid)),
                1 if scenario == "unresponsive" else 2,
            )
        except (RuntimeError, TimeoutError) as exc:
            store.update_job("fixture", status="failed", error=str(exc))
        else:
            store.update_job("fixture", status="partial")


@pytest.mark.parametrize(
    "scenario", ["timeout", "crash", "runtime", "unresponsive", "cancel"]
)
def test_supervised_pool_retains_commits_and_cleans_child_processes(
    monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    with temporary_mcp_workspace() as workspace:
        store, _ = _queued_job(
            workspace.root,
            monkeypatch,
            directions=[-1, 1, 2, -1],
            trial_timeout_seconds=20 if scenario == "cancel" else 2,
        )
        handle = ProcessRunner(workspace=store.workspace).spawn_module(
            "tests.mcp.test_optimization_manifest", scenario
        )
        store.update_job("fixture", **handle.job_fields())
        control = store.workspace.root / ".workers" / handle.control_id
        service = RunService(workspace=store.workspace, store=store.store)
        try:
            deadline = time.monotonic() + 55
            while time.monotonic() < deadline:
                if scenario == "cancel":
                    job = store.get_job("fixture")
                    if (
                        job
                        and job.get("dataset_id")
                        and service.get_optimization_trials("fixture")
                        .trials.items[1]
                        .state
                        == "running"
                    ):
                        cancelled = JobService(job_store=store).cancel_job("fixture")
                        assert cancelled["success"], cancelled
                        break
                elif (control / "finished.json").exists():
                    break
                time.sleep(0.05)
            else:
                pytest.fail(
                    f"Supervised optimization {scenario} did not finish in time"
                )
        finally:
            request_worker_stop(store.workspace.root, handle, timeout=15)

        finished = json.loads((control / "finished.json").read_bytes())
        assert finished["cleaned"] is True and finished["error"] is None
        assert not list(store.workspace.root.rglob("*.tmp"))
        trials = service.get_optimization_trials("fixture").trials.items
        assert JobService(job_store=store).list_jobs()[0]["completed"] == sum(
            trial.state == "completed" for trial in trials
        )
        assert trials[0].state == "completed"
        assert trials[1].state == (
            "timed_out" if scenario == "timeout" else "interrupted"
        )
        assert (
            service.get_run_metrics("fixture", scope="trial/0/train")["metrics"][
                "total_return"
            ]
            == -1.0
        )
        if scenario == "timeout":
            assert [trial.state for trial in trials[2:]] == ["completed", "completed"]
        if scenario != "timeout":
            assert trials[3].state == "not_attempted"
        if scenario in {"crash", "runtime", "unresponsive"}:
            job = store.get_job("fixture")
            assert job is not None and job["status"] == "failed"
            if scenario == "runtime":
                assert "runtime_exit: fixture daemon failure" in job["error"]
                assert "fixture-control" in job["error"]
            if scenario == "unresponsive":
                assert "process exceeded its timeout" in job["error"]


def test_watchdog_stops_dispatch_when_peer_finishes_at_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_trials = optimization._run_trials
    store, manifest = _queued_job(tmp_path, monkeypatch, directions=[-1, 1, 2])
    ledger = OptimizationLedger(store.store, "fixture", manifest, "scheduler-data")
    runner = cast("optimization.TrialRunner", SimpleNamespace(ledger=ledger))
    pending, completed = Future[float | None](), Future[float | None]()
    clock = [0.0]
    submitted: list[int] = []
    shutdown: list[bool] = []

    class Pool:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> Pool:
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def submit(self, task: object, index: int, test: bool) -> Future[float | None]:
            submitted.append(index)
            return pending if index == 0 else completed

        def kill_workers(self) -> None:
            shutdown.append(True)

    def first_completion(
        futures: dict[Future[float | None], tuple[int, float]],
        *,
        timeout: float,
        return_when: str,
    ) -> tuple[set[Future[float | None]], set[Future[float | None]]]:
        assert timeout > 0 and set(futures) == {pending, completed}
        clock[0] += timeout
        completed.set_result(None)
        return {completed}, {pending}

    monkeypatch.setattr(
        optimization, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    monkeypatch.setattr(optimization, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(optimization, "wait", first_completion)
    with pytest.raises(TimeoutError, match="Trial 0 process exceeded"):
        run_trials(runner, range(3), 2)
    assert submitted == [0, 1] and shutdown == [True]


if __name__ == "__main__":
    import sys

    _process_fixture(sys.argv[1])
