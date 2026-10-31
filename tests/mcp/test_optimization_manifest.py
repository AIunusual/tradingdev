"""Optimization workers execute only the accepted, verified specification."""

from __future__ import annotations

import json
import pickle
import signal
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pytest

from tradingdev.adapters.execution.process_runner import WorkerHandle
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService, LoadedDataset
from tradingdev.app.job_store import JobStore
from tradingdev.domain.backtest.result import BacktestResult
from tradingdev.domain.execution import ExecutionManifest, OptimizationSpec
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.bundled.kd_strategy.config import KDStrategyConfig
from tradingdev.domain.strategies.bundled.kd_strategy.strategy import KDStrategy
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.strategies.schemas import StrategySpec, StrategyStatus
from tradingdev.mcp.workers import optimization
from tradingdev.mcp.workers.optimization import ComboEvaluation, _run_optimization

if TYPE_CHECKING:
    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
    from tradingdev.domain.backtest.schemas import BacktestConfig, ParallelConfig


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
        result.metric_metadata["execution_context"] = config.model_dump(mode="json")
        result.metric_metadata["strategy_parameters"] = {**params, "warmup": 2}
    return ComboEvaluation(params, metric, result)


def _queued_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    metric: str = "total_return",
    trial_timeout_seconds: int = 300,
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
            kind="generated", constructor_kwargs={"direction": -1}
        ),
        config=config,
        optimization=OptimizationSpec.model_validate(
            {
                "param_ranges": {"direction": [-1, 1]},
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
    from tradingdev.domain.backtest.schemas import BacktestConfig, ParallelConfig

    strategy_config = {"id": "kd_crossover", "parameters": {"oversold": 15.0}}
    loader = StrategyLoader()
    execution = loader.resolve_execution(strategy_config)
    captured: list[dict[str, Any]] = []

    class ChangedDefaultConfig(KDStrategyConfig):
        k_period: int = 21

    monkeypatch.setattr(
        StrategyLoader,
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
    optimization._evaluate_combo(
        strategy_config,
        execution,
        config.model_dump(),
        frame,
        {"d_period": 7},
        "total_return",
        parallel.model_dump(),
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
    frame = pd.DataFrame(
        {
            "timestamp": timestamps + pd.Timedelta(nanoseconds=123),
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
    store, manifest = _queued_job(tmp_path, monkeypatch, metric="max_drawdown")
    assert manifest.optimization is not None
    assert manifest.optimization.direction == "minimize"

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
    job = store.get_job("fixture")
    assert job is not None
    if expected is None:
        assert job["status"] == "failed"
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


@pytest.mark.parametrize("failure", ["trial", "timeout", "batch", "oos", "artifacts"])
def test_optimization_failure_restores_alarm_without_partial_performance_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    store, _ = _queued_job(tmp_path, monkeypatch)

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
        if (failure, calls) in {("trial", 1), ("batch", 2), ("oos", 3)}:
            raise RuntimeError(f"fixture {failure} failure")
        evaluation = _evaluation(params, metric, 1.0, frame, bt_cfg)
        if failure == "artifacts" and calls == 3:
            # A corrupt raw series must be rejected before creating either file.
            evaluation.result.returns = np.array([0.0])
        return evaluation

    monkeypatch.setattr(BacktestService, "prepare_strategy", _prepared_strategy)
    monkeypatch.setattr(DataService, "load", load)
    monkeypatch.setattr(optimization, "_run_single_combo", evaluate)
    monkeypatch.setattr(optimization, "estimate_n_jobs", lambda *args, **kwargs: 1)
    previous_handler = signal.getsignal(signal.SIGALRM)
    _run_optimization("fixture")

    assert signal.getsignal(signal.SIGALRM) == previous_handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    job = store.get_job("fixture")
    assert job is not None
    assert job["status"] == ("estimation_timeout" if failure == "timeout" else "failed")
    assert store.list_runs() == []
    run_dir = store.workspace.runs / "fixture"
    assert (run_dir / "manifest.json").is_file()
    assert not (run_dir / "result.json").exists()
    assert not (run_dir / "performance.json").exists()
    assert not (run_dir / "observations.json").exists()
    assert not list(run_dir.glob("*.tmp"))


@pytest.mark.parametrize("phase", ["constructor", "generate_signals"])
def test_real_trial_alarm_during_contract_execution_retains_timeout_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """Use the real trial/contract/constructor path and a real one-second alarm."""
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

        # The fake class and data replace external inputs. The worker trial,
        # captured constructor inputs, contract fixture, alarm and SQLite are real.
        monkeypatch.setattr(BacktestService, "prepare_strategy", _prepared_strategy)
        monkeypatch.setattr(DataService, "load", load)
        monkeypatch.setattr(StrategyLoader, "load_class", lambda *args: SlowStrategy)
        try:
            _run_optimization("fixture")
            assert entered == [phase]
            assert signal.getsignal(signal.SIGALRM) == previous_handler
            assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
            job = store.get_job("fixture")
            assert job is not None
            assert job["status"] == "estimation_timeout", job
            assert "exceeded 1s timeout" in job["error"]
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
