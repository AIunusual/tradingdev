"""Generated parameter experiments preserve revisions and gate actual settings."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
import yaml
from tests.app.test_generated_strategy_job import FakeRunner
from tests.research_fixtures import research_for_config, with_fixture_market_source

from tradingdev.adapters.execution.strategy_factory import StrategyFactory
from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.optimization_trials import OptimizationLedger
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService, LoadedDataset
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.backtest.schemas import BacktestConfig, ParallelConfig
from tradingdev.domain.execution import (
    ExecutionManifest,
    ManifestError,
    OptimizationSpec,
)
from tradingdev.domain.performance.artifacts import bundles_from_pipeline
from tradingdev.domain.research_scope import optimization_backtest
from tradingdev.mcp.workers import optimization

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
    from tradingdev.domain.strategies.base import BaseStrategy
    from tradingdev.domain.strategies.execution import StrategyExecution

_CODE = """\
from typing import Any
import pandas as pd
from tradingdev.domain.strategies.base import BaseStrategy

class ExperimentStrategy(BaseStrategy):
    def __init__(
        self,
        settings: dict[str, int],
        failure: str = "none",
        scale: int = 2,
    ) -> None:
        if failure == "constructor":
            raise ValueError("selected constructor rejected")
        self.settings = settings
        self.failure = failure
        self.scale = scale

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.failure == "mutation":
            df["close"] = 0.0
        result = df.copy()
        result["signal"] = self.settings["direction"]
        if self.failure == "long_fixture" and len(df) >= 240:
            result["signal"] = 2
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"settings": self.settings, "failure": self.failure, "scale": self.scale}
"""


@dataclass
class Experiment:
    workspace: WorkspacePaths
    strategies: StrategyService
    loader: StrategyFactory
    data: DataService
    service: BacktestService
    jobs: JobService
    store: JobStore
    runner: FakeRunner
    config: dict[str, Any]
    revision_id: str
    files: dict[Path, bytes]
    loads: list[dict[str, Any]]


@pytest.fixture
def experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Experiment:
    workspace = WorkspacePaths(tmp_path / "workspace")
    monkeypatch.setenv("TRADINGDEV_WORKSPACE", str(workspace.root))
    strategies = StrategyService(workspace)
    monkeypatch.setattr(strategies, "_quality_gate_diagnostics", lambda _path: [])
    config = {
        "strategy": {
            "id": "experiment",
            "class_name": "ExperimentStrategy",
            "parameters": {"settings": {"direction": 1, "offset": 5}},
        },
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-05",
            "init_cash": 10000.0,
            "fees": 0,
            "slippage": 0,
            "periods_per_year": 365,
        },
        "validation": {"mode": "automatic", "n_splits": 2, "train_ratio": 0.5},
        "random_seed": 42,
    }
    code = _CODE
    if getattr(request, "param", None) == "fitted":
        code = code.replace(
            "    def generate_signals",
            "    def fit(self, df: pd.DataFrame) -> None:\n"
            '        self.settings["offset"] += 1\n\n'
            "    def generate_signals",
        )
    saved = strategies.save_draft(
        code,
        with_fixture_market_source(yaml.safe_dump(config)),
        research_for_config(config),
        strategy_id="experiment",
    )
    assert saved.success and saved.revision_id and saved.config_path
    checked = strategies.validate("experiment", saved.revision_id)
    assert checked["success"], checked["diagnostics"]
    assert strategies.dry_run("experiment", saved.revision_id)["success"]
    config = yaml.safe_load(Path(saved.config_path).read_text())
    files = {
        path: path.read_bytes()
        for path in workspace.generated_strategies.rglob("*")
        if path.is_file()
    }
    loader = StrategyFactory(workspace_root=workspace.root)
    data = DataService(workspace)
    loads: list[dict[str, Any]] = []
    prices = pd.Series([100.0 + i for i in range(80)])
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC"),
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": 100.0,
        }
    )

    def load(raw: dict[str, Any], _backtest: BacktestConfig) -> LoadedDataset:
        loads.append(deepcopy(raw))
        return LoadedDataset(
            frame=frame.copy(deep=True),
            processed_path=tmp_path / "fixture.parquet",
            dataset_id="experiment-fixture",
        )

    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "data_available", lambda *_args, **_kwargs: True)
    service = BacktestService(
        data_service=data, strategy_loader=loader, strategy_gate=strategies
    )
    store = JobStore(workspace=workspace)
    runner = FakeRunner()
    jobs = JobService(
        strategy_service=strategies,
        data_service=data,
        strategy_loader=loader,
        job_store=store,
        process_runner=runner,
    )
    return Experiment(
        workspace,
        strategies,
        loader,
        data,
        service,
        jobs,
        store,
        runner,
        config,
        saved.revision_id,
        files,
        loads,
    )


def _assert_revision_unchanged(experiment: Experiment) -> None:
    assert {
        path: path.read_bytes()
        for path in experiment.workspace.generated_strategies.rglob("*")
        if path.is_file()
    } == experiment.files


def test_two_experiments_share_revision_and_freeze_nested_parameters(
    experiment: Experiment,
) -> None:
    responses = [
        experiment.jobs.start_walk_forward(
            strategy_id="experiment",
            revision_id=experiment.revision_id,
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-05",
            parameters={"settings": {"direction": direction}, "scale": 3},
        )
        for direction in (1, -1)
    ]
    assert all(response["job_id"] for response in responses), responses
    assert responses[0]["job_id"] != responses[1]["job_id"]
    manifests = [
        ExecutionManifestStore(experiment.workspace).load(
            response["job_id"], expected_hash=response["manifest_hash"]
        )
        for response in responses
    ]
    for direction, manifest in zip((1, -1), manifests, strict=True):
        assert (
            manifest.config_copy()["strategy"]["revision_id"] == experiment.revision_id
        )
        expected = {
            "settings": {"direction": direction, "offset": 5},
            "failure": "none",
            "scale": 3,
        }
        assert manifest.config_copy()["strategy"]["parameters"] == expected
        assert manifest.strategy_execution.constructor_kwargs == expected
        run = experiment.service.run_manifest(manifest)
        assert run.pipeline.execution_manifest is manifest
        assert len(run.pipeline.fold_results) == 2
        manifest.verify()
    assert len(experiment.runner.calls) == 2
    _assert_revision_unchanged(experiment)


@pytest.mark.parametrize("walk_forward", [False, True])
def test_cli_effective_parameters_execute_without_saving_revision(
    experiment: Experiment,
    tmp_path: Path,
    walk_forward: bool,
) -> None:
    config = deepcopy(experiment.config)
    config["strategy"]["parameters"]["settings"]["direction"] = -1
    if not walk_forward:
        config.pop("validation")
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    run = experiment.service.run_config(path, walk_forward=walk_forward)
    assert (
        run.pipeline.config_snapshot["strategy"]["parameters"]["settings"]["direction"]
        == -1
    )
    assert run.pipeline.execution_manifest is not None
    assert run.pipeline.execution_manifest.strategy_execution.constructor_kwargs[
        "settings"
    ] == {"direction": -1, "offset": 5}
    _assert_revision_unchanged(experiment)


@pytest.mark.parametrize(
    "failure, reason",
    [
        ("constructor", "selected constructor rejected"),
        ("mutation", "input_mutated"),
        ("long_fixture", "invalid_signal_values"),
    ],
)
@pytest.mark.parametrize("walk_forward", [False, True])
def test_cli_rejects_experiment_contract_failures_before_loading_data(
    experiment: Experiment,
    failure: str,
    reason: str,
    walk_forward: bool,
) -> None:
    config = deepcopy(experiment.config)
    config["strategy"]["parameters"]["failure"] = failure
    if not walk_forward:
        config.pop("validation")
    with pytest.raises(ManifestError, match=reason):
        experiment.service.run_raw_config(config, walk_forward=walk_forward)
    assert not experiment.loads
    _assert_revision_unchanged(experiment)


@pytest.mark.parametrize(
    "failure, reason",
    [
        ("constructor", "selected constructor rejected"),
        ("mutation", "input_mutated"),
        ("long_fixture", "invalid_signal_values"),
    ],
)
def test_submission_rejects_bad_parameters_without_job_or_revision(
    experiment: Experiment,
    failure: str,
    reason: str,
) -> None:
    response = experiment.jobs.start_walk_forward(
        strategy_id="experiment",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-05",
        parameters={"failure": failure},
    )
    assert response["code"] == "invalid_execution_request", response
    assert reason in response["message"]
    assert not experiment.jobs.list_jobs()
    assert not experiment.runner.calls
    assert not list(experiment.workspace.runs.iterdir())
    _assert_revision_unchanged(experiment)


@pytest.mark.parametrize(
    "parameters",
    [
        {"unknown": 1},
        {"settings": {"unknown": 1}},
        {"scale": float("nan")},
        {"scale": {1: 2}},
    ],
)
def test_submission_rejects_unknown_or_non_json_parameters(
    experiment: Experiment,
    parameters: dict[str, Any],
) -> None:
    response = experiment.jobs.start_walk_forward(
        strategy_id="experiment",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-05",
        parameters=parameters,
    )
    assert response["code"] == "invalid_execution_request", response
    assert not experiment.jobs.list_jobs()
    assert not experiment.runner.calls
    _assert_revision_unchanged(experiment)


def test_worker_rechecks_captured_parameters_even_for_direct_manifest(
    experiment: Experiment,
) -> None:
    good = experiment.service.prepare_execution(experiment.config, kind="walk_forward")
    config = good.config_copy()
    config["strategy"]["parameters"]["failure"] = "long_fixture"
    bad = ExecutionManifest.create(
        kind="walk_forward",
        config=config,
        strategy_execution=experiment.loader.resolve_execution(config["strategy"]),
    )
    with pytest.raises(ManifestError, match="invalid_signal_values"):
        experiment.service.run_manifest(bad)
    assert not experiment.loads
    _assert_revision_unchanged(experiment)


def test_manifest_uses_frozen_defaults_during_check_and_actual_execution(
    experiment: Experiment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = experiment.service.prepare_execution(
        experiment.config, kind="walk_forward"
    )
    actual: list[int] = []
    monkeypatch.setenv(
        "TRADINGDEV_STRATEGY_IMAGE", "missing-image-must-not-be-resolved"
    )
    monkeypatch.setenv("TRADINGDEV_RUNTIME_LIMITS", '{"max_seconds": 0.01}')
    factory = StrategyFactory(workspace_root=experiment.workspace.root)
    original = factory.open_from_execution

    @contextmanager
    def observe(
        config: dict[str, Any], execution: StrategyExecution, *args: Any, **kwargs: Any
    ) -> Iterator[BaseStrategy]:
        scale = execution.constructor_kwargs["scale"]
        assert isinstance(scale, int)
        actual.append(scale)
        with original(config, execution, *args, **kwargs) as strategy:
            yield strategy

    monkeypatch.setattr(factory, "open_from_execution", observe)
    monkeypatch.setattr(experiment.service, "_strategy_loader", factory)
    run = experiment.service.run_manifest(manifest)
    assert actual == [2, 2, 2]
    assert all(fold.strategy_params["scale"] == 2 for fold in run.pipeline.fold_results)
    _assert_revision_unchanged(experiment)


@pytest.mark.parametrize(
    "failure, reason",
    [
        ("constructor", "selected constructor rejected"),
        ("mutation", "input_mutated"),
        ("long_fixture", "invalid_signal_values"),
    ],
)
def test_optimization_records_candidate_failures_from_synthetic_checks(
    experiment: Experiment,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    reason: str,
) -> None:
    config = deepcopy(experiment.config)
    config.pop("validation")
    search = OptimizationSpec.model_validate(
        {
            "param_ranges": {"failure": [failure]},
            "optimization_metric": "total_return",
            "train_start": "2024-01-01",
            "train_end": "2024-01-02",
            "test_start": "2024-01-03",
            "test_end": "2024-01-04",
        }
    )
    manifest = experiment.service.prepare_execution(
        config, kind="optimization", optimization=search
    )
    job_id = "candidate-check"
    dataset_id = "experiment-fixture"
    experiment.store.create_job(
        job_id=job_id, manifest=manifest, extra_payload={"dataset_id": dataset_id}
    )
    ledger = OptimizationLedger(experiment.store.store, job_id, manifest, dataset_id)
    ledger.initialize()

    open_strategy = StrategyFactory.open_from_execution

    @contextmanager
    def open_for_synthetic_check(
        factory: StrategyFactory,
        config: dict[str, Any],
        execution: StrategyExecution,
        engine: BaseBacktestEngine | None,
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[BaseStrategy]:
        assert engine is None, "Candidate reached historical execution"
        with open_strategy(
            factory, config, execution, engine, *args, **kwargs
        ) as strategy:
            yield strategy

    monkeypatch.setattr(
        StrategyFactory, "open_from_execution", open_for_synthetic_check
    )
    runner = optimization.TrialRunner(
        ledger=ledger,
        strategy_config=manifest.config_copy()["strategy"],
        execution=manifest.strategy_execution,
        config=optimization_backtest(manifest, "train"),
        data=pd.DataFrame(),
        parallel=ParallelConfig(),
        seed=42,
    )

    assert runner.run(0) is None

    snapshot = ledger.snapshot()
    assert snapshot is not None
    (trial,) = snapshot.trials
    assert trial.state == "failed" and trial.reason and reason in trial.reason
    assert trial.evidence_hash is None
    _assert_revision_unchanged(experiment)


@pytest.mark.parametrize("experiment", ["fitted"], indirect=True)
def test_walk_forward_keeps_each_fitted_parameter_snapshot(
    experiment: Experiment,
) -> None:
    run = experiment.service.run_raw_config(experiment.config, walk_forward=True)
    artifacts = bundles_from_pipeline("fitted", run.pipeline, run.metrics)
    for index, expected_offset in enumerate((6, 7)):
        for split in ("train", "test"):
            scope = artifacts.performance.scopes[f"fold/{index}/{split}"]
            assert scope.parameters["settings"] == {"direction": 1, "offset": 5}
            assert scope.metadata["strategy_parameters"] == {
                "settings": {"direction": 1, "offset": expected_offset},
                "failure": "none",
                "scale": 2,
            }
    _assert_revision_unchanged(experiment)
