"""Backtest service tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest
from pydantic import create_model

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService, LoadedDataset
from tradingdev.app.strategy_service import (
    StrategyNotExecutableError,
    StrategyService,
)
from tradingdev.domain.backtest.schemas import BacktestConfig, ParallelConfig
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.bundled.kd_strategy.config import KDStrategyConfig
from tradingdev.domain.strategies.bundled.kd_strategy.strategy import KDStrategy
from tradingdev.domain.strategies.bundled.quantile_strategy.config import (
    QuantileStrategyConfig,
)
from tradingdev.domain.strategies.bundled.quantile_strategy.strategy import (
    QuantileStrategy,
)
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.strategies.schemas import StrategySpec, StrategyStatus
from tradingdev.shared.utils.config import load_config

if TYPE_CHECKING:
    from pytest import MonkeyPatch

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine


def _frame(rows: int = 24) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=rows,
                freq="h",
                tz="UTC",
            ),
            "close": [100.0 + i for i in range(rows)],
        }
    )


def _raw_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "strategy": {
            "id": "fixture",
            "parameters": {},
        },
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "init_cash": 10_000.0,
            "fees": 0.0,
            "slippage": 0.0,
        },
    }
    config.update(overrides)
    return config


class _DataServiceStub(DataService):
    def __init__(self, dataset: LoadedDataset) -> None:
        super().__init__(WorkspacePaths(dataset.processed_path.parent / "workspace"))
        self.dataset = dataset
        self.loads: list[BacktestConfig] = []

    def load(
        self,
        raw_config: dict[str, Any],
        backtest_config: BacktestConfig,
    ) -> LoadedDataset:
        self.loads.append(backtest_config)
        return self.dataset


class _SignalStrategy(BaseStrategy):
    def __init__(self) -> None:
        self.fit_lengths: list[int] = []

    def fit(self, df: pd.DataFrame) -> None:
        self.fit_lengths.append(len(df))

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = [1 if i % 2 == 0 else -1 for i in range(len(result))]
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"name": "signal_stub"}


class _StrategyLoaderStub:
    def __init__(self, strategy: BaseStrategy) -> None:
        self.strategy = strategy
        self.engine: BaseBacktestEngine | None = None
        self.parallel_config: ParallelConfig | None = None

    def resolve_execution(self, strategy_cfg: dict[str, Any]) -> StrategyExecution:
        return StrategyExecution(kind="generated", constructor_kwargs={})

    def create_from_execution(
        self,
        strategy_cfg: dict[str, Any],
        execution: StrategyExecution,
        engine: BaseBacktestEngine,
        parallel_config: ParallelConfig | None = None,
    ) -> BaseStrategy:
        self.engine = engine
        self.parallel_config = parallel_config
        return self.strategy


class _GateStub:
    def __init__(self, source_path: str) -> None:
        self.resolved: list[str] = []
        self.source_path = source_path

    def resolve_executable(
        self, strategy_id: str, revision_id: str | None = None
    ) -> StrategySpec:
        self.resolved.append(strategy_id)
        return StrategySpec(
            strategy_id=strategy_id,
            class_name="Fixture",
            kind="bundled",
            source_path=self.source_path,
            config_path="",
            status=StrategyStatus.RUNNABLE,
        )


def _service(
    tmp_path: Path,
    *,
    rows: int = 24,
) -> tuple[BacktestService, _DataServiceStub, _StrategyLoaderStub, _SignalStrategy]:
    dataset = LoadedDataset(
        frame=_frame(rows),
        processed_path=tmp_path / "processed.parquet",
        dataset_id="dataset-fixture",
    )
    data_service = _DataServiceStub(dataset)
    strategy = _SignalStrategy()
    strategy_loader = _StrategyLoaderStub(strategy)
    source_path = tmp_path / "fixture.py"
    source_path.write_text("# fixture strategy\n", encoding="utf-8")
    service = BacktestService(
        data_service=cast("DataService", data_service),
        strategy_loader=cast("StrategyLoader", strategy_loader),
        strategy_gate=_GateStub(str(source_path)),
    )
    return service, data_service, strategy_loader, strategy


def test_run_raw_config_simple_backtest_serializes_metrics(tmp_path: Path) -> None:
    service, data_service, strategy_loader, _strategy = _service(tmp_path)

    run = service.run_raw_config(_raw_config())

    assert run.mode == "simple"
    assert run.pipeline.mode == "simple"
    assert run.pipeline.backtest_result is not None
    assert run.processed_path == tmp_path / "processed.parquet"
    assert run.dataset_id == "dataset-fixture"
    assert data_service.loads[0].symbol == "BTC/USDT"
    assert strategy_loader.engine is not None
    assert strategy_loader.parallel_config == ParallelConfig()
    assert "total_return" in run.metrics
    assert "daily_pnl_mean" in run.metrics
    assert "total_volume" in run.metrics
    assert run.metrics == run.pipeline.backtest_result.metrics
    manifest = run.pipeline.execution_manifest
    assert manifest is not None
    assert manifest.config_copy() == run.pipeline.config_snapshot
    manifest.verify()


def test_serialization_retains_metrics_outside_the_summary(tmp_path: Path) -> None:
    service, _data, _loader, _strategy = _service(tmp_path)
    metrics = {
        "daily_pnl_mean": 12.5,
        "total_volume": 1500.0,
        "monthly_volume_mean": 750.0,
        "future_metric": {"value": 1.25, "unavailable": float("nan")},
    }

    assert service.serialize_metrics(metrics) == {
        "daily_pnl_mean": 12.5,
        "total_volume": 1500.0,
        "monthly_volume_mean": 750.0,
        "future_metric": {"value": 1.25, "unavailable": None},
    }


def test_run_raw_config_walk_forward_uses_validation_section(tmp_path: Path) -> None:
    service, _data_service, _strategy_loader, strategy = _service(tmp_path, rows=40)
    raw_config = _raw_config(
        validation={
            "n_splits": 2,
            "train_ratio": 0.5,
            "target_metric": "total_return",
        }
    )

    run = service.run_raw_config(raw_config, walk_forward=True)

    assert run.mode == "walk_forward"
    assert run.pipeline.mode == "walk_forward"
    assert len(run.pipeline.fold_results) == 2
    assert run.metrics["n_folds"] == 2
    assert strategy.fit_lengths == [10, 10]


def _gate_service(tmp_path: Path) -> StrategyService:
    return StrategyService(WorkspacePaths(tmp_path / "workspace"))


def _stub_backtest_service(
    tmp_path: Path,
    gate: StrategyService | _GateStub,
) -> BacktestService:
    dataset = LoadedDataset(
        frame=_frame(),
        processed_path=tmp_path / "processed.parquet",
        dataset_id="dataset-fixture",
    )
    return BacktestService(
        data_service=cast("DataService", _DataServiceStub(dataset)),
        strategy_loader=cast("StrategyLoader", _StrategyLoaderStub(_SignalStrategy())),
        strategy_gate=gate,
    )


def test_run_raw_config_rejects_unknown_strategy(tmp_path: Path) -> None:
    service = _stub_backtest_service(tmp_path, _gate_service(tmp_path))

    with pytest.raises(StrategyNotExecutableError, match="not found"):
        service.run_raw_config(_raw_config())


def test_run_raw_config_rejects_draft_generated_strategy(tmp_path: Path) -> None:
    strategy_service = _gate_service(tmp_path)
    saved = strategy_service.save_draft(
        "fixture",
        "class Fixture:\n    pass\n",
        "strategy:\n  class_name: Fixture\n",
    )
    assert saved.success is True
    service = _stub_backtest_service(tmp_path, strategy_service)

    with pytest.raises(StrategyNotExecutableError, match="runnable or promoted"):
        service.run_raw_config(_raw_config())


def test_run_raw_config_allows_promoted_bundled_strategy(tmp_path: Path) -> None:
    service = _stub_backtest_service(tmp_path, _gate_service(tmp_path))
    raw_config = _raw_config()
    raw_config["strategy"] = {"id": "kd_crossover", "parameters": {}}

    run = service.run_raw_config(raw_config)

    assert run.mode == "simple"


@pytest.mark.parametrize("walk_forward", [False, True])
def test_run_manifest_uses_captured_defaults_in_real_strategy(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    sample_ohlcv_with_kd: pd.DataFrame,
    walk_forward: bool,
) -> None:
    dataset = LoadedDataset(
        frame=sample_ohlcv_with_kd,
        processed_path=tmp_path / "processed.parquet",
        dataset_id="captured-kd",
    )
    loader = StrategyLoader(workspace_root=tmp_path / "workspace")
    service = BacktestService(
        data_service=_DataServiceStub(dataset),
        strategy_loader=loader,
        strategy_gate=_gate_service(tmp_path),
    )
    config = _raw_config(strategy={"id": "kd_crossover", "parameters": {}})
    if walk_forward:
        config["validation"] = {"n_splits": 2, "train_ratio": 0.5}
    manifest = service.prepare_execution(
        config, kind="walk_forward" if walk_forward else "backtest"
    )
    assert manifest.config_copy()["strategy"]["parameters"] == {}

    changed_model = create_model(
        "ChangedKDConfig", k_period=(int, 21), __base__=KDStrategyConfig
    )
    monkeypatch.setattr(loader, "_bundled_config_model", lambda _cls: changed_model)
    observed: list[Any] = []
    original_generate = KDStrategy.generate_signals

    def observe(strategy: KDStrategy, df: pd.DataFrame) -> pd.DataFrame:
        observed.append(strategy.get_parameters()["k_period"])
        return original_generate(strategy, df)

    monkeypatch.setattr(KDStrategy, "generate_signals", observe)

    run = service.run_manifest(manifest)

    assert observed == ([14, 14, 14, 14] if walk_forward else [14])
    assert run.pipeline.execution_manifest is manifest
    manifest.verify()


def test_run_raw_config_rejects_source_path_mismatch(tmp_path: Path) -> None:
    gate = _GateStub(source_path=str(tmp_path / "registered.py"))
    service = _stub_backtest_service(tmp_path, gate)
    raw_config = _raw_config()
    raw_config["strategy"]["source_path"] = str(tmp_path / "other.py")

    with pytest.raises(StrategyNotExecutableError, match="does not match"):
        service.run_raw_config(raw_config)


def test_run_config_rejects_walk_forward_config_without_flag(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "walk_forward.yaml"
    config_path.write_text(
        """\
strategy:
  id: fixture
backtest:
  symbol: BTC/USDT
  timeframe: 1h
  start_date: "2024-01-01"
  end_date: "2024-01-02"
  init_cash: 10000.0
validation:
  n_splits: 1
""",
        encoding="utf-8",
    )
    service, _data_service, _strategy_loader, _strategy = _service(tmp_path)

    with pytest.raises(ValueError, match="prepare_walk_forward"):
        service.run_config(config_path)


def test_prepare_execution_freezes_defaults_without_mutating_input(
    tmp_path: Path,
) -> None:
    service, _data, _loader, _strategy = _service(tmp_path)
    config = _raw_config()
    original = deepcopy(config)

    manifest = service.prepare_execution(config, kind="backtest")
    frozen = manifest.config_copy()

    assert config == original
    assert frozen["backtest"]["re_entry_after_sl"] is True
    assert frozen["parallel"] == ParallelConfig().model_dump(mode="json")
    assert frozen["data"]["requirements"]["market"] == {
        "source": "binance_vision",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
    }
    assert frozen["data"]["processed_dir"] == str(
        tmp_path / "workspace" / "data" / "processed"
    )
    assert len(frozen["strategy"]["source_hash"]) == 64

    config["backtest"]["fees"] = 0.9
    config["strategy"]["parameters"]["new_parameter"] = 42
    run = service.run_manifest(manifest)
    assert run.pipeline.config_snapshot["backtest"]["fees"] == 0
    assert run.pipeline.config_snapshot["strategy"]["parameters"] == {}
    assert run.pipeline.execution_manifest == manifest


def test_manifest_rejects_source_change_before_loading_data(tmp_path: Path) -> None:
    service, data, _loader, _strategy = _service(tmp_path)
    manifest = service.prepare_execution(_raw_config(), kind="backtest")
    (tmp_path / "fixture.py").write_text("# changed\n", encoding="utf-8")

    with pytest.raises(StrategyNotExecutableError, match="source hash"):
        service.run_manifest(manifest)

    assert data.loads == []


def test_walk_forward_manifest_fixes_mode_and_validation_defaults(
    tmp_path: Path,
) -> None:
    service, _data, _loader, _strategy = _service(tmp_path, rows=40)
    config = _raw_config(validation={"n_splits": 2})
    manifest = service.prepare_execution(config, kind="walk_forward")
    config.pop("validation")

    run = service.run_manifest(manifest)

    assert run.mode == "walk_forward"
    assert len(run.pipeline.fold_results) == 2
    assert run.pipeline.config_snapshot["validation"]["train_ratio"] == 0.8
    assert run.pipeline.config_snapshot["validation"]["expanding"] is False


def test_run_config_reads_yaml_once_and_keeps_the_resolved_snapshot(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    service, _data, _loader, _strategy = _service(tmp_path)
    config_path = tmp_path / "run.yaml"
    reads: list[Path] = []

    def read_once(path: Path) -> dict[str, Any]:
        reads.append(path)
        assert len(reads) == 1
        return _raw_config()

    monkeypatch.setattr("tradingdev.app.backtest_service.load_config", read_once)
    run = service.run_config(config_path)

    assert reads == [config_path]
    assert run.pipeline.execution_manifest is not None
    assert run.pipeline.config_snapshot == (
        run.pipeline.execution_manifest.config_copy()
    )


def test_prepared_manifest_resolves_data_locations_before_environment_changes(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRADINGDEV_DATA_ROOT", str(tmp_path / "initial-data"))
    service, _data, _loader, _strategy = _service(tmp_path)
    config = _raw_config(
        data={
            "raw_dir": "relative/raw",
            "processed_dir": "relative/processed",
            "requirements": {
                "market": {"symbol": "BTC/USDT", "timeframe": "1h"},
                "features": [
                    {"type": "dvol", "source": "deribit", "column": "dvol"},
                    {
                        "type": "custom",
                        "source": "local",
                        "column": "sentiment",
                        "path": "features/sentiment.parquet",
                    },
                ],
            },
        }
    )
    manifest = service.prepare_execution(config, kind="backtest")
    monkeypatch.setenv("TRADINGDEV_DATA_ROOT", str(tmp_path / "later-data"))
    monkeypatch.chdir(tmp_path.parent)

    run = service.run_manifest(manifest)

    data = run.pipeline.config_snapshot["data"]
    assert data["raw_dir"] == str(tmp_path / "relative" / "raw")
    assert data["processed_dir"] == str(tmp_path / "relative" / "processed")
    dvol, custom = data["requirements"]["features"]
    assert dvol["path"] == str(
        tmp_path / "initial-data" / "processed" / "btc_dvol_1h_2024_2024.parquet"
    )
    assert dvol["raw_path"] == str(
        tmp_path / "initial-data" / "processed" / "btc_dvol_1h_2024_2024.csv"
    )
    assert custom["path"] == str(tmp_path / "features" / "sentiment.parquet")


@pytest.mark.parametrize("use_bundled_yaml", [False, True])
def test_bundled_manifest_source_is_independent_of_working_directory(
    tmp_path: Path, monkeypatch: MonkeyPatch, use_bundled_yaml: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    strategies = _gate_service(tmp_path)
    spec = strategies.resolve_executable("kd_crossover")
    config = load_config(Path(spec.config_path)) if use_bundled_yaml else _raw_config()
    if use_bundled_yaml:
        shadow = tmp_path / config["strategy"]["source_path"]
        shadow.parent.mkdir(parents=True)
        shadow.write_text("# unrelated file in the caller's directory\n")
    else:
        config["strategy"] = {"id": "kd_crossover"}
    service = _stub_backtest_service(tmp_path, strategies)

    manifest = service.prepare_execution(config, kind="backtest")
    source = Path(manifest.config_copy()["strategy"]["source_path"])
    assert source.is_absolute() and source.is_file()
    assert (
        source.read_text(encoding="utf-8")
        == strategies.get_strategy("kd_crossover")["source_code"]
    )
    monkeypatch.chdir(tmp_path.parent)
    assert service.run_manifest(manifest).pipeline.execution_manifest == manifest


def test_simple_backtest_refuses_an_untrained_quantile_strategy(
    tmp_path: Path,
) -> None:
    """The simple path must surface the refusal, not leaky metrics (issue #24 C).

    Bundled strategies skip the generated-code contract probe, so the refusal
    reaches the caller from ``generate_signals`` itself.
    """

    class _BundledLoader(_StrategyLoaderStub):
        def resolve_execution(self, strategy_cfg: dict[str, Any]) -> StrategyExecution:
            return StrategyExecution(kind="bundled", constructor_kwargs={})

    rows = 80
    closes = [100.0 + (index % 7) for index in range(rows)]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "open": closes,
            "high": [value + 1.0 for value in closes],
            "low": [value - 1.0 for value in closes],
            "close": closes,
            "volume": [1000.0] * rows,
        }
    )
    dataset = LoadedDataset(
        frame=frame,
        processed_path=tmp_path / "processed.parquet",
        dataset_id="dataset-fixture",
    )
    strategy = QuantileStrategy(config=QuantileStrategyConfig(horizon=3))
    source_path = tmp_path / "fixture.py"
    source_path.write_text("# fixture strategy\n", encoding="utf-8")
    service = BacktestService(
        data_service=cast("DataService", _DataServiceStub(dataset)),
        strategy_loader=cast("StrategyLoader", _BundledLoader(strategy)),
        strategy_gate=_GateStub(str(source_path)),
    )

    with pytest.raises(RuntimeError, match="no trained regime model"):
        service.run_raw_config(_raw_config())
