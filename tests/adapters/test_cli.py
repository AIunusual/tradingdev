"""CLI adapter tests."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from tradingdev.adapters.cli import main as cli_main
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.artifact_service import ArtifactService
from tradingdev.app.backtest_service import BacktestRun
from tradingdev.app.data_service import DataService, LoadedDataset
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.result import BacktestResult
from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine
from tradingdev.domain.strategies.bundled.kd_strategy.strategy import KDStrategy

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


class _LoggerStub:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args: Any) -> None:
        self.messages.append(message % args if args else message)


class _BacktestServiceStub:
    def __init__(self, run: BacktestRun) -> None:
        self.run = run
        self.calls: list[tuple[Path, bool]] = []

    def run_config(
        self, config_path: Path, *, walk_forward: bool = False
    ) -> BacktestRun:
        self.calls.append((config_path, walk_forward))
        return self.run


class _ArtifactServiceStub:
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.calls: list[dict[str, Any]] = []

    def cache_pipeline_result(
        self,
        *,
        pipeline: PipelineResult,
        config_path: Path,
        processed_path: Path,
        metrics: dict[str, Any],
        strategy_id: str,
    ) -> Path:
        self.calls.append(
            {
                "pipeline": pipeline,
                "config_path": config_path,
                "processed_path": processed_path,
                "metrics": metrics,
                "strategy_id": strategy_id,
            }
        )
        return self.cache_path


def _metrics() -> dict[str, Any]:
    return {
        "total_pnl": 100.0,
        "total_return": 0.01,
        "annual_return": 0.12,
        "max_drawdown": -0.02,
        "sharpe_ratio": 1.5,
        "win_rate": 0.6,
        "profit_factor": 1.2,
        "total_trades": 4,
        "total_volume": 1_000.0,
        "n_days": 2,
        "n_months": 1,
    }


def _simple_run(tmp_path: Path) -> BacktestRun:
    result = BacktestResult(
        metrics=_metrics(),
        equity_curve=np.array([10_000.0, 10_100.0]),
        mode="signal",
    )
    return BacktestRun(
        mode="simple",
        pipeline=PipelineResult(
            mode="simple",
            backtest_result=result,
            config_snapshot={"strategy": {"id": "cli_fixture"}},
        ),
        metrics=result.metrics,
        processed_path=tmp_path / "processed.parquet",
        dataset_id="dataset-cli",
    )


def test_cli_runs_backtest_and_caches_pipeline(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """\
strategy:
  id: cli_fixture
backtest:
  symbol: BTC/USDT
  timeframe: 1h
  start_date: "2024-01-01"
  end_date: "2024-01-02"
  init_cash: 10000.0
""",
        encoding="utf-8",
    )
    service = _BacktestServiceStub(_simple_run(tmp_path))
    artifacts = _ArtifactServiceStub(tmp_path / "cached.pkl")
    logger = _LoggerStub()
    monkeypatch.setattr(cli_main, "BacktestService", lambda: service)
    monkeypatch.setattr(cli_main, "ArtifactService", lambda: artifacts)
    monkeypatch.setattr(cli_main, "logger", logger)
    monkeypatch.setattr("sys.argv", ["tradingdev", "--config", str(config_path)])

    cli_main.main()

    assert service.calls == [(config_path, False)]
    assert artifacts.calls == [
        {
            "pipeline": service.run.pipeline,
            "config_path": config_path,
            "processed_path": tmp_path / "processed.parquet",
            "metrics": service.run.metrics,
            "strategy_id": "cli_fixture",
        }
    ]
    assert any("Backtest Performance Report" in msg for msg in logger.messages)
    assert any("Result cached" in msg for msg in logger.messages)


def test_cli_forwards_walk_forward_flag(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """\
strategy:
  id: cli_walk_forward
backtest:
  symbol: BTC/USDT
  timeframe: 1h
  start_date: "2024-01-01"
  end_date: "2024-01-02"
  init_cash: 10000.0
""",
        encoding="utf-8",
    )
    run = BacktestRun(
        mode="walk_forward",
        pipeline=PipelineResult(
            mode="walk_forward",
            fold_results=[],
            config_snapshot={"strategy": {"id": "cli_walk_forward"}},
        ),
        metrics={"n_folds": 0},
        processed_path=tmp_path / "processed.parquet",
        dataset_id="dataset-cli",
    )
    service = _BacktestServiceStub(run)
    artifacts = _ArtifactServiceStub(tmp_path / "cached.pkl")
    logger = _LoggerStub()
    monkeypatch.setattr(cli_main, "BacktestService", lambda: service)
    monkeypatch.setattr(cli_main, "ArtifactService", lambda: artifacts)
    monkeypatch.setattr(cli_main, "format_walk_forward_report", lambda _folds: "WF")
    monkeypatch.setattr(cli_main, "logger", logger)
    monkeypatch.setattr(
        "sys.argv",
        ["tradingdev", "--config", str(config_path), "--walk-forward"],
    )

    cli_main.main()

    assert service.calls == [(config_path, True)]
    assert artifacts.calls[0]["strategy_id"] == "cli_walk_forward"
    assert any("WF" in msg for msg in logger.messages)


@pytest.mark.parametrize(
    ("unavailable_metric", "raw_value", "report_label"),
    [
        ("profit_factor", float("inf"), "Profit Factor"),
        ("sharpe_ratio", float("nan"), "Sharpe Ratio"),
        ("annual_return", float("-inf"), "Annual Return"),
    ],
    ids=["infinite-profit-factor", "nan-sharpe-ratio", "negative-infinite-return"],
)
def test_cli_reports_and_caches_serialized_nonfinite_metrics(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    unavailable_metric: str,
    raw_value: float,
    report_label: str,
) -> None:
    """Use engine fixtures through real serialization, reporting, and storage."""
    workspace = WorkspacePaths(tmp_path / "workspace")
    monkeypatch.setenv("TRADINGDEV_WORKSPACE", str(workspace.root))
    monkeypatch.setenv("TRADINGDEV_DATA_ROOT", str(workspace.root / "data"))
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
            "close": [100.0, 101.0],
        }
    )
    processed_path = tmp_path / "processed.parquet"
    frame.to_parquet(processed_path)
    dataset = LoadedDataset(frame, processed_path, "dataset-cli-nonfinite")
    monkeypatch.setattr(DataService, "load", lambda *_args: dataset)
    monkeypatch.setattr(
        KDStrategy,
        "generate_signals",
        lambda _self, data: data.assign(signal=1),
    )
    metrics = _metrics()
    metrics[unavailable_metric] = raw_value
    engine_result = BacktestResult(
        metrics=metrics,
        equity_curve=np.array([10_000.0, 10_100.0]),
        mode="signal",
    )
    # Replace the numerical engine, keeping the service's real serialization
    # boundary. This test does not compile Numba code or create its disk caches.
    monkeypatch.setattr(SignalBacktestEngine, "run", lambda *_args: engine_result)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """\
strategy:
  id: kd_crossover
backtest:
  symbol: BTC/USDT
  timeframe: 1h
  start_date: "2024-01-01"
  end_date: "2024-01-02"
  init_cash: 10000.0
  fees: 0.0
  slippage: 0.0
""",
        encoding="utf-8",
    )
    logger = _LoggerStub()
    monkeypatch.setattr(cli_main, "logger", logger)
    monkeypatch.setattr("sys.argv", ["tradingdev", "--config", str(config_path)])

    cli_main.main()

    report = next(msg for msg in logger.messages if "Backtest results:" in msg)
    assert f"{report_label}:" in report
    metric_line = next(line for line in report.splitlines() if report_label in line)
    assert metric_line.split(":", 1)[1].strip() == "N/A"
    assert any("Result cached" in msg for msg in logger.messages)
    store = SQLiteStore(workspace)
    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0]["metrics"][unavailable_metric] is None
    assert runs[0]["metrics"]["total_trades"] == 4
    artifacts = ArtifactService(workspace=workspace, store=store)
    loaded = artifacts.load_pipeline_result(runs[0]["run_id"])
    assert loaded["success"] is True
    result = loaded["pipeline"].backtest_result
    assert result is not None
    assert not math.isfinite(result.metrics[unavailable_metric])
    assert len(list((workspace.processed_data / "cache").glob("*.pkl"))) == 1


@pytest.mark.parametrize(
    "validation",
    [
        "n_splits: 2",
        "mode: automatic\n  embargo_bars: 0",
        "mode: purged\n  label_horizon: 0",
    ],
)
def test_cli_rejects_invalid_fold_configuration_before_loading_data(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    validation: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "invalid.yaml"
    config.write_text(
        "strategy:\n  id: kd_crossover\nbacktest:\n  symbol: BTC/USDT\n"
        '  timeframe: 1h\n  start_date: "2024-01-01"\n  end_date: "2024-02-01"\n'
        f"validation:\n  {validation}\n"
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Invalid settings cannot load historical data")

    monkeypatch.setattr(DataService, "load", forbidden)
    monkeypatch.setattr(
        "sys.argv", ["tradingdev", "--config", str(config), "--walk-forward"]
    )
    with pytest.raises(ValueError, match="validation"):
        cli_main.main()
