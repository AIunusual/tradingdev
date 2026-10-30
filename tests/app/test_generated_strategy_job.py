"""Service composition: persist a job/config and request the correct worker.

The process runner is recorded, not executed. Real MCP/worker lifecycle coverage
lives in tests/integration/test_mcp_protocol.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from tests.data_cache import seed_market_cache
from tests.research_fixtures import research_for_config

from tradingdev.adapters.execution.process_runner import ProcessRunner, WorkerHandle
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService, LoadedDataset
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.strategy_service import StrategyNotExecutableError, StrategyService
from tradingdev.mcp.workers import backtest


class FakeRunner(ProcessRunner):
    """Process runner that records worker calls without spawning processes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def spawn_module(self, module: str, *args: str) -> WorkerHandle:
        self.calls.append((module, args))
        return WorkerHandle(4321, 100.0, "a" * 32)


_STRATEGY_CODE = """\
from __future__ import annotations

from typing import Any

import pandas as pd

from tradingdev.domain.strategies.base import BaseStrategy


class IntegrationStrategy(BaseStrategy):
    def __init__(self, backtest_engine: object | None = None) -> None:
        self._engine = backtest_engine

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = 0
        result.loc[result["close"] > result["open"], "signal"] = 1
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {}
"""

_YAML = """\
strategy:
  id: "integration_strategy"
  version: "0.1.0"
  class_name: "IntegrationStrategy"
  source_path: "workspace/generated_strategies/integration_strategy.py"
  parameters: {}
backtest:
  symbol: "ETH/USDT"
  timeframe: "1h"
  start_date: "2023-01-01"
  end_date: "2023-01-31"
  init_cash: 10000.0
data:
  requirements:
    market:
      source: "binance_api"
      symbol: "ETH/USDT"
      timeframe: "1h"
    features: []
"""


def test_generated_strategy_can_start_backtest_job(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    job_store = JobStore(workspace=workspace, store=store)
    workspace.processed_data.mkdir(parents=True, exist_ok=True)
    seed_market_cache(
        workspace.processed_data,
        pd.DataFrame({"timestamp": pd.to_datetime([], utc=True)}),
        source="binance_api",
    )

    strategy_service = StrategyService(workspace)
    strategy_service._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]
    saved = strategy_service.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML, strategy_outline="service scheduling test"),
        strategy_id="integration_strategy",
    )
    assert saved.success is True
    assert (
        strategy_service.validate("integration_strategy", saved.revision_id)["success"]
        is True
    )
    assert (
        strategy_service.dry_run("integration_strategy", saved.revision_id)["success"]
        is True
    )
    assert (
        strategy_service.promote("integration_strategy", saved.revision_id)["success"]
        is True
    )

    runner = FakeRunner()
    service = JobService(
        strategy_service=strategy_service,
        data_service=DataService(workspace),
        job_store=job_store,
        process_runner=runner,
        project_root=tmp_path,
    )

    response = service.start_backtest(
        strategy_id="integration_strategy",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )

    assert response["job_id"]
    assert response["data_available"] is True
    assert runner.calls == [
        (
            "tradingdev.mcp.workers.backtest",
            (response["job_id"],),
        )
    ]
    job = store.get_job(str(response["job_id"]))
    assert job is not None
    assert job["status"] == "queued"
    assert job["pid"] == 4321
    assert job["process_create_time"] == 100.0
    assert job["worker_control_id"] == "a" * 32
    assert job["job_type"] == "backtest"
    assert job["original_config_path"] == saved.config_path
    assert job["revision_id"] == saved.revision_id
    assert response["revision_id"] == saved.revision_id
    assert job["config_path"] == str(
        workspace.runs / response["job_id"] / "config.yaml"
    )

    effective_config = yaml.safe_load(
        (workspace.runs / response["job_id"] / "config.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert effective_config["backtest"]["symbol"] == "BTC/USDT"
    assert effective_config["backtest"]["start_date"].startswith("2024-01-01")
    assert effective_config["backtest"]["end_date"].startswith("2024-01-31")
    assert effective_config["data"]["requirements"]["market"]["symbol"] == "BTC/USDT"


def test_queued_revision_runs_after_new_draft_is_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real worker executes A after B becomes the current draft."""
    workspace = WorkspacePaths(tmp_path / "workspace")
    monkeypatch.setenv("TRADINGDEV_WORKSPACE", str(workspace.root))
    monkeypatch.setenv(
        "TRADINGDEV_WORKER_IDENTITY",
        json.dumps(WorkerHandle(4321, 100.0, "a" * 32).job_fields()),
    )
    strategies = StrategyService(workspace)
    strategies._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]
    saved = strategies.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML),
        strategy_id="integration_strategy",
    )
    assert saved.success and saved.revision_id is not None
    assert strategies.validate("integration_strategy", saved.revision_id)["success"]
    assert strategies.dry_run("integration_strategy", saved.revision_id)["success"]
    store = JobStore(workspace=workspace)
    jobs = JobService(
        strategy_service=strategies,
        job_store=store,
        process_runner=FakeRunner(),
    )
    started = jobs.start_backtest(
        strategy_id="integration_strategy",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
    )
    assert started["revision_id"] == saved.revision_id
    job = store.get_job(started["job_id"])
    assert job is not None
    replacement = strategies.save_draft(
        _STRATEGY_CODE.replace("= 1", "= -1"),
        _YAML,
        research_for_config(_YAML),
        strategy_id="integration_strategy",
        expected_revision_id=saved.revision_id,
    )
    assert replacement.success and replacement.revision_id != saved.revision_id
    prices = [100.0 + index for index in range(24)]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"),
            "open": [price - 0.5 for price in prices],
            "high": [price + 1 for price in prices],
            "low": [price - 1 for price in prices],
            "close": prices,
            "volume": [1000.0] * 24,
        }
    )
    dataset = LoadedDataset(
        frame=frame,
        processed_path=tmp_path / "processed.parquet",
        dataset_id="revision-fixture",
    )
    monkeypatch.setattr(DataService, "load", lambda *_args, **_kwargs: dataset)

    backtest._run_backtest(started["job_id"])

    status = jobs.get_job_status(started["job_id"])
    assert status["status"] == "done", status
    assert status["revision_id"] == saved.revision_id
    assert status["metrics"]["total_return"] > 0
    assert jobs.list_jobs()[0]["revision_id"] == saved.revision_id
    run = store.get_run(started["job_id"])
    assert run is not None and run["revision_id"] == saved.revision_id
    assert (workspace.runs / started["job_id"] / "strategy.py").read_text() == (
        _STRATEGY_CODE
    )
    current = strategies.load("integration_strategy")
    assert current is not None and current.status == "draft"


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("source_path", "other.py"),
        ("class_name", "UnvalidatedStrategy"),
        ("revision_id", None),
        ("fit", True),
    ],
)
def test_execution_rejects_revision_identity_overrides(
    tmp_path: Path, key: str, replacement: object
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    strategies = StrategyService(workspace)
    strategies._quality_gate_diagnostics = lambda _path: []  # type: ignore[assignment,method-assign]
    saved = strategies.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML),
        strategy_id="integration_strategy",
    )
    assert saved.success and saved.revision_id is not None
    assert strategies.validate("integration_strategy", saved.revision_id)["success"]
    assert strategies.dry_run("integration_strategy", saved.revision_id)["success"]
    assert saved.config_path is not None
    config = yaml.safe_load(Path(saved.config_path).read_text())
    config["strategy"][key] = replacement
    service = BacktestService(strategy_gate=strategies)

    with pytest.raises(StrategyNotExecutableError):
        service.prepare_strategy(config)


def test_result_persistence_rejects_changed_revision_source(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    strategies = StrategyService(workspace)
    saved = strategies.save_draft(
        _STRATEGY_CODE,
        _YAML,
        research_for_config(_YAML),
        strategy_id="integration_strategy",
    )
    assert saved.success and saved.revision_id is not None
    assert saved.source_path is not None
    store = JobStore(workspace=workspace)
    store.create_job(
        job_id="changed-source",
        strategy_name="integration_strategy",
        revision_id=saved.revision_id,
        config_path=saved.config_path,
    )
    Path(saved.source_path).write_text(_STRATEGY_CODE + "\n# changed\n")

    with pytest.raises(ValueError, match="content mismatch"):
        store.save_result("changed-source", {"total_return": 0.1})

    assert store.get_run("changed-source") is None
    assert not (workspace.runs / "changed-source" / "result.json").exists()
