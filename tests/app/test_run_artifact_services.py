"""Run and artifact service tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.artifact_service import ArtifactService
from tradingdev.app.job_store import JobStore
from tradingdev.app.run_lineage import load_config_payload
from tradingdev.app.run_service import RunService
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.result import BacktestResult
from tradingdev.domain.execution import ExecutionManifest, ManifestError
from tradingdev.domain.strategies.execution import StrategyExecution

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


def _backtest_result(total_return: float) -> BacktestResult:
    return BacktestResult(
        metrics={"total_return": total_return},
        equity_curve=np.array([100.0, 100.0 * (1 + total_return)]),
        returns=np.array([0.0, total_return]),
        init_cash=100.0,
    )


def test_run_service_compare_and_artifact_lookup(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    run_dir = workspace.runs / "run_a"
    run_dir.mkdir(parents=True)
    artifact_path = run_dir / "result.json"
    artifact_path.write_text('{"total_return": 0.1}', encoding="utf-8")
    store.create_run(
        run_id="run_a",
        job_id="job_a",
        strategy_id="fixture",
        artifact_dir=run_dir,
        metrics={"total_return": 0.1, "sharpe_ratio": 1.2},
    )
    store.create_run(
        run_id="run_b",
        job_id="job_b",
        strategy_id="fixture",
        artifact_dir=workspace.runs / "run_b",
        metrics={"total_return": 0.2, "sharpe_ratio": 1.5},
    )
    store.create_artifact(
        artifact_id="run_a:result_json",
        run_id="run_a",
        artifact_type="result_json",
        path=artifact_path,
        sha256=sha256_file(artifact_path),
        metadata={"job_id": "job_a"},
    )

    runs = RunService(workspace=workspace, store=store)
    artifacts = ArtifactService(workspace=workspace, store=store)

    comparison = runs.compare_runs(["run_a", "run_b"])
    assert comparison["success"] is True
    assert comparison["runs"][0]["metrics"]["total_return"] == 0.1
    artifact = artifacts.get_artifact("run_a:result_json", include_content=True)
    assert artifact["success"] is True
    assert artifact["content"] == '{"total_return": 0.1}'


def test_job_store_save_result_records_run_lineage(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    job_store = JobStore(workspace=workspace, store=store)

    strategy_source = tmp_path / "fixture_strategy.py"
    strategy_source.write_text(
        "from tradingdev.domain.strategies.base import BaseStrategy\n"
        "class FixtureStrategy(BaseStrategy):\n"
        "    pass\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "fixture.yaml"
    config_path.write_text(
        f"""\
strategy:
  id: fixture
  class_name: FixtureStrategy
  source_path: "{strategy_source}"
  parameters:
    random_state: 7
backtest:
  symbol: BTC/USDT
  timeframe: 1h
  start_date: "2024-01-01"
  end_date: "2024-01-02"
""",
        encoding="utf-8",
    )

    created = job_store.create_job(
        job_id="job_lineage",
        strategy_name="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
        config_path=str(config_path),
    )
    job_store.update_job(
        "job_lineage",
        status="running_backtest",
        dataset_id="dataset-fixture",
    )

    pipeline = PipelineResult(
        mode="simple",
        config_snapshot=load_config_payload(config_path) or {},
        backtest_result=_backtest_result(0.1),
    )
    # Persistence must retain what ran, even if the input file changed meanwhile.
    config_path.write_text("strategy:\n  id: different_strategy\n", encoding="utf-8")
    result_path = job_store.save_result(
        "job_lineage",
        {"total_return": 0.1},
        pipeline=pipeline,
    )

    assert created["created_at"]
    assert result_path == workspace.runs / "job_lineage" / "result.json"
    assert result_path.exists()
    assert (workspace.runs / "job_lineage" / "config.yaml").exists()
    assert (workspace.runs / "job_lineage" / "strategy.py").exists()
    assert (workspace.runs / "job_lineage" / "dataset_fingerprint.json").exists()
    assert (workspace.runs / "job_lineage" / "pipeline_result.pkl").exists()

    run = job_store.get_run("job_lineage")
    assert run is not None
    assert run["job_id"] == "job_lineage"
    assert run["strategy_id"] == "fixture"
    assert run["dataset_id"] == "dataset-fixture"
    assert run["artifact_dir"] == str(workspace.runs / "job_lineage")
    assert run["source_hash"] == sha256_file(strategy_source)
    assert run["random_seed"] is None

    artifacts = {
        item["artifact_type"]: item for item in job_store.list_artifacts("job_lineage")
    }
    assert set(artifacts) == {
        "config_snapshot",
        "dataset_fingerprint",
        "pipeline_result",
        "result_json",
        "strategy_source",
        "performance_json",
        "observations_json",
    }
    assert artifacts["config_snapshot"]["metadata"]["config_hash"] == run["config_hash"]
    assert artifacts["strategy_source"]["metadata"]["source_path"] == str(
        strategy_source
    )
    assert artifacts["strategy_source"]["metadata"]["source_hash"] == run["source_hash"]
    assert artifacts["dataset_fingerprint"]["metadata"]["fingerprint"]

    loaded = ArtifactService(workspace=workspace, store=store).load_pipeline_result(
        "job_lineage"
    )
    assert loaded["success"] is True
    assert loaded["pipeline"].mode == "simple"


def test_artifact_service_cache_pipeline_records_run_lineage(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    monkeypatch.setenv("TRADINGDEV_DATA_ROOT", str(workspace.root / "data"))
    # The code identity used to come from a git call inside the cache helper and
    # had to be stubbed to keep the key stable. It now comes from the strategy
    # revision pinned in the manifest, which is stable on its own.

    strategy_source = tmp_path / "cli_strategy.py"
    strategy_source.write_text("class CliStrategy: pass\n", encoding="utf-8")
    config_path = tmp_path / "cli.yaml"
    config_path.write_text(
        f"""\
random_seed: 42
strategy:
  id: cli_fixture
  source_path: "{strategy_source}"
  parameters:
    random_seed: 11
backtest:
  symbol: BTC/USDT
  timeframe: 1h
  start_date: "2024-01-01"
  end_date: "2024-01-02"
  init_cash: 10000
""",
        encoding="utf-8",
    )
    processed_path = tmp_path / "processed.parquet"
    processed_path.write_text("fixture", encoding="utf-8")

    service = ArtifactService(workspace=workspace, store=store)
    manifest = ExecutionManifest.create(
        kind="backtest",
        config=load_config_payload(config_path) or {},
        strategy_execution=StrategyExecution(
            kind="generated", constructor_kwargs={"random_seed": 11}
        ),
    )
    pipeline = PipelineResult(
        mode="simple",
        backtest_result=_backtest_result(0.2),
        config_snapshot=manifest.config_copy(),
        execution_manifest=manifest,
    )

    cached_path = service.cache_pipeline_result(
        pipeline=pipeline,
        config_path=config_path,
        processed_path=processed_path,
        metrics={"total_return": 0.2},
        strategy_id="cli_fixture",
    )

    run = store.list_runs()[0]
    assert run["source_hash"] == sha256_file(strategy_source)
    assert run["random_seed"] == 42
    assert run["manifest_hash"] == manifest.manifest_hash
    assert run["artifact_dir"] == str(workspace.runs / run["run_id"])
    artifact = service.get_artifact(
        f"{run['run_id']}:execution_manifest", include_content=True
    )
    assert json.loads(artifact["content"]) == manifest.model_dump(mode="json")

    # Saving an executed result never derives its identity from mutable YAML.
    config_path.write_text("strategy:\n  id: another_strategy\n", encoding="utf-8")
    second_path = service.cache_pipeline_result(
        pipeline=pipeline,
        config_path=config_path,
        processed_path=processed_path,
        metrics={"total_return": 0.2},
        strategy_id="cli_fixture",
    )
    assert second_path != cached_path
    assert second_path.exists()
    assert len(store.list_runs()) == 2

    # Reading follows the recorded artifact path even if current key inputs differ.
    config_path.unlink()
    processed_path.unlink()
    loaded = ArtifactService(workspace=workspace, store=store).load_pipeline_result(
        run["run_id"]
    )
    assert loaded["success"] is True
    assert loaded["artifact"]["path"] == str(cached_path)
    restored = loaded["pipeline"]
    assert isinstance(restored, PipelineResult)
    assert restored.mode == pipeline.mode
    assert restored.config_snapshot == manifest.config_copy()
    assert restored.execution_manifest == manifest


@pytest.mark.parametrize("changed_setting", ["backtest", "strategy_execution"])
def test_cli_cache_uses_manifest_even_after_original_config_is_removed(
    tmp_path: Path, monkeypatch: MonkeyPatch, changed_setting: str
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    monkeypatch.setenv("TRADINGDEV_DATA_ROOT", str(workspace.root / "data"))
    config_path = tmp_path / "cli.yaml"
    original = {
        "strategy": {"id": "fixture"},
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "init_cash": 10000,
        },
    }
    manifest = ExecutionManifest.create(
        kind="backtest",
        config=original,
        strategy_execution=StrategyExecution(
            kind="generated", constructor_kwargs={"window": 7}
        ),
    )
    pipeline = PipelineResult(
        mode="simple",
        backtest_result=_backtest_result(0.2),
        config_snapshot=manifest.config_copy(),
        execution_manifest=manifest,
    )
    service = ArtifactService(workspace=workspace, store=store)
    first = service.cache_pipeline_result(
        pipeline=pipeline,
        config_path=config_path,
        processed_path=tmp_path / "data.parquet",
        metrics={"total_return": 0.2},
        strategy_id="fixture",
    )
    assert not config_path.exists()
    changed_config = manifest.config_copy()
    strategy_execution = manifest.strategy_execution
    if changed_setting == "backtest":
        changed_config["backtest"]["end_date"] = "2024-02-29"
    else:
        strategy_execution = StrategyExecution(
            kind="generated", constructor_kwargs={"window": 14}
        )
    changed = ExecutionManifest.create(
        kind="backtest",
        config=changed_config,
        strategy_execution=strategy_execution,
    )
    if changed_setting == "strategy_execution":
        assert changed.config_copy() == manifest.config_copy()
    second = service.cache_pipeline_result(
        pipeline=PipelineResult(
            mode="simple",
            backtest_result=_backtest_result(0.3),
            config_snapshot=changed.config_copy(),
            execution_manifest=changed,
        ),
        config_path=config_path,
        processed_path=tmp_path / "data.parquet",
        metrics={"total_return": 0.3},
        strategy_id="fixture",
    )
    assert first != second
    assert {run["manifest_hash"] for run in store.list_runs()} == {
        manifest.manifest_hash,
        changed.manifest_hash,
    }


@pytest.mark.parametrize("invalid", ["missing", "modified_config", "modified_manifest"])
def test_cli_cache_rejects_results_without_their_intact_execution_manifest(
    tmp_path: Path, monkeypatch: MonkeyPatch, invalid: str
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    monkeypatch.setenv("TRADINGDEV_DATA_ROOT", str(workspace.root / "data"))
    manifest = ExecutionManifest.create(
        kind="backtest",
        strategy_execution=StrategyExecution(kind="generated", constructor_kwargs={}),
        config={
            "strategy": {"id": "fixture"},
            "backtest": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "init_cash": 10000,
            },
        },
    )
    pipeline = PipelineResult(
        mode="simple",
        config_snapshot=manifest.config_copy(),
        execution_manifest=manifest,
    )
    if invalid == "missing":
        pipeline.execution_manifest = None
    elif invalid == "modified_config":
        pipeline.config_snapshot["backtest"]["fees"] = 0.05
    else:
        manifest.config["random_seed"] = 12345
    with pytest.raises(ManifestError):
        ArtifactService(workspace=workspace, store=store).cache_pipeline_result(
            pipeline=pipeline,
            config_path=tmp_path / "cli.yaml",
            processed_path=tmp_path / "data.parquet",
            metrics={"total_return": 0.2},
            strategy_id="fixture",
        )
    assert store.list_runs() == []
    assert not list(workspace.root.rglob("*.pkl"))
