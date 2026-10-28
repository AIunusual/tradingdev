"""Complete performance publication, immutable reads and caller failure cleanup."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file
from tradingdev.adapters.storage.performance import (
    PerformanceArtifactError,
    PerformanceStore,
)
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.artifact_service import ArtifactService
from tradingdev.app.job_store import JobStore
from tradingdev.app.run_service import RunService
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.result import BacktestResult
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.performance.artifacts import (
    PerformanceArtifacts,
    build_artifacts,
    scope_from_backtest,
)
from tradingdev.domain.performance.catalog import METRIC_CATALOG
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.domain.validation.report import summarize_results
from tradingdev.domain.validation.walk_forward import WalkForwardResult
from tradingdev.shared.utils.cache import clear_cache

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _result(value: float = 0.1, *, volume: bool = False) -> BacktestResult:
    return BacktestResult(
        metrics={
            "total_pnl": value * 100,
            "total_return": None if volume else value,
            "sharpe_ratio": None,
        },
        equity_curve=np.array([0.0, value * 100])
        if volume
        else np.array([100.0, 100.0 * (1 + value)]),
        returns=None if volume else np.array([0.0, value]),
        timestamps=pd.date_range("2024-01-01", periods=2, tz="UTC").to_numpy(),
        trades=[
            {"entry_idx": 0, "exit_idx": 1, "status": "closed", "net_pnl": value * 100}
        ],
        init_cash=None if volume else 100.0,
        mode="volume" if volume else "signal",
        metric_metadata={
            "schema_version": 1,
            "providers": {"fixture": "1.0"},
            "settings": {
                "initial_cash": None if volume else 100.0,
                "periods_per_year": 365.0,
            },
            "unavailable": {"sharpe_ratio": "insufficient_data"},
        },
    )


def _pipeline(
    *, walk_forward: bool = False, volume: bool = False
) -> tuple[PipelineResult, dict[str, Any]]:
    config: dict[str, Any] = {
        "strategy": {"id": "fixture"},
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "mode": "volume" if volume else "signal",
            "init_cash": None if volume else 100.0,
        },
    }
    if walk_forward:
        config["validation"] = {"mode": "automatic"}
    manifest = ExecutionManifest.create(
        kind="walk_forward" if walk_forward else "backtest",
        config=config,
        strategy_execution=StrategyExecution(kind="generated", constructor_kwargs={}),
    )
    result = _result(volume=volume)
    if not walk_forward:
        return PipelineResult(
            mode="simple",
            backtest_result=result,
            config_snapshot=manifest.config_copy(),
            execution_manifest=manifest,
        ), result.metrics
    moment = datetime(2024, 1, 1, tzinfo=UTC)
    folds = [
        WalkForwardResult(
            fold_index=index,
            train_start=moment,
            train_end=moment,
            test_start=moment,
            test_end=moment,
            train_metrics=result.metrics,
            test_metrics=result.metrics,
            train_backtest=result,
            test_backtest=result,
            strategy_params={"window": index + 1},
        )
        for index in range(2)
    ]
    return PipelineResult(
        mode="walk_forward",
        fold_results=folds,
        config_snapshot=manifest.config_copy(),
        execution_manifest=manifest,
    ), summarize_results(folds)


def _writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: str,
    pipeline: PipelineResult,
) -> tuple[
    WorkspacePaths, SQLiteStore, str, Callable[[PipelineResult, dict[str, Any]], Path]
]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    if writer == "job":
        jobs = JobStore(workspace=workspace, store=store)
        jobs.create_job(
            job_id="run", strategy_name="fixture", manifest=pipeline.execution_manifest
        )
        return (
            workspace,
            store,
            "run",
            lambda result, metrics: jobs.save_result("run", metrics, pipeline=result),
        )
    monkeypatch.setenv("TRADINGDEV_DATA_ROOT", str(workspace.root / "data"))
    monkeypatch.setattr(
        "tradingdev.app.artifact_service.compute_cache_key", lambda **_: "fixture"
    )
    artifacts = ArtifactService(workspace=workspace, store=store)
    return (
        workspace,
        store,
        "cli_fixture",
        lambda result, metrics: artifacts.cache_pipeline_result(
            pipeline=result,
            config_path=tmp_path / "unused.yaml",
            processed_path=tmp_path / "unused.parquet",
            metrics=metrics,
            strategy_id="fixture",
        ),
    )


@pytest.mark.parametrize("writer", ["job", "cli"])
@pytest.mark.parametrize("walk_forward", [False, True])
@pytest.mark.parametrize("volume", [False, True])
def test_callers_preserve_complete_scope_values_provenance_and_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: str,
    walk_forward: bool,
    volume: bool,
) -> None:
    pipeline, projection = _pipeline(walk_forward=walk_forward, volume=volume)
    workspace, store, run_id, save = _writer(tmp_path, monkeypatch, writer, pipeline)

    saved_path = save(pipeline, projection)
    if writer == "cli":
        run_id = saved_path.stem
        assert run_id.startswith("cli_fixture_")
    details = PerformanceStore(workspace, store)
    performance = details.load(run_id)
    observations = details.load_observations(run_id)
    stored = store.get_run(run_id)
    assert stored is not None
    assert stored["metrics"] == projection
    assert performance.default_scope == ("test_summary" if walk_forward else "full")
    scalar_scope = "fold/0/test" if walk_forward else "full"
    assert performance.scopes[scalar_scope].values == _result(volume=volume).metrics
    context = performance.scopes[scalar_scope].metadata["execution_context"]
    assert isinstance(context, dict)
    assert context["symbol"] == "BTC/USDT"
    assert performance.scopes[scalar_scope].metadata["unavailable"] == {
        "sharpe_ratio": "insufficient_data"
    }
    assert observations.scopes[scalar_scope].timestamps == [
        "2024-01-01T00:00:00+00:00",
        "2024-01-02T00:00:00+00:00",
    ]
    assert observations.scopes[scalar_scope].returns == (None if volume else [0.0, 0.1])
    assert observations.scopes[scalar_scope].trades[0]["exit_idx"] == 1
    if walk_forward:
        assert set(observations.scopes) == {
            "fold/0/train",
            "fold/0/test",
            "fold/1/train",
            "fold/1/test",
        }
        assert (
            performance.scopes["test_summary"].metadata["aggregation"]
            == "fold_descriptive"
        )
    before = saved_path.read_bytes()
    second_path = save(pipeline, projection)
    if writer == "cli":
        assert second_path != saved_path
        assert second_path.exists()
    else:
        assert second_path == saved_path
    assert saved_path.read_bytes() == before
    assert not list(workspace.runs.rglob(".result-publication"))


@pytest.mark.parametrize("writer", ["job", "cli"])
@pytest.mark.parametrize(
    "failure", [OSError("disk failure"), TimeoutError("publication timeout")]
)
def test_caller_publication_failures_restore_files_and_remove_partial_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str, failure: Exception
) -> None:
    pipeline, projection = _pipeline()
    workspace, store, run_id, save = _writer(tmp_path, monkeypatch, writer, pipeline)
    before = {
        path: path.read_bytes() for path in workspace.runs.rglob("*") if path.is_file()
    }
    original_publish = PerformanceStore.publish

    def fail_after_publication(
        self: PerformanceStore, artifacts: PerformanceArtifacts
    ) -> None:
        original_publish(self, artifacts)
        raise failure

    monkeypatch.setattr(PerformanceStore, "publish", fail_after_publication)
    with pytest.raises(type(failure), match=str(failure)):
        save(pipeline, projection)

    assert store.list_runs() == []
    assert store.list_artifacts() == []
    assert {
        path: path.read_bytes() for path in workspace.runs.rglob("*") if path.is_file()
    } == before
    assert not list(workspace.root.rglob("*.pkl"))
    assert not list(workspace.runs.rglob(".result-publication"))


def test_existing_run_is_rejected_before_any_file_or_row_is_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, projection = _pipeline()
    workspace, store, run_id, save = _writer(tmp_path, monkeypatch, "job", pipeline)
    save(pipeline, projection)
    before_row = store.get_run(run_id)
    before_artifacts = store.list_artifacts(run_id)
    before_files = {
        path: path.read_bytes()
        for path in workspace.root.rglob("*")
        if path.is_file() and path.suffix != ".sqlite"
    }
    pipeline.backtest_result = _result(0.2)

    with pytest.raises(PerformanceArtifactError, match="different results"):
        save(pipeline, pipeline.backtest_result.metrics)

    assert store.get_run(run_id) == before_row
    assert store.list_artifacts(run_id) == before_artifacts
    assert {
        path: path.read_bytes()
        for path in workspace.root.rglob("*")
        if path.is_file() and path.suffix != ".sqlite"
    } == before_files


def test_reading_a_publication_marker_is_busy_instead_of_legacy(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    store.create_run(
        run_id="interrupted",
        job_id="interrupted",
        strategy_id="fixture",
        artifact_dir=workspace.runs / "interrupted",
        metrics={},
    )
    marker = workspace.runs / "interrupted" / ".result-publication"
    marker.mkdir(parents=True)

    with pytest.raises(PerformanceArtifactError) as caught:
        PerformanceStore(workspace, store).load("interrupted")
    assert caught.value.code == "performance_artifact_busy"
    assert marker.exists()


def test_missing_artifact_is_distinct_from_corrupted_record(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    details = PerformanceStore(workspace, store)
    with pytest.raises(PerformanceArtifactError) as caught:
        details.load("legacy")
    assert caught.value.code == "performance_artifact_unavailable"
    store.create_artifact(
        artifact_id="broken:performance_json",
        run_id="broken",
        artifact_type="performance_json",
        path=workspace.runs / "broken" / "performance.json",
    )
    with pytest.raises(PerformanceArtifactError) as caught:
        details.load("broken")
    assert caught.value.code == "performance_artifact_invalid"


@pytest.mark.parametrize(
    "corruption", ["checksum", "schema", "run_id", "manifest_hash"]
)
def test_reads_reject_corruption_identity_and_unknown_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    pipeline, projection = _pipeline()
    workspace, store, run_id, save = _writer(tmp_path, monkeypatch, "job", pipeline)
    save(pipeline, projection)
    path = workspace.runs / run_id / "performance.json"
    payload = json.loads(path.read_text())
    if corruption == "checksum":
        payload["scopes"]["full"]["values"]["total_return"] = 0.9
    elif corruption == "schema":
        payload["schema_version"] = 999
    elif corruption == "run_id":
        payload["run_id"] = "another"
    else:
        payload["manifest_hash"] = "a" * 64
    path.write_text(json.dumps(payload))
    if corruption != "checksum":
        with store.connect() as connection:
            connection.execute(
                "update artifacts set sha256 = ? where artifact_id = ?",
                (sha256_file(path), f"{run_id}:performance_json"),
            )
    with pytest.raises(PerformanceArtifactError) as caught:
        PerformanceStore(workspace, store).load(run_id)
    assert caught.value.code == "performance_artifact_invalid"


def test_definition_snapshot_is_read_without_current_catalog_or_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, projection = _pipeline()
    workspace, store, run_id, save = _writer(tmp_path, monkeypatch, "job", pipeline)
    save(pipeline, projection)
    original = METRIC_CATALOG["total_return"]
    monkeypatch.setitem(
        METRIC_CATALOG, "total_return", replace(original, description="changed")
    )
    monkeypatch.setattr(
        "pickle.load", lambda *_: pytest.fail("Performance reads cannot unpickle")
    )

    assert (
        PerformanceStore(workspace, store)
        .load(run_id)
        .definitions["total_return"]
        .description
        == original.description
    )


@pytest.mark.parametrize(
    "invalid", ["returns", "timestamps", "trade_index", "numeric_timestamps"]
)
def test_invalid_observation_alignment_fails_before_publication(invalid: str) -> None:
    result = _result()
    if invalid == "returns":
        result.returns = np.array([0.1])
    elif invalid == "timestamps":
        result.timestamps = np.array(["2024-01-01"], dtype="datetime64[D]")
    elif invalid == "trade_index":
        result.trades[0]["exit_idx"] = 5
    else:
        result.timestamps = np.array([0, 1])
    with pytest.raises(ValueError):
        scope_from_backtest(result)


def test_optimization_projection_cannot_disagree_with_saved_selected_trial(
    tmp_path: Path,
) -> None:
    pipeline, _ = _pipeline()
    assert pipeline.execution_manifest is not None
    workspace = WorkspacePaths(tmp_path / "workspace")
    jobs = JobStore(workspace=workspace)
    jobs.create_job(
        job_id="opt", strategy_name="fixture", manifest=pipeline.execution_manifest
    )
    train, train_obs = scope_from_backtest(
        _result(), split="train", trial_index=0, parameters={"window": 3}
    )
    test, test_obs = scope_from_backtest(
        _result(), split="test", parameters={"window": 3}
    )
    artifacts = build_artifacts(
        "opt",
        pipeline.execution_manifest.manifest_hash,
        "test",
        {"trial/0/train": train, "test": test},
        {"trial/0/train": train_obs, "test": test_obs},
        selected_train_scope="trial/0/train",
    )
    projection = {
        "train_metrics": {**train.values, "total_return": 0.9},
        "test_metrics": test.values,
        "best_params": {"window": 3},
        "optimization_metric": "total_return",
        "best_train_metric_value": 0.1,
        "best_oos_metric_value": 0.1,
    }
    with pytest.raises(ValueError, match="projection differs"):
        jobs.save_result(
            "opt",
            projection,
            performance=artifacts,
            execution_manifest=pipeline.execution_manifest,
        )
    assert jobs.get_run("opt") is None
    assert not (workspace.runs / "opt" / "result.json").exists()


def test_cli_save_after_cache_cleanup_creates_a_new_loadable_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, projection = _pipeline()
    workspace, store, _, save = _writer(tmp_path, monkeypatch, "cli", pipeline)
    first_path = save(pipeline, projection)
    first_id = first_path.stem
    first_details = PerformanceStore(workspace, store).load(first_id)

    assert clear_cache() == 1
    assert not first_path.exists()
    second_path = save(pipeline, projection)
    second_id = second_path.stem

    assert second_id != first_id
    assert second_path.exists()
    assert len(store.list_runs()) == 2
    restored = ArtifactService(workspace=workspace, store=store).load_pipeline_result(
        second_id
    )
    assert restored["success"] is True
    assert restored["pipeline"].backtest_result.metrics == projection
    assert PerformanceStore(workspace, store).load(first_id) == first_details
    assert not first_path.exists()


def test_cli_same_configuration_with_different_outputs_preserves_both_executions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, projection = _pipeline()
    workspace, store, _, save = _writer(tmp_path, monkeypatch, "cli", pipeline)
    first_path = save(pipeline, projection)
    original_bytes = first_path.read_bytes()
    pipeline.backtest_result = _result(0.2)

    second_path = save(pipeline, pipeline.backtest_result.metrics)

    assert second_path != first_path
    assert first_path.read_bytes() == original_bytes
    first_run = store.get_run(first_path.stem)
    second_run = store.get_run(second_path.stem)
    assert first_run is not None and second_run is not None
    assert first_run["manifest_hash"] == second_run["manifest_hash"]
    assert first_run["metrics"]["total_return"] == 0.1
    assert second_run["metrics"]["total_return"] == 0.2
    for path, expected in ((first_path, 0.1), (second_path, 0.2)):
        restored = ArtifactService(
            workspace=workspace, store=store
        ).load_pipeline_result(path.stem)
        assert restored["success"] is True
        assert restored["pipeline"].backtest_result.metrics["total_return"] == expected
        artifact = store.get_artifact(f"{path.stem}:pipeline_result")
        assert artifact is not None
        assert artifact["metadata"]["cache_key"] == "fixture"


@pytest.mark.parametrize(
    ("artifact_name", "corruption"),
    [("result.json", "missing"), ("pipeline_result.pkl", "corrupt")],
)
def test_job_retry_rejects_incomplete_registered_artifacts_without_rewriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
    corruption: str,
) -> None:
    pipeline, projection = _pipeline()
    workspace, store, run_id, save = _writer(tmp_path, monkeypatch, "job", pipeline)
    save(pipeline, projection)
    original_row = store.get_run(run_id)
    original_artifacts = store.list_artifacts(run_id)
    target = workspace.runs / run_id / artifact_name
    if corruption == "missing":
        target.unlink()
    else:
        target.write_bytes(b"corrupted pickle")

    with pytest.raises(PerformanceArtifactError, match="missing or corrupt") as caught:
        save(pipeline, projection)

    assert caught.value.code == "performance_artifact_invalid"
    assert store.get_run(run_id) == original_row
    assert store.list_artifacts(run_id) == original_artifacts
    assert not (workspace.runs / run_id / ".result-publication").exists()
    if corruption == "missing":
        assert not target.exists()
    else:
        assert target.read_bytes() == b"corrupted pickle"


@pytest.mark.parametrize("writer", ["job", "cli"])
@pytest.mark.parametrize("walk_forward", [False, True])
@pytest.mark.parametrize("bundled", [False, True])
def test_persisted_scopes_keep_fixed_parameters_separate_from_fitted_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: str,
    walk_forward: bool,
    bundled: bool,
) -> None:
    pipeline, projection = _pipeline(walk_forward=walk_forward)
    parameters = {"window": 3, "nested": {"warmup": 2}}
    config = pipeline.config_snapshot
    config["strategy"]["parameters"] = {"window": 3}
    manifest = ExecutionManifest.create(
        kind="walk_forward" if walk_forward else "backtest",
        config=config,
        strategy_execution=StrategyExecution(
            kind="bundled" if bundled else "generated",
            constructor_kwargs={"config": parameters} if bundled else parameters,
        ),
    )
    pipeline.execution_manifest = manifest
    pipeline.config_snapshot = manifest.config_copy()
    if pipeline.backtest_result is not None:
        pipeline.backtest_result.metric_metadata["strategy_parameters"] = {"window": 7}
    workspace, store, run_id, save = _writer(tmp_path, monkeypatch, writer, pipeline)
    path = save(pipeline, projection)
    if writer == "cli":
        run_id = path.stem
    performance_path = workspace.runs / run_id / "performance.json"
    original = performance_path.read_bytes()
    queries = RunService(workspace=workspace, store=store)
    scopes = queries.get_run(run_id)["run"]["available_scopes"]
    assert scopes
    for scope_id in scopes:
        detail = queries.get_run_metrics(run_id, scope=scope_id)
        assert detail["success"], detail
        assert detail["parameters"] == parameters
        assert detail["manifest_hash"] == manifest.manifest_hash
        if scope_id.startswith("fold/"):
            index = int(scope_id.split("/")[1])
            assert detail["metadata"]["strategy_parameters"] == {"window": index + 1}
        elif scope_id == "full":
            assert detail["metadata"]["strategy_parameters"] == {"window": 7}
        else:
            assert detail["metadata"]["fold_metadata"]["fold/0/test"][
                "strategy_parameters"
            ] == {"window": 1}
    assert performance_path.read_bytes() == original
