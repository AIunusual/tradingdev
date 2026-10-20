"""Saved performance discovery and queries through real workspace storage."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
from mcp.server import MCPServer
from mcp_types import CallToolResult

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.contracts.jobs import JobStatus
from tradingdev.app.contracts.research import (
    CompareRunsResponse,
    MetricCatalogResponse,
    MetricQueryError,
    RunMetricsResponse,
    RunRecord,
)
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.run_service import RunService
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.performance.artifacts import (
    FoldStats,
    MetricDefinitionSnapshot,
    ObservationsBundle,
    PerformanceArtifacts,
    PerformanceBundle,
    PerformanceScope,
    ScopeObservations,
)
from tradingdev.domain.performance.catalog import METRIC_CATALOG
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.domain.validation.report import summarize_results
from tradingdev.domain.validation.walk_forward import WalkForwardResult
from tradingdev.mcp.tools import runs

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def storage(tmp_path: Path) -> tuple[WorkspacePaths, SQLiteStore, RunService]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    return workspace, store, RunService(workspace=workspace, store=store)


def _scope(**updates: Any) -> PerformanceScope:
    values: dict[str, Any] = {
        "mode": "signal",
        "values": {
            "total_pnl": 11.0,
            "daily_pnl_mean": 5.5,
            "total_volume": 210.0,
            "win_rate": None,
        },
        "metadata": {
            "settings": {"periods_per_year": 365, "risk_free_rate": 0.0},
            "providers": {"empyrical-reloaded": "0.5.12", "vectorbt": "1.0.0"},
            "execution_context": {
                "symbol": "BTC/USDT",
                "init_cash": 100.0,
                "position_size": 100.0,
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
            },
            "observations": {
                "bar_count": 2,
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
            },
            "unavailable": {"win_rate": "no_trades"},
        },
    }
    values.update(updates)
    return PerformanceScope.model_validate(values)


def _publish(
    storage: tuple[WorkspacePaths, SQLiteStore, RunService],
    run_id: str,
    *,
    scopes: dict[str, PerformanceScope] | None = None,
    default_scope: str = "full",
    selected_train_scope: str | None = None,
    snapshot_changes: dict[str, dict[str, Any]] | None = None,
    projection: dict[str, Any] | None = None,
) -> PerformanceArtifacts:
    workspace, store, _ = storage
    scopes = scopes or {"full": _scope()}
    definitions = {
        name: MetricDefinitionSnapshot.model_validate(
            asdict(METRIC_CATALOG[name]) | (snapshot_changes or {}).get(name, {})
        )
        for name in {name for scope in scopes.values() for name in scope.values}
    }
    artifacts = PerformanceArtifacts(
        performance=PerformanceBundle(
            run_id=run_id,
            default_scope=default_scope,
            scopes=scopes,
            definitions=definitions,
            selected_train_scope=selected_train_scope,
        ),
        observations=ObservationsBundle(
            run_id=run_id,
            scopes={
                name: ScopeObservations(
                    init_cash=100.0,
                    equity_curve=[100.0, 111.0],
                    returns=[0.0, 0.11],
                    timestamps=["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
                    trades=[],
                )
                for name, scope in scopes.items()
                if scope.kind == "backtest"
            },
        ),
    )
    store.create_run(
        run_id=run_id,
        job_id=run_id,
        strategy_id="fixture",
        artifact_dir=workspace.runs / run_id,
        metrics=projection
        if projection is not None
        else scopes[default_scope].model_dump(mode="json")["values"],
    )
    PerformanceStore(workspace, store).publish(artifacts)
    return artifacts


def test_summary_uses_saved_flags_and_detail_keeps_every_value(
    storage: tuple[WorkspacePaths, SQLiteStore, RunService],
) -> None:
    _, store, service = storage
    _publish(
        storage,
        "run",
        snapshot_changes={
            "total_pnl": {"summary": False},
            "daily_pnl_mean": {"summary": True},
        },
    )
    run = service.get_run("run")["run"]
    RunRecord.model_validate(run)
    assert run["metrics"] == {"daily_pnl_mean": 5.5, "win_rate": None}
    assert run["available_metric_ids"] == [
        "daily_pnl_mean",
        "total_pnl",
        "total_volume",
        "win_rate",
    ]
    assert run["available_scopes"] == ["full"]
    assert run["default_scope"] == "full"
    assert run["details_available"] is True
    assert service.list_runs()[0] == run
    detail = RunMetricsResponse.model_validate(service.get_run_metrics("run"))
    assert detail.metrics["total_volume"] == 210
    assert detail.metrics["win_rate"] is None
    assert detail.metadata["unavailable"] == {"win_rate": "no_trades"}
    assert detail.definitions["total_pnl"].summary is False
    persisted = store.get_run("run")
    assert persisted is not None and persisted["metrics"]["total_volume"] == 210


def test_quantstats_migration_keeps_historical_provider_and_definitions(
    storage: tuple[WorkspacePaths, SQLiteStore, RunService],
) -> None:
    workspace, _, service = storage
    historical = _scope(values={"total_return": 0.11})
    _publish(
        storage,
        "historical",
        scopes={"full": historical},
        snapshot_changes={"total_return": {"provider": "empyrical"}},
    )
    original = {
        path: path.read_bytes()
        for path in (workspace.runs / "historical").rglob("*")
        if path.is_file()
    }
    metadata = historical.model_dump(mode="json")["metadata"]
    metadata["providers"] = {"quantstats": "0.0.86", "vectorbt": "1.0.0"}
    _publish(
        storage,
        "current",
        scopes={"full": historical.model_copy(update={"metadata": metadata})},
    )
    assert METRIC_CATALOG["total_return"].provider == "quantstats"
    old = RunMetricsResponse.model_validate(service.get_run_metrics("historical"))
    new = RunMetricsResponse.model_validate(service.get_run_metrics("current"))
    assert old.metrics == new.metrics == {"total_return": 0.11}
    assert old.definitions["total_return"].provider == "empyrical"
    old_providers = old.metadata["providers"]
    assert isinstance(old_providers, dict)
    assert old_providers["empyrical-reloaded"] == "0.5.12"
    assert new.definitions["total_return"].provider == "quantstats"
    comparison = CompareRunsResponse.model_validate(
        service.compare_runs(["historical", "current"], metric_ids=["total_return"])
    )
    reasons = comparison.metric_compatibility["total_return"].reasons
    assert "definition.provider_differs:current" in reasons
    assert "providers_differ:current" in reasons
    assert all(path.read_bytes() == content for path, content in original.items())


def test_scope_and_metric_selection_errors_include_available_choices(
    storage: tuple[WorkspacePaths, SQLiteStore, RunService],
) -> None:
    service = storage[2]
    _publish(storage, "run")
    one = service.get_run_metrics("run", metric_ids=["total_volume"])
    assert one["metrics"] == {"total_volume": 210.0}
    assert set(one["definitions"]) == {"total_volume"}
    unknown_metric = MetricQueryError.model_validate(
        service.get_run_metrics("run", metric_ids=["made_up"])
    )
    assert unknown_metric.code == "unknown_metric_id"
    assert "total_volume" in unknown_metric.available_metric_ids
    unknown_scope = MetricQueryError.model_validate(
        service.get_run_metrics("run", scope="train")
    )
    assert unknown_scope.code == "unknown_metric_scope"
    assert unknown_scope.available_scopes == ["full"]
    assert service.get_run_metrics("absent")["code"] == "run_not_found"


def test_fold_scope_summary_preserves_nulls_valid_counts_and_raw_fold(
    storage: tuple[WorkspacePaths, SQLiteStore, RunService],
) -> None:
    scalar = _scope(split="test", fold_index=0)
    summary = _scope(
        kind="fold_summary",
        split="test",
        values={
            "total_pnl": FoldStats(
                mean=11.0, std=0.0, min=11.0, max=11.0, valid_count=1
            ),
            "win_rate": FoldStats(
                mean=None, std=None, min=None, max=None, valid_count=0
            ),
        },
    )
    _publish(
        storage,
        "run",
        scopes={"fold/0/test": scalar, "test_summary": summary},
        default_scope="test_summary",
    )
    service = storage[2]
    detail = RunMetricsResponse.model_validate(service.get_run_metrics("run"))
    assert detail.kind == "fold_summary"
    assert detail.metrics["win_rate"] == {
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "valid_count": 0,
    }
    fold = service.get_run_metrics("run", scope="fold/0/test")
    assert fold["metrics"]["daily_pnl_mean"] == 5.5
    assert fold["fold_index"] == 0
    _publish(
        storage,
        "second",
        scopes={"test_summary": summary},
        default_scope="test_summary",
    )
    comparison = CompareRunsResponse.model_validate(
        service.compare_runs(["run", "second"], metric_ids=["total_pnl"])
    )
    assert comparison.comparable
    expected = summary.values["total_pnl"]
    assert isinstance(expected, FoldStats)
    assert comparison.runs[0].metrics["total_pnl"] == expected.model_dump()


def test_legacy_metrics_remain_readable_but_do_not_invent_details(
    storage: tuple[WorkspacePaths, SQLiteStore, RunService],
) -> None:
    workspace, store, service = storage
    for run_id in ("old_a", "old_b"):
        store.create_run(
            run_id=run_id,
            job_id=run_id,
            strategy_id="fixture",
            artifact_dir=workspace.runs / run_id,
            metrics={"daily_pnl_mean": 42},
        )
    run = service.get_run("old_a")["run"]
    assert run["metrics"] == {"daily_pnl_mean": 42}
    assert run["provenance"] == "legacy_metrics"
    assert not run["details_available"]
    assert run["available_scopes"] == []
    assert (
        service.get_run_metrics("old_a")["code"] == "performance_artifact_unavailable"
    )
    comparison = service.compare_runs(["old_a", "old_b"])
    assert not comparison["comparable"]
    assert (
        "unknown_provenance:old_a"
        in comparison["metric_compatibility"]["daily_pnl_mean"]["reasons"]
    )


def test_default_volume_comparison_uses_only_applicable_summary_metrics(
    storage: tuple[WorkspacePaths, SQLiteStore, RunService],
) -> None:
    scope = _scope(
        mode="volume",
        values={
            "total_pnl": 11.0,
            "max_drawdown_amount": 2.0,
            "total_return": None,
            "sharpe_ratio": None,
        },
    )
    for run_id in ("volume_a", "volume_b"):
        _publish(storage, run_id, scopes={"full": scope})
    service = storage[2]
    summary = service.get_run("volume_a")["run"]["metrics"]
    compared = service.compare_runs(["volume_a", "volume_b"])
    assert compared["comparable"]
    assert compared["runs"][0]["metrics"] == summary
    assert set(summary) == {"total_pnl", "max_drawdown_amount"}
    explicit = service.compare_runs(
        ["volume_a", "volume_b"], metric_ids=["sharpe_ratio"]
    )
    assert explicit["runs"][0]["metrics"] == {"sharpe_ratio": None}
    assert not explicit["comparable"]


@pytest.mark.parametrize("corruption", ["changed", "missing"])
def test_corrupt_artifact_cannot_fall_back_to_legacy_values(
    storage: tuple[WorkspacePaths, SQLiteStore, RunService], corruption: str
) -> None:
    workspace, _, service = storage
    _publish(storage, "run")
    path = workspace.runs / "run" / "performance.json"
    if corruption == "changed":
        path.write_text("{}", encoding="utf-8")
    else:
        path.unlink()
    assert service.get_run_metrics("run")["code"] == "performance_artifact_invalid"
    run = service.get_run("run")["run"]
    assert run["metrics"] == {}
    assert run["provenance"] == "invalid_artifact"
    assert run["detail_error"]["code"] == "performance_artifact_invalid"


@pytest.mark.parametrize(
    "difference",
    ["mode", "unit", "provider", "calendar", "currency", "capital", "scope"],
)
def test_comparison_flags_incompatible_meanings_and_context(
    storage: tuple[WorkspacePaths, SQLiteStore, RunService], difference: str
) -> None:
    _publish(storage, "first")
    scope = _scope()
    metadata = scope.model_dump(mode="json")["metadata"]
    changes: dict[str, dict[str, Any]] = {}
    if difference == "mode":
        scope = scope.model_copy(update={"mode": "volume"})
    elif difference == "unit":
        changes = {"total_pnl": {"unit": "fraction"}}
    elif difference == "provider":
        metadata["providers"]["vectorbt"] = "2.0.0"
    elif difference == "calendar":
        metadata["settings"]["periods_per_year"] = 252
    elif difference == "currency":
        metadata["execution_context"]["symbol"] = "ETH/BTC"
    elif difference == "capital":
        metadata["execution_context"]["init_cash"] = 200.0
    elif difference == "scope":
        scope = scope.model_copy(update={"split": "test"})
    scope = scope.model_copy(update={"metadata": metadata})
    _publish(storage, "second", scopes={"full": scope}, snapshot_changes=changes)
    result = storage[2].compare_runs(["first", "second"], metric_ids=["total_pnl"])
    comparison = CompareRunsResponse.model_validate(result)
    assert not comparison.comparable
    assert comparison.metric_compatibility["total_pnl"].reasons
    assert comparison.runs[0].metrics == {"total_pnl": 11.0}


def test_comparison_preserves_observed_interval_and_sample_count_differences(
    storage: tuple[WorkspacePaths, SQLiteStore, RunService],
) -> None:
    _publish(storage, "first")
    scope = _scope()
    metadata = scope.model_dump(mode="json")["metadata"]
    metadata["observations"]["bar_count"] = 500
    metadata["observations"]["end_date"] = "2024-03-01"
    _publish(
        storage,
        "second",
        scopes={"full": scope.model_copy(update={"metadata": metadata})},
    )
    comparison = storage[2].compare_runs(["first", "second"], metric_ids=["total_pnl"])
    assert comparison["context_differences"]["observations.bar_count"] == {
        "first": 2,
        "second": 500,
    }
    assert (
        comparison["context_differences"]["observations.end_date"]["second"]
        == "2024-03-01"
    )


def test_completed_job_returns_saved_summary_and_selected_train_scope(
    storage: tuple[WorkspacePaths, SQLiteStore, RunService],
) -> None:
    workspace, store, _ = storage
    jobs = JobStore(workspace=workspace, store=store)
    jobs.create_job(job_id="job", strategy_name="fixture", job_type="optimization")
    _publish(
        storage,
        "job",
        scopes={
            "trial/0/train": _scope(
                split="train", trial_index=0, parameters={"period": 4}
            ),
            "test": _scope(split="test"),
        },
        default_scope="test",
        selected_train_scope="trial/0/train",
        projection={
            "best_params": {"period": 4},
            "optimization_metric": "total_pnl",
            "direction": "maximize",
            "total_combinations": 1,
        },
    )
    jobs.update_job("job", status="done")
    status = JobStatus.model_validate(JobService(job_store=jobs).get_job_status("job"))
    assert status.metrics == {"total_pnl": 11.0, "win_rate": None}
    assert status.train_metrics == status.test_metrics == status.metrics
    assert status.best_params == {"period": 4}
    assert status.selected_train_scope == "trial/0/train"
    assert (
        status.available_metric_ids is not None
        and "daily_pnl_mean" in status.available_metric_ids
    )
    assert status.details_available


def test_read_tools_dispatch_catalog_and_recorded_detail_with_declared_schema(
    storage: tuple[WorkspacePaths, SQLiteStore, RunService],
) -> None:
    _publish(storage, "run")
    server = MCPServer("metric-query-test")
    runs.register(server, storage[2])

    async def check() -> None:
        catalog = await server.call_tool("get_metric_catalog", {"mode": "volume"})
        assert isinstance(catalog, CallToolResult)
        assert catalog.structured_content is not None
        typed = MetricCatalogResponse.model_validate(
            catalog.structured_content["result"]
        )
        assert "max_drawdown_amount" in {
            definition.id for definition in typed.definitions
        }
        assert "sharpe_ratio" not in {definition.id for definition in typed.definitions}
        descriptions = {
            definition.id: definition.description for definition in typed.definitions
        }
        assert "calendar daily periods" in descriptions["daily_pnl_mean"]
        assert "calendar monthly periods" in descriptions["monthly_pnl_mean"]
        result = await server.call_tool(
            "get_run_metrics",
            {"run_id": "run", "metric_ids": ["daily_pnl_mean", "total_volume"]},
        )
        assert isinstance(result, CallToolResult)
        assert result.structured_content is not None
        detail = RunMetricsResponse.model_validate(result.structured_content["result"])
        assert detail.metrics == {"daily_pnl_mean": 5.5, "total_volume": 210.0}
        invalid = await server.call_tool(
            "get_run_metrics", {"run_id": "run", "scope": "absent"}
        )
        assert isinstance(invalid, CallToolResult)
        assert invalid.structured_content is not None
        assert MetricQueryError.model_validate(
            invalid.structured_content["result"]
        ).available_scopes == ["full"]

    asyncio.run(check())


def test_unavailable_values_cannot_be_labeled_comparable(
    storage: tuple[WorkspacePaths, SQLiteStore, RunService],
) -> None:
    _publish(storage, "first")
    _publish(storage, "second")
    response = storage[2].compare_runs(["first", "second"], metric_ids=["win_rate"])
    assert not response["comparable"]
    assert (
        "metric_unavailable:first"
        in response["metric_compatibility"]["win_rate"]["reasons"]
    )
    summary = _scope(
        kind="fold_summary",
        split="test",
        values={
            "win_rate": FoldStats(
                mean=None, std=None, min=None, max=None, valid_count=0
            )
        },
    )
    _publish(
        storage,
        "fold_a",
        scopes={"test_summary": summary},
        default_scope="test_summary",
    )
    _publish(
        storage,
        "fold_b",
        scopes={"test_summary": summary},
        default_scope="test_summary",
    )
    response = storage[2].compare_runs(["fold_a", "fold_b"], metric_ids=["win_rate"])
    assert not response["comparable"]
    assert (
        "no_valid_folds:fold_a"
        in response["metric_compatibility"]["win_rate"]["reasons"]
    )


@pytest.mark.parametrize("walk_forward", [False, True], ids=["full", "fold-summary"])
def test_weekly_engine_results_keep_unavailable_metrics_through_storage_and_mcp(
    storage: tuple[WorkspacePaths, SQLiteStore, RunService], walk_forward: bool
) -> None:
    workspace, store, service = storage
    prices = [100.0, 100.0, 110.0, 121.0, 110.0, 115.0, 130.0, 130.0]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="7D", tz="UTC"),
            "open": prices,
            "close": prices,
            "signal": [1, 1, 1, 1, 1, 0, 0, 0],
        }
    )
    result = SignalBacktestEngine(
        init_cash=100.0, fees=0.0, slippage=0.0, freq="7D", periods_per_year=365
    ).run(frame)
    config: dict[str, Any] = {
        "strategy": {"id": "weekly_fixture"},
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": "1w",
            "start_date": "2024-01-01",
            "end_date": "2024-02-19",
            "init_cash": 100.0,
            "fees": 0.0,
            "slippage": 0.0,
            "periods_per_year": 365,
        },
    }
    if walk_forward:
        config["validation"] = {"target_metric": "total_return"}
    manifest = ExecutionManifest.create(
        kind="walk_forward" if walk_forward else "backtest",
        config=config,
        strategy_execution=StrategyExecution(kind="generated", constructor_kwargs={}),
    )
    pipeline = PipelineResult(
        mode="walk_forward" if walk_forward else "simple",
        backtest_result=None if walk_forward else result,
        config_snapshot=manifest.config_copy(),
        execution_manifest=manifest,
    )
    if walk_forward:
        start, end = datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 19, tzinfo=UTC)
        pipeline.fold_results = [
            WalkForwardResult(
                fold_index=index,
                train_start=start,
                train_end=end,
                test_start=start,
                test_end=end,
                train_metrics=result.metrics,
                test_metrics=result.metrics,
                train_backtest=result,
                test_backtest=result,
            )
            for index in range(2)
        ]
    projection = (
        summarize_results(pipeline.fold_results) if walk_forward else result.metrics
    )
    jobs = JobStore(workspace=workspace, store=store)
    for run_id in ("weekly_a", "weekly_b"):
        jobs.create_job(
            job_id=run_id, strategy_name="weekly_fixture", manifest=manifest
        )
        jobs.save_result(run_id, projection, pipeline=pipeline)

    persisted = store.get_run("weekly_a")
    assert persisted is not None and persisted["metrics"] == projection
    summary = service.get_run("weekly_a")["run"]["metrics"]
    expected_missing = (
        {"mean": None, "std": None, "min": None, "max": None, "valid_count": 0}
        if walk_forward
        else None
    )
    assert summary["annual_return"] == expected_missing
    assert summary["sharpe_ratio"] == expected_missing
    assert "daily_pnl_mean" not in summary
    assert (
        summary["total_return"]["mean"] if walk_forward else summary["total_return"]
    ) == pytest.approx(0.3)

    server = MCPServer("weekly-metric-query-test")
    runs.register(server, service)
    metric_ids = [
        "annual_return",
        "sharpe_ratio",
        "daily_pnl_mean",
        "monthly_pnl_mean",
        "total_return",
    ]

    async def check() -> None:
        response = await server.call_tool(
            "get_run_metrics", {"run_id": "weekly_a", "metric_ids": metric_ids}
        )
        assert isinstance(response, CallToolResult)
        assert response.structured_content is not None
        detail = RunMetricsResponse.model_validate(
            response.structured_content["result"]
        )
        for metric_id in metric_ids[:-1]:
            assert detail.metrics[metric_id] == expected_missing
        scalar_response = await server.call_tool(
            "get_run_metrics",
            {
                "run_id": "weekly_a",
                "scope": "fold/0/test" if walk_forward else "full",
                "metric_ids": metric_ids,
            },
        )
        assert isinstance(scalar_response, CallToolResult)
        assert scalar_response.structured_content is not None
        scalar = RunMetricsResponse.model_validate(
            scalar_response.structured_content["result"]
        )
        unavailable = scalar.metadata["unavailable"]
        assert isinstance(unavailable, dict)
        for metric_id in metric_ids[:-1]:
            assert scalar.metrics[metric_id] is None
            assert unavailable[metric_id] == "unsupported_daily_sampling"
        assert scalar.metrics["total_return"] == pytest.approx(0.3)
        comparison_response = await server.call_tool(
            "compare_runs",
            {"run_ids": ["weekly_a", "weekly_b"], "metric_ids": ["sharpe_ratio"]},
        )
        assert isinstance(comparison_response, CallToolResult)
        assert comparison_response.structured_content is not None
        comparison = CompareRunsResponse.model_validate(
            comparison_response.structured_content["result"]
        )
        assert not comparison.comparable
        reason = "no_valid_folds" if walk_forward else "metric_unavailable"
        assert (
            f"{reason}:weekly_a"
            in comparison.metric_compatibility["sharpe_ratio"].reasons
        )

    asyncio.run(check())
