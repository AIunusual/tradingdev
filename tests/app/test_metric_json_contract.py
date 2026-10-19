"""Regression coverage for finite nested metrics across JSON storage boundaries."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.artifact_service import ArtifactService
from tradingdev.app.job_store import JobStore
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.result import BacktestResult
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.domain.validation.report import summarize_results
from tradingdev.domain.validation.walk_forward import WalkForwardResult
from tradingdev.shared.utils.json_values import normalize_json_value

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


def _undefined_walk_forward_fold() -> WalkForwardResult:
    moment = datetime(2024, 1, 1, tzinfo=UTC)
    result = BacktestResult(
        metrics={"sharpe_ratio": None},
        equity_curve=np.array([100.0]),
        init_cash=100.0,
    )
    return WalkForwardResult(
        fold_index=0,
        train_start=moment,
        train_end=moment,
        test_start=moment,
        test_end=moment,
        test_metrics={"sharpe_ratio": float("inf")},
        train_backtest=result,
        test_backtest=result,
    )


def _undefined_walk_forward_summary() -> dict[str, object]:
    return summarize_results([_undefined_walk_forward_fold()])


def _reject_nonfinite(constant: str) -> object:
    raise AssertionError(f"Non-standard JSON numeric constant: {constant}")


@pytest.mark.parametrize(
    ("value", "expected", "error"),
    [
        (
            {"nested": [np.int64(12), np.float32(0.5), np.bool_(False), "012", None]},
            {"nested": [12, 0.5, False, "012", None]},
            None,
        ),
        (
            {"nested": [{"values": (float("nan"), np.float64("inf"), -float("inf"))}]},
            {"nested": [{"values": [None, None, None]}]},
            None,
        ),
        ({"nested": [object()]}, None, "Unsupported JSON value type"),
        ({"nested": [np.array([1, 2])]}, None, "Unsupported JSON value type"),
        ({"nested": {1: "numeric", "1": "text"}}, None, "keys must be strings"),
    ],
)
def test_json_values_preserve_types_or_reject_lossy_conversion(
    value: object, expected: object, error: str | None
) -> None:
    if error:
        with pytest.raises(TypeError, match=error):
            normalize_json_value(value)
        return
    normalized = normalize_json_value(value)
    assert normalized == expected
    assert normalized is not value
    assert json.dumps(normalized, allow_nan=False) == json.dumps(
        expected, allow_nan=False
    )


@pytest.mark.parametrize("writer", ["job", "sqlite", "cli"])
def test_nested_walk_forward_metrics_are_standard_json_across_writers(
    tmp_path: Path, monkeypatch: MonkeyPatch, writer: str
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    jobs = JobStore(workspace=workspace, store=store)
    metrics = _undefined_walk_forward_summary()
    original_stats = metrics["sharpe_ratio"]
    assert isinstance(original_stats, dict)
    assert original_stats["mean"] is None
    expected = {
        "n_folds": 1,
        "sharpe_ratio": {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "valid_count": 0,
        },
    }
    if writer == "job":
        jobs.create_job(job_id="walk_forward", strategy_name="fixture")
        path = jobs.save_result("walk_forward", metrics)
        assert (
            json.loads(
                path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite
            )
            == expected
        )
        assert jobs.load_result(str(path)) == expected
    elif writer == "sqlite":
        store.create_run(
            run_id="walk_forward",
            job_id="walk_forward",
            strategy_id="fixture",
            artifact_dir=tmp_path,
            metrics=metrics,
        )
    else:
        monkeypatch.setenv("TRADINGDEV_DATA_ROOT", str(workspace.root / "data"))
        monkeypatch.setattr("tradingdev.shared.utils.cache.CACHE_DIR", None)

        def cache_key(
            *,
            manifest_hash: str,
            processed_path: Path,
            code_fingerprint: str,
        ) -> str:
            return "metric-contract"

        monkeypatch.setattr(
            "tradingdev.app.artifact_service.compute_cache_key", cache_key
        )
        config_path = tmp_path / "strategy.yaml"
        config_path.write_text("strategy:\n  id: cli_fixture\n", encoding="utf-8")
        processed_path = tmp_path / "data.parquet"
        processed_path.write_bytes(b"cache identity fixture")
        manifest = ExecutionManifest.create(
            kind="walk_forward",
            strategy_execution=StrategyExecution(
                kind="generated", constructor_kwargs={}
            ),
            config={
                "strategy": {"id": "cli_fixture"},
                "backtest": {
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "start_date": "2024-01-01",
                    "end_date": "2024-02-01",
                    "init_cash": 10000,
                },
                "validation": {},
            },
        )
        ArtifactService(workspace=workspace, store=store).cache_pipeline_result(
            pipeline=PipelineResult(
                mode="walk_forward",
                fold_results=[_undefined_walk_forward_fold()],
                config_snapshot=manifest.config_copy(),
                execution_manifest=manifest,
            ),
            config_path=config_path,
            processed_path=processed_path,
            metrics=metrics,
            strategy_id="cli_fixture",
        )
    run_id = store.list_runs()[0]["run_id"] if writer == "cli" else "walk_forward"
    if writer == "cli":
        assert run_id.startswith("cli_metric-contract_")
    run = store.get_run(run_id)
    assert run is not None
    assert run["metrics"] == expected
    assert store.list_runs()[0]["metrics"] == expected
    with store.connect() as connection:
        row = connection.execute(
            "select metrics from runs where run_id = ?", (run_id,)
        ).fetchone()
    assert row is not None
    assert json.loads(row["metrics"], parse_constant=_reject_nonfinite) == expected
    json.dumps(run, allow_nan=False)
    assert original_stats["mean"] is None


@pytest.mark.parametrize("storage", ["file", "sqlite"])
def test_legacy_metrics_are_normalized_without_rewriting(
    tmp_path: Path, storage: str
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    original = '{"folds": [{"mean": NaN, "limits": [Infinity, -Infinity]}]}'
    expected = {"folds": [{"mean": None, "limits": [None, None]}]}
    if storage == "file":
        jobs = JobStore(workspace=workspace, store=store)
        path = tmp_path / "legacy-result.json"
        path.write_bytes(original.encode())
        assert jobs.load_result(str(path)) == expected
        assert path.read_bytes() == original.encode()
        assert jobs.load_result(str(tmp_path / "missing.json")) is None
    else:
        store.create_run(
            run_id="legacy",
            job_id="legacy",
            strategy_id="fixture",
            artifact_dir=tmp_path,
            metrics={},
        )
        with store.connect() as connection:
            connection.execute(
                "update runs set metrics = ? where run_id = ?", (original, "legacy")
            )
        run = store.get_run("legacy")
        assert run is not None
        assert run["metrics"] == expected
        assert store.list_runs()[0]["metrics"] == expected
        json.dumps(run, allow_nan=False)
        with store.connect() as connection:
            row = connection.execute(
                "select metrics from runs where run_id = ?", ("legacy",)
            ).fetchone()
        assert row is not None
        assert row["metrics"] == original


@pytest.mark.parametrize("writer", ["job", "sqlite"])
def test_invalid_metrics_leave_no_result_artifacts_or_run_records(
    tmp_path: Path, writer: str
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    jobs = JobStore(workspace=workspace, store=store)
    jobs.create_job(job_id="invalid", strategy_name="fixture")

    with pytest.raises(TypeError, match="Unsupported JSON value type"):
        if writer == "job":
            jobs.save_result("invalid", {"nested": {"bad": object()}})
        else:
            store.create_run(
                run_id="invalid",
                job_id="invalid",
                strategy_id="fixture",
                artifact_dir=tmp_path,
                metrics={"nested": {"bad": object()}},
            )
    assert not (workspace.runs / "invalid").exists()
    assert store.get_run("invalid") is None
    assert store.list_artifacts("invalid") == []
