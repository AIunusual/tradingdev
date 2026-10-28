"""Persisted execution identity stays fixed across jobs, results and old records."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.job_store import JobStore
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.execution import ExecutionManifest, ManifestError
from tradingdev.domain.strategies.execution import StrategyExecution

if TYPE_CHECKING:
    from pathlib import Path


def _manifest(*, fees: float = 0.0006) -> ExecutionManifest:
    return ExecutionManifest.create(
        strategy_execution=StrategyExecution(kind="generated", constructor_kwargs={}),
        kind="backtest",
        config={
            "strategy": {"id": "fixture"},
            "backtest": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-07",
                "init_cash": 10000,
                "fees": fees,
            },
        },
    )


def _submitted(tmp_path: Path) -> tuple[JobStore, ExecutionManifest]:
    store = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    manifest = _manifest()
    store.create_job(job_id="submitted", manifest=manifest)
    return store, manifest


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("manifest_hash", "a" * 64),
        ("strategy_name", "other"),
        ("revision_id", "other-revision"),
        ("job_type", "optimization"),
    ],
)
def test_submitted_job_identity_cannot_be_changed_by_progress_updates(
    tmp_path: Path, field: str, replacement: str
) -> None:
    store, manifest = _submitted(tmp_path)
    original = store.get_job("submitted")
    with pytest.raises(ManifestError, match="execution identity"):
        store.update_job("submitted", **{field: replacement})
    assert store.get_job("submitted") == original
    assert store.load_manifest("submitted") == manifest
    store.update_job("submitted", status="running_backtest", data_downloaded=True)
    assert store.load_manifest("submitted") == manifest


@pytest.mark.parametrize(
    "field", ["job_id", "manifest_hash", "strategy_name", "revision_id", "job_type"]
)
def test_extra_payload_cannot_replace_identity_before_publication(
    tmp_path: Path, field: str
) -> None:
    store = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    with pytest.raises(ManifestError, match="execution identity"):
        store.create_job(
            job_id="submitted", manifest=_manifest(), extra_payload={field: "other"}
        )
    assert store.list_all_jobs() == []
    assert not (store.workspace.runs / "submitted" / "manifest.json").exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("strategy_name", "other"),
        ("revision_id", "other-revision"),
        ("job_type", "optimization"),
    ],
)
def test_load_rejects_persisted_job_identity_that_disagrees_with_manifest(
    tmp_path: Path, field: str, replacement: str
) -> None:
    store, _ = _submitted(tmp_path)
    record = store.get_job("submitted")
    assert record is not None
    record[field] = replacement
    store.store.upsert_job(record)
    with pytest.raises(ManifestError, match="identity differ"):
        store.load_manifest("submitted")


def test_a_valid_replacement_manifest_cannot_change_an_accepted_job(
    tmp_path: Path,
) -> None:
    store, _ = _submitted(tmp_path)
    replacement = _manifest(fees=0.02)
    path = store.workspace.runs / "submitted" / "manifest.json"
    path.write_text(replacement.model_dump_json(), encoding="utf-8")
    with pytest.raises(ManifestError, match="expected hash"):
        store.load_manifest("submitted")
    with pytest.raises(ManifestError, match="expected hash"):
        store.save_result("submitted", {}, execution_manifest=replacement)
    assert store.get_run("submitted") is None
    assert not path.with_name("result.json").exists()


def test_result_uses_held_manifest_when_config_projection_was_changed(
    tmp_path: Path,
) -> None:
    store, manifest = _submitted(tmp_path)
    run_dir = store.workspace.runs / "submitted"
    manifest_bytes = (run_dir / "manifest.json").read_bytes()
    projection = run_dir / "config.yaml"
    projection.write_text("invalid: [\n", encoding="utf-8")

    result = store.save_result(
        "submitted", {"total_return": 0.25}, execution_manifest=manifest
    )

    assert json.loads(result.read_text()) == {"total_return": 0.25}
    assert yaml.safe_load(projection.read_text()) == manifest.config_copy()
    assert (run_dir / "manifest.json").read_bytes() == manifest_bytes
    run = store.get_run("submitted")
    assert run is not None
    assert run["manifest_hash"] == manifest.manifest_hash
    assert run["config_hash"] == sha256_file(projection)
    artifacts = {
        item["artifact_type"]: item for item in store.list_artifacts("submitted")
    }
    assert artifacts["execution_manifest"]["metadata"]["manifest_hash"] == (
        manifest.manifest_hash
    )
    assert artifacts["execution_manifest"]["sha256"] == sha256_file(
        run_dir / "manifest.json"
    )


@pytest.mark.parametrize(
    "mismatch", ["missing", "manifest", "snapshot", "pipeline", "persisted"]
)
def test_unbound_or_mismatched_results_are_rejected_without_partial_output(
    tmp_path: Path, mismatch: str
) -> None:
    store, manifest = _submitted(tmp_path)
    run_dir = store.workspace.runs / "submitted"
    kwargs: dict[str, Any] = {"execution_manifest": manifest}
    if mismatch == "missing":
        kwargs = {}
    elif mismatch == "manifest":
        kwargs["execution_manifest"] = _manifest(fees=0.02)
    elif mismatch in {"snapshot", "pipeline"}:
        changed = manifest.config_copy()
        changed["backtest"]["fees"] = 0.02
        if mismatch == "snapshot":
            kwargs["config_snapshot"] = changed
        else:
            kwargs = {
                "pipeline": PipelineResult(
                    mode="simple", config_snapshot=changed, execution_manifest=manifest
                )
            }
    else:
        (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ManifestError):
        store.save_result("submitted", {"total_return": 0.25}, **kwargs)

    assert store.get_run("submitted") is None
    assert store.list_artifacts("submitted") == []
    assert not (run_dir / "result.json").exists()


def test_legacy_result_remains_readable_without_becoming_resumable(
    tmp_path: Path,
) -> None:
    store = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    store.create_job(job_id="legacy", strategy_name="fixture")
    store.update_job("legacy", status="done")
    run_dir = store.workspace.runs / "legacy"
    run_dir.mkdir(exist_ok=True)
    result_path = run_dir / "result.json"
    original = '{"profit_factor": Infinity, "nested": {"value": NaN}}'
    result_path.write_text(original, encoding="utf-8")
    store.store.create_run(
        run_id="legacy",
        job_id="legacy",
        strategy_id="fixture",
        artifact_dir=run_dir,
        metrics={"profit_factor": float("inf")},
    )

    with pytest.raises(ManifestError, match="submit a new job"):
        store.load_manifest("legacy")

    run = store.get_run("legacy")
    assert run is not None and run["manifest_hash"] is None
    assert run["metrics"] == {"profit_factor": None}
    assert store.load_result(str(result_path)) == {
        "profit_factor": None,
        "nested": {"value": None},
    }
    assert result_path.read_text() == original
    assert not (run_dir / "manifest.json").exists()


def test_migration_adds_nullable_manifest_hash_to_existing_runs(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    workspace.ensure()
    with sqlite3.connect(workspace.root / "tradingdev.sqlite") as conn:
        conn.executescript("""
            create table runs (
                run_id text primary key,
                job_id text not null,
                strategy_id text not null,
                revision_id text,
                config_hash text,
                source_hash text,
                random_seed integer,
                dataset_id text,
                metrics text,
                artifact_dir text not null,
                created_at text not null
            );
            insert into runs values (
                'legacy', 'legacy', 'fixture', 'revision-before-manifests',
                'original-config-hash', null, null, null,
                '{"total_return": 0.5}', '/legacy', '2024-01-01'
            );
        """)

    store = SQLiteStore(workspace)
    legacy = store.get_run("legacy")
    assert legacy is not None
    assert legacy["manifest_hash"] is None
    assert legacy["revision_id"] == "revision-before-manifests"
    assert legacy["config_hash"] == "original-config-hash"
    assert legacy["metrics"] == {"total_return": 0.5}
    store.create_run(
        run_id="new",
        job_id="new",
        strategy_id="fixture",
        artifact_dir=workspace.runs / "new",
        metrics={},
        manifest_hash=_manifest().manifest_hash,
    )
    assert (store.get_run("new") or {})["manifest_hash"] == _manifest().manifest_hash
