"""Typed job/run store facade backed by workspace SQLite metadata."""

from __future__ import annotations

import json
import logging
import re
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict

from tradingdev.adapters.storage.comparisons import ComparisonStore
from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore, get_sqlite_store
from tradingdev.app.run_lineage import (
    load_config_payload,
    read_strategy_snapshot,
)
from tradingdev.app.run_publisher import RunPublicationBinding, RunPublisher
from tradingdev.domain.comparison import ComparisonManifest
from tradingdev.domain.comparison_results import ComparisonLedger
from tradingdev.domain.execution import ExecutionManifest, ManifestError
from tradingdev.shared.utils.json_values import normalize_json_object

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from tradingdev.domain.backtest.pipeline_result import PipelineResult
    from tradingdev.domain.execution_target import ExecutionTarget
    from tradingdev.domain.performance.artifacts import PerformanceArtifacts

logger = logging.getLogger(__name__)

ACTIVE_JOB_STATUSES = {
    "queued",
    "downloading_data",
    "running_backtest",
    "estimating",
    "optimizing",
    "testing_oos",
}
TERMINAL_JOB_STATUSES = {"done", "failed", "cancelled", "estimation_timeout"}


class JobRecord(BaseModel):
    """Typed job record persisted in SQLite."""

    model_config = ConfigDict(extra="allow")

    job_id: str
    status: str
    strategy_name: str | None = None
    revision_id: str | None = None
    manifest_hash: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    config_path: str | None = None
    job_type: str = "backtest"
    pid: int | None = None
    process_create_time: float | None = None
    worker_control_id: str | None = None
    created_at: str
    started_at: str | None = None
    ended_at: str | None = None
    data_downloaded: bool = False
    result_path: str | None = None
    error: str | None = None


class JobStore:
    """Application-level facade for job, run, and artifact metadata."""

    def __init__(
        self,
        *,
        workspace: WorkspacePaths | None = None,
        store: SQLiteStore | None = None,
    ) -> None:
        if workspace is not None:
            self._workspace = workspace
        elif store is not None:
            self._workspace = WorkspacePaths(store.db_path.parent)
        else:
            self._workspace = WorkspacePaths()
        self._workspace.ensure()
        self._store = store or get_sqlite_store(self._workspace)
        self._manifests = ExecutionManifestStore(self._workspace)

    @property
    def workspace(self) -> WorkspacePaths:
        """Return the runtime workspace used by this facade."""
        return self._workspace

    @property
    def store(self) -> SQLiteStore:
        """Return the backing SQLite adapter."""
        return self._store

    def create_job(
        self,
        *,
        job_id: str,
        strategy_name: str | None = None,
        revision_id: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        config_path: str | None = None,
        job_type: str = "backtest",
        extra_payload: dict[str, Any] | None = None,
        manifest: ExecutionTarget | None = None,
    ) -> dict[str, Any]:
        """Create a new job record and persist it."""
        with self.submission_lock(job_id), ExitStack() as reservations:
            self._check_comparison_reservation(job_id)
            if self._store.get_job(job_id) is not None:
                raise ValueError(f"Job already exists: {job_id}")
            if self._store.get_run(job_id) is not None:
                raise ValueError(f"Job ID belongs to an existing run: {job_id}")
            if extra_payload is not None and "job_id" in extra_payload:
                raise ManifestError("Extra payload cannot replace execution identity")
            result_path = self._workspace.runs / job_id / "result.json"
            if manifest is not None:
                if extra_payload is not None and any(
                    key in extra_payload
                    for key in (
                        "job_id",
                        "manifest_hash",
                        "strategy_name",
                        "revision_id",
                        "job_type",
                    )
                ):
                    raise ManifestError(
                        "Extra payload cannot replace execution identity"
                    )
                if isinstance(manifest, ComparisonManifest):
                    for row in ComparisonLedger.create(job_id, manifest).candidates:
                        reservations.enter_context(self.submission_lock(row.run_id))
                        if (
                            self._store.get_job(row.run_id) is not None
                            or self._store.get_run(row.run_id) is not None
                        ):
                            raise ManifestError("Comparison child ID is already in use")
                    config = manifest.common_config()
                else:
                    config = manifest.config_copy()
                strategy = config["strategy"]
                if strategy_name is not None and strategy_name != strategy.get("id"):
                    raise ManifestError(
                        "Job and manifest refer to different strategies"
                    )
                if revision_id is not None and revision_id != strategy.get(
                    "revision_id"
                ):
                    raise ManifestError("Job and manifest refer to different revisions")
                strategy_name = strategy["id"]
                revision_id = strategy.get("revision_id")
                job_type = manifest.kind
                symbol = config["backtest"]["symbol"]
                timeframe = config["backtest"]["timeframe"]
                start_date = config["backtest"]["start_date"]
                end_date = config["backtest"]["end_date"]
                if (
                    isinstance(manifest, ExecutionManifest)
                    and manifest.optimization is not None
                ):
                    start_date = manifest.optimization.train_start.isoformat()
                    end_date = manifest.optimization.test_end.isoformat()
                if isinstance(manifest, ComparisonManifest):
                    comparisons = ComparisonStore(self._workspace)
                    comparisons.initialize(job_id, manifest)
                    path = comparisons.path(job_id)
                    result_path = comparisons.path(job_id, "ledger.json")
                else:
                    path = self._manifests.publish(job_id, manifest)
                projection = path.with_name("config.yaml")
                projection.write_text(
                    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                config_path = str(projection)
            record = JobRecord(
                job_id=job_id,
                status="queued",
                job_type=job_type,
                strategy_name=strategy_name,
                revision_id=revision_id,
                manifest_hash=manifest.manifest_hash if manifest is not None else None,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                config_path=config_path,
                created_at=_now_iso(),
                result_path=str(result_path),
            )
            record_payload = _record_to_dict(record)
            if extra_payload is not None:
                record_payload.update(extra_payload)
            self._store.upsert_job(record_payload)
            logger.debug("Created job %s", job_id)
            return record_payload

    def submission_lock(self, job_id: str) -> AbstractContextManager[None]:
        """Serialize one job's creation and launch through the storage adapter."""
        return self._manifests.submission_lock(job_id)

    def _check_comparison_reservation(self, identifier: str) -> None:
        """A submitted comparison owns its deterministic child IDs before execution."""
        match = re.fullmatch(r"(.+)_candidate_([1-8])", identifier)
        if match is None:
            return
        parent_id = match[1]
        parent = self.get_job(parent_id)
        if parent is None or parent.get("job_type") != "comparison":
            return
        ledger = ComparisonStore(self._workspace).load_ledger(
            parent_id, expected_hash=str(parent["manifest_hash"])
        )
        if any(row.run_id == identifier for row in ledger.candidates):
            raise ManifestError("ID is reserved by a comparison candidate")

    def load_manifest(self, job_id: str) -> ExecutionTarget:
        """Read a job's pinned specification; legacy jobs cannot resume execution."""
        job = self.get_job(job_id)
        if job is None or not isinstance(job.get("manifest_hash"), str):
            raise ManifestError("Job has no execution manifest; submit a new job")
        if job.get("job_type") == "comparison":
            comparison = ComparisonStore(self._workspace).load_manifest(
                job_id, expected_hash=job["manifest_hash"]
            )
            strategy = comparison.common_config()["strategy"]
            manifest: ExecutionTarget = comparison
        else:
            manifest = self._manifests.load(job_id, expected_hash=job["manifest_hash"])
            strategy = manifest.config_copy()["strategy"]
        if (
            manifest.kind != job.get("job_type")
            or strategy.get("id") != job.get("strategy_name")
            or strategy.get("revision_id") != job.get("revision_id")
        ):
            raise ManifestError("Job and execution manifest identity differ")
        return manifest

    def update_job(
        self,
        job_id: str,
        **fields: Any,
    ) -> dict[str, Any] | None:
        """Atomically merge fields, preserving the first persisted terminal state."""

        def update(current: dict[str, Any]) -> dict[str, Any]:
            if "job_id" in fields and fields["job_id"] != current["job_id"]:
                raise ManifestError("Cannot change a job's execution identity: job_id")
            if current.get("manifest_hash") is not None:
                for key in (
                    "manifest_hash",
                    "strategy_name",
                    "revision_id",
                    "job_type",
                ):
                    if key in fields and fields[key] != current.get(key):
                        raise ManifestError(
                            f"Cannot change a job's execution identity: {key}"
                        )
            patch = dict(fields)
            status = patch.get("status")
            if (
                status is not None
                and status not in ACTIVE_JOB_STATUSES | TERMINAL_JOB_STATUSES
            ):
                raise ValueError(f"Unknown job status: {status}")
            if current["status"] not in ACTIVE_JOB_STATUSES:
                if status is not None:
                    return current
                for key in ("started_at", "ended_at", "error"):
                    patch.pop(key, None)
            if (
                status in ACTIVE_JOB_STATUSES - {"queued"}
                and current.get("started_at") is None
            ):
                patch.setdefault("started_at", _now_iso())
            if status in TERMINAL_JOB_STATUSES:
                patch.setdefault("ended_at", _now_iso())
            current.update(patch)
            return _record_to_dict(JobRecord.model_validate(current))

        return self._store.mutate_job(job_id, update)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return a single job record, or None if not found."""
        return self._store.get_job(job_id)

    def list_all_jobs(self) -> list[dict[str, Any]]:
        """Return all job records sorted by creation time descending."""
        return self._store.list_jobs()

    def save_result(
        self,
        job_id: str,
        metrics: dict[str, Any],
        *,
        pipeline: PipelineResult | None = None,
        performance: PerformanceArtifacts | None = None,
        config_snapshot: dict[str, Any] | None = None,
        execution_manifest: ExecutionManifest | None = None,
    ) -> Path:
        """Serialize metrics to a run artifact and record run metadata."""
        with self.submission_lock(job_id):
            return self._save_result_locked(
                job_id,
                metrics,
                pipeline=pipeline,
                performance=performance,
                config_snapshot=config_snapshot,
                execution_manifest=execution_manifest,
            )

    def _save_result_locked(
        self,
        job_id: str,
        metrics: dict[str, Any],
        *,
        pipeline: PipelineResult | None,
        performance: PerformanceArtifacts | None,
        config_snapshot: dict[str, Any] | None,
        execution_manifest: ExecutionManifest | None,
    ) -> Path:
        if pipeline is not None and performance is not None:
            raise ValueError(
                "Supply either pipeline or performance artifacts, not both"
            )
        safe = normalize_json_object(metrics)

        job = self.get_job(job_id)
        self._check_comparison_reservation(job_id)
        if job is not None and job.get("job_type") == "comparison":
            raise ManifestError(
                "Comparison results must be published as candidate runs"
            )
        if job is None and self.get_run(job_id) is not None:
            raise ValueError(f"Result ID belongs to an existing run: {job_id}")
        config_path = _resolve_optional_path(job.get("config_path")) if job else None
        config_payload = (
            pipeline.config_snapshot
            if pipeline is not None
            else config_snapshot
            if config_snapshot is not None
            else load_config_payload(config_path)
            if config_path is not None and not (job or {}).get("manifest_hash")
            else None
        )
        executed = (
            getattr(pipeline, "execution_manifest", None)
            if pipeline is not None
            else execution_manifest
        )
        if executed is not None and not isinstance(executed, ExecutionManifest):
            raise ManifestError("Invalid executed manifest")
        expected: ExecutionManifest | None = None
        if job is not None and job.get("manifest_hash") is not None:
            if executed is None:
                raise ManifestError("Result must retain the executed manifest")
            executed.verify(expected_hash=job["manifest_hash"])
            loaded = self.load_manifest(job_id)
            if not isinstance(loaded, ExecutionManifest):
                raise ManifestError("A comparison cannot publish one parent result")
            expected = loaded
            if config_payload is not None and config_payload != executed.config_copy():
                raise ManifestError("Result config differs from the executed manifest")
            config_payload = executed.config_copy()
        elif executed is not None:
            raise ManifestError("Result manifest is not bound to a submitted job")
        if job is None:
            # Historical summary-only callers can save without a submitted job;
            # this does not create a run or authorize an executed manifest.
            if pipeline is not None or performance is not None:
                raise ValueError("Performance artifacts require an existing job")
            read_strategy_snapshot(config_payload, self._workspace, strategy_id="")
            result_path = self._workspace.runs / job_id / "result.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(safe, indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            return result_path
        result_path = RunPublisher(self._workspace, self._store).publish(
            RunPublicationBinding(
                job_id=job_id, run_id=job_id, expected_manifest=expected
            ),
            safe,
            pipeline=pipeline,
            performance=performance,
            config_snapshot=config_payload,
            execution_manifest=executed,
        )
        logger.debug("Saved result for job %s -> %s", job_id, result_path)
        return result_path

    def load_result(self, result_path: str) -> dict[str, Any] | None:
        """Load a cached result JSON. Returns None if file is missing."""
        path = Path(result_path)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return normalize_json_object(raw) if isinstance(raw, dict) else None

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return a completed run."""
        return self._store.get_run(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        """Return completed runs."""
        return self._store.list_runs()

    def list_artifacts(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """Return stored artifact metadata."""
        return self._store.list_artifacts(run_id)

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Return stored artifact metadata."""
        return self._store.get_artifact(artifact_id)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _record_to_dict(record: JobRecord) -> dict[str, Any]:
    return record.model_dump(mode="json")


def _resolve_optional_path(value: object) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def get_default_job_store() -> JobStore:
    """Return a default job store without import-time workspace side effects."""
    return JobStore()


def create_job(
    job_id: str,
    strategy_name: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    config_path: str | None = None,
    job_type: str = "backtest",
    extra_payload: dict[str, Any] | None = None,
    revision_id: str | None = None,
    manifest: ExecutionTarget | None = None,
) -> dict[str, Any]:
    """Create a new job record in the default store."""
    return get_default_job_store().create_job(
        job_id=job_id,
        strategy_name=strategy_name,
        revision_id=revision_id,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        config_path=config_path,
        job_type=job_type,
        extra_payload=extra_payload,
        manifest=manifest,
    )


def update_job(job_id: str, **fields: Any) -> None:
    """Update a job in the default store."""
    get_default_job_store().update_job(job_id, **fields)


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return a job from the default store."""
    return get_default_job_store().get_job(job_id)


def load_manifest(job_id: str) -> ExecutionTarget:
    """Load a pinned execution specification from the default workspace."""
    return get_default_job_store().load_manifest(job_id)


def list_all_jobs() -> list[dict[str, Any]]:
    """Return jobs from the default store."""
    return get_default_job_store().list_all_jobs()


def save_result(
    job_id: str,
    metrics: dict[str, Any],
    *,
    pipeline: PipelineResult | None = None,
    performance: PerformanceArtifacts | None = None,
    config_snapshot: dict[str, Any] | None = None,
    execution_manifest: ExecutionManifest | None = None,
) -> Path:
    """Persist a result in the default store."""
    return get_default_job_store().save_result(
        job_id,
        metrics,
        pipeline=pipeline,
        performance=performance,
        config_snapshot=config_snapshot,
        execution_manifest=execution_manifest,
    )


def load_result(result_path: str) -> dict[str, Any] | None:
    """Load a result from disk."""
    return get_default_job_store().load_result(result_path)


def get_run(run_id: str) -> dict[str, Any] | None:
    """Return a completed run from the default store."""
    return get_default_job_store().get_run(run_id)


def list_runs() -> list[dict[str, Any]]:
    """Return completed runs from the default store."""
    return get_default_job_store().list_runs()


def list_artifacts(run_id: str | None = None) -> list[dict[str, Any]]:
    """Return artifacts from the default store."""
    return get_default_job_store().list_artifacts(run_id)


def get_artifact(artifact_id: str) -> dict[str, Any] | None:
    """Return one artifact from the default store."""
    return get_default_job_store().get_artifact(artifact_id)
