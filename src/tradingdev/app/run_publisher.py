"""Publish verified run artifacts independently of their owning job's lifecycle."""

from __future__ import annotations

import json
import pickle
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import sha256_file, sha256_text
from tradingdev.adapters.storage.performance import (
    PerformanceArtifactError,
    PerformanceStore,
)
from tradingdev.app.run_lineage import (
    StrategySourceSnapshot,
    extract_random_seed,
    load_config_payload,
    read_strategy_snapshot,
)
from tradingdev.app.strategy_service import StrategyNotExecutableError
from tradingdev.domain.execution import ExecutionManifest, ManifestError
from tradingdev.domain.performance.artifacts import (
    PerformanceArtifacts,
    bundles_from_pipeline,
)
from tradingdev.shared.utils.json_values import normalize_json_object

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore
    from tradingdev.domain.backtest.pipeline_result import PipelineResult


@dataclass(frozen=True)
class RunPublicationBinding:
    """Expected identity supplied by the trusted submission owner.

    The caller must establish the job-to-run membership before constructing this
    binding. In particular, a child manifest must come from that job's accepted
    submission, never from the result being saved. This value does not establish
    membership by itself. Legacy jobs can publish only their original run ID.
    """

    job_id: str
    run_id: str
    expected_manifest: ExecutionManifest | None


@dataclass(frozen=True)
class _Publication:
    """One validated snapshot used for every file and metadata record."""

    binding: RunPublicationBinding
    job: dict[str, Any]
    metrics: dict[str, Any]
    manifest: ExecutionManifest | None
    config_path: Path | None
    config_content: str | None
    config_hash: str | None
    source: StrategySourceSnapshot
    random_seed: int | None
    dataset_fingerprint: dict[str, str]
    details: PerformanceArtifacts | None
    pipeline: PipelineResult | None


class RunPublisher:
    """Validate and publish one run under an explicitly supplied submission binding."""

    def __init__(self, workspace: WorkspacePaths, store: SQLiteStore) -> None:
        self._workspace = workspace
        self._store = store
        self._manifests = ExecutionManifestStore(workspace)

    def publish(
        self,
        binding: RunPublicationBinding,
        metrics: dict[str, Any],
        *,
        pipeline: PipelineResult | None = None,
        performance: PerformanceArtifacts | None = None,
        config_snapshot: dict[str, Any] | None = None,
        execution_manifest: ExecutionManifest | None = None,
    ) -> Path:
        """Persist a run without updating or completing its owning job."""
        # A child result and a new job share the same workspace ID namespace.
        # Keep the absence check and the complete publication under the same
        # lock used by both direct job creation and worker submission.
        reservation = (
            self._manifests.submission_lock(binding.run_id)
            if binding.run_id != binding.job_id
            else nullcontext()
        )
        with reservation:
            context = self._prepare(
                binding,
                metrics,
                pipeline=pipeline,
                performance=performance,
                config_snapshot=config_snapshot,
                execution_manifest=execution_manifest,
            )
            return self._publish(context)

    def _publish(self, context: _Publication) -> Path:
        binding = context.binding
        run_id = binding.run_id
        manifest_path = self._manifests.path(run_id)
        run_dir = manifest_path.parent
        result_path = run_dir / "result.json"
        performance_store = PerformanceStore(self._workspace, self._store)
        publication = (
            performance_store.publication(
                context.details,
                context.metrics,
                [
                    run_dir / name
                    for name in (
                        "result.json",
                        "manifest.json",
                        "config.yaml",
                        "strategy.py",
                        "dataset_fingerprint.json",
                        "pipeline_result.pkl",
                    )
                ],
            )
            if context.details is not None
            else nullcontext(False)
        )
        with publication:
            existing = self._store.get_run(run_id)
            if existing is not None:
                self._verify_existing(context, existing, run_dir)
                return result_path
            run_dir.mkdir(parents=True, exist_ok=True)
            if context.manifest is not None:
                self._manifests.publish(run_id, context.manifest)
            result_path.write_text(
                json.dumps(
                    context.metrics, indent=2, ensure_ascii=False, allow_nan=False
                ),
                encoding="utf-8",
            )
            self._store.create_run(
                run_id=run_id,
                job_id=binding.job_id,
                strategy_id=str(context.job.get("strategy_name", "")),
                revision_id=context.source.revision_id,
                manifest_hash=context.manifest.manifest_hash
                if context.manifest is not None
                else None,
                artifact_dir=run_dir,
                metrics=context.metrics,
                config_hash=context.config_hash,
                source_hash=context.source.source_hash,
                random_seed=context.random_seed,
                dataset_id=context.dataset_fingerprint["dataset_id"],
            )
            self._register(context, "result_json", result_path)
            if context.manifest is not None:
                self._register(
                    context,
                    "execution_manifest",
                    manifest_path,
                    {
                        "manifest_hash": context.manifest.manifest_hash,
                        "schema_version": context.manifest.schema_version,
                    },
                )
            if context.config_content is not None:
                snapshot_path = run_dir / "config.yaml"
                snapshot_path.write_bytes(context.config_content.encode("utf-8"))
                self._register(
                    context,
                    "config_snapshot",
                    snapshot_path,
                    {
                        "source_path": str(context.config_path),
                        "config_hash": context.config_hash,
                    },
                )
                if (
                    context.source.path is not None
                    and context.source.content is not None
                ):
                    strategy_snapshot = run_dir / "strategy.py"
                    strategy_snapshot.write_bytes(context.source.content)
                    self._register(
                        context,
                        "strategy_source",
                        strategy_snapshot,
                        {
                            "source_path": str(context.source.path),
                            "source_hash": context.source.source_hash,
                            "revision_id": context.source.revision_id,
                        },
                    )
            fingerprint_path = run_dir / "dataset_fingerprint.json"
            fingerprint_path.write_text(
                json.dumps(context.dataset_fingerprint, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self._register(
                context,
                "dataset_fingerprint",
                fingerprint_path,
                context.dataset_fingerprint,
            )
            if context.pipeline is not None:
                pipeline_path = run_dir / "pipeline_result.pkl"
                pipeline_path.write_bytes(pickle.dumps(context.pipeline))
                self._register(
                    context, "pipeline_result", pipeline_path, {"format": "pickle"}
                )
            if context.details is not None:
                performance_store.publish(context.details)
        return result_path

    def _prepare(
        self,
        binding: RunPublicationBinding,
        metrics: dict[str, Any],
        *,
        pipeline: PipelineResult | None,
        performance: PerformanceArtifacts | None,
        config_snapshot: dict[str, Any] | None,
        execution_manifest: ExecutionManifest | None,
    ) -> _Publication:
        if pipeline is not None and performance is not None:
            raise ValueError(
                "Supply either pipeline or performance artifacts, not both"
            )
        safe = normalize_json_object(metrics)
        job = self._store.get_job(binding.job_id)
        if job is None:
            raise ValueError("Run publication requires an existing job")
        config_path = Path(str(job["config_path"])) if job.get("config_path") else None
        if config_path is not None and not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        config_payload = (
            pipeline.config_snapshot
            if pipeline is not None
            else config_snapshot
            if config_snapshot is not None
            else load_config_payload(config_path)
            if config_path is not None and not job.get("manifest_hash")
            else None
        )
        executed = (
            pipeline.execution_manifest if pipeline is not None else execution_manifest
        )
        if executed is not None and not isinstance(executed, ExecutionManifest):
            raise ManifestError("Invalid executed manifest")
        expected = binding.expected_manifest
        if expected is not None:
            expected.verify()
            if executed is None:
                raise ManifestError("Result must retain the executed manifest")
            executed.verify(expected_hash=expected.manifest_hash)
            parent_hash = job.get("manifest_hash")
            if not isinstance(parent_hash, str):
                raise ManifestError("Result manifest is not bound to a submitted job")
            parent = self._manifests.load(binding.job_id, expected_hash=parent_hash)
            parent_strategy = parent.config_copy()["strategy"]
            strategy = expected.config_copy()["strategy"]
            if (
                parent.kind != job.get("job_type")
                or parent_strategy.get("id") != job.get("strategy_name")
                or parent_strategy.get("revision_id") != job.get("revision_id")
                or strategy.get("id") != job.get("strategy_name")
                or strategy.get("revision_id") != job.get("revision_id")
            ):
                raise ManifestError("Job and execution manifest identity differ")
            if binding.run_id == binding.job_id:
                expected.verify(expected_hash=parent_hash)
            elif self._store.get_job(binding.run_id) is not None:
                raise ManifestError("Child run ID belongs to an existing job")
            if config_payload is not None and config_payload != executed.config_copy():
                raise ManifestError("Result config differs from the executed manifest")
            config_payload = executed.config_copy()
        elif (
            executed is not None
            or job.get("manifest_hash") is not None
            or binding.run_id != binding.job_id
        ):
            raise ManifestError("Result manifest is not bound to a submitted job")
        details = (
            bundles_from_pipeline(binding.run_id, pipeline, safe)
            if pipeline is not None
            else PerformanceArtifacts.model_validate_json(performance.model_dump_json())
            if performance is not None
            else None
        )
        if details is not None and (
            details.performance.run_id != binding.run_id
            or details.performance.manifest_hash
            != (executed.manifest_hash if executed is not None else None)
        ):
            raise ManifestError("Performance artifacts differ from the executed result")
        if binding.run_id != binding.job_id and details is None:
            raise ValueError(
                "Child run publication requires complete performance artifacts"
            )
        source = read_strategy_snapshot(
            config_payload,
            self._workspace,
            strategy_id=str(job.get("strategy_name") or ""),
        )
        if job.get("revision_id") != source.revision_id:
            raise StrategyNotExecutableError(
                "Job and result configuration refer to different revisions"
            )
        config_content = (
            yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True)
            if config_payload is not None
            else None
        )
        return _Publication(
            binding=binding,
            job=job,
            metrics=safe,
            manifest=executed,
            config_path=config_path,
            config_content=config_content,
            config_hash=sha256_text(config_content)
            if config_content is not None
            else None,
            source=source,
            random_seed=extract_random_seed(config_payload),
            dataset_fingerprint=_dataset_fingerprint(job),
            details=details,
            pipeline=pipeline,
        )

    def _register(
        self,
        context: _Publication,
        artifact_type: str,
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._store.create_artifact(
            artifact_id=f"{context.binding.run_id}:{artifact_type}",
            run_id=context.binding.run_id,
            artifact_type=artifact_type,
            path=path,
            sha256=sha256_file(path),
            metadata={**(metadata or {}), "job_id": context.binding.job_id},
        )

    def _verify_existing(
        self, context: _Publication, existing: dict[str, Any], run_dir: Path
    ) -> None:
        expected = {
            "job_id": context.binding.job_id,
            "strategy_id": str(context.job.get("strategy_name", "")),
            "revision_id": context.source.revision_id,
            "manifest_hash": context.manifest.manifest_hash
            if context.manifest is not None
            else None,
            "artifact_dir": str(run_dir),
            "metrics": context.metrics,
            "config_hash": context.config_hash,
            "source_hash": context.source.source_hash,
            "random_seed": context.random_seed,
            "dataset_id": context.dataset_fingerprint["dataset_id"],
        }
        if any(existing.get(key) != value for key, value in expected.items()):
            raise PerformanceArtifactError(
                "performance_artifact_invalid",
                "An existing run has different identity or results",
            )
        # PerformanceStore performs the same integrity check under its publication
        # lock; summary-only legacy results still need protection against rewriting.
        if context.details is None:
            for artifact in self._store.list_artifacts(context.binding.run_id):
                path = Path(artifact["path"])
                if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
                    raise PerformanceArtifactError(
                        "performance_artifact_invalid",
                        "Existing artifact is missing or corrupt",
                    )


def _dataset_fingerprint(job: dict[str, Any]) -> dict[str, str]:
    dataset_id = str(job.get("dataset_id") or "")
    payload = {
        "dataset_id": dataset_id,
        "symbol": str(job.get("symbol") or ""),
        "timeframe": str(job.get("timeframe") or ""),
        "start_date": str(job.get("start_date") or ""),
        "end_date": str(job.get("end_date") or ""),
    }
    if not dataset_id:
        payload["dataset_id"] = sha256_text(json.dumps(payload, sort_keys=True))
    payload["fingerprint"] = sha256_text(json.dumps(payload, sort_keys=True))
    return payload
