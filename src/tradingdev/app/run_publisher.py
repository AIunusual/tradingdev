"""Publish verified run artifacts independently of their owning job's lifecycle."""

from __future__ import annotations

import json
import pickle
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from tradingdev.adapters.storage.comparisons import ComparisonStore
from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import sha256_file, sha256_text
from tradingdev.adapters.storage.performance import (
    PerformanceArtifactError,
    PerformanceStore,
)
from tradingdev.adapters.storage.submitted_research import (
    ResearchSnapshot,
    SubmittedResearchStore,
)
from tradingdev.app.execution_dataset import execution_data_binding
from tradingdev.app.run_lineage import (
    StrategySourceSnapshot,
    extract_random_seed,
    load_config_payload,
    read_strategy_snapshot,
)
from tradingdev.app.strategy_service import StrategyNotExecutableError
from tradingdev.domain.execution import ExecutionManifest, ManifestError
from tradingdev.domain.performance.artifacts import (
    ObservationsBundle,
    PerformanceArtifacts,
    PerformanceBundle,
    bundles_from_pipeline,
    validate_projection,
)
from tradingdev.domain.research_scope import verify_research_scopes
from tradingdev.shared.utils.json_values import normalize_json_object

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore
    from tradingdev.domain.approved_research import ApprovedResearch
    from tradingdev.domain.backtest.pipeline_result import PipelineResult
    from tradingdev.domain.execution_target import ExecutionTarget


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
    research_evidence: ApprovedResearch | None
    research_target: ExecutionTarget | None


class RunPublisher:
    """Validate and publish one run under an explicitly supplied submission binding."""

    def __init__(self, workspace: WorkspacePaths, store: SQLiteStore) -> None:
        self._workspace = workspace
        self._store = store
        self._manifests = ExecutionManifestStore(workspace)
        self._research = SubmittedResearchStore(workspace)

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

    def verify_published(self, binding: RunPublicationBinding) -> None:
        """Validate a stopped comparison worker's complete publication for recovery.

        No strategy code is imported and no pipeline pickle is loaded. A leftover
        publication marker is removed only after every registered artifact and the
        complete performance pair have verified against the approved parent/child.
        The caller must first confirm that the owning worker has stopped.
        """
        manifest = binding.expected_manifest
        job = self._store.get_job(binding.job_id)
        if manifest is None or job is None or job.get("job_type") != "comparison":
            raise ManifestError("Recovery requires an approved comparison candidate")
        manifest.verify()
        comparisons = ComparisonStore(self._workspace)
        parent_hash = str(job["manifest_hash"])
        parent = comparisons.load_manifest(binding.job_id, expected_hash=parent_hash)
        ledger = comparisons.load_ledger(binding.job_id, expected_hash=parent_hash)
        row = next(
            (row for row in ledger.candidates if row.run_id == binding.run_id), None
        )
        if row is None or row.child_manifest_hash != manifest.manifest_hash:
            raise ManifestError("Published result is not a comparison member")
        receipt = comparisons.load_receipt(binding.job_id, expected_hash=parent_hash)
        if receipt.data_binding_json != execution_data_binding(manifest):
            raise ManifestError("Published result data binding differs")
        config = manifest.config_copy()
        strategy = config["strategy"]
        if parent.common_config()["strategy"] != {
            key: value for key, value in strategy.items() if key != "parameters"
        }:
            raise ManifestError("Published result strategy binding differs")
        run = self._store.get_run(binding.run_id)
        run_dir = self._manifests.path(binding.run_id).parent
        expected = {
            "job_id": binding.job_id,
            "strategy_id": strategy["id"],
            "revision_id": strategy.get("revision_id"),
            "manifest_hash": manifest.manifest_hash,
            "source_hash": strategy.get("source_hash"),
            "random_seed": config["random_seed"],
            "dataset_id": f"sha256:{receipt.snapshot_sha256}",
            "artifact_dir": str(run_dir),
        }
        if run is None or any(run.get(key) != value for key, value in expected.items()):
            raise ManifestError("Published result identity differs from its comparison")
        required = {
            "result_json": "result.json",
            "execution_manifest": "manifest.json",
            "config_snapshot": "config.yaml",
            "dataset_fingerprint": "dataset_fingerprint.json",
            "performance_json": "performance.json",
            "observations_json": "observations.json",
        }
        approved = self._research.load_for_job(job, parent)
        if approved is not None:
            required["research_evidence"] = "research_evidence.json"
        if strategy.get("source_hash") is not None:
            required["strategy_source"] = "strategy.py"
        artifacts = self._store.list_artifacts(binding.run_id)
        by_type = {item["artifact_type"]: item for item in artifacts}
        if not required.keys() <= by_type.keys():
            raise ManifestError("Published result is incomplete")
        for item in artifacts:
            path = Path(item["path"])
            if (
                item["run_id"] != binding.run_id
                or item["artifact_id"] != f"{binding.run_id}:{item['artifact_type']}"
                or path.parent != run_dir
                or path.is_symlink()
                or path.resolve().parent != run_dir.resolve()
                or not path.is_file()
                or sha256_file(path) != item["sha256"]
            ):
                raise ManifestError("Published result artifact is missing or corrupt")
            if (
                item["artifact_type"] in required
                and path.name != required[item["artifact_type"]]
            ):
                raise ManifestError("Published result artifact path differs")
            if (
                item["artifact_type"] not in {"performance_json", "observations_json"}
                and item["metadata"].get("job_id") != binding.job_id
            ):
                raise ManifestError("Published artifact belongs to a different job")
        self._manifests.load(
            binding.run_id,
            expected_hash=manifest.manifest_hash,
            expected_file_sha256=by_type["execution_manifest"]["sha256"],
        )
        if (
            yaml.safe_load((run_dir / "config.yaml").read_text()) != config
            or sha256_file(run_dir / "config.yaml") != run.get("config_hash")
            or json.loads((run_dir / "result.json").read_text()) != run["metrics"]
            or (
                strategy.get("source_hash") is not None
                and sha256_file(run_dir / "strategy.py") != strategy["source_hash"]
            )
        ):
            raise ManifestError(
                "Published result snapshots differ from approved execution"
            )
        fingerprint = json.loads((run_dir / "dataset_fingerprint.json").read_text())
        if fingerprint != _dataset_fingerprint(
            {
                **job,
                "dataset_id": expected["dataset_id"],
                "source_dataset_id": receipt.source_dataset_id,
            }
        ):
            raise ManifestError("Published result dataset fingerprint differs")
        details = PerformanceArtifacts(
            performance=PerformanceBundle.model_validate_json(
                (run_dir / "performance.json").read_bytes()
            ),
            observations=ObservationsBundle.model_validate_json(
                (run_dir / "observations.json").read_bytes()
            ),
        )
        if (
            details.performance.run_id != binding.run_id
            or details.performance.manifest_hash != manifest.manifest_hash
        ):
            raise ManifestError("Published performance identity differs")
        validate_projection(details, run["metrics"])
        if approved is not None:
            verify_research_scopes(manifest, details.performance, details.observations)
        if self._research.load_run(run, self._store, manifest) != approved:
            raise ManifestError("Published research differs from approved submission")
        marker = run_dir / ".result-publication"
        if marker.exists():
            if marker.is_symlink():
                raise ManifestError("Invalid publication marker")
            marker.rmdir()

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
                        "research_evidence.json",
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
            if context.research_evidence is not None:
                assert context.research_target is not None
                research_path = self._research.publish_run(
                    binding.job_id,
                    run_id,
                    context.research_evidence,
                    context.research_target,
                )
                self._register(
                    context,
                    "research_evidence",
                    research_path,
                    {"research_evidence_hash": context.research_evidence.content_hash},
                )
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
        reservation = re.fullmatch(r"(.+)_candidate_([1-8])", binding.run_id)
        if reservation is not None and reservation[1] != binding.job_id:
            owner = self._store.get_job(reservation[1])
            if owner is not None and owner.get("job_type") == "comparison":
                ledger = ComparisonStore(self._workspace).load_ledger(
                    reservation[1], expected_hash=str(owner["manifest_hash"])
                )
                if any(row.run_id == binding.run_id for row in ledger.candidates):
                    raise ManifestError("Run ID is reserved by a different comparison")
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
        parent: ExecutionTarget | None = None
        if expected is not None:
            expected.verify()
            if executed is None:
                raise ManifestError("Result must retain the executed manifest")
            executed.verify(expected_hash=expected.manifest_hash)
            parent_hash = job.get("manifest_hash")
            if not isinstance(parent_hash, str):
                raise ManifestError("Result manifest is not bound to a submitted job")
            if job.get("job_type") == "comparison":
                comparisons = ComparisonStore(self._workspace)
                comparison = comparisons.load_manifest(
                    binding.job_id, expected_hash=parent_hash
                )
                parent = comparison
                ledger = comparisons.load_ledger(
                    binding.job_id, expected_hash=parent_hash
                )
                member = next(
                    (row for row in ledger.candidates if row.run_id == binding.run_id),
                    None,
                )
                if (
                    member is None
                    or member.child_manifest_hash != expected.manifest_hash
                    or member.status not in {"running", "succeeded"}
                    or (
                        self._store.get_run(binding.run_id) is None
                        and (
                            member.status != "running"
                            or job.get("status") != "running_backtest"
                        )
                    )
                ):
                    raise ManifestError("Run is not an active member of its comparison")
                receipt = comparisons.load_receipt(
                    binding.job_id, expected_hash=parent_hash
                )
                if (
                    receipt.data_binding_json != execution_data_binding(expected)
                    or job.get("dataset_id") != f"sha256:{receipt.snapshot_sha256}"
                    or job.get("source_dataset_id") != receipt.source_dataset_id
                ):
                    raise ManifestError("Comparison result data identity differs")
                parent_strategy = comparison.common_config()["strategy"]
            else:
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
        evidence = None
        if parent is not None:
            evidence = self._research.load_for_job(job, parent)
            if evidence is not None:
                assert expected is not None
                snapshot = ResearchSnapshot(
                    job_id=binding.job_id,
                    run_id=binding.run_id,
                    target=parent,
                    research_evidence=evidence,
                )
                if snapshot.manifest_for_run().model_dump(
                    mode="json"
                ) != expected.model_dump(mode="json"):
                    raise ManifestError(
                        "Research evidence differs from the published manifest"
                    )
        elif (
            job.get("research_evidence_hash") is not None
            or self._research.path(binding.job_id).exists()
        ):
            raise ManifestError(
                "Research evidence requires an accepted execution target"
            )
        if evidence is None and (
            self._research.path(binding.run_id, run=True).exists()
            or self._store.get_artifact(f"{binding.run_id}:research_evidence")
            is not None
        ):
            raise ManifestError("Run research has no declared submission evidence")
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
        if evidence is not None:
            if details is None or executed is None:
                raise ManifestError(
                    "Research publication requires complete performance scopes"
                )
            verify_research_scopes(executed, details.performance, details.observations)
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
            research_evidence=evidence,
            research_target=parent,
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
        if (
            self._research.load_run(existing, self._store, context.manifest)
            != context.research_evidence
        ):
            raise ManifestError("Existing run research differs from its submission")
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
    if job.get("source_dataset_id"):
        payload["source_dataset_id"] = str(job["source_dataset_id"])
    payload["fingerprint"] = sha256_text(json.dumps(payload, sort_keys=True))
    return payload
