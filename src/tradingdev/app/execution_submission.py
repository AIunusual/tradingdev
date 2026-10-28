"""Submit already prepared execution specifications to background workers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from tradingdev.adapters.storage.submitted_research import SubmittedResearchStore
from tradingdev.domain.execution import ExecutionManifest, ManifestError

if TYPE_CHECKING:
    from tradingdev.adapters.execution.process_runner import ProcessRunner
    from tradingdev.app.job_store import JobStore
    from tradingdev.app.prepared_execution import PreparedSubmission
    from tradingdev.domain.execution_target import ExecutionTarget


class ExecutionSubmissionService:
    """Persist and launch a fixed manifest without resolving configuration again."""

    def __init__(self, job_store: JobStore, process_runner: ProcessRunner) -> None:
        self._job_store = job_store
        self._process_runner = process_runner

    def submit(
        self, prepared: PreparedSubmission, *, job_id: str | None = None
    ) -> dict[str, Any]:
        """Create one job and retain startup failures, including interruptions."""
        prepared.manifest.verify()
        selected_id = job_id if job_id is not None else uuid4().hex[:12]
        with self._job_store.submission_lock(selected_id):
            return self._submit_locked(prepared, selected_id)

    def _submit_locked(
        self, prepared: PreparedSubmission, selected_id: str
    ) -> dict[str, Any]:
        manifest = prepared.manifest
        manifest.verify()
        evidence = prepared.research_evidence
        if evidence is not None:
            evidence.verify_current_policy(manifest)
            SubmittedResearchStore.validate_publication_size(manifest, evidence)
        if self._job_store.get_job(selected_id) is not None:
            raise ValueError(f"Job already exists: {selected_id}")
        extra: dict[str, Any] = {
            "original_config_path": str(prepared.original_config_path),
        }
        if (
            isinstance(manifest, ExecutionManifest)
            and manifest.optimization is not None
        ):
            extra["total_combinations"] = manifest.optimization.total_combinations
        if evidence is not None:
            extra["research_evidence_hash"] = evidence.content_hash
        self._job_store.create_job(
            job_id=selected_id, manifest=manifest, extra_payload=extra
        )
        module = (
            "tradingdev.mcp.workers.comparison"
            if manifest.kind == "comparison"
            else "tradingdev.mcp.workers.optimization"
            if manifest.kind == "optimization"
            else "tradingdev.mcp.workers.backtest"
        )
        try:
            if evidence is not None:
                SubmittedResearchStore(self._job_store.workspace).save(
                    selected_id, evidence, manifest
                )
            identity = self._process_runner.spawn_module(module, selected_id)
        except BaseException as exc:
            self._job_store.update_job(
                selected_id,
                status="failed",
                error=f"Worker failed to start: {type(exc).__name__}: {exc}",
            )
            if manifest.kind == "comparison":
                from tradingdev.adapters.storage.comparisons import ComparisonStore

                ComparisonStore(self._job_store.workspace).finish_unfinished(
                    selected_id,
                    expected_hash=manifest.manifest_hash,
                    reason=f"Worker failed to start: {type(exc).__name__}: {exc}"[
                        :4000
                    ],
                )
            raise
        self._job_store.update_job(selected_id, **identity.job_fields())
        config = (
            manifest.config_copy()
            if isinstance(manifest, ExecutionManifest)
            else manifest.common_config()
        )
        return {
            "job_id": selected_id,
            "revision_id": config["strategy"].get("revision_id"),
            "manifest_hash": manifest.manifest_hash,
        }


def verify_submission_research(
    job_store: JobStore, job_id: str, target: ExecutionTarget
) -> None:
    """Check queued disclosure before acquiring data or executing strategy code."""
    job = job_store.get_job(job_id)
    if job is None:
        raise ManifestError("Research execution requires a submitted job")
    research = SubmittedResearchStore(job_store.workspace)
    evidence = research.load_for_job(job, target)
    if evidence is not None:
        evidence.verify_current_policy(target)
    elif (
        research.path(job_id, run=True).exists()
        or job_store.get_artifact(f"{job_id}:research_evidence") is not None
    ):
        raise ManifestError("Research execution has no declared submission evidence")
