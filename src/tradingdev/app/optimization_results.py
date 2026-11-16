"""Read optimization evidence without granting partial jobs completed-run status."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.optimization_trials import (
    TEST_POSITION,
    OptimizationLedger,
    TrialSnapshot,
)
from tradingdev.adapters.storage.performance import PerformanceArtifactError
from tradingdev.app.contracts.common import DEFAULT_PAGE_LIMIT
from tradingdev.app.contracts.optimization import (
    OptimizationProgress,
    OptimizationTrial,
    OptimizationTrialsResponse,
)
from tradingdev.app.pagination import paginate
from tradingdev.domain.execution import ManifestError
from tradingdev.domain.optimization.grid_search import finite_metric_value

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore
    from tradingdev.domain.optimization.trials import TrialRecord, TrialStateView


@dataclass(frozen=True)
class OptimizationResultView:
    """Verified ledger and the persisted lifecycle that owns it."""

    ledger: OptimizationLedger
    snapshot: TrialSnapshot
    job: dict[str, Any]

    def progress(self) -> OptimizationProgress:
        trials = self.snapshot.trials
        leader = self.snapshot.leading_index
        status = self.job["status"]
        outcome = (
            "complete"
            if status == "done"
            else "partial"
            if status in {"partial", "failed", "cancelled"}
            else "in_progress"
        )
        return OptimizationProgress(
            outcome=outcome,
            manifest_hash=self.ledger.manifest.manifest_hash,
            dataset_id=self.ledger.dataset_id,
            objective=self.ledger.search.optimization_metric,
            direction=self.ledger.search.direction,
            total_combinations=len(trials),
            attempted=sum(trial.state != "not_attempted" for trial in trials),
            completed=sum(trial.state == "completed" for trial in trials),
            rankable=sum(
                finite_metric_value(trial.target_value) is not None for trial in trials
            ),
            search_complete=self.snapshot.search_complete,
            estimation=(
                "available"
                if self.job.get("estimated_total_seconds") is not None
                else "unavailable"
                if outcome != "in_progress"
                or self.job["status"] in {"optimizing", "testing_oos"}
                else "pending"
            ),
            stop_reason=self.job.get("error"),
            leading_trial_index=leader,
            leading_target_value=trials[leader].target_value
            if leader is not None
            else None,
            test=self.describe(self.snapshot.test),
        )

    def describe(self, record: TrialRecord) -> OptimizationTrial:
        state: TrialStateView = record.state
        reason = record.reason
        if state == "running" and self.job["status"] in {"failed", "cancelled"}:
            state = "interrupted"
            reason = (
                self.job.get("error")
                or "Execution ended before a result was committed."
            )
        return OptimizationTrial(
            trial_index=record.selected_index
            if record.position == TEST_POSITION
            else record.position,
            parameters=self.ledger.parameters(record),
            state=state,
            scope=self.scope_id(record) if state == "completed" else None,
            target_value=record.target_value,
            reason=reason,
        )

    @staticmethod
    def scope_id(record: TrialRecord) -> str:
        return (
            "test"
            if record.position == TEST_POSITION
            else f"trial/{record.position}/train"
        )


class OptimizationResults:
    """Verify job/manifest/dataset binding before exposing a ledger to callers."""

    def __init__(self, workspace: WorkspacePaths, store: SQLiteStore) -> None:
        self._manifests = ExecutionManifestStore(workspace)
        self._store = store

    def load(
        self,
        job_id: str,
        *,
        verify_observations: bool = True,
    ) -> OptimizationResultView | None:
        """Return committed evidence, when present, independently of job success."""
        job = self._store.get_job(job_id)
        if (
            job is None
            or job["job_type"] != "optimization"
            or not job.get("manifest_hash")
        ):
            return None
        try:
            manifest = self._manifests.load(job_id, expected_hash=job["manifest_hash"])
            dataset_id = job.get("dataset_id")
            ledger = OptimizationLedger(self._store, job_id, manifest, dataset_id)
            try:
                snapshot = ledger.snapshot(verify_evidence=verify_observations)
            except ManifestError:
                latest = self._store.get_job(job_id)
                if (
                    dataset_id is not None
                    or latest is None
                    or not latest.get("dataset_id")
                ):
                    raise
                # A first attempt may commit between the job and ledger reads.
                job = latest
                ledger = OptimizationLedger(
                    self._store, job_id, manifest, job["dataset_id"]
                )
                snapshot = ledger.snapshot(verify_evidence=verify_observations)
            if snapshot is None:
                return None
            # The lifecycle may have committed while the snapshot was being read.
            # Retain the earlier lifecycle so a concurrent completion cannot
            # promote an older snapshot to a complete result.
            if job["status"] == "done" and snapshot.test.state != "completed":
                raise ManifestError(
                    "Completed optimization has no observed test evidence"
                )
            return OptimizationResultView(ledger, snapshot, job)
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            raise PerformanceArtifactError(
                "optimization_evidence_invalid", str(exc)
            ) from exc

    def page(
        self, job_id: str, *, offset: int = 0, limit: int = DEFAULT_PAGE_LIMIT
    ) -> OptimizationTrialsResponse:
        view = self.load(job_id)
        if view is None:
            raise PerformanceArtifactError(
                "optimization_evidence_unavailable",
                f"No trial ledger recorded for {job_id}",
            )
        return OptimizationTrialsResponse(
            success=True,
            job_id=job_id,
            progress=view.progress(),
            trials=paginate(
                view.snapshot.trials, project=view.describe, offset=offset, limit=limit
            ),
        )
