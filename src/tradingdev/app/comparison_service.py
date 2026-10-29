"""Execute one approved fixed comparison and preserve all candidate outcomes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.execution.process_runner import WorkerHandle
from tradingdev.adapters.storage.comparisons import ComparisonStore
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService
from tradingdev.app.execution_dataset import ExecutionDataset, execution_data_binding
from tradingdev.app.execution_submission import verify_submission_research
from tradingdev.app.run_lineage import read_strategy_snapshot
from tradingdev.app.run_publisher import RunPublicationBinding, RunPublisher
from tradingdev.app.strategy_service import StrategyNotExecutableError, StrategyService
from tradingdev.domain.comparison import ComparisonManifest
from tradingdev.domain.execution import ManifestError
from tradingdev.domain.strategies.loader import StrategyLoader

if TYPE_CHECKING:
    from tradingdev.app.job_store import JobStore
    from tradingdev.domain.comparison_results import ComparisonLedger, FailureStatus

logger = logging.getLogger(__name__)


class ComparisonService:
    """Run independent candidates over one captured frame; never retry a job."""

    def __init__(
        self,
        *,
        job_store: JobStore,
        backtest_service: BacktestService | None = None,
    ) -> None:
        self.jobs = job_store
        self.comparisons = ComparisonStore(job_store.workspace)
        self.publisher = RunPublisher(job_store.workspace, job_store.store)
        self.backtest = backtest_service or BacktestService(
            data_service=DataService(job_store.workspace),
            strategy_gate=StrategyService(job_store.workspace),
            strategy_loader=StrategyLoader(workspace_root=job_store.workspace.root),
        )

    def _manifest(self, job_id: str) -> ComparisonManifest:
        manifest = self.jobs.load_manifest(job_id)
        if not isinstance(manifest, ComparisonManifest):
            raise ManifestError("Job does not contain a comparison manifest")
        return manifest

    def execute(
        self, job_id: str, *, worker_handle: WorkerHandle | None = None
    ) -> ComparisonLedger:
        """Claim a queued bundle once, then retain successful and failed candidates."""
        with self.jobs.submission_lock(job_id):
            manifest = self._manifest(job_id)
            job = self.jobs.get_job(job_id)
            ledger = self.comparisons.load_ledger(
                job_id, expected_hash=manifest.manifest_hash
            )
            if (
                job is None
                or job["status"] != "queued"
                or any(row.status != "pending" for row in ledger.candidates)
                or self.comparisons.path(job_id, "receipt.json").exists()
            ):
                raise ManifestError("An approved comparison cannot be replayed")
            if worker_handle is not None:
                existing_handle = WorkerHandle.from_job(job)
                if existing_handle is not None and existing_handle != worker_handle:
                    raise ManifestError("Comparison belongs to a different worker")
            self.jobs.update_job(
                job_id,
                status="downloading_data",
                **(worker_handle.job_fields() if worker_handle is not None else {}),
            )
        try:
            verify_submission_research(self.jobs, job_id, manifest)
            first = manifest.candidates[0].manifest
            # Check shared source identity before any real data is acquired.
            read_strategy_snapshot(
                first.config_copy(),
                self.jobs.workspace,
                strategy_id=first.config_copy()["strategy"]["id"],
            )
            captured = self.backtest.load_execution_dataset(first)
            if captured.data_binding_json != execution_data_binding(first):
                raise ManifestError("Captured comparison data binding differs")
            receipt = self.comparisons.publish_snapshot(
                job_id,
                expected_hash=manifest.manifest_hash,
                snapshot_bytes=captured.payload,
                data_binding_json=captured.data_binding_json,
                source_dataset_id=captured.source_dataset_id,
            )
            verified_receipt, snapshot = self.comparisons.load_snapshot(
                job_id, expected_hash=manifest.manifest_hash
            )
            if verified_receipt != receipt:
                raise ManifestError("Comparison snapshot receipt changed")
            dataset = ExecutionDataset(
                payload=snapshot,
                data_binding_json=receipt.data_binding_json,
                processed_path=self.comparisons.path(job_id, "snapshot.parquet"),
                source_dataset_id=receipt.source_dataset_id,
            )
            if dataset.dataset_id != f"sha256:{receipt.snapshot_sha256}":
                raise ManifestError("Comparison snapshot identity changed")
            self.jobs.update_job(
                job_id,
                status="running_backtest",
                data_downloaded=True,
                dataset_id=dataset.dataset_id,
                source_dataset_id=dataset.source_dataset_id,
                data_binding_json=dataset.data_binding_json,
            )
            # Every candidate trades the same market, so one baseline is captured
            # once and reused instead of being reloaded per candidate.
            benchmark = self.backtest.prepare_benchmark(dataset.frame_for(first), first)
            for candidate, row in zip(
                manifest.candidates, ledger.candidates, strict=True
            ):
                # Revalidate saved parent and ledger before each independent run.
                self._manifest(job_id).verify(expected_hash=manifest.manifest_hash)
                self.comparisons.transition(
                    job_id,
                    expected_hash=manifest.manifest_hash,
                    candidate_id=candidate.candidate_id,
                    status="running",
                )
                try:
                    run = self.backtest.run_manifest(
                        candidate.manifest,
                        loaded_dataset=dataset,
                        benchmark=benchmark,
                    )
                except ManifestError, StrategyNotExecutableError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self.comparisons.transition(
                        job_id,
                        expected_hash=manifest.manifest_hash,
                        candidate_id=candidate.candidate_id,
                        status="failed",
                        reason=_reason(exc),
                    )
                    continue
                if run.dataset_id != dataset.dataset_id:
                    raise ManifestError("Candidate executed with a different dataset")
                self.publisher.publish(
                    RunPublicationBinding(job_id, row.run_id, candidate.manifest),
                    run.metrics,
                    pipeline=run.pipeline,
                )
                self.comparisons.transition(
                    job_id,
                    expected_hash=manifest.manifest_hash,
                    candidate_id=candidate.candidate_id,
                    status="succeeded",
                )
            ledger = self.comparisons.load_ledger(
                job_id, expected_hash=manifest.manifest_hash
            )
            if not ledger.is_complete:
                raise ManifestError("Comparison completed with unfinished candidates")
            self.jobs.update_job(job_id, status="done")
            return ledger
        except Exception as exc:  # noqa: BLE001
            logger.exception("Comparison %s stopped", job_id)
            reason = _reason(exc)
            try:
                return self.reconcile(job_id, reason=reason)
            finally:
                self.jobs.update_job(job_id, status="failed", error=reason)

    def reconcile(
        self,
        job_id: str,
        *,
        reason: str,
        active_status: FailureStatus = "failed",
    ) -> ComparisonLedger:
        """Finalize after worker shutdown, recovering only complete bound results.

        The caller must confirm worker cleanup before invoking this method. During
        execution it is also safe from the worker's own synchronous failure path.
        A corrupt or partial publication is retained for diagnosis and never
        promoted to a successful candidate.
        """
        manifest = self._manifest(job_id)
        ledger = self.comparisons.load_ledger(
            job_id, expected_hash=manifest.manifest_hash
        )
        for candidate, row in zip(manifest.candidates, ledger.candidates, strict=True):
            if row.status != "running" or self.jobs.get_run(row.run_id) is None:
                continue
            try:
                self.publisher.verify_published(
                    RunPublicationBinding(job_id, row.run_id, candidate.manifest)
                )
            except OSError, TypeError, ValueError:
                logger.warning("Could not recover comparison child %s", row.run_id)
                continue
            self.comparisons.transition(
                job_id,
                expected_hash=manifest.manifest_hash,
                candidate_id=row.candidate_id,
                status="succeeded",
            )
        return self.comparisons.finish_unfinished(
            job_id,
            expected_hash=manifest.manifest_hash,
            reason=reason[:4000],
            active_status=active_status,
        )

    def summary(self, job_id: str) -> dict[str, Any]:
        """Read bounded status evidence without reloading the market-data snapshot."""
        manifest = self._manifest(job_id)
        ledger = self.comparisons.load_ledger(
            job_id, expected_hash=manifest.manifest_hash
        )
        counts = dict.fromkeys(
            ("pending", "running", "succeeded", "failed", "not_run", "cancelled"), 0
        )
        for row in ledger.candidates:
            counts[row.status] += 1
        outcome = (
            "in_progress"
            if not ledger.is_complete
            else "all_succeeded"
            if counts["succeeded"] == len(ledger.candidates)
            else "partial_success"
            if counts["succeeded"]
            else "no_success"
        )
        receipt = None
        if self.comparisons.path(job_id, "receipt.json").exists():
            receipt = self.comparisons.load_receipt(
                job_id, expected_hash=manifest.manifest_hash
            )
            if receipt.data_binding_json != execution_data_binding(
                manifest.candidates[0].manifest
            ):
                raise ManifestError("Comparison receipt data binding differs")
        job = self.jobs.get_job(job_id)
        if (
            receipt is None
            and job is not None
            and (job.get("data_downloaded") or job.get("dataset_id"))
        ):
            raise ManifestError("Comparison execution data receipt is missing")
        if (
            receipt is not None
            and job is not None
            and any(
                job.get(key) is not None and job[key] != expected
                for key, expected in {
                    "dataset_id": f"sha256:{receipt.snapshot_sha256}",
                    "source_dataset_id": receipt.source_dataset_id,
                }.items()
            )
        ):
            raise ManifestError("Comparison job data identity differs from receipt")
        return {
            "manifest_hash": manifest.manifest_hash,
            "total_candidates": len(ledger.candidates),
            "counts": counts,
            "outcome": outcome,
            "is_complete": ledger.is_complete,
            "dataset_id": f"sha256:{receipt.snapshot_sha256}" if receipt else None,
            "source_dataset_id": receipt.source_dataset_id if receipt else None,
            "candidates": [
                {
                    "candidate_id": row.candidate_id,
                    "label": candidate.label,
                    "child_manifest_hash": row.child_manifest_hash,
                    "run_id": row.run_id,
                    "status": row.status,
                    "reason": row.reason,
                }
                for candidate, row in zip(
                    manifest.candidates, ledger.candidates, strict=True
                )
            ],
        }


def _reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:4000]
