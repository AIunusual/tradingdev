"""Verify complete fixed-comparison evidence without executing or acquiring data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.storage.comparisons import ComparisonStore
from tradingdev.adapters.storage.submitted_research import SubmittedResearchStore
from tradingdev.app.execution_dataset import ExecutionDataset
from tradingdev.domain.backtest.metrics import timestamp_index
from tradingdev.domain.comparison_results import canonical_json_object
from tradingdev.domain.execution import ManifestError

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore
    from tradingdev.app.trade_history_service import (
        LoadedHistoryRun,
        TradeHistoryService,
    )
    from tradingdev.domain.approved_research import ApprovedResearch
    from tradingdev.domain.comparison import ComparisonCandidate


@dataclass(frozen=True)
class ReportComparison:
    """One detached evidence snapshot, including every unsuccessful candidate."""

    evidence: dict[str, Any]
    runs: dict[str, LoadedHistoryRun]
    research_evidence: ApprovedResearch | None = None


def load_report_comparison(
    workspace: WorkspacePaths,
    store: SQLiteStore,
    history: TradeHistoryService,
    job_id: str,
) -> ReportComparison:
    """Require a terminal parent, finalized ledger and authentic shared snapshot."""
    comparisons = ComparisonStore(workspace)
    # Validate the identifier before the database lookup or any path access.
    comparisons.path(job_id)
    job = store.get_job(job_id)
    if job is None or job.get("job_type") != "comparison":
        raise ManifestError("Unknown fixed-comparison job")
    expected_hash = job.get("manifest_hash")
    if not isinstance(expected_hash, str):
        raise ManifestError("Comparison job has no accepted manifest hash")
    manifest = comparisons.load_manifest(job_id, expected_hash=expected_hash)
    research_evidence = SubmittedResearchStore(workspace).load_for_job(job, manifest)
    ledger = comparisons.load_ledger(job_id, expected_hash=expected_hash)
    if (
        job.get("status") not in {"done", "failed", "cancelled"}
        or not ledger.is_complete
    ):
        raise ManifestError(
            "Comparison report requires a terminal job and complete ledger"
        )
    if job["status"] == "done" and any(
        row.status not in {"succeeded", "failed"} for row in ledger.candidates
    ):
        raise ManifestError("Completed comparison job has interrupted candidates")

    receipt_path = comparisons.path(job_id, "receipt.json")
    snapshot_path = comparisons.path(job_id, "snapshot.parquet")
    claims_data = bool(job.get("data_downloaded")) or any(
        job.get(key) is not None
        for key in ("dataset_id", "source_dataset_id", "data_binding_json")
    )
    receipt = None
    dataset = None
    timestamps: list[str] | None = None
    if (
        receipt_path.exists()
        or snapshot_path.exists()
        or claims_data
        or any(row.status != "not_run" for row in ledger.candidates)
    ):
        receipt, snapshot = comparisons.load_snapshot(
            job_id, expected_hash=expected_hash
        )
        dataset = ExecutionDataset(
            payload=snapshot,
            data_binding_json=receipt.data_binding_json,
            processed_path=snapshot_path,
            source_dataset_id=receipt.source_dataset_id,
        )
        # This checks the accepted source/market binding and decodes the actual
        # hashed bytes. Neither today's configuration nor a data loader is used.
        try:
            frame = dataset.frame_for(manifest.candidates[0].manifest)
            if any(row.status == "succeeded" for row in ledger.candidates):
                # The engines prefer the timestamp column; stored datasets often
                # retain a RangeIndex after feature joins. Use their exact rule.
                converted = timestamp_index(frame)
                if converted is None:
                    raise ManifestError("Comparison snapshot has no market timestamps")
                timestamps = [timestamp.isoformat() for timestamp in converted]
        except (ValueError, TypeError, OSError, NotImplementedError) as exc:
            raise ManifestError(
                "Cannot read the comparison's recorded market data"
            ) from exc
        for key, value in {
            "dataset_id": dataset.dataset_id,
            "source_dataset_id": receipt.source_dataset_id,
            "data_binding_json": receipt.data_binding_json,
        }.items():
            if job.get(key) is not None and job[key] != value:
                raise ManifestError("Comparison job data differs from its receipt")

    runs: dict[str, LoadedHistoryRun] = {}
    candidates = []
    for candidate, row in zip(manifest.candidates, ledger.candidates, strict=True):
        parameters = _parameters(candidate)
        candidates.append(
            {
                **row.model_dump(mode="json"),
                "label": candidate.label,
                "rationale": candidate.rationale,
                "parameters": parameters,
            }
        )
        if row.status != "succeeded":
            continue
        assert dataset is not None and timestamps is not None
        loaded = history.load_run(row.run_id)
        _verify_run(loaded, candidate, job_id, dataset.dataset_id, timestamps)
        if loaded.research_evidence != research_evidence:
            raise ManifestError("Comparison child research differs from its parent")
        runs[row.run_id] = loaded
    evidence = {
        "research_evidence_status": "recorded"
        if research_evidence is not None
        else "not_recorded",
        "comparison_job_id": job_id,
        "status": job["status"],
        "counts": {
            status: sum(row.status == status for row in ledger.candidates)
            for status in ("succeeded", "failed", "cancelled", "not_run")
        },
        "manifest": manifest.model_dump(mode="json"),
        "ledger": ledger.model_dump(mode="json"),
        "candidates": candidates,
        "shared_config": manifest.common_config(),
        "receipt": receipt.model_dump(mode="json") if receipt is not None else None,
        "snapshot_sha256": dataset.content_sha256 if dataset is not None else None,
        "data_available": dataset is not None,
        "data_unavailable_reason": None
        if dataset is not None
        else "工作在共用行情快照保存前結束；沒有可用行情或績效結果。",
        "sources": {
            "manifest_canonical_sha256": hashlib.sha256(
                canonical_json_object(manifest.model_dump_json()).encode()
            ).hexdigest(),
            "ledger_canonical_sha256": hashlib.sha256(
                canonical_json_object(ledger.model_dump_json()).encode()
            ).hexdigest(),
        },
    }
    return ReportComparison(evidence, runs, research_evidence)


def _parameters(candidate: ComparisonCandidate) -> dict[str, Any]:
    execution = candidate.manifest.strategy_execution
    raw = execution.constructor_kwargs
    result = raw.get("config", {}) if execution.kind == "bundled" else raw
    if not isinstance(result, dict):
        raise ManifestError("Comparison constructor parameters must be an object")
    return dict(result)


def _verify_run(
    loaded: LoadedHistoryRun,
    candidate: ComparisonCandidate,
    job_id: str,
    dataset_id: str,
    timestamps: list[str],
) -> None:
    manifest = candidate.manifest
    run = loaded.run
    config = manifest.config_copy()
    if (
        run.get("job_id") != job_id
        or run.get("manifest_hash") != manifest.manifest_hash
        or run.get("dataset_id") != dataset_id
        or run.get("random_seed") != config.get("random_seed")
        or loaded.manifest is None
        or loaded.manifest.model_dump() != manifest.model_dump()
        or loaded.manifest_artifact is None
    ):
        raise ManifestError("Comparison child run differs from its accepted identity")
    performance = loaded.performance
    if (
        list(performance.scopes) != ["full"]
        or performance.default_scope != "full"
        or performance.selected_train_scope is not None
    ):
        raise ManifestError("Comparison child must have one full backtest scope")
    scope = performance.scopes["full"]
    observations = loaded.observations.scopes["full"]
    if (
        scope.kind != "backtest"
        or scope.split != "full"
        or scope.fold_index is not None
        or scope.trial_index is not None
        or scope.parameters != _parameters(candidate)
        or scope.mode != config["backtest"]["mode"]
        or scope.metadata.get("execution_context")
        != {**config["backtest"], "random_seed": config.get("random_seed")}
        or observations.timestamps != timestamps
        or observations.init_cash
        != (config["backtest"]["init_cash"] if scope.mode == "signal" else None)
        or run.get("metrics") != scope.values
    ):
        raise ManifestError(
            "Comparison child parameters, performance or data scope differ"
        )
