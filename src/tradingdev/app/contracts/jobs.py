"""Public job responses, distinct from persisted worker control records."""

from typing import Literal

from pydantic import Field, JsonValue

from tradingdev.app.contracts.common import ContractModel, ErrorResponse
from tradingdev.domain.comparison_results import CandidateStatus

type JobState = Literal[
    "queued",
    "downloading_data",
    "running_backtest",
    "estimating",
    "optimizing",
    "testing_oos",
    "done",
    "failed",
    "cancelled",
    "estimation_timeout",
]


class JobSummary(ContractModel):
    """Public job identity and progress, without process control secrets."""

    job_id: str
    job_type: str
    status: JobState
    strategy_name: str | None
    revision_id: str | None = None
    manifest_hash: str | None = None
    symbol: str | None
    timeframe: str | None
    start_date: str | None
    end_date: str | None
    elapsed_seconds: float | None
    elapsed_unavailable_reason: Literal["missing_or_invalid_timestamps"] | None = None
    data_downloaded: bool
    total_combinations: int | None = None
    completed: int | None = None


class ComparisonCandidateStatus(ContractModel):
    """One bounded candidate identity and outcome, including reserved run IDs."""

    candidate_id: str
    label: str
    child_manifest_hash: str
    run_id: str
    status: CandidateStatus
    reason: str | None = None


class ComparisonJobStatus(ContractModel):
    """Bounded complete ledger summary, without a fabricated parent run."""

    manifest_hash: str
    total_candidates: int = Field(ge=1, le=8)
    counts: dict[CandidateStatus, int]
    outcome: Literal["in_progress", "all_succeeded", "partial_success", "no_success"]
    is_complete: bool
    dataset_id: str | None
    source_dataset_id: str | None
    candidates: list[ComparisonCandidateStatus] = Field(min_length=1, max_length=8)


class JobStatus(ContractModel):
    """Lifecycle status; stage-specific fields are null when unavailable.

    A failed worker is a successfully retrieved job with status='failed',
    not a failed lookup. Metrics may contain nested optimization results.
    """

    status: JobState
    job_type: str
    strategy_name: str | None
    revision_id: str | None = None
    manifest_hash: str | None = None
    symbol: str | None
    timeframe: str | None
    start_date: str | None
    end_date: str | None
    elapsed_seconds: float | None
    elapsed_unavailable_reason: Literal["missing_or_invalid_timestamps"] | None = None
    ended_at: str | None = None
    run_id: str | None = None
    metrics: dict[str, JsonValue] | None = None
    comparison: ComparisonJobStatus | None = None
    details_available: bool | None = None
    provenance: (
        Literal["performance_artifact", "legacy_metrics", "invalid_artifact"] | None
    ) = None
    available_metric_ids: list[str] | None = None
    available_scopes: list[str] | None = None
    default_scope: str | None = None
    selected_train_scope: str | None = None
    detail_error: ErrorResponse | None = None
    error: str | None = None
    message: str | None = None
    data_downloaded: bool | None = None
    best_params: dict[str, JsonValue] | None = None
    train_metrics: dict[str, JsonValue] | None = None
    test_metrics: dict[str, JsonValue] | None = None
    optimization_metric: str | None = None
    direction: Literal["maximize", "minimize"] | None = None
    total_combinations: int | None = None
    time_per_combo: float | None = None
    estimated_total_seconds: float | None = None
    n_parallel_workers: int | None = None
    completed: int | None = None
    estimated_remaining_seconds: float | None = None


class JobNotFound(ContractModel):
    """No persisted job matches the requested identifier."""

    status: Literal["not_found"]
    error: str
    code: Literal["job_not_found"]


class JobActionFailed(ErrorResponse):
    """An expected lifecycle or cleanup failure; no successful action is claimed."""

    code: Literal[
        "job_not_found",
        "invalid_job_state",
        "estimation_timeout",
        "worker_identity_unavailable",
        "worker_cleanup_failed",
        "execution_manifest_invalid",
    ]
    status: JobState | None = None
    pid: int | None = None


class JobCancelled(ContractModel):
    """Cancellation recorded after any required process cleanup acknowledgement."""

    success: Literal[True]
    job_id: str
    status: Literal["cancelled"]
    process_terminated: bool
