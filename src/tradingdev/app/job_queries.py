"""Project saved job state and verified run details for discovery and polling."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.storage.performance import PerformanceArtifactError
from tradingdev.app.comparison_service import ComparisonService
from tradingdev.app.contracts.common import DEFAULT_PAGE_LIMIT, Page
from tradingdev.app.contracts.jobs import JobState, JobSummary
from tradingdev.app.job_store import TERMINAL_JOB_STATUSES
from tradingdev.app.optimization_results import OptimizationResults
from tradingdev.app.pagination import paginate
from tradingdev.app.run_service import RunService

if TYPE_CHECKING:
    from tradingdev.app.job_lifecycle import JobObservation
    from tradingdev.app.job_store import JobStore


class JobQueries:
    """Read result details without performing worker control or state transitions."""

    def __init__(self, job_store: JobStore) -> None:
        self._job_store = job_store
        self._runs = RunService(workspace=job_store.workspace, store=job_store.store)
        self._optimization = OptimizationResults(job_store.workspace, job_store.store)

    def status(self, job_id: str, observation: JobObservation) -> dict[str, Any]:
        """Attach result details to the state returned by lifecycle recovery."""
        job = observation.job
        if job is None:
            return {
                "status": "not_found",
                "error": f"No job with ID: {job_id}",
                "code": "job_not_found",
            }
        status = str(job["status"])
        response: dict[str, Any] = {
            "status": status,
            "job_type": job.get("job_type", "backtest"),
            "strategy_name": job.get("strategy_name"),
            "revision_id": job.get("revision_id"),
            "manifest_hash": job.get("manifest_hash"),
            "symbol": job.get("symbol"),
            "timeframe": job.get("timeframe"),
            "start_date": job.get("start_date"),
            "end_date": job.get("end_date"),
            **_elapsed_fields(job, datetime.now(UTC)),
            "ended_at": job.get("ended_at"),
        }
        if observation.cleanup_error is not None:
            response["cleanup_error"] = observation.cleanup_error

        if job.get("job_type") == "comparison":
            response["data_downloaded"] = job.get("data_downloaded", False)
            if job.get("error"):
                response["error"] = job["error"]
            try:
                response["comparison"] = ComparisonService(
                    job_store=self._job_store
                ).summary(job_id)
            except (OSError, TypeError, ValueError) as exc:
                response["detail_error"] = {
                    "success": False,
                    "code": "execution_manifest_invalid",
                    "error": str(exc),
                }
            if observation.detail_error is not None:
                response["detail_error"] = observation.detail_error
            return response

        if job.get("job_type") == "optimization":
            self._add_optimization_progress(response, job)
        if status == "done":
            run_response = self._runs.get_run(job_id)
            run = run_response.get("run")
            if run is not None:
                response["run_id"] = run["run_id"]
                response["metrics"] = run["metrics"]
                for key in (
                    "details_available",
                    "provenance",
                    "available_metric_ids",
                    "available_scopes",
                    "default_scope",
                    "selected_train_scope",
                    "detail_error",
                ):
                    response[key] = run[key]
            else:
                response.update(
                    metrics=None,
                    details_available=False,
                    provenance="legacy_metrics",
                    available_metric_ids=[],
                    available_scopes=[],
                    default_scope=None,
                    detail_error=run_response,
                )
            if job.get("job_type") == "optimization":
                # Non-metric search outcome fields retain their result contract.
                stored_run = self._job_store.get_run(job_id)
                result = stored_run.get("metrics", {}) if stored_run else {}
                for key in ("optimization_metric", "direction", "total_combinations"):
                    response.setdefault(key, (result or {}).get(key))
                if run is not None and run["details_available"]:
                    response["test_metrics"] = run["metrics"]
                    selected_train = run["selected_train_scope"]
                    if selected_train is not None:
                        detail = self._runs.get_run_metrics(
                            job_id, scope=selected_train
                        )
                        if detail.get("success"):
                            response["train_metrics"] = _saved_metric_summary(detail)
                            response["best_params"] = detail["parameters"]
                        else:
                            response["detail_error"] = {
                                key: detail[key] for key in ("success", "code", "error")
                            }
                elif run is not None and run["provenance"] == "legacy_metrics":
                    response["best_params"] = (result or {}).get("best_params")
                    response["train_metrics"] = run["metrics"].get("train_metrics")
                    response["test_metrics"] = run["metrics"].get("test_metrics")
        elif status in {"failed", "partial", "cancelled"}:
            response["error"] = job.get("error", "Unknown error")
        elif status == "estimating":
            response["total_combinations"] = job.get("total_combinations")
            response["message"] = "Running trial combination to estimate total time..."
        elif status == "optimizing":
            response.setdefault("completed", job.get("completed"))
            response["total_combinations"] = job.get("total_combinations")
            response["estimated_remaining_seconds"] = job.get(
                "estimated_remaining_seconds"
            )
        else:
            response["data_downloaded"] = job.get("data_downloaded", False)
        return response

    def _add_optimization_progress(
        self, response: dict[str, Any], job: dict[str, Any]
    ) -> None:
        for key in ("time_per_combo", "estimated_total_seconds", "n_parallel_workers"):
            response[key] = job.get(key)
        try:
            view = self._optimization.load(
                job["job_id"],
                verify_observations=job["status"] in TERMINAL_JOB_STATUSES,
            )
        except PerformanceArtifactError as exc:
            response["detail_error"] = {
                "success": False,
                "code": exc.code,
                "error": str(exc),
            }
            return
        if view is not None:
            progress = view.progress()
            response.update(
                optimization=progress.model_dump(mode="json"),
                optimization_metric=progress.objective,
                direction=progress.direction,
                completed=progress.completed,
                total_combinations=progress.total_combinations,
            )

    def list_jobs(self) -> list[dict[str, Any]]:
        """List job summaries newest first."""
        jobs = self._job_store.list_all_jobs()
        now = datetime.now(UTC)
        return [self._summarize_job(job, now) for job in jobs]

    def list_jobs_page(
        self,
        *,
        status: JobState | None = None,
        job_type: str | None = None,
        strategy_name: str | None = None,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> Page[JobSummary]:
        """Page saved job progress without polling or changing worker state."""
        rows = [
            job
            for job in self._job_store.list_all_jobs()
            if (status is None or job["status"] == status)
            and (job_type is None or job.get("job_type", "backtest") == job_type)
            and (strategy_name is None or job.get("strategy_name") == strategy_name)
        ]
        rows.sort(key=lambda row: str(row["job_id"]))
        rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
        now = datetime.now(UTC)
        return paginate(
            rows,
            project=lambda row: JobSummary.model_validate(
                self._summarize_job(row, now)
            ),
            offset=offset,
            limit=limit,
        )

    def _summarize_job(self, job: dict[str, Any], now: datetime) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "job_id": job["job_id"],
            "job_type": job.get("job_type", "backtest"),
            "status": job["status"],
            "strategy_name": job.get("strategy_name"),
            "revision_id": job.get("revision_id"),
            "manifest_hash": job.get("manifest_hash"),
            "symbol": job.get("symbol"),
            "timeframe": job.get("timeframe"),
            "start_date": job.get("start_date"),
            "end_date": job.get("end_date"),
            **_elapsed_fields(job, now),
            "data_downloaded": job.get("data_downloaded", False),
        }
        if job.get("job_type") == "optimization":
            try:
                view = self._optimization.load(job["job_id"], verify_observations=False)
            except PerformanceArtifactError:
                # The detailed status exposes verification errors to the caller.
                return summary
            if view is None:
                summary["total_combinations"] = job.get("total_combinations")
                summary["completed"] = job.get("completed")
            else:
                progress = view.progress()
                summary["total_combinations"] = progress.total_combinations
                summary["completed"] = progress.completed
        return summary


def _saved_metric_summary(detail: dict[str, Any]) -> dict[str, Any]:
    """Select the original snapshot's summary flags, never today's catalog."""
    return {
        key: value
        for key, value in detail["metrics"].items()
        if detail["definitions"][key]["summary"]
        and detail["mode"] in detail["definitions"][key]["modes"]
    }


def _elapsed_fields(job: dict[str, Any], now: datetime) -> dict[str, Any]:
    """Duration from creation to the saved terminal time, or current active age."""
    try:
        created = datetime.fromisoformat(str(job["created_at"]))
        end = (
            datetime.fromisoformat(str(job.get("ended_at")))
            if job["status"] in TERMINAL_JOB_STATUSES
            else now
        )
        if created.utcoffset() is None or end.utcoffset() is None or end < created:
            raise ValueError("Invalid lifecycle timestamps")
        return {
            "elapsed_seconds": round((end - created).total_seconds(), 1),
            "elapsed_unavailable_reason": None,
        }
    except KeyError, TypeError, ValueError, OverflowError:
        return {
            "elapsed_seconds": None,
            "elapsed_unavailable_reason": "missing_or_invalid_timestamps",
        }
