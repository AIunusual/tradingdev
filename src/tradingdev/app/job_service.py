"""Application service for background jobs."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from tradingdev.adapters.execution.process_runner import (
    ProcessIdentity,
    ProcessRunner,
    WorkerHandle,
    request_worker_stop,
)
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.contracts.common import DEFAULT_PAGE_LIMIT, Page
from tradingdev.app.contracts.jobs import JobState, JobSummary
from tradingdev.app.data_service import DataService
from tradingdev.app.execution_submission import (
    ExecutionSubmissionService,
    PreparedExecution,
)
from tradingdev.app.job_config import (
    apply_backtest_overrides,
    apply_run_overrides,
    bind_strategy_revision,
)
from tradingdev.app.job_store import JobStore, get_default_job_store
from tradingdev.app.pagination import paginate
from tradingdev.app.run_service import RunService
from tradingdev.app.strategy_service import (
    StrategyNotExecutableError,
    StrategyService,
)
from tradingdev.domain.backtest.schemas import BacktestRunConfig
from tradingdev.domain.strategies.capabilities import (
    StrategyExecutionModeError,
    ensure_execution_mode,
)
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.shared.utils.config import load_config

if TYPE_CHECKING:
    from tradingdev.domain.strategies.schemas import StrategySpec


class JobService:
    """Create and query background jobs."""

    _ACTIVE_STATUSES = {
        "queued",
        "downloading_data",
        "running_backtest",
        "estimating",
        "optimizing",
        "testing_oos",
    }
    _TERMINAL_STATUSES = {"done", "failed", "cancelled", "estimation_timeout"}

    def __init__(
        self,
        *,
        strategy_service: StrategyService | None = None,
        data_service: DataService | None = None,
        strategy_loader: StrategyLoader | None = None,
        job_store: JobStore | None = None,
        process_runner: ProcessRunner | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._job_store = job_store or get_default_job_store()
        self._runs = RunService(
            workspace=self._job_store.workspace, store=self._job_store.store
        )
        self._strategy_service = strategy_service or StrategyService(
            self._job_store.workspace
        )
        self._data_service = data_service or DataService(self._job_store.workspace)
        self._strategy_loader = strategy_loader or StrategyLoader(
            workspace_root=self._job_store.workspace.root
        )
        self._project_root = (project_root or self._default_project_root()).resolve()
        self._process_runner = process_runner or ProcessRunner(
            self._project_root, workspace=self._job_store.workspace
        )
        self._submission = ExecutionSubmissionService(
            self._job_store, self._process_runner
        )

    def start_backtest(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        revision_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        backtest_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare and submit a simple backtest job."""
        prepared = self.prepare_backtest(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            revision_id=revision_id,
            parameters=parameters,
            backtest_overrides=backtest_overrides,
        )
        return self._submit_prepared(prepared)

    def prepare_backtest(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        revision_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        backtest_overrides: dict[str, Any] | None = None,
    ) -> PreparedExecution | dict[str, Any]:
        """Capture a simple backtest without creating a job or launching a worker."""
        try:
            spec, error = self._resolve_strategy_run_config(strategy_id, revision_id)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            return self._invalid_execution_request(exc)
        if spec is None:
            return {
                "job_id": "",
                "message": error,
                "data_available": False,
                "code": "strategy_not_executable",
            }
        config_path = Path(spec.config_path)
        try:
            raw_config = load_config(config_path)
            run_config = BacktestRunConfig.model_validate(raw_config)
            ensure_execution_mode(
                spec.strategy_id, spec.execution_capabilities, "backtest"
            )
        except StrategyExecutionModeError as exc:
            return {
                "job_id": "",
                "message": str(exc),
                "data_available": False,
                "code": exc.code,
                "execution_guidance": exc.guidance.model_dump(mode="json"),
            }
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            return self._invalid_execution_request(exc)
        if run_config.is_walk_forward:
            return {
                "job_id": "",
                "message": (
                    "Config contains validation settings; use prepare_walk_forward."
                ),
                "data_available": False,
                "code": "invalid_run_mode",
            }
        return self._prepare_execution(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            config_path=config_path,
            raw_config=raw_config,
            walk_forward=False,
            spec=spec,
            parameters=parameters,
            backtest_overrides=backtest_overrides,
        )

    def start_walk_forward(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        revision_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        backtest_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare and submit a walk-forward job."""
        prepared = self.prepare_walk_forward(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            revision_id=revision_id,
            parameters=parameters,
            backtest_overrides=backtest_overrides,
        )
        return self._submit_prepared(prepared)

    def prepare_walk_forward(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        revision_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        backtest_overrides: dict[str, Any] | None = None,
    ) -> PreparedExecution | dict[str, Any]:
        """Capture a walk-forward without creating a job or launching a worker."""
        try:
            spec, error = self._resolve_strategy_run_config(strategy_id, revision_id)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            return self._invalid_execution_request(exc)
        if spec is None:
            return {
                "job_id": "",
                "message": error,
                "data_available": False,
                "code": "strategy_not_executable",
            }
        try:
            resolved = self._resolve_walk_forward_config(
                strategy_id=strategy_id,
                config_path=Path(spec.config_path),
                allow_fallback=spec.kind == "bundled",
            )
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            return self._invalid_execution_request(exc)
        if resolved is None:
            return {
                "job_id": "",
                "message": "Config has no validation section for walk-forward.",
                "data_available": False,
                "code": "invalid_run_mode",
            }
        config_path, raw_config = resolved
        return self._prepare_execution(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            config_path=config_path,
            raw_config=raw_config,
            walk_forward=True,
            spec=spec,
            parameters=parameters,
            backtest_overrides=backtest_overrides,
        )

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Return current job status and completed result payload."""
        job = self._job_store.get_job(job_id)
        if job is None:
            return {
                "status": "not_found",
                "error": f"No job with ID: {job_id}",
                "code": "job_not_found",
            }

        status = str(job["status"])
        if (
            status in self._ACTIVE_STATUSES
            and (status != "queued" or job.get("pid") is not None)
            and not self._is_process_alive(job)
        ):
            status = "failed"
            failure = {
                "status": status,
                "error": "Worker process terminated unexpectedly.",
                "ended_at": datetime.now(UTC).isoformat(),
            }
            self._job_store.update_job(job_id, **failure)
            job.update(failure)

        created_at = datetime.fromisoformat(str(job["created_at"]))
        elapsed = round((datetime.now(UTC) - created_at).total_seconds(), 1)
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
            "elapsed_seconds": elapsed,
        }

        if status == "done":
            response["ended_at"] = job.get("ended_at")
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
                response.update(
                    {
                        key: (result or {}).get(key)
                        for key in (
                            "best_params",
                            "optimization_metric",
                            "direction",
                            "total_combinations",
                        )
                    }
                )
                if run is not None and run["details_available"]:
                    response["test_metrics"] = run["metrics"]
                    selected_train = run["selected_train_scope"]
                    if selected_train is not None:
                        detail = self._runs.get_run_metrics(
                            job_id, scope=selected_train
                        )
                        if detail.get("success"):
                            response["train_metrics"] = _saved_metric_summary(detail)
                        else:
                            response["detail_error"] = {
                                key: detail[key] for key in ("success", "code", "error")
                            }
                elif run is not None and run["provenance"] == "legacy_metrics":
                    response["train_metrics"] = run["metrics"].get("train_metrics")
                    response["test_metrics"] = run["metrics"].get("test_metrics")
        elif status == "failed":
            response["error"] = job.get("error", "Unknown error")
        elif status == "estimating":
            response["total_combinations"] = job.get("total_combinations")
            response["message"] = "Running trial combination to estimate total time..."
        elif status == "optimizing":
            response["completed"] = job.get("completed", 0)
            response["total_combinations"] = job.get("total_combinations")
            response["estimated_remaining_seconds"] = job.get(
                "estimated_remaining_seconds"
            )
        else:
            response["data_downloaded"] = job.get("data_downloaded", False)
        return response

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
        rows.sort(key=lambda row: str(row["created_at"]), reverse=True)
        now = datetime.now(UTC)
        return paginate(
            rows,
            project=lambda row: JobSummary.model_validate(
                self._summarize_job(row, now)
            ),
            offset=offset,
            limit=limit,
        )

    @staticmethod
    def _summarize_job(job: dict[str, Any], now: datetime) -> dict[str, Any]:
        created_at = datetime.fromisoformat(str(job["created_at"]))
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
            "elapsed_seconds": round((now - created_at).total_seconds(), 1),
            "data_downloaded": job.get("data_downloaded", False),
        }
        if job.get("job_type") == "optimization":
            summary["total_combinations"] = job.get("total_combinations")
            summary["completed"] = job.get("completed", 0)
        return summary

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued or running background job."""
        job = self._job_store.get_job(job_id)
        if job is None:
            return {
                "success": False,
                "error": f"No job with ID: {job_id}",
                "code": "job_not_found",
            }

        status = str(job["status"])
        if status in self._TERMINAL_STATUSES:
            return {
                "success": False,
                "error": f"Job is already terminal: {status}",
                "code": "invalid_job_state",
                "status": status,
            }
        if status not in self._ACTIVE_STATUSES:
            return {
                "success": False,
                "error": f"Job status is not cancellable: {status}",
                "code": "invalid_job_state",
                "status": status,
            }

        process_terminated = False
        handle = WorkerHandle.from_job(job)
        if handle is None:
            if status != "queued" or job.get("pid") is not None:
                return {
                    "success": False,
                    "error": (
                        "Worker control identity is unavailable; "
                        "cannot confirm process cleanup."
                    ),
                    "code": "worker_identity_unavailable",
                    "status": status,
                    "pid": job.get("pid"),
                }
        else:
            try:
                process_terminated = request_worker_stop(
                    self._job_store.workspace.root, handle
                )
            except (OSError, ValueError, RuntimeError) as exc:
                return {
                    "success": False,
                    "error": f"Worker cleanup could not be confirmed: {exc}",
                    "code": "worker_cleanup_failed",
                    "status": status,
                    "pid": handle.pid,
                }

        self._job_store.update_job(
            job_id,
            status="cancelled",
            error="Cancelled by user.",
            ended_at=datetime.now(UTC).isoformat(),
        )
        return {
            "success": True,
            "job_id": job_id,
            "status": "cancelled",
            "process_terminated": process_terminated,
        }

    def _prepare_execution(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        config_path: Path,
        raw_config: dict[str, Any],
        walk_forward: bool,
        spec: StrategySpec,
        parameters: dict[str, Any] | None = None,
        backtest_overrides: dict[str, Any] | None = None,
    ) -> PreparedExecution | dict[str, Any]:
        kind: Literal["backtest", "walk_forward"] = (
            "walk_forward" if walk_forward else "backtest"
        )
        try:
            bind_strategy_revision(raw_config, spec)
            effective_config = apply_run_overrides(
                raw_config,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )
            apply_backtest_overrides(effective_config, backtest_overrides)
            manifest = BacktestService(
                data_service=self._data_service,
                strategy_gate=self._strategy_service,
                strategy_loader=self._strategy_loader,
            ).prepare_execution(effective_config, kind=kind, parameters=parameters)
        except (
            OSError,
            TypeError,
            ValueError,
            yaml.YAMLError,
            StrategyNotExecutableError,
        ) as exc:
            return self._invalid_execution_request(exc)

        return PreparedExecution(manifest, config_path)

    def _submit_prepared(
        self, prepared: PreparedExecution | dict[str, Any]
    ) -> dict[str, Any]:
        if isinstance(prepared, dict):
            return prepared
        backtest = prepared.manifest.config_copy()["backtest"]
        data_available = self._data_service.data_available(
            backtest["symbol"],
            backtest["timeframe"],
            backtest["start_date"],
            backtest["end_date"],
        )
        submitted = self._submission.submit(prepared)
        data_msg = (
            "Data already cached locally."
            if data_available
            else "Data not fully cached; worker will download it automatically."
        )
        return {
            **submitted,
            "message": f"Job started. Job ID: {submitted['job_id']}. {data_msg}",
            "data_available": data_available,
        }

    def _resolve_walk_forward_config(
        self,
        *,
        strategy_id: str,
        config_path: Path,
        allow_fallback: bool,
    ) -> tuple[Path, dict[str, Any]] | None:
        raw_config = load_config(config_path)
        run_config = BacktestRunConfig.model_validate(raw_config)
        if run_config.is_walk_forward:
            return config_path, raw_config

        fallback = config_path.with_name("walkforward_config.yaml")
        if allow_fallback and fallback.exists():
            fallback_raw = load_config(fallback)
            fallback_run_config = BacktestRunConfig.model_validate(fallback_raw)
            fallback_strategy = fallback_raw.get("strategy", {})
            fallback_id = (
                fallback_strategy.get("id")
                if isinstance(fallback_strategy, dict)
                else None
            )
            if fallback_id == strategy_id and fallback_run_config.is_walk_forward:
                return fallback, fallback_raw
        return None

    @staticmethod
    def _invalid_execution_request(exc: Exception) -> dict[str, Any]:
        """Reject preparation errors before creating a job or starting a worker."""
        return {
            "job_id": "",
            "message": str(exc),
            "data_available": False,
            "code": "invalid_execution_request",
        }

    def _resolve_strategy_run_config(
        self, strategy_id: str, revision_id: str | None
    ) -> tuple[StrategySpec | None, str]:
        try:
            spec = self._strategy_service.resolve_executable(strategy_id, revision_id)
        except StrategyNotExecutableError as exc:
            return None, str(exc)
        path = Path(spec.config_path)
        return (spec, "") if path.exists() else (None, f"Config not found: {path}")

    @staticmethod
    def _process_identity(job: dict[str, Any]) -> ProcessIdentity | None:
        return ProcessIdentity.from_values(
            job.get("pid"), job.get("process_create_time")
        )

    def _is_process_alive(self, job: dict[str, Any]) -> bool:
        identity = self._process_identity(job)
        return identity is not None and identity.get_process() is not None

    def _default_project_root(self) -> Path:
        configured = os.environ.get("TRADINGDEV_PROJECT_ROOT")
        if configured:
            return Path(configured).expanduser().resolve()
        return Path.cwd().resolve()


def _saved_metric_summary(detail: dict[str, Any]) -> dict[str, Any]:
    """Select the original snapshot's summary flags, never today's catalog."""
    return {
        key: value
        for key, value in detail["metrics"].items()
        if detail["definitions"][key]["summary"]
        and detail["mode"] in detail["definitions"][key]["modes"]
    }
