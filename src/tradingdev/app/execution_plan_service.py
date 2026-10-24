"""Prepare reviewable research plans and apply decisions relayed by clients."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from tradingdev.adapters.execution.process_runner import ProcessRunner, WorkerHandle
from tradingdev.adapters.presentation.confirmation import (
    render_confirmation_html,
    render_confirmation_summary,
    render_confirmation_text,
)
from tradingdev.adapters.storage.execution_plans import ExecutionPlanStore
from tradingdev.app.contracts.plans import (
    ExecutionDecision,
    PlanRejected,
    PlanResponse,
    PlanStarted,
)
from tradingdev.app.execution_submission import (
    ExecutionSubmissionService,
    PreparedExecution,
)
from tradingdev.app.job_config import bind_strategy_revision
from tradingdev.app.job_store import JobStore
from tradingdev.app.preflight_service import PreflightError, PreflightService
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.execution_plan import ExecutionPlan
from tradingdev.domain.presentation.confirmation import (
    ConfirmationPresentation,
    ConfirmationPresentationError,
    build_confirmation_document,
)

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.domain.preflight import PreflightRequest


class PlanOperationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ConfirmationChallenge:
    """Internal reservation binding a submission to one immutable plan."""

    plan: ExecutionPlan
    token: str


class ExecutionPlanService:
    """Share preparation and immutable content across client adapters."""

    def __init__(
        self,
        workspace: WorkspacePaths,
        *,
        job_store: JobStore | None = None,
        preflight: PreflightService | None = None,
        process_runner: ProcessRunner | None = None,
    ) -> None:
        self.workspace = workspace
        self.jobs = job_store or JobStore(workspace=workspace)
        self.plans = ExecutionPlanStore(workspace, self.jobs.store)
        self.preflight = preflight or PreflightService(workspace)
        self.submission = ExecutionSubmissionService(
            self.jobs, process_runner or ProcessRunner(workspace=workspace)
        )
        self.strategies = StrategyService(workspace, store=self.jobs.store)

    def prepare(
        self, request: PreflightRequest, presentation: ConfirmationPresentation
    ) -> PlanResponse | PlanRejected:
        """Return a ready document only after bounded end-to-end trial success."""
        try:
            checked = self.preflight.prepare(request)
            plan_id = uuid4().hex
            manifest = checked.prepared.manifest
            document = build_confirmation_document(
                manifest, presentation, checked.receipt, plan_id=plan_id
            )
            created = datetime.now(UTC)
            payload = {
                "schema_version": 1,
                "plan_id": plan_id,
                "manifest": manifest.model_dump(mode="json"),
                "original_config_path": str(checked.prepared.original_config_path),
                "preflight": checked.receipt.model_dump(mode="json"),
                "document": document.model_dump(mode="json"),
                "created_at": created.isoformat().replace("+00:00", "Z"),
                "expires_at": (created + timedelta(hours=1))
                .isoformat()
                .replace("+00:00", "Z"),
            }
            plan = ExecutionPlan.model_validate(
                {**payload, "plan_hash": ExecutionPlan.digest(payload)}
            )
            text = render_confirmation_text(document)
            self.plans.publish(plan, text, render_confirmation_html(document))
            return self._response(plan, {"state": "ready", "job_id": None})
        except ConfirmationPresentationError as exc:
            return PlanRejected(
                success=False,
                code="invalid_confirmation_presentation",
                error=str(exc),
                required_parameter_paths=exc.required_parameter_paths,
            )
        except PreflightError as exc:
            return PlanRejected(
                success=False,
                code=exc.code,
                error=str(exc),
                execution_guidance=exc.execution_guidance,
            )
        except (OSError, ValueError, sqlite3.Error) as exc:
            return PlanRejected(
                success=False, code="plan_preparation_failed", error=str(exc)
            )

    def get(self, plan_id: str) -> PlanResponse | PlanRejected:
        try:
            plan, record = self.plans.get(plan_id)
            return self._response(plan, record)
        except (LookupError, OSError, ValueError, sqlite3.Error) as exc:
            return self._rejected(exc)

    def begin_confirmation(self, plan_id: str) -> ConfirmationChallenge | PlanStarted:
        """Reserve a submission after the client relays the user's approval."""
        with self.plans.lock(plan_id):
            plan, record = self.plans.get(plan_id)
            if record["state"] == "submitted":
                return self._started(plan, record["job_id"])
            if record["state"] == "submitting":
                job = self._bound_job(plan)
                if job is None or WorkerHandle.from_job(job) is None:
                    raise PlanOperationError(
                        "submission_incomplete",
                        "尚無完整的工作啟動紀錄；請查詢計畫與工作狀態，"
                        "不會再次啟動工作。",
                    )
                return self._record_submission(plan)
            if record["state"] != "ready":
                raise PlanOperationError(
                    "plan_not_ready",
                    "此計畫正在等待確認、已取消或執行失敗，請查看計畫狀態。",
                )
            self._verify_ready(plan)
            token = uuid4().hex
            self.plans.transition(
                plan_id, "ready", "awaiting_confirmation", token=token
            )
            return ConfirmationChallenge(plan, token)

    def decide(
        self, plan_id: str, *, decision: ExecutionDecision
    ) -> PlanStarted | PlanRejected:
        """Apply the client's report of the user's explicit conversation decision.

        The caller must collect that decision after presenting the saved HTML.
        This service validates the plan, not the human origin of the decision.
        """
        if decision not in {"approve", "decline", "revise"}:
            return PlanRejected(
                success=False,
                code="invalid_execution_decision",
                error="不支援的確認決策。",
            )
        if decision != "approve":
            cancelled = self.cancel(plan_id)
            if isinstance(cancelled, PlanRejected):
                return cancelled
            return PlanRejected(
                success=False,
                code="confirmation_adjustment_requested"
                if decision == "revise"
                else "confirmation_declined",
                error=(
                    "使用者要求修改設定，原計畫已取消，正式回測未啟動。"
                    "請收集修改內容，重新準備、試跑並等待使用者確認新計畫。"
                    if decision == "revise"
                    else "使用者未同意，計畫已取消，正式回測未啟動。"
                ),
            )
        try:
            challenge = self.begin_confirmation(plan_id)
            if isinstance(challenge, PlanStarted):
                return challenge
            return self.finish_confirmation(challenge, accepted=True)
        except (LookupError, OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
            return self._rejected(exc)

    def finish_confirmation(
        self, challenge: ConfirmationChallenge, *, accepted: bool
    ) -> PlanStarted | PlanRejected:
        """Complete one reservation, checking it still names the same ready plan."""
        plan_id = challenge.plan.plan_id
        try:
            with self.plans.lock(plan_id):
                plan, record = self.plans.get(plan_id)
                if (
                    record["state"] != "awaiting_confirmation"
                    or record["confirmation_token"] != challenge.token
                    or plan.plan_hash != challenge.plan.plan_hash
                ):
                    raise PlanOperationError(
                        "confirmation_stale", "確認已失效，未啟動新的工作。"
                    )
                if not accepted:
                    self.plans.transition(plan_id, "awaiting_confirmation", "cancelled")
                    return PlanRejected(
                        success=False,
                        code="confirmation_declined",
                        error="使用者未同意，正式回測未啟動。",
                    )
                try:
                    self._verify_ready(plan)
                except (OSError, ValueError, RuntimeError) as exc:
                    self.plans.transition(
                        plan_id, "awaiting_confirmation", "failed", error=str(exc)
                    )
                    raise
                self.plans.transition(plan_id, "awaiting_confirmation", "submitting")
                try:
                    self.submission.submit(
                        PreparedExecution(
                            plan.manifest, Path(plan.original_config_path)
                        ),
                        job_id=plan_id,
                    )
                except BaseException as exc:
                    self.plans.transition(
                        plan_id, "submitting", "failed", error=str(exc)
                    )
                    raise
                return self._record_submission(plan)
        except (LookupError, OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
            return self._rejected(exc)

    def abandon_confirmation(self, challenge: ConfirmationChallenge) -> None:
        """Release an unused reservation without authorizing execution."""
        with self.plans.lock(challenge.plan.plan_id):
            _plan, record = self.plans.get(challenge.plan.plan_id)
            if (
                record["state"] == "awaiting_confirmation"
                and record["confirmation_token"] == challenge.token
            ):
                self.plans.transition(
                    challenge.plan.plan_id, "awaiting_confirmation", "ready"
                )

    def cancel(self, plan_id: str) -> PlanResponse | PlanRejected:
        try:
            with self.plans.lock(plan_id):
                plan, record = self.plans.get(plan_id)
                if record["state"] in {"ready", "awaiting_confirmation"}:
                    self.plans.transition(plan_id, record["state"], "cancelled")
                elif record["state"] != "cancelled":
                    raise PlanOperationError(
                        "plan_not_cancellable", "已啟動的工作請使用取消工作功能。"
                    )
            return self.get(plan_id)
        except (LookupError, OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
            return self._rejected(exc)

    def _verify_ready(self, plan: ExecutionPlan) -> None:
        plan.verify()
        if datetime.now(UTC) >= plan.expires_at:
            raise PlanOperationError("plan_expired", "計畫已過期，請重新準備與試跑。")
        config = plan.manifest.config_for_execution()
        strategy = config["strategy"]
        selected = self.strategies.resolve_executable(
            strategy["id"], strategy.get("revision_id")
        )
        bind_strategy_revision(config, selected)

    def _response(self, plan: ExecutionPlan, record: dict[str, Any]) -> PlanResponse:
        job_id = record.get("job_id")
        if job_id is None and record["state"] in {"submitting", "failed"}:
            job = self._bound_job(plan)
            if job is not None:
                job_id = plan.plan_id
        return PlanResponse(
            plan_id=plan.plan_id,
            status=record["state"],
            manifest_hash=plan.manifest.manifest_hash,
            expires_at=plan.expires_at.isoformat(),
            confirmation_summary=render_confirmation_summary(
                plan.document, mode=plan.manifest.config_copy()["backtest"]["mode"]
            ),
            html_path=str(self.plans.directory(plan.plan_id) / "confirmation.html"),
            artifact_id=f"{plan.plan_id}:confirmation_html",
            preflight=plan.preflight,
            job_id=job_id,
            error=record.get("error"),
        )

    def _bound_job(self, plan: ExecutionPlan) -> dict[str, Any] | None:
        """Recover only the deterministic job with this plan's exact manifest."""
        job = self.jobs.get_job(plan.plan_id)
        if job is not None:
            self.jobs.load_manifest(plan.plan_id).verify(
                expected_hash=plan.manifest.manifest_hash
            )
        return job

    def _record_submission(self, plan: ExecutionPlan) -> PlanStarted:
        """A failed final status write must not hide an already launched job."""
        started = self._started(plan, plan.plan_id)
        try:
            self.plans.transition(
                plan.plan_id, "submitting", "submitted", job_id=plan.plan_id
            )
        except (OSError, ValueError, sqlite3.Error) as exc:
            return started.model_copy(
                update={
                    "message": (
                        "工作已提交，但確認計畫的最終狀態尚未寫入。"
                        f"請以回傳的工作識別查詢進度；重試不會重複執行。原因：{exc}"
                    )
                }
            )
        return started

    def _started(self, plan: ExecutionPlan, job_id: str) -> PlanStarted:
        manifest = self.jobs.load_manifest(job_id)
        manifest.verify(expected_hash=plan.manifest.manifest_hash)
        return PlanStarted(
            plan_id=plan.plan_id,
            job_id=job_id,
            manifest_hash=manifest.manifest_hash,
            message="已依使用者確認的設定提交工作；可查詢工作狀態。",
        )

    @staticmethod
    def _rejected(exc: Exception) -> PlanRejected:
        return PlanRejected(
            success=False,
            code=exc.code
            if isinstance(exc, PlanOperationError)
            else "execution_plan_invalid",
            error=str(exc),
        )
