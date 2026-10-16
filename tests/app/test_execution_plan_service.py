"""Approval state and persisted identity with substituted trial/process boundaries."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from tests.integration.execution_fixtures import confirmation_presentation
from tests.mcp.execution_fixtures import (
    FixedPreflight,
    PlanContext,
    RecordingRunner,
    plan_context,
)
from tests.preflight_fixtures import make_preflight_fixture

from tradingdev.app.contracts.plans import PlanRejected, PlanResponse, PlanStarted
from tradingdev.app.execution_plan_service import (
    ConfirmationChallenge,
    ExecutionPlanService,
    PlanOperationError,
)
from tradingdev.app.execution_submission import PreparedExecution
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.preflight_service import PreflightError
from tradingdev.domain.execution_plan import ExecutionPlan
from tradingdev.domain.preflight import PreflightRequest
from tradingdev.domain.presentation.confirmation import ConfirmationPresentation

if TYPE_CHECKING:
    from tradingdev.app.contracts.plans import ExecutionDecision


def _prepare(context: PlanContext) -> PlanResponse:
    arguments = dict(context.arguments)
    presentation = ConfirmationPresentation.model_validate(
        arguments.pop("presentation")
    )
    minimum = arguments.pop("minimum_history_bars")
    sample = arguments.pop("sample_bars")
    result = context.service.prepare(
        PreflightRequest(
            kind="backtest",
            arguments=arguments,
            minimum_history_bars=minimum,
            sample_bars=sample,
        ),
        presentation,
    )
    assert isinstance(result, PlanResponse), result
    return result


def _challenge(context: PlanContext, plan_id: str) -> ConfirmationChallenge:
    result = context.service.begin_confirmation(plan_id)
    assert isinstance(result, ConfirmationChallenge)
    return result


def _rewrite_payload(
    context: PlanContext, plan_id: str, payload: dict[str, Any]
) -> None:
    with context.service.jobs.store.connect() as conn:
        conn.execute(
            "UPDATE execution_plans SET payload=? WHERE plan_id=?",
            (json.dumps(payload), plan_id),
        )


def test_cancellation_invalidates_an_open_challenge(tmp_path: Path) -> None:
    context = plan_context(tmp_path)
    prepared = _prepare(context)
    challenge = _challenge(context, prepared.plan_id)
    cancelled = context.service.cancel(prepared.plan_id)
    assert isinstance(cancelled, PlanResponse) and cancelled.status == "cancelled"
    result = context.service.finish_confirmation(challenge, accepted=True)
    assert isinstance(result, PlanRejected) and result.code == "confirmation_stale"
    assert not context.runner.calls
    assert context.service.jobs.get_job(prepared.plan_id) is None


def test_old_challenge_cannot_answer_a_reopened_form(tmp_path: Path) -> None:
    context = plan_context(tmp_path)
    prepared = _prepare(context)
    first = _challenge(context, prepared.plan_id)
    context.service.abandon_confirmation(first)
    second = _challenge(context, prepared.plan_id)
    rejected = context.service.finish_confirmation(first, accepted=True)
    assert isinstance(rejected, PlanRejected) and rejected.code == "confirmation_stale"
    assert not context.runner.calls
    started = context.service.finish_confirmation(second, accepted=True)
    assert isinstance(started, PlanStarted)


def test_parallel_acceptance_launches_one_job(tmp_path: Path) -> None:
    context = plan_context(tmp_path)
    prepared = _prepare(context)
    challenge = _challenge(context, prepared.plan_id)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(context.service.finish_confirmation, challenge, accepted=True)
            for _ in range(2)
        ]
        results = [future.result(timeout=10) for future in futures]
    assert sum(isinstance(result, PlanStarted) for result in results) == 1
    assert sum(isinstance(result, PlanRejected) for result in results) == 1
    assert len(context.runner.calls) == 1
    assert isinstance(context.service.begin_confirmation(prepared.plan_id), PlanStarted)


@pytest.mark.parametrize("target", ["manifest", "preflight", "document"])
def test_changed_persisted_content_is_rejected_before_question(
    tmp_path: Path,
    target: str,
) -> None:
    context = plan_context(tmp_path)
    prepared = _prepare(context)
    plan, _record = context.service.plans.get(prepared.plan_id)
    payload = plan.model_dump(mode="json")
    if target == "manifest":
        payload[target]["config"]["backtest"]["fees"] = 0.5
    elif target == "preflight":
        payload[target]["trade_count"] = 99
    else:
        payload[target]["summary"] = "Different settings"
    _rewrite_payload(context, prepared.plan_id, payload)
    with pytest.raises(ValueError):
        context.service.begin_confirmation(prepared.plan_id)
    assert not context.runner.calls


@pytest.mark.parametrize("filename", ["confirmation.html", "confirmation.txt"])
def test_changed_confirmation_artifact_cannot_be_approved(
    tmp_path: Path, filename: str
) -> None:
    context = plan_context(tmp_path)
    prepared = _prepare(context)
    challenge = _challenge(context, prepared.plan_id)
    path = context.service.plans.directory(prepared.plan_id) / filename
    path.write_text("altered", encoding="utf-8")
    result = context.service.finish_confirmation(challenge, accepted=True)
    assert isinstance(result, PlanRejected)
    assert "document has changed" in result.error
    assert not context.runner.calls


@pytest.mark.parametrize("after_question", [False, True])
def test_expiration_prevents_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_question: bool
) -> None:
    context = plan_context(tmp_path)
    prepared = _prepare(context)
    challenge = _challenge(context, prepared.plan_id) if after_question else None

    class Later(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> Later:
            return cls.fromtimestamp(
                (datetime.now(UTC) + timedelta(hours=2)).timestamp(), tz
            )

    monkeypatch.setattr("tradingdev.app.execution_plan_service.datetime", Later)
    if challenge is None:
        with pytest.raises(PlanOperationError, match="過期"):
            context.service.begin_confirmation(prepared.plan_id)
    else:
        result = context.service.finish_confirmation(challenge, accepted=True)
        assert isinstance(result, PlanRejected) and result.code == "plan_expired"
        saved = context.service.get(prepared.plan_id)
        assert (
            isinstance(saved, PlanResponse) and saved.status == "failed" and saved.error
        )
    assert not context.runner.calls


def test_rehashed_new_plan_content_does_not_accept_an_old_challenge(
    tmp_path: Path,
) -> None:
    context = plan_context(tmp_path)
    prepared = _prepare(context)
    challenge = _challenge(context, prepared.plan_id)
    payload = challenge.plan.model_dump(mode="json", exclude={"plan_hash"})
    payload["document"]["summary"] = "Updated explanation"
    payload["plan_hash"] = ExecutionPlan.digest(payload)
    _rewrite_payload(context, prepared.plan_id, payload)
    result = context.service.finish_confirmation(challenge, accepted=True)
    assert isinstance(result, PlanRejected) and result.code == "confirmation_stale"
    assert not context.runner.calls


def test_challenge_from_another_plan_cannot_be_used(tmp_path: Path) -> None:
    context = plan_context(tmp_path)
    first = _challenge(context, _prepare(context).plan_id)
    second = _challenge(context, _prepare(context).plan_id)
    result = context.service.finish_confirmation(
        replace(first, token=second.token), accepted=True
    )
    assert isinstance(result, PlanRejected) and result.code == "confirmation_stale"
    assert not context.runner.calls


def test_submission_failure_is_visible_and_cannot_launch_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = plan_context(tmp_path)
    prepared = _prepare(context)
    challenge = _challenge(context, prepared.plan_id)

    def fail(*_args: object) -> None:
        raise OSError("launch unavailable")

    monkeypatch.setattr(context.runner, "spawn_module", fail)
    result = context.service.finish_confirmation(challenge, accepted=True)
    assert isinstance(result, PlanRejected)
    saved = context.service.get(prepared.plan_id)
    assert isinstance(saved, PlanResponse) and saved.status == "failed"
    assert saved.error == "launch unavailable"
    job = context.service.jobs.get_job(prepared.plan_id)
    assert job is not None and job["status"] == "failed"
    with pytest.raises(PlanOperationError):
        context.service.begin_confirmation(prepared.plan_id)


def test_failed_trial_never_publishes_a_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = plan_context(tmp_path)

    def fail(_request: PreflightRequest) -> None:
        raise PreflightError("preflight_timeout", "sample timed out")

    monkeypatch.setattr(context.preflight, "prepare", fail)
    with pytest.raises(AssertionError, match="preflight_timeout"):
        _prepare(context)
    with context.service.jobs.store.connect() as conn:
        assert conn.execute("SELECT count(*) FROM execution_plans").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 0
    assert not (context.service.workspace.root / "execution_plans").exists()
    assert not context.runner.calls


def test_artifact_registration_failure_rolls_back_plan_and_files(
    tmp_path: Path,
) -> None:
    context = plan_context(tmp_path)
    with context.service.jobs.store.connect() as conn:
        conn.execute(
            "CREATE TRIGGER reject_confirmation BEFORE INSERT ON artifacts "
            "BEGIN SELECT RAISE(ABORT, 'artifact failure'); END"
        )
    with pytest.raises(AssertionError, match="artifact failure"):
        _prepare(context)
    with context.service.jobs.store.connect() as conn:
        assert conn.execute("SELECT count(*) FROM execution_plans").fetchone()[0] == 0
    directory = context.service.workspace.root / "execution_plans"
    assert list(directory.iterdir()) == []
    assert not context.runner.calls


def test_final_state_write_failure_retains_and_recovers_the_started_job(
    tmp_path: Path,
) -> None:
    context = plan_context(tmp_path)
    prepared = _prepare(context)
    challenge = _challenge(context, prepared.plan_id)
    with context.service.jobs.store.connect() as conn:
        conn.execute(
            "CREATE TRIGGER reject_submitted BEFORE UPDATE ON execution_plans "
            "WHEN NEW.state='submitted' "
            "BEGIN SELECT RAISE(ABORT, 'final state unavailable'); END"
        )
    result = context.service.finish_confirmation(challenge, accepted=True)
    assert isinstance(result, PlanStarted)
    assert result.job_id == prepared.plan_id
    assert "尚未寫入" in result.message
    saved = context.service.get(prepared.plan_id)
    assert isinstance(saved, PlanResponse) and saved.status == "submitting"
    assert saved.job_id == prepared.plan_id
    repeated = context.service.begin_confirmation(prepared.plan_id)
    assert isinstance(repeated, PlanStarted) and repeated.job_id == result.job_id
    assert len(context.runner.calls) == 1
    with context.service.jobs.store.connect() as conn:
        conn.execute("DROP TRIGGER reject_submitted")
    recovered = context.service.begin_confirmation(prepared.plan_id)
    assert isinstance(recovered, PlanStarted) and recovered.job_id == result.job_id
    saved = context.service.get(prepared.plan_id)
    assert isinstance(saved, PlanResponse) and saved.status == "submitted"
    assert saved.job_id == prepared.plan_id and len(context.runner.calls) == 1


@pytest.mark.parametrize("failed_job", [False, True])
def test_interrupted_submission_without_worker_identity_is_not_relaunched(
    tmp_path: Path,
    failed_job: bool,
) -> None:
    context = plan_context(tmp_path)
    prepared = _prepare(context)
    challenge = _challenge(context, prepared.plan_id)
    context.service.plans.transition(
        prepared.plan_id, "awaiting_confirmation", "submitting"
    )
    context.service.jobs.create_job(
        job_id=prepared.plan_id, manifest=challenge.plan.manifest
    )
    if failed_job:
        context.service.jobs.update_job(
            prepared.plan_id, status="failed", error="spawn failed"
        )
    saved = context.service.get(prepared.plan_id)
    assert isinstance(saved, PlanResponse) and saved.job_id == prepared.plan_id
    with pytest.raises(PlanOperationError, match="尚無完整"):
        context.service.begin_confirmation(prepared.plan_id)
    assert not context.runner.calls


def test_recovery_rejects_job_bound_to_a_different_manifest(tmp_path: Path) -> None:
    context = plan_context(tmp_path)
    prepared = _prepare(context)
    challenge = _challenge(context, prepared.plan_id)
    context.service.plans.transition(
        prepared.plan_id, "awaiting_confirmation", "submitting"
    )
    other_config = challenge.plan.manifest.config_copy()
    other_config["backtest"]["fees"] = 0.5
    changed = type(challenge.plan.manifest).create(
        kind="backtest",
        config=other_config,
        strategy_execution=challenge.plan.manifest.strategy_execution,
    )
    context.service.jobs.create_job(job_id=prepared.plan_id, manifest=changed)
    with pytest.raises(ValueError, match="expected hash"):
        context.service.begin_confirmation(prepared.plan_id)
    assert not context.runner.calls


@pytest.mark.parametrize("changed_file", ["source_path", "config_path"])
def test_changed_generated_revision_after_question_cannot_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_file: str,
) -> None:
    workspace, request = make_preflight_fixture(tmp_path, monkeypatch)
    store = JobStore(workspace=workspace)
    arguments: dict[str, Any] = dict(request.arguments)
    captured = JobService(job_store=store).prepare_backtest(**arguments)
    assert isinstance(captured, PreparedExecution)
    preflight = FixedPreflight(captured)
    runner = RecordingRunner()
    service = ExecutionPlanService(
        workspace, job_store=store, preflight=preflight, process_runner=runner
    )
    presentation = ConfirmationPresentation.model_validate(
        confirmation_presentation(
            captured.manifest.strategy_execution.constructor_kwargs
        )
    )
    prepared = service.prepare(request, presentation)
    assert isinstance(prepared, PlanResponse), prepared
    challenge = service.begin_confirmation(prepared.plan_id)
    assert isinstance(challenge, ConfirmationChallenge)
    spec = service.strategies.resolve_executable(
        "sample_strategy", str(request.arguments["revision_id"])
    )
    path = Path(getattr(spec, changed_file))
    path.write_text(
        path.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8"
    )
    result = service.finish_confirmation(challenge, accepted=True)
    assert isinstance(result, PlanRejected), result
    assert service.jobs.get_job(prepared.plan_id) is None
    assert not runner.calls
    current = service.get(prepared.plan_id)
    assert isinstance(current, PlanResponse) and current.status == "failed"


@pytest.mark.parametrize(
    "decision,code",
    [
        ("approve", None),
        ("decline", "confirmation_declined"),
        ("revise", "confirmation_adjustment_requested"),
    ],
)
def test_conversation_decision_applies_to_the_prepared_plan(
    tmp_path: Path, decision: ExecutionDecision, code: str | None
) -> None:
    context = plan_context(tmp_path)
    prepared = _prepare(context)
    assert prepared.confirmation_summary
    assert "confirmation_text" not in prepared.model_dump()
    assert "<html" in Path(prepared.html_path).read_text(encoding="utf-8")
    assert context.service.get(prepared.plan_id) == prepared
    assert context.runner.calls == []
    assert context.service.jobs.list_all_jobs() == []
    result = context.service.decide(prepared.plan_id, decision=decision)
    if code is None:
        assert isinstance(result, PlanStarted)
        assert result.manifest_hash == prepared.manifest_hash
        assert context.service.decide(prepared.plan_id, decision="approve") == result
        assert len(context.runner.calls) == 1
    else:
        assert isinstance(result, PlanRejected) and result.code == code
        saved = context.service.get(prepared.plan_id)
        assert isinstance(saved, PlanResponse) and saved.status == "cancelled"
        later = context.service.decide(prepared.plan_id, decision="approve")
        assert isinstance(later, PlanRejected) and later.code == "plan_not_ready"
        assert context.runner.calls == []
        assert context.service.jobs.list_all_jobs() == []


def test_conversation_approval_checks_expiration_after_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = plan_context(tmp_path)
    prepared = _prepare(context)

    class Later(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> Later:
            return cls.fromtimestamp(
                (datetime.now(UTC) + timedelta(hours=2)).timestamp(), tz
            )

    monkeypatch.setattr("tradingdev.app.execution_plan_service.datetime", Later)
    result = context.service.decide(prepared.plan_id, decision="approve")
    assert isinstance(result, PlanRejected) and result.code == "plan_expired"
    assert context.runner.calls == []
    assert context.service.jobs.list_all_jobs() == []


@pytest.mark.parametrize(
    "target",
    ["manifest", "preflight", "document", "confirmation.html", "confirmation.txt"],
)
def test_conversation_approval_rejects_changes_to_reviewed_evidence(
    tmp_path: Path, target: str
) -> None:
    context = plan_context(tmp_path)
    prepared = _prepare(context)
    if target.startswith("confirmation."):
        (context.service.plans.directory(prepared.plan_id) / target).write_text(
            "altered", encoding="utf-8"
        )
    else:
        plan, _record = context.service.plans.get(prepared.plan_id)
        payload = plan.model_dump(mode="json")
        if target == "manifest":
            payload[target]["config"]["backtest"]["fees"] = 0.5
        elif target == "preflight":
            payload[target]["trade_count"] = 99
        else:
            payload[target]["summary"] = "Unreviewed settings"
        _rewrite_payload(context, prepared.plan_id, payload)
    result = context.service.decide(prepared.plan_id, decision="approve")
    assert isinstance(result, PlanRejected)
    assert context.runner.calls == []
    assert context.service.jobs.list_all_jobs() == []


def test_parallel_conversation_approvals_submit_only_one_job(tmp_path: Path) -> None:
    context = plan_context(tmp_path)
    prepared = _prepare(context)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(context.service.decide, prepared.plan_id, decision="approve")
            for _ in range(2)
        ]
        results = [future.result(timeout=10) for future in futures]
    started = [result for result in results if isinstance(result, PlanStarted)]
    assert started
    assert {result.job_id for result in started} == {prepared.plan_id}
    assert len(context.runner.calls) == 1
    assert len(context.service.jobs.list_all_jobs()) == 1


@pytest.mark.parametrize("decision", ["decline", "revise"])
def test_conversation_rejection_cannot_cancel_an_already_submitted_job(
    tmp_path: Path, decision: ExecutionDecision
) -> None:
    context = plan_context(tmp_path)
    prepared = _prepare(context)
    started = context.service.decide(prepared.plan_id, decision="approve")
    assert isinstance(started, PlanStarted)
    result = context.service.decide(prepared.plan_id, decision=decision)
    assert isinstance(result, PlanRejected) and result.code == "plan_not_cancellable"
    assert context.service.decide(prepared.plan_id, decision="approve") == started
    assert len(context.runner.calls) == 1
