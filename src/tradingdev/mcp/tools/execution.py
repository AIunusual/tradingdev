"""Offline program checks, HTML review and explicit conversational decisions."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import anyio
from mcp_types import ToolAnnotations
from pydantic import JsonValue, ValidationError  # noqa: TC002

from tradingdev.app.contracts.common import ErrorResponse  # noqa: TC001
from tradingdev.app.contracts.execution_requests import (
    BacktestInput,
    ComparisonInput,
    OptimizationInput,
    ParameterCandidateInput,  # noqa: TC001
    WalkForwardInput,
)
from tradingdev.app.contracts.plans import (
    ExecutionDecision,  # noqa: TC001
    PlanRejected,
    PlanResponse,  # noqa: TC001
    PlanStarted,  # noqa: TC001
)
from tradingdev.app.contracts.research_evidence import ResearchEvidenceSaved  # noqa: TC001
from tradingdev.domain.preflight import PreflightRequest
from tradingdev.domain.research import ResearchSpec  # noqa: TC001
from tradingdev.domain.research_evidence import (
    EvidenceId,  # noqa: TC001
    ResearchEvidenceDraft,  # noqa: TC001
)
from tradingdev.domain.validation.config import WalkForwardConfig  # noqa: TC001
from tradingdev.mcp.tools.input_errors import invalid_arguments

if TYPE_CHECKING:
    from mcp.server import MCPServer

    from tradingdev.app.execution_plan_service import ExecutionPlanService
    from tradingdev.domain.execution import ExecutionKind


def register(mcp: MCPServer, service: ExecutionPlanService) -> None:
    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        )
    )
    def save_research_evidence(
        draft: ResearchEvidenceDraft,
    ) -> ResearchEvidenceSaved | ErrorResponse:
        """Save client-reported rules, sources and parameter origins as a new record.

        Read get_strategy_contract(section="evidence") for exact binding requirements.
        Use only references actually supplied by the user or retrieved by your host;
        without retrieval use model_proposal. The server never fetches URLs or proves
        browsing/source truth. Reference parameter origins require explicit source
        value claims. Revisions create a new evidence_id; saved records are immutable.
        """
        return service.evidence.save(draft)

    prepare_annotations = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=True,
    )

    async def prepare(
        kind: ExecutionKind | Literal["comparison"],
        arguments: dict[str, Any],
        evidence_id: EvidenceId,
    ) -> PlanResponse | PlanRejected:
        try:
            parsed = (
                ComparisonInput.model_validate(arguments)
                if kind == "comparison"
                else OptimizationInput.model_validate(arguments)
                if kind == "optimization"
                else WalkForwardInput.model_validate(arguments)
                if kind == "walk_forward"
                else BacktestInput.model_validate(arguments)
            )
            request = PreflightRequest(
                kind=kind,
                arguments=parsed.model_dump(mode="json", exclude_unset=True),
            )
        except ValidationError as exc:
            raise invalid_arguments(exc) from exc
        result = await anyio.to_thread.run_sync(
            partial(service.prepare, request, evidence_id)
        )
        if isinstance(result, PlanRejected) and result.execution_guidance is not None:
            return result.model_copy(
                update={
                    "next_tool": "prepare_walk_forward",
                    "error": (
                        f"{result.error} Use prepare_walk_forward after confirming "
                        "the strategy has valid validation settings. Read "
                        "get_strategy to inspect its configuration first."
                    ),
                }
            )
        return result

    @mcp.tool(annotations=prepare_annotations)
    async def prepare_backtest(
        strategy_id: str,
        research: ResearchSpec,
        evidence_id: EvidenceId,
        revision_id: str | None = None,
        parameters: dict[str, JsonValue] | None = None,
        backtest_overrides: dict[str, JsonValue] | None = None,
    ) -> PlanResponse | PlanRejected:
        """Prepare one frozen backtest plan using synthetic program checks only.

        Requires a runnable/promoted revision and saved evidence_id. parameters
        recursively overrides base parameters; costs belong only in research,
        backtest_overrides changes other execution settings. Explicit inheritance
        of a generated strategy requires revision_id.
        Follow the shared confirmation workflow: show confirmation_summary and
        HTML confirmation_link verbatim, end the turn and await explicit approval."""
        args = {
            "strategy_id": strategy_id,
            "research": research,
            "revision_id": revision_id,
            "parameters": parameters,
            "backtest_overrides": backtest_overrides,
        }
        return await prepare("backtest", args, evidence_id)

    @mcp.tool(annotations=prepare_annotations)
    async def prepare_comparison(
        strategy_id: str,
        research: ResearchSpec,
        candidates: list[ParameterCandidateInput],
        evidence_id: EvidenceId,
        revision_id: str | None = None,
        backtest_overrides: dict[str, JsonValue] | None = None,
    ) -> PlanResponse | PlanRejected:
        """Prepare 1–8 fixed parameter sets for independent backtests on shared data.

        Each candidate needs a label, rationale and parameter overrides. Ordinary
        backtest strategies only; training-required research needs walk-forward.
        evidence_id must cover every candidate's effective parameter leaves;
        see get_strategy_contract(section="evidence"). Preparation is synthetic only.
        Follow the shared confirmation workflow: show confirmation_summary and
        HTML confirmation_link verbatim, end the turn and await whole-set approval.
        Report the completed set with generate_report(comparison_job_id=job_id)."""
        return await prepare(
            "comparison",
            {
                "strategy_id": strategy_id,
                "research": research,
                "candidates": candidates,
                "revision_id": revision_id,
                "backtest_overrides": backtest_overrides,
            },
            evidence_id,
        )

    @mcp.tool(annotations=prepare_annotations)
    async def prepare_walk_forward(
        strategy_id: str,
        research: ResearchSpec,
        evidence_id: EvidenceId,
        revision_id: str | None = None,
        parameters: dict[str, JsonValue] | None = None,
        backtest_overrides: dict[str, JsonValue] | None = None,
        validation_config: WalkForwardConfig | None = None,
    ) -> PlanResponse | PlanRejected:
        """Prepare training/validation using synthetic program checks only.

        validation_config replaces the complete validation section; omission keeps
        the selected configuration. Choose mode=automatic folds, mode=explicit
        inclusive UTC day windows, or mode=purged folds with positive label_horizon.
        Explicit windows must fit the research period. Supply saved evidence_id.
        Follow the shared confirmation workflow: show confirmation_summary and
        HTML confirmation_link verbatim, end the turn and await explicit approval."""
        args = {
            "strategy_id": strategy_id,
            "research": research,
            "revision_id": revision_id,
            "parameters": parameters,
            "backtest_overrides": backtest_overrides,
            "validation_config": validation_config,
        }
        return await prepare("walk_forward", args, evidence_id)

    @mcp.tool(annotations=prepare_annotations)
    async def prepare_optimization(
        strategy_id: str,
        research: ResearchSpec,
        param_ranges: dict[str, list[JsonValue]],
        optimization_metric: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        evidence_id: EvidenceId,
        revision_id: str | None = None,
        parameters: dict[str, JsonValue] | None = None,
        backtest_overrides: dict[str, JsonValue] | None = None,
    ) -> PlanResponse | PlanRejected:
        """Prepare an explicitly requested parameter search; no candidate backtests.

        Dates are inclusive UTC days: train_start < train_end < test_start < test_end.
        Research covers exactly train_start 00:00 through test_end 23:59:59.999999 UTC.
        evidence_id must cover base and every merged search-axis option;
        see get_strategy_contract(section="evidence").
        Follow the shared confirmation workflow: show confirmation_summary and
        HTML confirmation_link verbatim, end the turn and await explicit approval."""
        args = {
            "strategy_id": strategy_id,
            "research": research,
            "param_ranges": param_ranges,
            "optimization_metric": optimization_metric,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "revision_id": revision_id,
            "parameters": parameters,
            "backtest_overrides": backtest_overrides,
        }
        return await prepare("optimization", args, evidence_id)

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def get_execution_plan(plan_id: str) -> PlanResponse | PlanRejected:
        """Read saved confirmation and program-check evidence without rerunning code.

        Use the shared confirmation workflow when presenting a ready plan."""
        return service.get(plan_id)

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def cancel_execution_plan(plan_id: str) -> PlanResponse | PlanRejected:
        """Cancel a ready or awaiting plan; use cancel_job for a started job."""
        return service.cancel(plan_id)

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=True,
        )
    )
    async def submit_execution_decision(
        plan_id: str, decision: ExecutionDecision
    ) -> PlanStarted | PlanRejected:
        """Relay the user's explicit later decision about this exact reviewed plan.

        approve starts execution; decline/revise cancel. Follow the shared decision
        workflow: changes or expiry need a new plan and fresh approval. The backend
        verifies plan integrity; it cannot prove the decision's human origin."""
        return await anyio.to_thread.run_sync(
            partial(service.decide, plan_id, decision=decision)
        )
