"""Offline program checks, HTML review and explicit conversational decisions."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import anyio
from mcp_types import ToolAnnotations
from pydantic import JsonValue, ValidationError  # noqa: TC002

from tradingdev.app.contracts.execution_requests import (
    BacktestInput,
    OptimizationInput,
    WalkForwardInput,
)
from tradingdev.app.contracts.plans import (
    ExecutionDecision,  # noqa: TC001
    PlanRejected,
    PlanResponse,  # noqa: TC001
    PlanStarted,  # noqa: TC001
)
from tradingdev.domain.backtest.schemas import WalkForwardConfig  # noqa: TC001
from tradingdev.domain.preflight import PreflightRequest
from tradingdev.domain.presentation.confirmation import (
    ConfirmationPresentation,  # noqa: TC001
)
from tradingdev.domain.research import ResearchSpec  # noqa: TC001
from tradingdev.mcp.tools.input_errors import invalid_arguments

if TYPE_CHECKING:
    from mcp.server import MCPServer

    from tradingdev.app.execution_plan_service import ExecutionPlanService
    from tradingdev.domain.execution import ExecutionKind


def register(mcp: MCPServer, service: ExecutionPlanService) -> None:
    prepare_annotations = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=True,
    )

    async def prepare(
        kind: ExecutionKind,
        arguments: dict[str, Any],
        presentation: ConfirmationPresentation,
    ) -> PlanResponse | PlanRejected:
        try:
            parsed = (
                OptimizationInput.model_validate(arguments)
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
            partial(service.prepare, request, presentation)
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
        presentation: ConfirmationPresentation,
        revision_id: str | None = None,
        parameters: dict[str, JsonValue] | None = None,
        backtest_overrides: dict[str, JsonValue] | None = None,
    ) -> PlanResponse | PlanRejected:
        """Prepare a frozen plan with program checks; no market data or backtest.

        Requires a runnable/promoted revision. Supply presentation labels/units for
        every constructor parameter including defaults. Missing labels return exact
        required_parameter_paths. parameters recursively overrides base parameters;
        research requires strategy outline, market, timeframe and period, or explicit
        inheritance with an explicit generated revision_id.
        Costs belong only in research;
        backtest_overrides changes other execution settings.
        Present the returned confirmation_summary verbatim and HTML link, end the turn,
        and wait for a later explicit user decision before submitting this exact plan.
        """
        args = {
            "strategy_id": strategy_id,
            "research": research,
            "revision_id": revision_id,
            "parameters": parameters,
            "backtest_overrides": backtest_overrides,
        }
        return await prepare("backtest", args, presentation)

    @mcp.tool(annotations=prepare_annotations)
    async def prepare_walk_forward(
        strategy_id: str,
        research: ResearchSpec,
        presentation: ConfirmationPresentation,
        revision_id: str | None = None,
        parameters: dict[str, JsonValue] | None = None,
        backtest_overrides: dict[str, JsonValue] | None = None,
        validation_config: WalkForwardConfig | None = None,
    ) -> PlanResponse | PlanRejected:
        """Prepare a walk-forward plan without market data, training or backtests.

        Requires validation training/test windows and a runnable/promoted revision.
        validation_config replaces the complete validation section; use automatic
        folds or explicit inclusive UTC day windows within the research period.
        Omission keeps the selected configuration.
        Supply human parameter presentation as for prepare_backtest. Program checks
        do not prove training or folds will succeed. Present confirmation_summary
        verbatim and its HTML link; end the turn and await the user's later decision.
        """
        args = {
            "strategy_id": strategy_id,
            "research": research,
            "revision_id": revision_id,
            "parameters": parameters,
            "backtest_overrides": backtest_overrides,
            "validation_config": validation_config,
        }
        return await prepare("walk_forward", args, presentation)

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
        presentation: ConfirmationPresentation,
        revision_id: str | None = None,
        parameters: dict[str, JsonValue] | None = None,
        backtest_overrides: dict[str, JsonValue] | None = None,
    ) -> PlanResponse | PlanRejected:
        """Prepare a pinned search with program and grid checks; no candidate backtests.

        Dates are inclusive UTC days: train_start < train_end < test_start < test_end.
        Research covers exactly train_start 00:00 through
        test_end 23:59:59.999999 UTC.
        Use human parameter descriptions. The HTML contains ranges, objective
        direction, costs and program-check limits. Present its
        link and verbatim confirmation_summary; end the turn and await a later decision.
        An approved search has no second worker-side confirmation.
        """
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
        return await prepare("optimization", args, presentation)

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def get_execution_plan(plan_id: str) -> PlanResponse | PlanRejected:
        """Read saved confirmation and program-check evidence without rerunning code."""
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

        Only approval after presenting its summary/HTML permits approve; the initial
        request, silence, questions and ambiguous replies are not consent. decline and
        revise cancel without launching; changes/expiry require a newly prepared plan
        and fresh approval. The backend verifies identity, expiry and source integrity
        and deduplicates submissions; it cannot prove the decision's human origin.
        """
        return await anyio.to_thread.run_sync(
            partial(service.decide, plan_id, decision=decision)
        )
