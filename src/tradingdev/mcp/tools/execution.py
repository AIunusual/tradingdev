"""Bounded preparation, HTML review and explicit conversational decisions."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import anyio
from mcp_types import ToolAnnotations
from pydantic import JsonValue, ValidationError  # noqa: TC002

from tradingdev.app.contracts.plans import (
    ExecutionDecision,  # noqa: TC001
    PlanRejected,
    PlanResponse,  # noqa: TC001
    PlanStarted,  # noqa: TC001
)
from tradingdev.domain.preflight import (
    MinimumHistoryBars,  # noqa: TC001 — SDK resolves tool annotations.
    PreflightRequest,
    SampleBars,  # noqa: TC001 — SDK resolves tool annotations.
)
from tradingdev.domain.presentation.confirmation import (
    ConfirmationPresentation,  # noqa: TC001
)
from tradingdev.mcp.schemas import BacktestInput, OptimizationInput
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
        arguments: BacktestInput | OptimizationInput,
        minimum_history_bars: int,
        sample_bars: int,
        presentation: ConfirmationPresentation,
    ) -> PlanResponse | PlanRejected:
        try:
            request = PreflightRequest(
                kind=kind,
                arguments=arguments.model_dump(mode="json"),
                minimum_history_bars=minimum_history_bars,
                sample_bars=sample_bars,
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
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        presentation: ConfirmationPresentation,
        minimum_history_bars: MinimumHistoryBars,
        revision_id: str | None = None,
        parameters: dict[str, JsonValue] | None = None,
        backtest_overrides: dict[str, JsonValue] | None = None,
        sample_bars: SampleBars = 1024,
    ) -> PlanResponse | PlanRejected:
        """Prepare a frozen backtest plan with a bounded trial; no full job starts.

        Requires a runnable/promoted revision. Supply minimum_history_bars per window,
        presentation labels/units for every constructor parameter including defaults,
        and sample_bars as the total trial budget. Missing labels return exact
        required_parameter_paths. parameters recursively overrides base parameters;
        backtest_overrides changes cost/capital/execution fields, not dates/market.
        Present the returned confirmation_summary verbatim and HTML link, end the turn,
        and wait for a later explicit user decision before submitting this exact plan.
        """
        args = BacktestInput(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            revision_id=revision_id,
            parameters=parameters,
            backtest_overrides=backtest_overrides,
        )
        return await prepare(
            "backtest",
            args,
            minimum_history_bars,
            sample_bars,
            presentation,
        )

    @mcp.tool(annotations=prepare_annotations)
    async def prepare_walk_forward(
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        presentation: ConfirmationPresentation,
        minimum_history_bars: MinimumHistoryBars,
        revision_id: str | None = None,
        parameters: dict[str, JsonValue] | None = None,
        backtest_overrides: dict[str, JsonValue] | None = None,
        sample_bars: SampleBars = 1024,
    ) -> PlanResponse | PlanRejected:
        """Prepare a frozen walk-forward plan using a bounded representative fold.

        Requires validation training/test windows and a runnable/promoted revision.
        Supply minimum history and human parameter presentation as for prepare_backtest.
        The sample does not prove all folds will succeed. Present confirmation_summary
        verbatim and its HTML link; end the turn and await the user's later decision.
        """
        args = BacktestInput(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            revision_id=revision_id,
            parameters=parameters,
            backtest_overrides=backtest_overrides,
        )
        return await prepare(
            "walk_forward",
            args,
            minimum_history_bars,
            sample_bars,
            presentation,
        )

    @mcp.tool(annotations=prepare_annotations)
    async def prepare_optimization(
        strategy_id: str,
        symbol: str,
        timeframe: str,
        param_ranges: dict[str, list[JsonValue]],
        optimization_metric: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        presentation: ConfirmationPresentation,
        minimum_history_bars: MinimumHistoryBars,
        revision_id: str | None = None,
        parameters: dict[str, JsonValue] | None = None,
        backtest_overrides: dict[str, JsonValue] | None = None,
        sample_bars: SampleBars = 1024,
    ) -> PlanResponse | PlanRejected:
        """Prepare a pinned search after checking the grid and trialing one candidate.

        Dates are inclusive UTC days: train_start < train_end < test_start < test_end.
        Use human parameter descriptions and the strategy's minimum history. The HTML
        contains ranges, objective direction, costs and sample coverage. Present its
        link and verbatim confirmation_summary; end the turn and await a later decision.
        An approved search has no second worker-side confirmation.
        """
        args = OptimizationInput(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            param_ranges=param_ranges,
            optimization_metric=optimization_metric,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            revision_id=revision_id,
            parameters=parameters,
            backtest_overrides=backtest_overrides,
        )
        return await prepare(
            "optimization",
            args,
            minimum_history_bars,
            sample_bars,
            presentation,
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def get_execution_plan(plan_id: str) -> PlanResponse | PlanRejected:
        """Read saved confirmation and trial evidence without rerunning a strategy."""
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
