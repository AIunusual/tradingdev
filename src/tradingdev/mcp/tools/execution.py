"""Bounded preparation, HTML review and explicit conversational decisions."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import anyio
from mcp_types import ToolAnnotations
from pydantic import JsonValue, ValidationError  # noqa: TC002

from tradingdev.app.contracts.plans import (
    ExecutionDecision,  # noqa: TC001
    PlanRejected,  # noqa: TC001
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
        return await anyio.to_thread.run_sync(
            partial(service.prepare, request, presentation)
        )

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
        """Prepare a frozen plan after a bounded market-data/engine trial; no full job.

        Complete source validation and dry-run first. Supply the strategy's minimum
        history needed per sample window; sample_bars is the total budget (64..4096).
        presentation uses human labels/units, never values: parameter_descriptions
        keys are JSON pointers into captured constructor arguments (generated
        /fast_period; bundled /config/k_period and /fit_config/...). Include defaults.
        Missing descriptions return required_parameter_paths for repair. Values,
        costs, coverage and HTML are rendered by the backend. Quote
        confirmation_summary verbatim, preferably as a Markdown blockquote.
        Preserve wording, labels, values, units and train/test roles; do not
        paraphrase, rename, omit or recalculate them. For a locally accessible
        html_path, use the clickable HTML link [閱讀確認單](<FULL_HTML_PATH>), replacing
        FULL_HTML_PATH with the exact returned html_path. Do not add file:// or
        invent another URI prefix. Put the link on a separate normal Markdown line,
        outside the quote and code blocks, then STOP this turn.
        Wait for the user to review and explicitly approve, decline or request edits
        in a later conversation turn before calling submit_execution_decision.
        The initial task request is not approval of a not-yet-reviewed plan.
        backtest_overrides can change known cost/capital/execution fields, not
        symbol/timeframe/dates. Edits require a new plan and another successful trial.
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
        """Prepare walk-forward confirmation after a bounded representative fold.

        Uses the same presentation and sample rules as prepare_backtest. A sample
        does not prove all folds or full-history training will succeed. Human
        confirmation in a later conversation turn is required before submission.
        Quote confirmation_summary verbatim, preserving labels, values, units and
        train/test roles; do not paraphrase. Prefer a Markdown blockquote. For a
        locally accessible html_path, use the clickable HTML link
        [閱讀確認單](<FULL_HTML_PATH>), replacing FULL_HTML_PATH with the exact returned
        html_path. Do not add file:// or invent another URI prefix. Put the link
        on a separate normal Markdown line outside quotes/code blocks; end and wait.
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

        Dates are inclusive UTC days, train_start < train_end < test_start < test_end.
        Present the complete ranges, objective direction, costs and sample coverage
        in the HTML. Use human parameter descriptions as prepare_backtest.
        Quote confirmation_summary verbatim, preserving labels, values, units and
        train/test roles; do not paraphrase. Prefer a Markdown blockquote. For a
        locally accessible html_path, use the clickable HTML link
        [閱讀確認單](<FULL_HTML_PATH>), replacing FULL_HTML_PATH with the exact returned
        html_path. Do not add file:// or invent another URI prefix. Put the link
        on a separate normal Markdown line outside quotes/code blocks; end and wait.
        There is no additional worker-side confirmation after the search starts.
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
        """Relay an explicit user decision AFTER they reviewed this plan's HTML.

        First quote confirmation_summary verbatim; preserve labels, values, units
        and train/test roles without paraphrasing. Prefer a Markdown blockquote.
        For a locally accessible html_path, use the clickable HTML link
        [閱讀確認單](<FULL_HTML_PATH>), replacing FULL_HTML_PATH with the exact returned
        html_path. Do not add file:// or invent another URI prefix. Put the link
        on a separate normal Markdown line outside quotes/code blocks; end and wait.
        Only a later clear user approval of this exact plan
        permits decision='approve'. The initial request, silence, questions or
        ambiguous replies are not consent. Ask for clarification if unclear.
        Use 'decline' for refusal and 'revise' for requested changes; both cancel
        the plan without launching. For changes, prepare a new plan and wait for
        fresh user approval. Never approve a replacement plan automatically.
        The backend trusts the caller to relay the user's decision; it cannot
        independently prove human consent. It verifies the immutable plan,
        expiry and source integrity, and deduplicates successful submissions.
        No native MCP elicitation support is required.
        """
        return await anyio.to_thread.run_sync(
            partial(service.decide, plan_id, decision=decision)
        )
