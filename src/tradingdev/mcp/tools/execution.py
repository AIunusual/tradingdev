"""Bounded preparation followed by client-mediated user confirmation."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import anyio
from mcp.server.fastmcp import Context  # noqa: TC002
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from tradingdev.app.contracts.plans import PlanRejected, PlanResponse, PlanStarted
from tradingdev.app.execution_plan_service import (
    ExecutionPlanService,
    PlanOperationError,
)
from tradingdev.domain.preflight import PreflightRequest
from tradingdev.domain.presentation.confirmation import (
    ConfirmationPresentation,  # noqa: TC001
)
from tradingdev.mcp.schemas import BacktestInput, OptimizationInput

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from pydantic.json_schema import JsonSchemaValue


def _elicitation_form_schema(schema: JsonSchemaValue) -> None:
    """Emit MCP's restricted root schema while keeping strict response validation."""
    for keyword in ("title", "description", "additionalProperties"):
        schema.pop(keyword, None)


class UserApproval(BaseModel):
    """Only the client elicitation response can supply this model."""

    model_config = ConfigDict(
        extra="forbid", strict=True, json_schema_extra=_elicitation_form_schema
    )
    approved: bool = Field(default=False, title="同意依上述設定開始正式執行")


def register(mcp: FastMCP, service: ExecutionPlanService) -> None:
    prepare_annotations = ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )

    async def prepare(
        request: PreflightRequest, presentation: ConfirmationPresentation
    ) -> PlanResponse | PlanRejected:
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
        minimum_history_bars: int,
        revision_id: str | None = None,
        parameters: dict[str, JsonValue] | None = None,
        backtest_overrides: dict[str, JsonValue] | None = None,
        sample_bars: int = 1024,
    ) -> PlanResponse | PlanRejected:
        """Prepare a frozen plan after a bounded market-data/engine trial; no full job.

        Complete source validation and dry-run first. Supply the strategy's minimum
        history needed per sample window; sample_bars is the total budget (64..4096).
        presentation uses human labels/units, never values: parameter_descriptions
        keys are JSON pointers into captured constructor arguments (generated
        /fast_period; bundled /config/k_period and /fit_config/...). Include defaults.
        Missing descriptions return required_parameter_paths for repair. Values,
        costs, coverage, text and HTML are rendered by the backend. Present the
        confirmation_text and HTML link, then request_execution_confirmation(plan_id).
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
            PreflightRequest(
                kind="backtest",
                arguments=args.model_dump(mode="json"),
                minimum_history_bars=minimum_history_bars,
                sample_bars=sample_bars,
            ),
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
        minimum_history_bars: int,
        revision_id: str | None = None,
        parameters: dict[str, JsonValue] | None = None,
        backtest_overrides: dict[str, JsonValue] | None = None,
        sample_bars: int = 1024,
    ) -> PlanResponse | PlanRejected:
        """Prepare walk-forward confirmation after a bounded representative fold.

        Uses the same presentation and sample rules as prepare_backtest. A sample
        does not prove all folds or full-history training will succeed. Human
        confirmation through request_execution_confirmation is required to launch.
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
            PreflightRequest(
                kind="walk_forward",
                arguments=args.model_dump(mode="json"),
                minimum_history_bars=minimum_history_bars,
                sample_bars=sample_bars,
            ),
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
        minimum_history_bars: int,
        revision_id: str | None = None,
        parameters: dict[str, JsonValue] | None = None,
        backtest_overrides: dict[str, JsonValue] | None = None,
        sample_bars: int = 1024,
    ) -> PlanResponse | PlanRejected:
        """Prepare a pinned search after checking the grid and trialing one candidate.

        Dates are inclusive UTC days, train_start < train_end < test_start < test_end.
        Present the complete ranges, objective direction, costs and sample coverage
        from confirmation_text. Use human parameter descriptions as prepare_backtest.
        request_execution_confirmation collects human consent for the whole search;
        there is no additional worker-side confirmation after the search starts.
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
            PreflightRequest(
                kind="optimization",
                arguments=args.model_dump(mode="json"),
                minimum_history_bars=minimum_history_bars,
                sample_bars=sample_bars,
            ),
            presentation,
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    def get_execution_plan(plan_id: str) -> PlanResponse | PlanRejected:
        """Read saved confirmation and trial evidence without rerunning a strategy."""
        return service.get(plan_id)

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    def cancel_execution_plan(plan_id: str) -> PlanResponse | PlanRejected:
        """Cancel a ready or awaiting plan; use cancel_job for a started job."""
        return service.cancel(plan_id)

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    async def request_execution_confirmation(
        plan_id: str,
        ctx: Context[Any, Any],
    ) -> PlanStarted | PlanRejected:
        """Ask the actual user through MCP form elicitation, then launch the saved plan.

        No model-supplied approval is accepted. The client must show the complete
        server message to the user and return their decision, not auto-answer it.
        Unsupported clients, declines, cancellation, timeout, stale plans and
        changed strategy bytes do not launch a formal job. Repeating a successfully
        submitted plan returns its existing job instead of launching another.
        """
        try:
            client = ctx.session.client_params
            capability = client.capabilities.elicitation if client else None
        except (ValueError, AssertionError):
            capability = None
        if capability is None or (
            capability.form is None and capability.url is not None
        ):
            return PlanRejected(
                success=False,
                code="confirmation_unsupported",
                error=(
                    "此客戶端未宣告支援 MCP 表單確認，正式回測未啟動。"
                    "請使用會向使用者收集確認的客戶端。"
                ),
            )
        try:
            challenge = await anyio.to_thread.run_sync(
                partial(service.begin_confirmation, plan_id)
            )
        except (LookupError, OSError, ValueError, RuntimeError) as exc:
            return PlanRejected(
                success=False,
                code=exc.code
                if isinstance(exc, PlanOperationError)
                else "execution_plan_invalid",
                error=str(exc),
            )
        if isinstance(challenge, PlanStarted):
            return challenge
        try:
            with anyio.fail_after(300):
                answer = await ctx.elicit(
                    message=challenge.message, schema=UserApproval
                )
            accepted = answer.action == "accept" and answer.data.approved
            return await anyio.to_thread.run_sync(
                partial(service.finish_confirmation, challenge, accepted=accepted)
            )
        except BaseException as exc:
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(
                    partial(service.abandon_confirmation, challenge)
                )
            if not isinstance(exc, Exception):
                raise
            return PlanRejected(
                success=False,
                code="confirmation_failed",
                error=f"未收到有效的使用者確認，未啟動新的工作：{exc}",
            )
