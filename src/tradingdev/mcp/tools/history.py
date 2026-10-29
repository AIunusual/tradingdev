"""Saved-run discovery, paginated observations, and standard research reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from mcp_types import ToolAnnotations

from tradingdev.app.contracts.common import ErrorResponse
from tradingdev.app.contracts.history import (
    FindRunsResponse,
    HistoryQueryError,
    RunAccountHistoryResponse,
    RunEquityResponse,
    RunExecutionsResponse,
    RunTradesResponse,
)
from tradingdev.app.contracts.reports import (
    ReportAnalysis,
    ReportCommentary,
    ReportResponse,
    ReportSectionCatalog,
)
from tradingdev.app.diagnostic_service import DiagnosticService, RunDiagnostics

if TYPE_CHECKING:
    from mcp.server import MCPServer

    from tradingdev.app.report_service import ReportService
    from tradingdev.app.trade_history_service import TradeHistoryService


def register(
    mcp: MCPServer, history: TradeHistoryService, reports: ReportService
) -> None:
    """Keep all clients on the same verified historical data services."""
    read_only = ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @mcp.tool(annotations=read_only)
    def get_run_diagnostics(
        run_id: str, scope: str | None = None
    ) -> RunDiagnostics | ErrorResponse:
        """Inspect bounded heuristic findings with saved metric evidence and limits.

        Omitted scope selects the saved default; aggregate scopes are explicitly
        unavailable. Suggestions are untested experiments, not causes or approval.
        Market regimes/volatility sensitivity are not measured by this provider.
        Read relevant trades/equity before making more specific claims.
        """
        return DiagnosticService(history).get(run_id, scope)

    @mcp.tool(annotations=read_only)
    def find_runs(
        strategy_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> FindRunsResponse | HistoryQueryError:
        """Find saved run/scopes by historical effective parameter subsets.

        Each optimization trial is a separate match. Select a run_id and scope
        before requesting trades; the same parameters can have different dates,
        costs, revisions, and datasets. limit is 1..500. No strategy is rerun.
        """
        result = history.find_runs(
            strategy_id=strategy_id,
            parameters=parameters,
            symbol=symbol,
            timeframe=timeframe,
            offset=offset,
            limit=limit,
        )
        if result.get("success") is False:
            return HistoryQueryError.model_validate(result)
        return FindRunsResponse.model_validate(result)

    @mcp.tool(annotations=read_only)
    def get_run_trades(
        run_id: str,
        scope: str | None = None,
        offset: int = 0,
        limit: int = 50,
        status: Literal["open", "closed"] | None = None,
        direction: Literal["long", "short"] | None = None,
        entry_start: str | None = None,
        entry_end: str | None = None,
    ) -> RunTradesResponse | HistoryQueryError:
        """Read saved position/trade records with UTC dates and stable pagination.

        Open positions have a final mark, not an executed exit. These are paired
        trades, not individual orders. Filter dates refer to entry time; date-only
        end includes that entire UTC day, timestamps are inclusive instants.
        Omitted scope selects the saved default; aggregate fold scopes have no
        individual trades. limit is 1..500. Never recalculates or reruns a strategy.
        """
        result = history.get_run_trades(
            run_id,
            scope=scope,
            offset=offset,
            limit=limit,
            status=status,
            direction=direction,
            entry_start=entry_start,
            entry_end=entry_end,
        )
        if result.get("success") is False:
            return HistoryQueryError.model_validate(result)
        return RunTradesResponse.model_validate(result)

    @mcp.tool(annotations=read_only)
    def get_run_equity(
        run_id: str,
        scope: str | None = None,
        offset: int = 0,
        limit: int = 100,
        start: str | None = None,
        end: str | None = None,
    ) -> RunEquityResponse | HistoryQueryError:
        """Read aligned saved per-bar equity or cumulative volume-mode PnL.

        Date-only end includes that entire UTC day; timestamps are inclusive
        instants. limit is 1..500. Volume mode has no initial capital and its
        observed values must not be described as account balances.
        """
        result = history.get_run_equity(
            run_id, scope=scope, offset=offset, limit=limit, start=start, end=end
        )
        if result.get("success") is False:
            return HistoryQueryError.model_validate(result)
        return RunEquityResponse.model_validate(result)

    @mcp.tool(annotations=read_only)
    def get_run_executions(
        run_id: str,
        scope: str | None = None,
        offset: int = 0,
        limit: int = 50,
        status: Literal["filled", "ignored", "rejected"] | None = None,
        side: Literal["buy", "sell"] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> RunExecutionsResponse | HistoryQueryError:
        """Read saved order attempts, fill costs and before/after account states.

        Distinct from paired trades: one reversal fill may close and open positions.
        OHLC are bar market references, requested_price is before slippage, and
        filled_price is simulated execution. Equity marks before/after at the
        request price. Time identifies the bar, not an exact intrabar execution.
        Generic VectorBT accounting is not an exchange margin wallet. Check
        availability: legacy/volume results may not record this ledger. Never
        reconstructs absent records. limit 1..500; inclusive UTC dates, a date-only
        end includes the whole day. side filters filled orders only.
        """
        result = history.get_run_executions(
            run_id,
            scope=scope,
            offset=offset,
            limit=limit,
            status=status,
            side=side,
            start=start,
            end=end,
        )
        if result.get("success") is False:
            return HistoryQueryError.model_validate(result)
        return RunExecutionsResponse.model_validate(result)

    @mcp.tool(annotations=read_only)
    def get_run_account_history(
        run_id: str,
        scope: str | None = None,
        offset: int = 0,
        limit: int = 100,
        start: str | None = None,
        end: str | None = None,
    ) -> RunAccountHistoryResponse | HistoryQueryError:
        """Read saved end-of-bar cash, free cash, signed position and equity.

        Equity/asset_value use bar close. This is VectorBT generic accounting,
        not exchange margin, funding or liquidation balances. Check availability
        before interpreting an empty page; old or volume-mode results have no
        account ledger. No replay or reconstruction. limit 1..500. Date-only end
        includes the entire UTC day; timestamp bounds are inclusive instants.
        """
        result = history.get_run_account_history(
            run_id, scope=scope, offset=offset, limit=limit, start=start, end=end
        )
        if result.get("success") is False:
            return HistoryQueryError.model_validate(result)
        return RunAccountHistoryResponse.model_validate(result)

    @mcp.tool(annotations=read_only)
    def get_report_sections() -> ReportSectionCatalog:
        """Discover available report sections, their meanings and suggested recipes.

        Recipes are optional. The client chooses sections and writes interpretation;
        the backend renders saved evidence. Read section descriptions before choosing
        ledger (order/account reconciliation) or trades (paired-trade PnL).
        """
        return ReportSectionCatalog.model_validate(reports.get_report_sections())

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def generate_report(
        run_ids: list[str] | None = None,
        sections: list[str] | None = None,
        commentary: list[ReportCommentary] | None = None,
        comparison_job_id: str | None = None,
        analysis: ReportAnalysis | None = None,
        ml_entry_ids: list[str] | None = None,
        ml_ranking_metric: str = "ic_mean",
        ml_allow_parameter_axes: bool = False,
    ) -> ReportResponse | ErrorResponse:
        """Render saved evidence: run_ids (1..8), comparison_job_id, or ML entries.

        run_ids and comparison_job_id are mutually exclusive. ML can stand alone or
        accompany either: select 1..50 distinct ml_entry_ids from leaderboard or
        recorded evaluate_signal results; never infer model/run associations.
        ML is cost-free signal research, not a backtest; unsaved prediction/IC series
        are not reconstructed. Parameter-axis ranking requires explicit opt-in.
        A comparison needs a terminal job and complete ledger; every candidate,
        including failed/cancelled/not_run reasons, remains visible with any sections.
        Use its parent job ID even when zero candidates succeeded.

        Discover section IDs with get_report_sections. Omit sections for the complete
        standard report plus selected ML evidence (ML alone defaults to ['ml']),
        select detail order, or use [] to omit optional sections.
        overview is first when selected. Result switching requires a detail section.
        Every report embeds all loaded scopes/config/trades/equity even with []; section
        selection does not redact data. Supply analysis with summary, evidence-based
        observations, hypotheses_and_limits, next_steps and exact backtest run/scope
        references. ML-only: empty references; name selected entry IDs in prose.
        Explain each candidate; identify unanalysed scopes. Specific loss/regime claims
        require inspected evidence. Missing analysis is visibly not provided.
        Commentary is optional supplementary plain text, labelled
        separately from calculations (at most 20 entries, 20,000 total characters).
        Returns an artifact and local path; no strategy or market download runs. Deliver
        the link with selected IDs/scopes and the returned navigation_instructions
        in the final answer, not only an intermediate message. Preserve reports and
        source evidence after server shutdown.
        """
        result = reports.generate_report(
            run_ids,
            sections=sections,
            commentary=[item.model_dump() for item in commentary]
            if commentary is not None
            else None,
            comparison_job_id=comparison_job_id,
            analysis=analysis.model_dump() if analysis is not None else None,
            ml_entry_ids=ml_entry_ids,
            ml_ranking_metric=ml_ranking_metric,
            ml_allow_parameter_axes=ml_allow_parameter_axes,
        )
        if result.get("success") is False:
            return ErrorResponse.model_validate(result)
        return ReportResponse.model_validate(result)
