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
    ReportCommentary,
    ReportResponse,
    ReportSectionCatalog,
)

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
        """Discover backend-rendered sections and suggested report recipes.

        The client chooses sections and writes interpretation; the server renders
        the HTML, tables, and charts. Recipes are optional, not enforced layouts.
        ledger combines order attempts and bar-close balances; trades is the
        supplementary paired-trade PnL view. Legacy section IDs executions and
        account_history are no longer valid; the history query tools remain.
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
        run_ids: list[str],
        sections: list[str] | None = None,
        commentary: list[ReportCommentary] | None = None,
    ) -> ReportResponse | ErrorResponse:
        """Compose an offline HTML report from 1..8 saved runs and client prose.

        Discover section IDs/recipes with get_report_sections. Omit sections for
        the standard recipe, provide IDs in desired detail order, or [] to render no
        built-in sections while keeping source identity and supplied commentary.
        overview always occupies the first navigation level when selected.
        Every report embeds the complete loaded data for all scopes of the runs,
        including saved trades, equity and execution configuration, even with [].
        Section selection does not remove underlying data from the HTML.
        The overview compares all run/scopes. Detail links and result switching
        require settings, metrics, equity, ledger or trades; without these,
        selected provenance and limitations appear in the main view.
        metrics includes native QuantStats
        performance analysis. ledger combines order attempts and bar-close
        valuations with Bokeh status filters and column checkboxes. Static tables
        remain readable without JavaScript; CSV/JSON download buttons require it
        and preserve full precision.
        Commentary contains plain-text title/text objects (at most 20, total
        20,000 characters); HTML is escaped and text is
        labelled client interpretation, distinct from recorded calculations.
        trades is supplementary paired-trade PnL, not the reconciliation ledger.
        Returns a registered artifact and local path. Existing backtests remain
        unchanged. No strategy, model, external website, or market download runs.
        """
        result = reports.generate_report(
            run_ids,
            sections=sections,
            commentary=[item.model_dump() for item in commentary]
            if commentary is not None
            else None,
        )
        if result.get("success") is False:
            return ErrorResponse.model_validate(result)
        return ReportResponse.model_validate(result)
