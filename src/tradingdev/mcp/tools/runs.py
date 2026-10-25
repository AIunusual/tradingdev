"""Run MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp_types import ToolAnnotations

from tradingdev.app.contracts.common import (
    DEFAULT_PAGE_LIMIT,
    ErrorResponse,
    Page,
    PageLimit,
    PageOffset,
)
from tradingdev.app.contracts.research import (
    CompareRunsResponse,
    MetricCatalogResponse,
    MetricQueryError,
    RunMetricsResponse,
    RunResponse,
    RunSummary,
)

if TYPE_CHECKING:
    from mcp.server import MCPServer

    from tradingdev.app.run_service import RunService


def register(mcp: MCPServer, service: RunService) -> None:
    """Register run tools."""

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def list_runs(
        strategy_id: str | None = None,
        revision_id: str | None = None,
        offset: PageOffset = 0,
        limit: PageLimit = DEFAULT_PAGE_LIMIT,
    ) -> Page[RunSummary]:
        """Discover completed run identities; get_run reads metrics and scope choices.

        Exact filters precede paging; order is created_at descending, run_id ascending.
        total counts matches; follow next_offset. Restart at offset=0 after inventory
        changes because pages do not retain a cross-call snapshot.
        """
        return service.list_runs_page(
            strategy_id=strategy_id, revision_id=revision_id, offset=offset, limit=limit
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def get_run(run_id: str) -> RunResponse | ErrorResponse:
        """Read a run summary; get_run_metrics reads any available metric IDs."""
        payload = service.get_run(run_id)
        if payload.get("success") is False:
            return ErrorResponse.model_validate(payload)
        return RunResponse.model_validate(payload)

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def compare_runs(
        run_ids: list[str],
        metric_ids: list[str] | None = None,
        scope: str | None = None,
    ) -> CompareRunsResponse | MetricQueryError | ErrorResponse:
        """Compare recorded scopes without ranking; check each metric's compatibility.

        Omit scope for each run's default. metric_ids selects stored metrics;
        omission selects summaries. Fold statistics retain valid_count.
        Incompatible definitions, modes, or settings are explicitly reported.
        """
        payload = service.compare_runs(run_ids, metric_ids=metric_ids, scope=scope)
        if payload.get("success") is False:
            if "available_metric_ids" in payload:
                return MetricQueryError.model_validate(payload)
            return ErrorResponse.model_validate(payload)
        return CompareRunsResponse.model_validate(payload)

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def get_metric_catalog(
        mode: str | None = None,
    ) -> MetricCatalogResponse | ErrorResponse:
        """Discover current metric IDs, units, meanings, and optimization direction.

        Optional mode is signal or volume. Historical runs retain their own
        definitions; get_run_metrics returns that saved snapshot.
        """
        payload = service.get_metric_catalog(mode)
        if payload.get("success") is False:
            return ErrorResponse.model_validate(payload)
        return MetricCatalogResponse.model_validate(payload)

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def get_run_metrics(
        run_id: str,
        metric_ids: list[str] | None = None,
        scope: str | None = None,
    ) -> RunMetricsResponse | MetricQueryError:
        """Read all or selected stored metrics, including values omitted from summaries.

        Omit scope to use default_scope. Scopes include full, fold/0/train,
        fold/0/test, test_summary, trial/0/train, and test when recorded.
        Response contains original definitions/settings and reasons for nulls.
        Unknown IDs/scopes return available choices. No strategy is rerun.
        Fold summaries are distributions across folds, not whole-period returns.
        """
        payload = service.get_run_metrics(run_id, metric_ids=metric_ids, scope=scope)
        if payload.get("success") is False:
            return MetricQueryError.model_validate(payload)
        return RunMetricsResponse.model_validate(payload)
