"""Run MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp_types import ToolAnnotations

from tradingdev.adapters.storage.performance import PerformanceArtifactError
from tradingdev.app.contracts.common import (
    DEFAULT_PAGE_LIMIT,
    ErrorResponse,
    Page,
    PageLimit,
    PageOffset,
)
from tradingdev.app.contracts.optimization import OptimizationTrialsResponse  # noqa: TC001
from tradingdev.app.contracts.research import (
    CompareRunsResponse,
    MetricCatalogResponse,
    MetricQueryError,
    RunMetricsResponse,
    RunResponse,
    RunScopeSelection,
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
        """Read a run summary; use display_metrics for ready-to-present values.

        get_run_metrics reads any available metric IDs with saved definitions.
        """
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
        selections: list[RunScopeSelection],
        metric_ids: list[str] | None = None,
    ) -> CompareRunsResponse | MetricQueryError | ErrorResponse:
        """Compare saved run scopes without ranking; check each metric's compatibility.

        Each selection names one run and optionally one saved scope, so two scopes
        of the same run can be compared; omit scope for that run's default. Two
        selections that resolve to the same measurement are refused. metric_ids
        selects stored metrics; omission selects summaries. Fold statistics retain
        valid_count. Rows report metric_source: bundle or legacy_summary.
        Incompatible definitions, modes, or settings are explicitly reported.
        Use display_metrics for presentation; raw metrics retain saved units.
        Rolling traces are summaries with artifact_id and json_pointer;
        get_artifact(include_content=true) pages the complete saved evidence.
        """
        payload = service.compare_runs(
            [(item.run_id, item.scope) for item in selections],
            metric_ids=metric_ids,
        )
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
    def get_optimization_trials(
        job_id: str,
        offset: PageOffset = 0,
        limit: PageLimit = DEFAULT_PAGE_LIMIT,
    ) -> OptimizationTrialsResponse | ErrorResponse:
        """Page approved optimization trial positions and their recorded outcomes.

        Use get_job_status for lifecycle. completed counts fully observed trials,
        rankable counts finite objectives, and search_complete requires all
        approved positions to be observed. The leader is among observed trials.
        Test evidence is separate. Follow trials.next_offset. Pass job_id as
        run_id and an explicit recorded scope to get_run_metrics for values.
        """
        try:
            return service.get_optimization_trials(job_id, offset=offset, limit=limit)
        except PerformanceArtifactError as exc:
            return ErrorResponse(success=False, code=exc.code, error=str(exc))

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

        Partial optimization jobs require an explicit scope and use the job ID as
        run_id. Discover scopes with get_optimization_trials. A partial response
        has result_completeness=partial and no default_scope.
        Otherwise omit scope to use default_scope. Scopes include full, fold/0/train,
        fold/0/test, test_summary, trial/0/train, and test when recorded.
        Response contains original definitions/settings and reasons for nulls.
        Use display_metrics for presentation; percent signs are already included.
        Unknown IDs/scopes return available choices. No strategy is rerun.
        Fold summaries are distributions across folds, not whole-period returns.
        Rolling traces are summaries with artifact_id and json_pointer;
        get_artifact(include_content=true) pages the complete saved evidence.
        """
        payload = service.get_run_metrics(run_id, metric_ids=metric_ids, scope=scope)
        if payload.get("success") is False:
            return MetricQueryError.model_validate(payload)
        return RunMetricsResponse.model_validate(payload)
