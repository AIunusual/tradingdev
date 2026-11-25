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
from tradingdev.app.contracts.optimization import (  # noqa: TC001
    OptimizationTrialsResponse,
    SelectionEvidencePage,
)
from tradingdev.app.contracts.research import (
    CompareRunsResponse,
    MetricCatalogResponse,
    MetricQueryError,
    RunMetricsResponse,
    RunResponse,
    RunScopeSelection,
    RunSummary,
)
from tradingdev.app.contracts.statistics import (
    StatisticsAnalysisResponse,
    StatisticsQueryError,
)
from tradingdev.app.statistics_service import StatisticsService

if TYPE_CHECKING:
    from mcp.server import MCPServer

    from tradingdev.app.run_service import RunService


def register(
    mcp: MCPServer,
    service: RunService,
    statistics: StatisticsService | None = None,
) -> None:
    """Register run tools."""
    statistics = statistics or StatisticsService()

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
        """Page grid positions and recorded outcomes, with separate test status.

        completed counts fully observed trials, including undefined objectives;
        rankable counts finite objectives. The leader is among observed trials.
        Out-of-range objectives are null with reason metric_outside_numeric_range.
        Follow trials.next_offset. Read values with get_run_metrics(job ID, explicit
        scope). search_complete requires every trial to be observed.
        Use get_job_status for lifecycle.
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
    def get_selection_evidence(
        run_id: str,
        performance_sha256: str,
        offset: PageOffset = 0,
        limit: PageLimit = DEFAULT_PAGE_LIMIT,
    ) -> SelectionEvidencePage | ErrorResponse:
        """Page the saved training trials behind a ranked optimization run.

        Use run_id and performance_sha256 from selection_evidence.reference;
        follow trials.next_offset. Includes parameters, training objective and
        Sharpe, settings and null reasons. Trial independence is unknown, and
        training values cannot establish deflated Sharpe for test returns.
        """
        try:
            return service.get_selection_evidence(
                run_id, performance_sha256, offset=offset, limit=limit
            )
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
        """Read saved metrics, definitions, settings and null reasons for one scope.

        Omit scope for a published run's default. Partial optimizations require
        the job ID and explicit scope from get_optimization_trials. Unknown IDs
        or scopes return choices. Fold summaries describe distributions over folds.
        Use display_metrics as formatted. For published rolling traces, use
        artifact_id and json_pointer with get_artifact for details.
        """
        payload = service.get_run_metrics(run_id, metric_ids=metric_ids, scope=scope)
        if payload.get("success") is False:
            return MetricQueryError.model_validate(payload)
        return RunMetricsResponse.model_validate(payload)

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def analyze_run_statistics(
        run_id: str, scope: str | None = None, save_artifact: bool = False
    ) -> StatisticsAnalysisResponse | StatisticsQueryError:
        """Analyze one saved run/scope from validated observations; no rerun.

        Uses frozen capital, frequency, annualization and risk-free settings.
        Summaries may be available when inference is not; each absence has a
        reason, next step and source. Return inference is unavailable for
        volume mode, timestamp gaps or incomplete settings. PSR is iid; DSR
        only for a verified selected training scope. save_artifact publishes
        a derived immutable Markdown artifact; source reports stay unchanged.
        """
        payload = statistics.analyze(run_id, scope, save_artifact=save_artifact)
        if payload.get("success") is False:
            return StatisticsQueryError.model_validate(payload)
        return StatisticsAnalysisResponse.model_validate(payload)
