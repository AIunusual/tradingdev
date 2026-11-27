"""Run MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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
from tradingdev.app.contracts.paired_comparison import (  # noqa: TC001
    PairedComparisonResponse,
    SeriesSelection,
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
from tradingdev.app.paired_comparison_service import PairedComparisonService

if TYPE_CHECKING:
    from mcp.server import MCPServer

    from tradingdev.app.run_service import RunService


def register(
    mcp: MCPServer,
    service: RunService,
    paired: PairedComparisonService | None = None,
) -> None:
    """Register run tools and the shared paired-comparison entry point."""

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
    def compare_saved_series(
        a: SeriesSelection,
        b: SeriesSelection,
        mode: Literal["descriptive", "inferential"] = "descriptive",
        reps: int = 1000,
        seed: int = 20261127,
        block_size: int | None = None,
        confidence: float = 0.95,
        save_artifact: bool = False,
    ) -> PairedComparisonResponse | ErrorResponse:
        """Compare two saved series on their exact shared UTC timestamps.

        Sources are run/scope observations, a saved product benchmark, a
        verified zero-cost price artifact, or a saved pointwise signal series.
        run and benchmark selections require run_id; benchmarks must state
        benchmark_mode=product or zero_cost; zero_cost also requires
        price_artifact_id; signal selections require artifact_id and column.
        The statistic is mean with direction a minus b; fewer than 20 jointly
        valid observations return insufficient_observations; an omitted
        block_size is estimated automatically from the paired difference.
        Run returns prefer saved per-period values and are derived from saved
        equity and capital only when absent; warm-up nulls keep their rows and
        are excluded from the paired mask, never shifted earlier.
        The service does not verify that both sides use the same instrument,
        date range, fees or slippage, so an available result is not necessarily
        a controlled comparison. reps accepts 100..10000, seed must be a
        nonnegative integer and confidence must be a float strictly between
        0 and 1; the service enforces these bounds.
        Run pairs must match saved init_cash, periods_per_year and
        risk_free_rate; IC pairs must match evaluation settings. This never
        runs a strategy, trains a model, or downloads prices.
        Warm-up NaNs keep their coordinates; only jointly valid dates pair.
        mode=descriptive labels p as non-inferential; mode=inferential also
        requires an exact UTC daily/hourly calendar and a recorded producer,
        but still does not guarantee nominal bootstrap coverage or correct
        nonstationarity. No multiple-comparison correction is applied, and
        claim_scope=verified_out_of_sample requires BOTH sides to be test
        run scopes with saved research evidence; benchmarks and signals never
        qualify. Response samples cap excluded and missing coordinates at 20;
        the published artifact holds the full audit.
        A zero-cost price baseline is unavailable unless a verified raw price
        artifact exists; saved net benchmark values are never reversed into closes.
        save_artifact publishes immutable Markdown and self-contained offline
        HTML artifacts from one analysis; page either with get_artifact using
        the returned ID and hash.
        """
        from tradingdev.app.paired_comparison_service import PairedComparisonError

        paired_service = paired or PairedComparisonService()
        try:
            payload = paired_service.compare(
                a,
                b,
                mode=mode,
                reps=reps,
                seed=seed,
                block_size=block_size,
                confidence=confidence,
                save_artifact=save_artifact,
            )
        except PairedComparisonError as exc:
            return ErrorResponse(success=False, code=exc.code, error=str(exc))
        return PairedComparisonResponse.model_validate(payload)
