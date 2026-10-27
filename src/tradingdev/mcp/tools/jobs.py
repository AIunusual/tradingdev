"""Job MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp_types import ToolAnnotations
from pydantic import TypeAdapter

from tradingdev.app.contracts.common import (
    DEFAULT_PAGE_LIMIT,
    Page,
    PageLimit,
    PageOffset,
)
from tradingdev.app.contracts.jobs import (
    JobActionFailed,
    JobCancelled,
    JobNotFound,
    JobState,
    JobStatus,
    JobSummary,
)

if TYPE_CHECKING:
    from mcp.server import MCPServer

    from tradingdev.app.job_service import JobService


def register(mcp: MCPServer, service: JobService) -> None:
    """Register job tools."""

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def get_job_status(job_id: str) -> JobStatus | JobNotFound:
        """Check progress, reconciling a vanished worker to persisted failed status.

        When done, use run_id with get_run. Human confirmation is collected
        before creating a job. failed is a job state, not a
        lookup error; not_found means the job ID does not exist.
        """
        return TypeAdapter(JobStatus | JobNotFound).validate_python(
            service.get_job_status(job_id)
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def list_jobs(
        status: JobState | None = None,
        job_type: str | None = None,
        strategy_name: str | None = None,
        offset: PageOffset = 0,
        limit: PageLimit = DEFAULT_PAGE_LIMIT,
    ) -> Page[JobSummary]:
        """Discover persisted job progress; get_job_status checks worker liveness.

        Exact filters precede paging; order is created_at descending, job_id ascending.
        total counts matches; follow next_offset. Restart at offset=0 after inventory
        changes because pages do not retain a cross-call snapshot.
        """
        return service.list_jobs_page(
            status=status,
            job_type=job_type,
            strategy_name=strategy_name,
            offset=offset,
            limit=limit,
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def cancel_job(job_id: str) -> JobCancelled | JobActionFailed:
        """Stop an active job and acknowledge cleanup before marking it cancelled.

        success=False means cancellation was not confirmed; inspect code and
        get_job_status. Completed jobs cannot be cancelled.
        """
        return TypeAdapter(JobCancelled | JobActionFailed).validate_python(
            service.cancel_job(job_id)
        )
