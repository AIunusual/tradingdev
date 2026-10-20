"""Job MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp_types import ToolAnnotations
from pydantic import TypeAdapter

from tradingdev.app.contracts.jobs import (
    JobActionFailed,
    JobCancelled,
    JobNotFound,
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
    def list_jobs() -> list[JobSummary]:
        """List persisted job summaries; use get_job_status to check worker liveness."""
        return [JobSummary.model_validate(job) for job in service.list_jobs()]

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
