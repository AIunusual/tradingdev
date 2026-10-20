"""Feature request MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp_types import ToolAnnotations

from tradingdev.app.contracts.research import (
    FeatureRequestArtifact,
    RecordedFeatureRequest,
    RecordFeatureRequestResponse,
)

if TYPE_CHECKING:
    from mcp.server import MCPServer

    from tradingdev.app.feature_request_service import FeatureRequestService


def register(
    mcp: MCPServer,
    service: FeatureRequestService,
) -> None:
    """Register feature-request tools."""

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        )
    )
    def record_feature_request(
        title: str,
        description: str,
        source_tool: str = "",
    ) -> RecordFeatureRequestResponse:
        """Record an unsupported capability locally; no external issue is created.

        Each call creates a new request. Use list_feature_requests to review it.
        """
        request = service.record(
            title=title,
            description=description,
            source_tool=source_tool or "record_feature_request",
        )
        return RecordFeatureRequestResponse(
            success=True,
            message="Feature request recorded.",
            feature_request=RecordedFeatureRequest.model_validate(request),
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def list_feature_requests() -> list[FeatureRequestArtifact]:
        """List local requests; use get_artifact with an artifact_id to read a file."""
        return [
            FeatureRequestArtifact.model_validate(row)
            for row in service.list_requests()
        ]
