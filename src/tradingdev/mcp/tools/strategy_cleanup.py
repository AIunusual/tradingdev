"""Explicit generated draft retention and cleanup boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp_types import ToolAnnotations

# MCPServer resolves return types at runtime when registering its output schema.
from tradingdev.app.contracts.common import ErrorResponse  # noqa: TC001
from tradingdev.app.contracts.strategy_cleanup import (
    StrategyCleanupResult,  # noqa: TC001
)

if TYPE_CHECKING:
    from mcp.server import MCPServer

    from tradingdev.app.strategy_cleanup_service import StrategyCleanupService


def register(mcp: MCPServer, service: StrategyCleanupService) -> None:
    """Register a preview-first cleanup tool with an explicit apply selector."""

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def cleanup_strategy_drafts(
        strategy_id: str,
        revision_ids: list[str] | None = None,
        apply: bool = False,
    ) -> StrategyCleanupResult | ErrorResponse:
        """Preview unused drafts by default; never clean up automatically.

        Explain the preview and obtain the user's deletion authorization before
        apply=true, unless the user already explicitly authorized these deletions.
        Apply requires explicit revision_ids and rechecks all protection rules.
        Current, non-draft, referenced, and unverifiable revisions are retained.
        Per-revision failures are reported; filesystem deletion is not transactional.
        """
        return service.cleanup(strategy_id, revision_ids, apply=apply)
