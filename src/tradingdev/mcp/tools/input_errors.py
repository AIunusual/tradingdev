"""Public diagnostics for validation performed at explicit tool input boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server.mcpserver.exceptions import ToolError

if TYPE_CHECKING:
    from pydantic import ValidationError


def invalid_arguments(error: ValidationError) -> ToolError:
    """Describe invalid inputs without echoing entire request values or context."""
    details = []
    for issue in error.errors(
        include_url=False, include_context=False, include_input=False
    ):
        field = ".".join(str(part) for part in issue["loc"]) or "request"
        details.append(f"{field}: {issue['msg']}")
    return ToolError("Invalid arguments: " + "; ".join(details))
