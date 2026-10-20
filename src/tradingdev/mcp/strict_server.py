"""Reject misspelled MCP arguments before MCPServer's permissive parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

if TYPE_CHECKING:
    from mcp.server.mcpserver import Context
    from mcp_types import CallToolResult, InputRequiredResult, Tool


class StrictMCPServer(MCPServer):
    """Keep the advertised argument names and dispatch boundary in agreement.

    MCPServer's generated argument models ignore extra fields. These public overrides
    reject unknown top-level arguments while preserving SDK type conversion,
    optional defaults, and the declared contents of dynamic parameter maps.
    """

    async def list_tools(self) -> list[Tool]:
        """Advertise closed argument objects without mutating SDK tool metadata."""
        return [
            tool.model_copy(
                update={
                    "input_schema": tool.input_schema | {"additionalProperties": False}
                }
            )
            for tool in await super().list_tools()
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Context[Any, Any] | None = None,
    ) -> CallToolResult | InputRequiredResult:
        """Reject unknown keys before business logic or worker creation runs."""
        for tool in await self.list_tools():
            if tool.name == name:
                properties = tool.input_schema.get("properties", {})
                unknown = sorted(arguments.keys() - properties.keys())
                if unknown:
                    allowed = ", ".join(sorted(properties)) or "(none)"
                    raise ToolError(
                        f"Unknown arguments for {name}: {', '.join(unknown)}. "
                        f"Allowed arguments: {allowed}"
                    )
                break
        return await super().call_tool(name, arguments, context)
