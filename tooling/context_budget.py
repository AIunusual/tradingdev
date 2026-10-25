"""Measure serialized context without confusing characters, bytes, and tokens."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


def text_size(text: str) -> dict[str, int]:
    """Measure exact text sizes; tokenizer-independent sizes are not token counts."""
    return {"characters": len(text), "utf8_bytes": len(text.encode("utf-8"))}


def json_size(value: Any) -> dict[str, int]:
    """Measure compact Unicode JSON with no dropped fields or string truncation."""
    return text_size(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


@dataclass
class RequestSizes:
    """Track the actual serialized HTTP bodies observed by a local test client."""

    requests: int = 0
    first_bytes: int = 0
    peak_bytes: int = 0

    def observe(self, body: bytes) -> str:
        size = text_size(body.decode("utf-8"))
        self.requests += 1
        if self.requests == 1:
            self.first_bytes = len(body)
        self.peak_bytes = max(self.peak_bytes, len(body))
        return (
            f"request_chars={size['characters']} request_utf8_bytes={len(body)} "
            f"request_growth_bytes={len(body) - self.first_bytes} "
            f"peak_request_utf8_bytes={self.peak_bytes}"
        )


def tool_context_report(
    instructions: str,
    tools: list[dict[str, Any]],
    *,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """Separate MCP wire metadata from one explicit model-facing tool projection."""
    known = {tool["name"] for tool in tools}
    requested = known if names is None else set(names)
    if missing := requested - known:
        raise ValueError(f"Unknown tool names: {', '.join(sorted(missing))}")
    selected = sorted(
        (tool for tool in tools if tool["name"] in requested),
        key=lambda tool: tool["name"],
    )
    model_tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description") or "",
                "parameters": tool["inputSchema"],
            },
        }
        for tool in selected
    ]
    return {
        "measurement": "compact Unicode JSON / UTF-8; not token counts",
        "model_projection": "Chat Completions name/description/input schema",
        "limitations": (
            "Excludes client instructions, user messages, history and output reserve. "
            "Other clients may select or serialize tools differently. MCP wire "
            "metadata is not assumed to be model input."
        ),
        "tool_count": len(selected),
        "tool_names": [tool["name"] for tool in selected],
        "instructions": text_size(instructions),
        "model_tools": json_size(model_tools),
        "mcp_wire_tools": json_size(selected),
        "per_tool": [
            {
                "name": tool["function"]["name"],
                **json_size(tool),
            }
            for tool in model_tools
        ],
    }


@contextmanager
def audit_environment() -> Iterator[Path]:
    """Confine service artifacts and import-time caches, restoring the caller env."""
    with TemporaryDirectory(prefix="tradingdev-context-audit-") as directory:
        root = Path(directory)
        updates = {
            "TRADINGDEV_WORKSPACE": str(root / "workspace"),
            "TRADINGDEV_DATA_ROOT": str(root / "data"),
            "MPLCONFIGDIR": str(root / "matplotlib"),
            "NUMBA_CACHE_DIR": str(root / "numba"),
            "XDG_CACHE_HOME": str(root / "cache"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        previous = {name: os.environ.get(name) for name in updates}
        previous_bytecode_setting = sys.dont_write_bytecode
        os.environ.update(updates)
        sys.dont_write_bytecode = True
        try:
            yield root
        finally:
            sys.dont_write_bytecode = previous_bytecode_setting
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


async def inspect_context(names: list[str] | None = None) -> dict[str, Any]:
    """Inspect registration only; no transport, worker, market fetch or model runs."""
    with audit_environment() as root:
        from tradingdev.adapters.storage.filesystem import WorkspacePaths
        from tradingdev.mcp.prompts import SERVER_INSTRUCTIONS
        from tradingdev.mcp.server import create_server

        server = create_server(WorkspacePaths(root / "workspace"))
        tools = [
            tool.model_dump(mode="json", by_alias=True)
            for tool in await server.list_tools()
        ]
        return tool_context_report(SERVER_INSTRUCTIONS, tools, names=names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", nargs="+", help="Measure only these tool names")
    args = parser.parse_args()
    try:
        report = asyncio.run(inspect_context(args.tools))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
