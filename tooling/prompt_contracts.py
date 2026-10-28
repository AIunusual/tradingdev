"""Check discoverable MCP contracts and explicit UTF-8 context growth budgets."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from tooling.context_budget import audit_environment, json_size, tool_context_report

# Maintenance budgets, not model token limits or a client compatibility promise.
BUDGETS = {
    "instructions": 11000,
    "model_tools": 70000,
    "initial_total": 80000,
    "strategy_section": 16000,
    "indicator_reference": 10000,
    "discovery_response": 20000,
}
_DISCOVERY_TOOLS = (
    "get_ml_contract",
    "get_metric_catalog",
    "get_report_sections",
    "list_data_sources",
)
_SECTION_REFERENCE = re.compile(
    r"\bsection\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([A-Za-z_][A-Za-z0-9_-]*))"
)


@dataclass
class ContractSnapshot:
    instructions: str
    tools: list[dict[str, Any]]
    sections: dict[str, dict[str, Any]]
    indicators: dict[str, dict[str, Any]]
    discovery: dict[str, dict[str, Any]]


def _section_values(schema: dict[str, Any]) -> list[str]:
    value = schema["properties"]["section"]
    visited = set()
    while "$ref" in value:
        reference = value["$ref"]
        if not reference.startswith("#/") or reference in visited:
            raise ValueError(
                "Strategy section schema requires a finite local reference"
            )
        visited.add(reference)
        value = schema
        for key in reference[2:].split("/"):
            value = value[key.replace("~1", "/").replace("~0", "~")]
    choices = value["enum"]
    if (
        not isinstance(choices, list)
        or not choices
        or not all(isinstance(item, str) for item in choices)
    ):
        raise ValueError("Strategy sections must be a nonempty string enum")
    return choices


def _prose(value: Any) -> list[str]:
    """Exclude program/schema payloads from prose-reference interpretation."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _prose(item)]
    if isinstance(value, dict):
        return [
            text
            for key, item in value.items()
            if "source" not in key
            and "schema" not in key
            and key not in {"example_strategy_code", "example_yaml_config"}
            for text in _prose(item)
        ]
    return []


def check_snapshot(snapshot: ContractSnapshot) -> dict[str, Any]:
    """Reject broken discovery and growth; do not claim to check prose semantics."""
    tools = {tool["name"]: tool for tool in snapshot.tools}
    if len(tools) != len(snapshot.tools):
        raise ValueError("Duplicate registered tool names")
    contract = tools["get_strategy_contract"]
    declared = _section_values(contract["inputSchema"])
    sections = set(declared)
    if len(sections) != len(declared) or set(snapshot.sections) != sections:
        raise ValueError("Strategy sections disagree with the registered input schema")

    def check_section(section: str, payload: dict[str, Any]) -> None:
        menu = [item["id"] for item in payload["available_sections"]]
        if len(menu) != len(set(menu)) or set(menu) != sections:
            raise ValueError(f"{section}: discovery menu disagrees with section schema")
        if payload["section"] != section:
            raise ValueError(f"{section}: returned the wrong section")
        Draft202012Validator(contract["outputSchema"]).validate(payload)

    for section, payload in snapshot.sections.items():
        check_section(section, payload)
    catalog = snapshot.sections["indicators"]["content"]["catalog"]
    names = [item["name"] for item in catalog]
    if len(names) != len(set(names)) or set(names) != set(snapshot.indicators):
        raise ValueError("Indicator catalog and selected references disagree")
    for name, payload in snapshot.indicators.items():
        check_section("indicators", payload)
        if payload["content"]["catalog"] != catalog:
            raise ValueError(f"{name}: selected reference has a different catalog")
        if payload["content"]["selected"]["name"] != name:
            raise ValueError(f"{name}: returned a different indicator")
    if set(snapshot.discovery) != set(_DISCOVERY_TOOLS):
        raise ValueError("Missing static discovery response")
    for name, payload in snapshot.discovery.items():
        Draft202012Validator(tools[name]["outputSchema"]).validate(payload)
        if payload.get("success") is False:
            raise ValueError(f"{name}: discovery failed")
    texts = [snapshot.instructions]
    texts.extend(tool.get("description") or "" for tool in snapshot.tools)
    texts.extend(_prose(snapshot.sections))
    texts.extend(_prose(snapshot.indicators))
    for text in texts:
        for quoted, single_quoted, bare in _SECTION_REFERENCE.findall(text):
            name = quoted or single_quoted or bare
            if name not in sections:
                raise ValueError(f"Unknown strategy section reference: {name}")
    report = tool_context_report(snapshot.instructions, snapshot.tools)
    sizes: dict[str, int] = {}

    def bound(label: str, size: int, budget: str) -> None:
        sizes[label] = size
        if size > BUDGETS[budget]:
            raise ValueError(
                f"{label}: {size} UTF-8 bytes exceeds {budget} budget "
                f"{BUDGETS[budget]}; review content ownership before changing budgets"
            )

    for name in ("instructions", "model_tools"):
        bound(name, report[name]["utf8_bytes"], name)
    bound("initial_total", sum(sizes.values()), "initial_total")
    for group, payloads, budget in (
        ("strategy", snapshot.sections, "strategy_section"),
        ("indicator", snapshot.indicators, "indicator_reference"),
        ("discovery", snapshot.discovery, "discovery_response"),
    ):
        for name, payload in payloads.items():
            bound(f"{group}/{name}", json_size(payload)["utf8_bytes"], budget)
    return {
        "measurement": report["measurement"],
        "model_projection": report["model_projection"],
        "limitations": report["limitations"],
        "checked_utf8_bytes": sizes,
        "budgets_utf8_bytes": BUDGETS,
        "mcp_wire_tools": report["mcp_wire_tools"],
    }


async def collect_snapshot() -> ContractSnapshot:
    """Read actual registration/discovery in a disposable workspace; no worker."""
    with audit_environment() as root:
        from mcp_types import CallToolResult

        from tradingdev.adapters.storage.filesystem import WorkspacePaths
        from tradingdev.mcp.prompts import SERVER_INSTRUCTIONS
        from tradingdev.mcp.server import create_server

        server = create_server(WorkspacePaths(root / "workspace"))
        tools = [
            tool.model_dump(mode="json", by_alias=True)
            for tool in await server.list_tools()
        ]

        tool_schemas = {tool["name"]: tool for tool in tools}

        async def read(name: str, **arguments: str) -> dict[str, Any]:
            Draft202012Validator(tool_schemas[name]["inputSchema"]).validate(arguments)
            result = await server.call_tool(name, arguments)
            if not isinstance(result, CallToolResult) or not isinstance(
                result.structured_content, dict
            ):
                raise ValueError(f"{name}: missing structured discovery response")
            return result.structured_content

        core = await read("get_strategy_contract", section="core")
        sections = {"core": core}
        for item in core["available_sections"]:
            if item["id"] != "core":
                sections[item["id"]] = await read(
                    "get_strategy_contract", section=item["id"]
                )
        indicators = {
            item["name"]: await read(
                "get_strategy_contract", section="indicators", indicator=item["name"]
            )
            for item in sections["indicators"]["content"]["catalog"]
        }
        discovery = {name: await read(name) for name in _DISCOVERY_TOOLS}
        return ContractSnapshot(
            SERVER_INSTRUCTIONS, tools, sections, indicators, discovery
        )


def main() -> None:
    report = check_snapshot(asyncio.run(collect_snapshot()))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
