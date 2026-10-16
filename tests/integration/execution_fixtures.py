"""Explicit presentation metadata for test execution plans."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tests.integration.mcp_harness import MCPClient


def confirmation_presentation(parameters: dict[str, Any]) -> dict[str, Any]:
    """Describe each captured constructor leaf, including nested parameter maps."""
    descriptions: dict[str, dict[str, str]] = {}

    def visit(value: Any, pointer: str) -> None:
        if isinstance(value, dict) and value:
            for key, child in value.items():
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                visit(child, f"{pointer}/{escaped}")
        elif isinstance(value, list) and value:
            for index, child in enumerate(value):
                visit(child, f"{pointer}/{index}")
        else:
            descriptions[pointer] = {
                "label": f"研究參數 {pointer.rsplit('/', 1)[-1]}",
                "description": "此值控制本次策略的訊號規則，僅套用於這次歷史模擬。",
                "unit": "策略單位",
            }

    for key, value in parameters.items():
        visit(value, "/" + key.replace("~", "~0").replace("/", "~1"))
    return {
        "title": "歷史策略研究設定確認",
        "summary": "核對策略參數與行情期間，批准後執行歷史模擬。",
        "parameter_descriptions": descriptions,
    }


def execution_options(parameters: dict[str, Any]) -> dict[str, Any]:
    """Provide deliberately small bounded preflight settings for cached fixtures."""
    return {
        "presentation": confirmation_presentation(parameters),
        "minimum_history_bars": 20,
        "sample_bars": 128,
    }


async def confirm_prepared(
    client: MCPClient, prepared: dict[str, Any]
) -> dict[str, Any]:
    """Submit the test user's explicit approval for one ready plan."""
    assert prepared["success"] is True, prepared
    assert prepared["status"] == "ready", prepared
    assert not prepared.get("job_id"), prepared
    confirmed: dict[str, Any] = await client.call(
        "submit_execution_decision", plan_id=prepared["plan_id"], decision="approve"
    )
    assert confirmed["success"] is True, confirmed
    assert confirmed["plan_id"] == prepared["plan_id"]
    assert confirmed["manifest_hash"] == prepared["manifest_hash"]
    assert confirmed["job_id"]
    return confirmed
