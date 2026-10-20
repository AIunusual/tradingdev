"""Preparation input errors remain actionable through both MCP protocols."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from jsonschema import Draft202012Validator
from mcp import Client
from mcp.server.mcpserver.exceptions import UnexpectedToolError
from mcp_types import TextContent
from pydantic import ValidationError
from tests.mcp.execution_fixtures import plan_context

from tradingdev.app.contracts.plans import PlanResponse

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch

    from tradingdev.domain.preflight import PreflightRequest
    from tradingdev.domain.presentation.confirmation import ConfirmationPresentation


_PREPARE_TOOLS = [
    "prepare_backtest",
    "prepare_walk_forward",
    "prepare_optimization",
]


def _arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name != "prepare_optimization":
        return arguments
    return {
        key: value
        for key, value in arguments.items()
        if key not in {"start_date", "end_date"}
    } | {
        "param_ranges": {"k_period": [9, 14]},
        "optimization_metric": "sharpe_ratio",
        "train_start": "2024-01-01",
        "train_end": "2024-01-04",
        "test_start": "2024-01-05",
        "test_end": "2024-01-08",
    }


@pytest.mark.parametrize("protocol_mode", ["legacy", "auto"])
@pytest.mark.parametrize("tool_name", _PREPARE_TOOLS)
@pytest.mark.parametrize(
    ("changes", "field", "diagnostic", "schema_rejects"),
    [
        ({"sample_bars": 32}, "sample_bars", "64", True),
        ({"sample_bars": 4097}, "sample_bars", "4096", True),
        ({"minimum_history_bars": 0}, "minimum_history_bars", "1", True),
        ({"minimum_history_bars": 4097}, "minimum_history_bars", "4096", True),
        (
            {"minimum_history_bars": 129, "sample_bars": 128},
            "minimum_history_bars",
            "must not exceed sample_bars",
            False,
        ),
    ],
)
def test_invalid_preparation_inputs_fail_before_preflight_or_plan_creation(
    tmp_path: Path,
    protocol_mode: str,
    tool_name: str,
    changes: dict[str, int],
    field: str,
    diagnostic: str,
    schema_rejects: bool,
) -> None:
    context = plan_context(tmp_path)
    arguments = _arguments(tool_name, context.arguments) | changes
    arguments["parameters"] = {"private_note": "PRIVATE_INPUT"}

    async def check() -> None:
        async with Client(context.server, mode=protocol_mode) as client:
            tool = next(
                tool
                for tool in (await client.list_tools()).tools
                if tool.name == tool_name
            )
            validator = Draft202012Validator(tool.input_schema)
            assert validator.is_valid(arguments) is not schema_rejects
            result = await client.call_tool(tool_name, arguments)
            assert result.is_error
            assert result.structured_content is None
            text = " ".join(
                item.text for item in result.content if isinstance(item, TextContent)
            )
            assert field in text
            assert diagnostic in text
            assert text != f"Error executing tool {tool_name}"
            assert "PRIVATE_INPUT" not in text

    asyncio.run(check())
    assert context.preflight.requests == []
    assert context.runner.calls == []
    assert context.service.jobs.list_all_jobs() == []
    assert context.service.jobs.list_runs() == []
    assert not list(context.service.workspace.runs.iterdir())
    assert not (context.service.workspace.root / "execution_plans").exists()
    with context.service.jobs.store.connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM execution_plans").fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("protocol_mode", ["legacy", "auto"])
@pytest.mark.parametrize("tool_name", _PREPARE_TOOLS)
def test_preparation_service_validation_errors_remain_private(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    protocol_mode: str,
    tool_name: str,
) -> None:
    context = plan_context(tmp_path)
    arguments = _arguments(tool_name, context.arguments)

    def invalid_internal_dto(
        _request: PreflightRequest, _presentation: ConfirmationPresentation
    ) -> None:
        PlanResponse.model_validate({"private_record": "PRIVATE_PLAN"})

    monkeypatch.setattr(context.service, "prepare", invalid_internal_dto)
    with pytest.raises(
        UnexpectedToolError, match=f"^Error executing tool {tool_name}$"
    ) as caught:
        asyncio.run(context.server.call_tool(tool_name, arguments))
    assert isinstance(caught.value.__cause__, ValidationError)
    assert "PRIVATE_PLAN" in str(caught.value.__cause__)

    async def check() -> None:
        async with Client(context.server, mode=protocol_mode) as client:
            result = await client.call_tool(tool_name, arguments)
            assert result.is_error
            assert result.structured_content is None
            assert result.content == [
                TextContent(type="text", text=f"Error executing tool {tool_name}")
            ]
            assert "PRIVATE_PLAN" not in str(
                result.model_dump(mode="json", by_alias=True)
            )

    asyncio.run(check())
    assert context.preflight.requests == []
    assert context.runner.calls == []
    assert context.service.jobs.list_all_jobs() == []
