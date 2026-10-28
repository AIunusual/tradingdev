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

from tradingdev.app.contracts.execution_requests import BacktestInput
from tradingdev.app.contracts.plans import PlanResponse
from tradingdev.domain.research import NewResearchSpec

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch

    from tradingdev.domain.preflight import PreflightRequest


_PREPARE_TOOLS = [
    "prepare_backtest",
    "prepare_walk_forward",
    "prepare_optimization",
]


def test_request_serialization_preserves_omissions_and_the_research_discriminator() -> (
    None
):
    research = NewResearchSpec.model_validate(
        {
            "strategy_outline": "均線交叉買賣",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "fees": 0,
        }
    )
    request = BacktestInput(strategy_id="fixture", research=research)
    encoded = request.model_dump(mode="json", exclude_unset=True)
    assert encoded["research"]["kind"] == "new"
    assert encoded["research"]["fees"] == 0
    assert "init_cash" not in encoded["research"]
    assert "slippage" not in encoded["research"]
    decoded = BacktestInput.model_validate(encoded)
    assert decoded.research.model_fields_set == research.model_fields_set | {"kind"}


def _arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name != "prepare_optimization":
        return arguments
    return arguments | {
        "research": {
            **arguments["research"],
            "end_date": "2024-01-08T23:59:59.999999",
        },
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
    "field", ["strategy_outline", "symbol", "timeframe", "start_date", "end_date"]
)
def test_missing_research_choice_is_rejected_before_any_preparation(
    tmp_path: Path, protocol_mode: str, tool_name: str, field: str
) -> None:
    context = plan_context(tmp_path)
    arguments = _arguments(tool_name, context.arguments)
    arguments["research"] = dict(arguments["research"])
    del arguments["research"][field]

    async def check() -> None:
        async with Client(context.server, mode=protocol_mode) as client:
            result = await client.call_tool(tool_name, arguments)
            assert result.is_error
            assert field in str(result.content)

    asyncio.run(check())
    assert context.preflight.requests == []
    assert context.runner.calls == []
    assert context.service.jobs.list_all_jobs() == []


@pytest.mark.parametrize("tool_name", _PREPARE_TOOLS)
def test_prepare_schema_has_one_research_source_for_market_and_period(
    tmp_path: Path, tool_name: str
) -> None:
    context = plan_context(tmp_path)
    tool = next(
        tool
        for tool in asyncio.run(context.server.list_tools())
        if tool.name == tool_name
    )
    assert "research" in tool.input_schema["required"]
    assert not {"symbol", "timeframe", "start_date", "end_date"}.intersection(
        tool.input_schema["properties"]
    )


@pytest.mark.parametrize("tool_name", _PREPARE_TOOLS)
def test_inheritance_requires_an_explicit_immutable_revision_before_preparation(
    tmp_path: Path, tool_name: str
) -> None:
    context = plan_context(tmp_path)
    arguments = _arguments(tool_name, context.arguments) | {
        "research": {"kind": "inherit_selected_revision"}
    }

    async def check() -> None:
        async with Client(context.server) as client:
            result = await client.call_tool(tool_name, arguments)
            assert result.is_error
            assert "revision_id" in str(result.content)
            assert "get_strategy" in str(result.content)

    asyncio.run(check())
    assert context.preflight.requests == []
    assert context.runner.calls == []


@pytest.mark.parametrize("tool_name", ["prepare_backtest", "prepare_optimization"])
def test_validation_replacement_is_only_available_for_walk_forward(
    tmp_path: Path, tool_name: str
) -> None:
    context = plan_context(tmp_path)
    arguments = _arguments(tool_name, context.arguments) | {
        "validation_config": {"n_splits": 2}
    }

    async def check() -> None:
        async with Client(context.server) as client:
            result = await client.call_tool(tool_name, arguments)
            assert result.is_error
            assert "validation_config" in str(result.content)

    asyncio.run(check())
    assert context.preflight.requests == []


@pytest.mark.parametrize("protocol_mode", ["legacy", "auto"])
@pytest.mark.parametrize("tool_name", _PREPARE_TOOLS)
@pytest.mark.parametrize(
    "field",
    [
        "sample_bars",
        "minimum_history_bars",
    ],
)
def test_removed_market_sample_inputs_fail_before_validation_or_plan_creation(
    tmp_path: Path,
    protocol_mode: str,
    tool_name: str,
    field: str,
) -> None:
    context = plan_context(tmp_path)
    arguments = _arguments(tool_name, context.arguments) | {field: 128}
    arguments["parameters"] = {"private_note": "PRIVATE_INPUT"}

    async def check() -> None:
        async with Client(context.server, mode=protocol_mode) as client:
            tool = next(
                tool
                for tool in (await client.list_tools()).tools
                if tool.name == tool_name
            )
            validator = Draft202012Validator(tool.input_schema)
            assert field not in tool.input_schema["properties"]
            assert not validator.is_valid(arguments)
            result = await client.call_tool(tool_name, arguments)
            assert result.is_error
            assert result.structured_content is None
            text = " ".join(
                item.text for item in result.content if isinstance(item, TextContent)
            )
            assert field in text
            assert "Unknown arguments" in text
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

    def invalid_internal_dto(_request: PreflightRequest, _evidence_id: str) -> None:
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


@pytest.mark.parametrize("protocol_mode", ["legacy", "auto"])
@pytest.mark.parametrize(
    "validation",
    [
        {"n_splits": 2},
        {"mode": "automatic", "embargo_bars": 0},
        {"mode": "automatic", "n_splits": True},
        {"mode": "purged", "label_horizon": 0},
        {"mode": "purged", "label_horizon": 1, "train_ratio": 0.8},
        {"mode": "explicit", "train_start": "2024-01-01"},
    ],
)
def test_invalid_fold_mode_is_rejected_before_preflight(
    tmp_path: Path,
    protocol_mode: str,
    validation: dict[str, Any],
) -> None:
    context = plan_context(tmp_path)
    arguments = context.arguments | {"validation_config": validation}
    tool = next(
        tool
        for tool in asyncio.run(context.server.list_tools())
        if tool.name == "prepare_walk_forward"
    )
    assert not Draft202012Validator(tool.input_schema).is_valid(arguments)

    async def check() -> None:
        async with Client(context.server, mode=protocol_mode) as client:
            result = await client.call_tool("prepare_walk_forward", arguments)
            assert result.is_error
            assert "validation_config" in str(result.content)

    asyncio.run(check())
    assert context.preflight.requests == []
    assert context.runner.calls == []
    assert context.service.jobs.list_all_jobs() == []
