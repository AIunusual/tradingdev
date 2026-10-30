"""Data tool contracts through in-process MCP dispatch with a local cache."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
from mcp import Client
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import UnexpectedToolError
from mcp_types import CallToolResult, TextContent
from pydantic import TypeAdapter, ValidationError
from tests.data_cache import seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.contracts.common import Page
from tradingdev.app.contracts.data import (
    CachedDataset,
    DatasetInspection,
    DataSourceDescription,
    EnsureDataResponse,
)
from tradingdev.app.data_service import DataService
from tradingdev.mcp.strict_server import StrictMCPServer
from tradingdev.mcp.tools import data

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


def test_source_discovery_provides_settings_without_fetching(tmp_path: Path) -> None:
    server = MCPServer("source-discovery")
    data.register(server, DataService(WorkspacePaths(tmp_path / "workspace")))
    result = asyncio.run(server.call_tool("list_data_sources", {}))
    assert isinstance(result, CallToolResult)
    assert result.structured_content is not None
    sources = TypeAdapter(list[DataSourceDescription]).validate_python(
        result.structured_content["result"]
    )
    choices = {item.source: item for item in sources}
    vision = choices["binance_vision"]
    assert vision.default_market_type == "futures/um"
    assert set(vision.market_types) == {"futures/um", "spot"}
    assert "BTC/USDT" in vision.symbol_examples
    yahoo = choices["yahoo_finance"]
    assert yahoo.timeframes is not None and "1h" in yahoo.timeframes
    assert "2330.TW" in yahoo.symbol_examples
    assert "futures" in yahoo.asset_classes
    assert all(item.limitations for item in sources)


def test_data_tools_validate_local_cache_and_inspection(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = DataService(workspace)
    path = seed_market_cache(
        workspace.processed_data,
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [10.0, 11.0, 12.0],
            }
        ),
    )
    config_path = tmp_path / "fixture.yaml"
    config_path.write_text(
        "backtest:\n  symbol: BTC/USDT\n  timeframe: 1h\n"
        '  start_date: "2024-01-01"\n  end_date: "2024-01-31"\n'
        "  init_cash: 10000\n"
        "data:\n  requirements:\n    market:\n      symbol: BTC/USDT\n"
        "      timeframe: 1h\n    features:\n      - type: custom\n"
        "        source: local\n        column: close\n"
        f'        path: "{path}"\n',
        encoding="utf-8",
    )
    mcp = MCPServer("data-contract-test")
    data.register(mcp, service)

    async def call(name: str, arguments: dict[str, object]) -> dict[str, object]:
        result = await mcp.call_tool(name, arguments)
        assert isinstance(result, CallToolResult)
        assert result.structured_content is not None
        payload = result.structured_content
        assert isinstance(payload, dict)
        return payload

    async def check() -> None:
        listed = Page[CachedDataset].model_validate(
            await call("list_available_data", {})
        )
        assert listed.items[0].years_available == [2024]
        basic = DatasetInspection.model_validate(await call("inspect_dataset", {}))
        assert basic.requirements is None
        detailed = DatasetInspection.model_validate(
            await call("inspect_dataset", {"config_path": str(config_path)})
        )
        assert detailed.market is not None
        assert detailed.market.rows == 3
        assert detailed.requirements is not None
        assert detailed.requirements.features[0].raw_path is None
        assert detailed.features[0].missing_values == 0
        available = EnsureDataResponse.model_validate(
            await call(
                "ensure_data",
                {
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                },
            )
        )
        assert available.rows == 3
        assert available.processed_path == str(path)

    asyncio.run(check())


@pytest.mark.parametrize("protocol_mode", ["legacy", "auto"])
def test_data_tool_rejects_undeclared_cache_fields(
    tmp_path: Path, monkeypatch: MonkeyPatch, protocol_mode: str
) -> None:
    service = DataService(WorkspacePaths(tmp_path / "workspace"))
    monkeypatch.setattr(
        service,
        "list_available_data",
        lambda: [
            {
                "symbol": "BTC",
                "timeframe": "1h",
                "years_available": [],
                "source": "binance_vision",
                "market_type": "futures/um",
                "bad": "PRIVATE_CACHE_RECORD",
            }
        ],
    )
    mcp = MCPServer("data-drift-test")
    data.register(mcp, service)
    with pytest.raises(
        UnexpectedToolError, match="^Error executing tool list_available_data$"
    ) as caught:
        asyncio.run(mcp.call_tool("list_available_data", {}))
    cause = caught.value.__cause__
    assert isinstance(cause, ValidationError)
    assert any(error["type"] == "extra_forbidden" for error in cause.errors())
    assert "PRIVATE_CACHE_RECORD" in str(cause)

    async def check() -> None:
        async with Client(mcp, mode=protocol_mode) as client:
            result = await client.call_tool("list_available_data", {})
            assert result.is_error
            assert result.structured_content is None
            assert result.content == [
                TextContent(
                    type="text", text="Error executing tool list_available_data"
                )
            ]
            wire = result.model_dump(mode="json", by_alias=True)
            assert wire["isError"] is True
            assert wire["structuredContent"] is None
            assert "PRIVATE_CACHE_RECORD" not in str(wire)

    asyncio.run(check())


def test_data_contract_rejects_missing_required_fields() -> None:
    with pytest.raises(ValidationError, match="dataset_id"):
        EnsureDataResponse.model_validate({"rows": 0, "processed_path": "/unused"})


_ENSURE_ARGUMENTS = {
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
}


@pytest.mark.parametrize("protocol_mode", ["legacy", "auto"])
@pytest.mark.parametrize(
    ("changes", "diagnostic"),
    [
        ({"start_date": "invalid-start-date"}, "start_date"),
        ({"end_date": "invalid-end-date"}, "end_date"),
        ({"source": "unknown_source"}, "Unknown data source 'unknown_source'"),
    ],
)
def test_ensure_data_reports_invalid_inputs_before_data_access(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    protocol_mode: str,
    changes: dict[str, str],
    diagnostic: str,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = DataService(workspace)

    def unexpected_access(**_kwargs: object) -> None:
        pytest.fail("Invalid inputs must not reach data retrieval or caching")

    monkeypatch.setattr(service, "ensure", unexpected_access)
    server = StrictMCPServer("data-input-errors")
    data.register(server, service)

    async def check() -> None:
        async with Client(server, mode=protocol_mode) as client:
            result = await client.call_tool("ensure_data", _ENSURE_ARGUMENTS | changes)
            assert result.is_error
            assert result.structured_content is None
            text = " ".join(
                item.text for item in result.content if isinstance(item, TextContent)
            )
            assert diagnostic in text
            assert text != "Error executing tool ensure_data"
            if "source" in changes:
                assert "Known sources:" in text
                assert "binance_vision" in text
            else:
                assert "valid datetime or date" in text
                assert changes[diagnostic] not in text

    asyncio.run(check())
    assert not list(workspace.raw_data.iterdir())
    assert not list(workspace.processed_data.iterdir())


@pytest.mark.parametrize("protocol_mode", ["legacy", "auto"])
@pytest.mark.parametrize("path_kind", ["missing", "directory"])
def test_inspect_dataset_reports_invalid_config_paths_before_inspection(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    protocol_mode: str,
    path_kind: str,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = DataService(workspace)
    config_path = tmp_path / "strategy.yaml"
    if path_kind == "directory":
        config_path.mkdir()

    def unexpected_access(_path: Path | None) -> None:
        pytest.fail("An invalid config path must not reach dataset inspection")

    monkeypatch.setattr(service, "inspect_dataset", unexpected_access)
    server = StrictMCPServer("data-config-errors")
    data.register(server, service)

    async def check() -> None:
        async with Client(server, mode=protocol_mode) as client:
            result = await client.call_tool(
                "inspect_dataset", {"config_path": str(config_path)}
            )
            assert result.is_error
            assert result.structured_content is None
            assert result.content == [
                TextContent(
                    type="text",
                    text=(
                        "Error executing tool inspect_dataset: "
                        f"config_path must point to an existing file: {config_path}"
                    ),
                )
            ]

    asyncio.run(check())
    assert not list(workspace.raw_data.iterdir())
    assert not list(workspace.processed_data.iterdir())


@pytest.mark.parametrize("protocol_mode", ["legacy", "auto"])
@pytest.mark.parametrize("tool_name", ["ensure_data", "inspect_dataset"])
@pytest.mark.parametrize("failure", ["validation", "value", "missing_file"])
def test_data_services_internal_errors_remain_private(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    protocol_mode: str,
    tool_name: str,
    failure: str,
) -> None:
    service = DataService(WorkspacePaths(tmp_path / "workspace"))
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text("backtest: {}\n", encoding="utf-8")
    arguments = (
        _ENSURE_ARGUMENTS
        if tool_name == "ensure_data"
        else {"config_path": str(config_path)}
    )

    def fail_in_service(*_args: object, **_kwargs: object) -> None:
        if failure == "validation":
            EnsureDataResponse.model_validate({"private_record": "PRIVATE_DATA"})
        elif failure == "value":
            raise ValueError("PRIVATE_DATA")
        else:
            raise FileNotFoundError("PRIVATE_DATA")

    monkeypatch.setattr(
        service,
        "ensure" if tool_name == "ensure_data" else "inspect_dataset",
        fail_in_service,
    )
    server = StrictMCPServer("data-service-errors")
    data.register(server, service)
    with pytest.raises(
        UnexpectedToolError, match=f"^Error executing tool {tool_name}$"
    ) as caught:
        asyncio.run(server.call_tool(tool_name, arguments))
    assert "PRIVATE_DATA" in str(caught.value.__cause__)

    async def check() -> None:
        async with Client(server, mode=protocol_mode) as client:
            result = await client.call_tool(tool_name, arguments)
            assert result.is_error
            assert result.structured_content is None
            assert result.content == [
                TextContent(type="text", text=f"Error executing tool {tool_name}")
            ]
            assert "PRIVATE_DATA" not in str(
                result.model_dump(mode="json", by_alias=True)
            )

    asyncio.run(check())


@pytest.mark.parametrize("protocol_mode", ["legacy", "auto"])
def test_inspect_dataset_output_validation_stays_outside_input_boundary(
    tmp_path: Path, monkeypatch: MonkeyPatch, protocol_mode: str
) -> None:
    service = DataService(WorkspacePaths(tmp_path / "workspace"))
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text("backtest: {}\n", encoding="utf-8")
    payload: dict[str, Any] = service.inspect_dataset() | {
        "private_record": "PRIVATE_OUTPUT"
    }
    monkeypatch.setattr(service, "inspect_dataset", lambda _path: payload)
    server = StrictMCPServer("data-output-errors")
    data.register(server, service)

    async def check() -> None:
        async with Client(server, mode=protocol_mode) as client:
            result = await client.call_tool(
                "inspect_dataset", {"config_path": str(config_path)}
            )
            assert result.is_error
            assert result.structured_content is None
            assert result.content == [
                TextContent(type="text", text="Error executing tool inspect_dataset")
            ]
            assert "PRIVATE_OUTPUT" not in str(
                result.model_dump(mode="json", by_alias=True)
            )

    asyncio.run(check())
