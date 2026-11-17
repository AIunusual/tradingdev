"""Real MCP discovery, bounded row reads and verified export for issue #62."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from mcp.server import MCPServer
from mcp_types import CallToolResult
from tests.data_cache import seed_market_cache

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.ml_datasets import MLDatasetStore
from tradingdev.app.ml_service import MLService
from tradingdev.mcp.tools import ml

if TYPE_CHECKING:
    from pathlib import Path

_ROWS = 300


def _dispatch(
    server: MCPServer, name: str, arguments: dict[str, object]
) -> dict[str, Any]:
    async def call() -> dict[str, Any]:
        result = await server.call_tool(name, arguments)
        assert isinstance(result, CallToolResult)
        payload = result.structured_content
        assert isinstance(payload, dict)
        response = payload.get("result", payload)
        assert isinstance(response, dict)
        return response

    return asyncio.run(call())


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    paths = WorkspacePaths(tmp_path / "workspace")
    paths.ensure()
    positions = np.arange(_ROWS, dtype=float)
    close = 100.0 + 10.0 * np.sin(positions / 5.0) + positions * 0.01
    seed_market_cache(
        paths.processed_data,
        pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=_ROWS, freq="h", tz="UTC"
                ),
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.full(_ROWS, 10.0),
            }
        ),
    )
    return paths


def test_mcp_pages_rows_and_reports_missing_evidence(workspace: WorkspacePaths) -> None:
    service = MLService(workspace=workspace)
    server = MCPServer("ml-series-tool-test")
    ml.register(server, service)
    built = service.build_dataset(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-31",
        feature_set="returns",
        horizon=1,
    )
    dataset_id = str(built["dataset"]["dataset_id"])
    frame, manifest = MLDatasetStore(workspace.ml_datasets).load(dataset_id)
    column = str(manifest["feature_columns"][0])

    recorded = _dispatch(
        server,
        "evaluate_signal",
        {
            "dataset_id": dataset_id,
            "column": column,
            "threshold": float(frame[column].median()),
            "window": 20,
            "record": True,
        },
    )
    artifact_id = str(recorded["series_artifact_id"])
    assert recorded["producer_source"] == "backend_rule"

    discovered = _dispatch(server, "get_evaluation_series_page", {})
    assert discovered["mode"] == "discovery"
    assert artifact_id in {item["artifact_id"] for item in discovered["items"]}

    pages = [
        _dispatch(
            server,
            "get_evaluation_series_page",
            {"artifact_id": artifact_id, "offset": offset, "limit": 100},
        )
        for offset in (0, 100, 200)
    ]
    assert [page["mode"] for page in pages] == ["rows", "rows", "rows"]
    assert [page["next_offset"] for page in pages] == [100, 200, None]
    assembled = [row for page in pages for row in page["rows"]]

    exported = _dispatch(
        server, "export_evaluation_series", {"artifact_id": artifact_id}
    )
    assert assembled == exported["rows"]
    assert len(exported["rows"]) == len(frame)
    assert all(row["pearson_ic"] is None for row in exported["rows"][:19])
    assert exported["rows"][19]["pearson_ic"] is not None

    missing = _dispatch(server, "export_evaluation_series", {"artifact_id": "0" * 64})
    assert missing["success"] is False
    assert "unavailable" in missing["error"]
