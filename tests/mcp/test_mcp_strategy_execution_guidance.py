"""Training guidance crosses real MCP and preflight worker boundaries."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from mcp_types import CallToolResult
from tests.integration.execution_fixtures import execution_options
from tests.integration.mcp_harness import temporary_mcp_workspace
from tests.mcp.execution_fixtures import plan_context

from tradingdev.app.preflight_service import PreflightError


@pytest.mark.parametrize("tool", ["prepare_backtest", "prepare_optimization"])
def test_real_mcp_preflight_returns_training_guidance_without_market_or_jobs(
    tool: str,
) -> None:
    async def check() -> None:
        with temporary_mcp_workspace() as workspace:
            async with workspace.connect() as client:
                for strategy_id in (
                    "quantile_volume",
                    "xgboost_direction",
                    "safety_first_volume",
                ):
                    arguments: dict[str, Any] = {
                        "strategy_id": strategy_id,
                        "symbol": "BTC/USDT",
                        "timeframe": "1h",
                        **execution_options({}),
                    }
                    if tool == "prepare_optimization":
                        arguments.update(
                            param_ranges={"n_estimators": [10]},
                            optimization_metric="total_pnl",
                            train_start="2024-01-01",
                            train_end="2024-01-03",
                            test_start="2024-01-04",
                            test_end="2024-01-07",
                        )
                    else:
                        arguments.update(start_date="2024-01-01", end_date="2024-01-07")
                    response = await client.call(tool, **arguments)
                    assert response["success"] is False
                    assert response["code"] == "strategy_requires_training"
                    assert response["next_tool"] == "prepare_walk_forward"
                    assert "prepare_walk_forward" in response["error"]
                    assert "validation" in response["error"]
                    guidance = response["execution_guidance"]
                    assert guidance["strategy_id"] == strategy_id
                    assert guidance["recommended_mode"] == "walk_forward"
                    assert guidance["required_configuration"] == ["validation"]
                with closing(
                    sqlite3.connect(workspace.workspace / "tradingdev.sqlite")
                ) as connection:
                    for table in ("jobs", "runs", "execution_plans"):
                        assert (
                            connection.execute(
                                f"SELECT count(*) FROM {table}"
                            ).fetchone()[0]
                            == 0
                        )
                assert not list(workspace.data_root.rglob("*.parquet"))

    asyncio.run(check())


def test_nontraining_rejection_keeps_its_code_without_training_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_request: object) -> None:
        raise PreflightError("preflight_timeout", "sample timed out")

    with TemporaryDirectory(prefix="mcp-guidance-", dir=tmp_path) as directory:
        root = Path(directory)
        context = plan_context(root)
        monkeypatch.setattr(context.preflight, "prepare", fail)
        result = asyncio.run(
            context.server.call_tool("prepare_backtest", context.arguments)
        )
        assert isinstance(result, CallToolResult)
        assert result.structured_content is not None
        payload = result.structured_content["result"]
        assert payload["success"] is False
        assert payload["code"] == "preflight_timeout"
        assert payload["error"] == "sample timed out"
        assert payload["next_tool"] is None
        assert payload["execution_guidance"] is None
        text = next(block.text for block in result.content if block.type == "text")
        assert json.loads(text) == payload
        assert context.runner.calls == []
    assert not root.exists()
