"""Real stdio creation, server restart and fixed collection continuation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from tests.app.test_parameter_plateau import _record
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore


def test_plateau_survives_stdio_restart_without_experiments() -> None:
    with temporary_mcp_workspace() as workspace:
        paths = WorkspacePaths(workspace.workspace)
        sql = SQLiteStore(paths)
        ids = _record(paths, [0.5, None, 0.6])

        async def create() -> dict[str, Any]:
            async with workspace.connect() as client:
                series = await client.call("get_evaluation_series_page")
                assert series["success"] and series["total"] == 0
                tools = (await client.session.list_tools()).tools
                description = next(
                    tool.description
                    for tool in tools
                    if tool.name == "get_evaluation_series_page"
                )
                assert description and "leaderboard" in description
                assert "analyze_parameter_plateau" in description
                result = await client.call(
                    "analyze_parameter_plateau",
                    metric="signal.ic_mean",
                    axes={"/threshold": [1, 2, 3]},
                    entry_ids=ids,
                    limit=1,
                )
                assert isinstance(result, dict)
                return result

        first = asyncio.run(create())
        assert first["success"], first
        _record(paths, [9.0, 9.0, 9.0])

        async def continue_read() -> None:
            async with workspace.connect() as client:
                second = await client.call(
                    "get_parameter_plateau",
                    analysis_id=first["analysis_id"],
                    offset=1,
                    limit=1,
                )
                assert second["summary"] == first["summary"]
                assert second["source_hash"] == first["source_hash"]
                assert second["cells"]["items"][0]["status"] == "undefined"
                text = await client.read_artifact_text(first["artifact"]["artifact_id"])
                assert (
                    hashlib.sha256(text.encode()).hexdigest()
                    == first["artifact"]["sha256"]
                )
                assert json.loads(text)["cells"][1]["value"] is None
                assert text == Path(first["artifact"]["path"]).read_text()

        asyncio.run(continue_read())
        assert sql.list_runs() == []
    assert not workspace.root.exists()
