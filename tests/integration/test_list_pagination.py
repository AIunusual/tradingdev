"""List consumers follow advertised pages without hiding the MCP page contract."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, call, create_autospec

import pytest
from mcp import ClientSession

from tests.integration.mcp_harness import MCPClient
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore

if TYPE_CHECKING:
    from tests.integration.mcp_harness import MCPWorkspace


def test_explicit_reader_follows_pages_and_preserves_the_requested_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MCPClient(create_autospec(ClientSession, instance=True))
    pages = [
        {
            "items": [{"artifact_id": "first"}],
            "offset": 0,
            "limit": 1,
            "total": 2,
            "next_offset": 1,
        },
        {
            "items": [{"artifact_id": "second"}],
            "offset": 1,
            "limit": 1,
            "total": 2,
            "next_offset": None,
        },
    ]
    invoke = AsyncMock(side_effect=pages)
    monkeypatch.setattr(client, "call", invoke)

    assert asyncio.run(
        client.read_all_pages("list_artifacts", limit=1, run_id="run")
    ) == [{"artifact_id": "first"}, {"artifact_id": "second"}]
    assert invoke.await_args_list == [
        call("list_artifacts", offset=0, limit=1, run_id="run"),
        call("list_artifacts", offset=1, limit=1, run_id="run"),
    ]


@pytest.mark.parametrize("damage", ["early_end", "no_progress", "changed_total"])
def test_explicit_reader_rejects_incomplete_or_inconsistent_pages(
    monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    client = MCPClient(create_autospec(ClientSession, instance=True))
    first: dict[str, Any] = {
        "items": [{"artifact_id": "first"}],
        "offset": 0,
        "limit": 1,
        "total": 2,
        "next_offset": 1,
    }
    second = {
        "items": [{"artifact_id": "second"}],
        "offset": 1,
        "limit": 1,
        "total": 3,
        "next_offset": None,
    }
    if damage == "early_end":
        first["next_offset"] = None
    elif damage == "no_progress":
        first["next_offset"] = 0
    invoke = AsyncMock(side_effect=[first, second])
    monkeypatch.setattr(client, "call", invoke)

    with pytest.raises(AssertionError):
        asyncio.run(client.read_all_pages("list_artifacts", limit=1))


@pytest.mark.integration
@pytest.mark.anyio
async def test_large_inventories_page_through_the_real_stdio_server(
    mcp_workspace: MCPWorkspace,
) -> None:
    store = SQLiteStore(WorkspacePaths(mcp_workspace.workspace))
    expected = {f"artifact-{index:02}" for index in range(25)}
    expected_runs = {f"run-{index:02}" for index in range(25)}
    for artifact_id in sorted(expected):
        store.create_artifact(
            artifact_id=artifact_id,
            run_id="paged-run",
            artifact_type="fixture",
            path=mcp_workspace.root / artifact_id,
            metadata={"large_metadata": "x" * 10_000},
        )
    for run_id in sorted(expected_runs):
        store.create_run(
            run_id=run_id,
            job_id=f"job-{run_id}",
            strategy_id="paged-strategy",
            artifact_dir=mcp_workspace.root / run_id,
            metrics={"large_metric": "x" * 10_000},
        )

    async with mcp_workspace.connect() as client:
        for name, filters, identity, all_ids in (
            ("list_artifacts", {"run_id": "paged-run"}, "artifact_id", expected),
            (
                "list_runs",
                {"strategy_id": "paged-strategy"},
                "run_id",
                expected_runs,
            ),
        ):
            first = await client.call(name, **filters)
            assert first["offset"] == 0 and first["limit"] == 20
            assert first["total"] == 25 and first["next_offset"] == 20
            assert len(first["items"]) == 20
            assert all(
                "metadata" not in item and "metrics" not in item
                for item in first["items"]
            )
            items = await client.read_all_pages(name, limit=20, **filters)
            assert {item[identity] for item in items} == all_ids
            assert len(items) == 25
            maximum = await client.call(name, limit=100, **filters)
            assert len(maximum["items"]) == 25 and maximum["next_offset"] is None
            rejected = await client.session.call_tool(name, filters | {"limit": 101})
            assert rejected.is_error
        empty = await client.call("list_artifacts", run_id="missing")
        assert empty == {
            "items": [],
            "offset": 0,
            "limit": 20,
            "total": 0,
            "next_offset": None,
        }
