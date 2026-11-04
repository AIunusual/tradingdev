"""The leaderboard tool's second source: saved run scopes ranked by a run. metric.

The signal path is covered in ``test_ml_leaderboard_tool.py``. This module checks
the run dispatch through real saved artifacts and the typed MCP response.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from mcp.server import MCPServer
from mcp_types import CallToolResult
from tests.app.test_run_ranking import _publish_run, _scope

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.contracts.ml import (
    LeaderboardPageResponse,
    RankedRunRow,
    SkippedRunRow,
)
from tradingdev.app.ml_service import MLService
from tradingdev.mcp.tools import ml

if TYPE_CHECKING:
    from pathlib import Path

SHARPE = "sharpe_ratio"


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    paths = WorkspacePaths(tmp_path / "workspace")
    paths.ensure()
    return paths


@pytest.fixture
def mcp(workspace: WorkspacePaths) -> MCPServer:
    server = MCPServer("test-ml-runs")
    ml.register(server, MLService(workspace=workspace))
    return server


def _dispatch(mcp: MCPServer, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async def call() -> dict[str, Any]:
        result = await mcp.call_tool(name, arguments)
        assert isinstance(result, CallToolResult)
        assert result.structured_content is not None
        payload = result.structured_content
        assert isinstance(payload, dict)
        response = payload.get("result", payload)
        assert isinstance(response, dict)
        return response

    return asyncio.run(call())


def _run_rows(board: LeaderboardPageResponse) -> list[RankedRunRow]:
    """The placed rows of a run fixture, narrowed to their own model.

    The page carries either kind of row; these tests publish runs, so a signal
    row here would mean the tool answered from the wrong source.
    """
    rows = [row for row in board.ranked if isinstance(row, RankedRunRow)]
    assert len(rows) == len(board.ranked), "run fixture returned a signal row"
    return rows


@pytest.fixture
def published(workspace: WorkspacePaths) -> list[dict[str, str]]:
    """Two saved runs of different quality, left the way execution leaves them."""
    store = SQLiteStore(workspace)
    for run_id, sharpe in (("alpha", 0.5), ("beta", 1.5)):
        _publish_run(workspace, store, run_id, scopes={"full": {SHARPE: sharpe}})
    return [
        {"run_id": "alpha", "scope": "full"},
        {"run_id": "beta", "scope": "full"},
    ]


def test_a_run_metric_ranks_the_selected_scopes(
    mcp: MCPServer, published: list[dict[str, str]]
) -> None:
    payload = _dispatch(
        mcp, "leaderboard", {"metric": "run.sharpe_ratio", "runs": published}
    )

    board = LeaderboardPageResponse.model_validate(payload)
    assert board.source == "run"
    assert board.metric == "run.sharpe_ratio"
    assert board.direction == "maximize"
    assert board.n_candidates == board.n_ranked == 2
    assert [row.run_id for row in _run_rows(board)] == ["beta", "alpha"]
    assert all(isinstance(row, RankedRunRow) for row in board.ranked)
    assert {row.source for row in board.ranked} == {"run"}
    assert board.n_skipped == 0 and board.skipped == []


def test_scope_parameters_and_observation_metadata_survive_the_tool(
    mcp: MCPServer, workspace: WorkspacePaths
) -> None:
    store = SQLiteStore(workspace)
    scopes = {}
    for index, value in enumerate((0.5, 1.5)):
        scope = _scope({SHARPE: value}).model_copy(
            update={"parameters": {"window": 10 + index}, "split": "train"}
        )
        scope.metadata["observations"] = {
            "start_date": f"2024-02-0{index + 1}",
            "end_date": f"2024-02-0{index + 2}",
        }
        scope.metadata["strategy_parameters"] = {"fitted_threshold": index + 0.5}
        scopes[f"trial/{index}/train"] = scope
    _publish_run(workspace, store, "trials", scope_models=scopes, window=99)
    payload = _dispatch(
        mcp,
        "leaderboard",
        {
            "metric": "run.sharpe_ratio",
            "runs": [{"run_id": "trials", "scope": name} for name in scopes],
        },
    )
    board = LeaderboardPageResponse.model_validate(payload)
    rows = _run_rows(board)
    assert [row.scope for row in rows] == ["trial/1/train", "trial/0/train"]
    assert [row.parameters for row in rows] == [{"window": 11}, {"window": 10}]
    assert [row.metadata for row in rows] == [
        scopes["trial/1/train"].metadata,
        scopes["trial/0/train"].metadata,
    ]
    assert all(row.conditions.start_date.startswith("2024-01-01") for row in rows)
    assert any("conditional" in caveat for caveat in board.caveats)


def test_a_run_ranking_carries_no_signal_only_fields(
    mcp: MCPServer, published: list[dict[str, str]]
) -> None:
    """A backtest selection has no evaluation context or method mix to report.

    Absent rather than empty: an empty dict would read as "measured, and the
    answer was nothing", which is a different claim from "not applicable".
    """
    payload = _dispatch(
        mcp, "leaderboard", {"metric": "run.sharpe_ratio", "runs": published}
    )

    assert payload["context"] is None
    assert payload["families"] is None
    assert payload["parameter_axes"] is None
    # The row still carries the conditions the value was measured under.
    row = payload["ranked"][0]
    assert row["conditions"]["symbol"] == "BTC/USDT"
    assert row["conditions"]["fees"] == pytest.approx(0.0006)


def test_a_run_selection_pages_like_a_signal_ranking(
    mcp: MCPServer, published: list[dict[str, str]]
) -> None:
    first = LeaderboardPageResponse.model_validate(
        _dispatch(
            mcp,
            "leaderboard",
            {"metric": "run.sharpe_ratio", "runs": published, "limit": 1},
        )
    )
    assert [row.run_id for row in _run_rows(first)] == ["beta"]
    assert first.n_ranked == 2 and first.next_offset == 1

    second = LeaderboardPageResponse.model_validate(
        _dispatch(
            mcp,
            "leaderboard",
            {"metric": "run.sharpe_ratio", "runs": published, "offset": 1, "limit": 1},
        )
    )
    assert [row.run_id for row in _run_rows(second)] == ["alpha"]
    assert second.next_offset is None


def test_an_unreadable_scope_is_reported_as_a_skipped_run_row(
    mcp: MCPServer,
) -> None:
    payload = _dispatch(
        mcp,
        "leaderboard",
        {
            "metric": "run.sharpe_ratio",
            "runs": [{"run_id": "missing", "scope": "full"}],
        },
    )

    board = LeaderboardPageResponse.model_validate(payload)
    assert board.source == "run" and board.n_ranked == 0
    (skipped,) = board.skipped
    assert isinstance(skipped, SkippedRunRow)
    assert skipped.candidate_id == "run:missing#full"
    assert skipped.reason == "unknown run: missing"


@pytest.mark.parametrize(
    ("metric", "extra", "expected"),
    [
        ("run.sharpe_ratio", {}, "at least one run selection"),
        (
            "run.sharpe_ratio",
            {"runs": [{"run_id": "a"}], "symbol": "BTC/USDT"},
            "not signal filters",
        ),
        ("signal.ic_mean", {"runs": [{"run_id": "a"}]}, "not run selections"),
        ("backtest.sharpe_ratio", {}, "needs a 'signal.' or 'run.' prefix"),
    ],
)
def test_the_tool_refuses_a_request_that_mixes_the_two_sources(
    mcp: MCPServer, metric: str, extra: dict[str, Any], expected: str
) -> None:
    """A half-filled table would look like a ranking of two things."""
    payload = _dispatch(mcp, "leaderboard", {"metric": metric, **extra})

    assert payload["success"] is False
    assert payload["code"] == "invalid_leaderboard_request"
    assert expected in payload["error"]


def test_run_ranking_refuses_the_signal_context_override(
    mcp: MCPServer, published: list[dict[str, str]]
) -> None:
    payload = _dispatch(
        mcp,
        "leaderboard",
        {"metric": "run.sharpe_ratio", "runs": published, "allow_mixed": True},
    )
    assert payload["success"] is False
    assert payload["code"] == "invalid_leaderboard_request"
    assert "not signal filters" in payload["error"]


def test_the_advertised_schema_offers_runs_and_a_metric(
    mcp: MCPServer,
) -> None:
    async def tools() -> list[Any]:
        return list(await mcp.list_tools())

    entries = {tool.name: tool for tool in asyncio.run(tools())}
    properties = entries["leaderboard"].input_schema["properties"]
    assert "runs" in properties and "metric" in properties
    assert entries["leaderboard"].input_schema["required"] == ["metric"]


def test_a_duplicate_selection_is_refused_through_the_tool(
    mcp: MCPServer, published: list[dict[str, str]]
) -> None:
    """Two names for one measurement must not become a two-row ranking."""
    payload = _dispatch(
        mcp,
        "leaderboard",
        {"metric": "run.sharpe_ratio", "runs": [published[0], published[0]]},
    )

    assert payload["success"] is False
    assert payload["code"] == "duplicate_selections"
    assert "alpha" in payload["error"]


def test_an_omitted_scope_that_resolves_to_a_selected_one_is_refused(
    mcp: MCPServer, published: list[dict[str, str]]
) -> None:
    """The inputs differ; the resolved measurement does not."""
    payload = _dispatch(
        mcp,
        "leaderboard",
        {
            "metric": "run.sharpe_ratio",
            "runs": [{"run_id": "alpha"}, {"run_id": "alpha", "scope": "full"}],
        },
    )

    assert payload["success"] is False
    assert payload["code"] == "duplicate_selections"
