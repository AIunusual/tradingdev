"""Reject premature attempts and artifacts without using a model to grade prose."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from tests.e2e.conversation import ModelTurn
from tests.e2e.llm_client import ToolCall
from tests.e2e.novice_intake import MissingInputUser, assert_missing_input_turn
from tests.integration.mcp_harness import MCPWorkspace
from tradingdev.adapters.storage.execution_plans import ExecutionPlanStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore

if TYPE_CHECKING:
    from pathlib import Path


def _workspace(path: Path) -> MCPWorkspace:
    workspace = MCPWorkspace(path)
    paths = WorkspacePaths(workspace.workspace)
    ExecutionPlanStore(paths, SQLiteStore(paths))
    return workspace


@pytest.mark.parametrize("result", [{"success": True}, {"success": False}])
@pytest.mark.parametrize(
    "tool", ["save_strategy", "prepare_backtest", "ensure_data", "train_model"]
)
def test_missing_choices_reject_successful_and_failed_early_attempts(
    tmp_path: Path, tool: str, result: dict[str, bool]
) -> None:
    workspace = _workspace(tmp_path)
    turn = ModelTurn(1, "請補充回測期間與 K 棒週期。", [ToolCall(tool, {}, result)])
    with pytest.raises(AssertionError, match="including rejected attempts"):
        assert_missing_input_turn(workspace, turn, read_only_tools={"list_strategies"})


@pytest.mark.parametrize(
    "artifact", ["generated_strategies", "models", "execution_plans"]
)
def test_missing_choices_reject_unreported_artifacts(
    tmp_path: Path, artifact: str
) -> None:
    workspace = _workspace(tmp_path)
    directory = workspace.workspace / artifact
    directory.mkdir(exist_ok=True)
    (directory / "unexpected.json").write_text("{}")
    turn = ModelTurn(1, "請問要使用哪個期間？", [])
    with pytest.raises(AssertionError, match="Research artifacts appeared"):
        assert_missing_input_turn(workspace, turn, read_only_tools=set())


@pytest.mark.parametrize("directory", ["ordinary", "研究 (case) 100%20 ?#"])
def test_missing_user_stops_without_supplying_hidden_choices(
    tmp_path: Path, directory: str
) -> None:
    workspace = _workspace(tmp_path / directory)
    user = MissingInputUser(workspace, {"list_strategies"})
    turn = ModelTurn(
        1, "你要用每幾分鐘或幾小時一根 K 棒？", [ToolCall("list_strategies", {}, {})]
    )
    assert asyncio.run(user(turn)) is None
    assert user.turns == [turn]
