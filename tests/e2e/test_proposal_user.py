"""Proposal selection must remain distinct from execution approval."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from tests.e2e.conversation import ModelTurn
from tests.e2e.llm_client import ToolCall
from tests.e2e.test_conversation_user import _decision, _prepare, _present
from tests.e2e.test_llm_research_proposal import ProposalUser
from tests.integration.mcp_harness import MCPWorkspace
from tests.mcp.execution_fixtures import plan_context

if TYPE_CHECKING:
    from pathlib import Path


def test_preliminary_selection_still_requires_saved_plan_and_separate_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = plan_context(tmp_path)
    user = ProposalUser(MCPWorkspace(tmp_path))
    # This case tests consent boundaries; candidate semantics are checked by
    # ProposalUser's separate validator in the live product discovery scenario.
    monkeypatch.setattr(user, "_verify_proposal", lambda: None)
    reply = asyncio.run(user(ModelTurn(1, "建議比較 10 根與 20 根。你要選哪些？", [])))
    assert reply and "還沒有同意" in reply
    assert not user.replies and not user.plans and not context.runner.calls
    prepared = _prepare(context)
    approval = asyncio.run(user(_present(prepared, number=2)))
    assert approval and "同意依這份設定正式執行" in approval
    accepted = _decision(context, prepared.result["plan_id"], "approve")
    assert asyncio.run(user(ModelTurn(3, "完成", [accepted]))) is None
    user.assert_complete()


@pytest.mark.parametrize(
    "tool",
    ["ensure_data", "train_model", "submit_execution_decision", "prepare_optimization"],
)
def test_proposal_selection_never_authorizes_early_execution(
    tmp_path: Path, tool: str
) -> None:
    plan_context(tmp_path)
    user = ProposalUser(MCPWorkspace(tmp_path))
    asyncio.run(user(ModelTurn(1, "比較 10 根與 20 根。", [])))
    with pytest.raises(AssertionError, match="before consent"):
        asyncio.run(
            user(ModelTurn(2, "已嘗試執行", [ToolCall(tool, {}, {"success": False})]))
        )


def test_repeated_preliminary_questions_are_bounded(tmp_path: Path) -> None:
    plan_context(tmp_path)
    user = ProposalUser(MCPWorkspace(tmp_path))
    asyncio.run(user(ModelTurn(1, "比較 10 根與 20 根。", [])))
    with pytest.raises(AssertionError, match="Only one preliminary"):
        asyncio.run(user(ModelTurn(2, "仍不提供確認單。", [])))
