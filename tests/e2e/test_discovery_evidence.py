"""Offline checks for natural intent and execution-bound discovery evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.e2e.codex_harness import LIFECYCLE_TOOLS
from tests.e2e.conversation import ModelTurn
from tests.e2e.discovery_workflow import NATURAL_REQUEST, verify_discovery_outcome
from tests.e2e.llm_client import ToolCall
from tests.integration.mcp_harness import MCPWorkspace
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore

if TYPE_CHECKING:
    from pathlib import Path


def test_natural_request_contains_requirements_without_tool_answers() -> None:
    assert all(name not in NATURAL_REQUEST for name in LIFECYCLE_TOOLS)
    assert "2024-01-01" in NATURAL_REQUEST and "2024-01-08" in NATURAL_REQUEST
    assert "fast_period=5" in NATURAL_REQUEST and "slow_period=20" in NATURAL_REQUEST
    assert "績效、風險與限制" in NATURAL_REQUEST
    assert "generate_report" not in NATURAL_REQUEST


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "contract",
        "wrong_section",
        "empty_contract",
        "save",
        "revision",
        "lookup",
        "extra_run",
    ],
)
def test_discovery_requires_saved_execution_identity_before_report_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str | None
) -> None:
    workspace = MCPWorkspace(tmp_path)
    store = SQLiteStore(WorkspacePaths(workspace.workspace))
    store.create_run(
        run_id="run",
        job_id="job",
        strategy_id="llm_sma",
        revision_id="revision",
        artifact_dir=tmp_path,
        metrics={},
    )
    calls = [
        ToolCall(
            "list_strategies",
            {},
            {"items": [], "offset": 0, "limit": 20, "total": 0, "next_offset": None},
        ),
        ToolCall(
            "get_strategy_contract",
            {},
            {"section": "core", "content": {"requirements": ["contract"]}},
        ),
        ToolCall("save_strategy", {"strategy_id": "llm_sma"}, {"success": True}),
        ToolCall(
            "get_job_status",
            {},
            {
                "status": "done",
                "run_id": "run",
                "revision_id": "revision",
            },
        ),
        ToolCall("get_run", {"run_id": "run"}, {"success": True}),
    ]
    if damage == "contract":
        calls.pop(1)
    elif damage == "wrong_section":
        calls[1].result["section"] = "example"
    elif damage == "empty_contract":
        calls[1].result["content"] = {}
    elif damage == "save":
        calls[2].result["success"] = False
    elif damage == "revision":
        calls[3].result["revision_id"] = "wrong-revision"
    elif damage == "lookup":
        calls.pop()
    elif damage == "extra_run":
        store.create_run(
            run_id="other",
            job_id="other",
            strategy_id="llm_sma",
            artifact_dir=tmp_path,
            metrics={},
        )
    verified: list[str] = []

    def verify(*args: object, **kwargs: object) -> str:
        verified.append(str(args[2]))
        assert kwargs["expected_commentary"] is None
        return "report"

    monkeypatch.setattr("tests.e2e.discovery_workflow.verify_report_delivery", verify)
    turn = ModelTurn(2, "Complete", calls)
    if damage is None:
        assert verify_discovery_outcome(workspace, calls, turn) == "run"
        assert verified == ["run"]
    else:
        with pytest.raises(AssertionError):
            verify_discovery_outcome(workspace, calls, turn)
        assert verified == []
