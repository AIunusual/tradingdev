"""Offline rejection-evidence checks, plus the same real MCP path without an LLM."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import pytest

from tests.e2e.conversation import ModelTurn
from tests.e2e.llm_client import ToolCall
from tests.e2e.strategy_execution_guidance import (
    GUIDANCE_PROMPT,
    GUIDANCE_TOOLS,
    STRATEGY_ID,
    assert_guidance_workflow,
    assert_guidance_workspace,
    guidance_arguments,
)
from tests.integration.mcp_harness import MCPWorkspace, temporary_mcp_workspace

if TYPE_CHECKING:
    from collections.abc import Iterator


def _report(calls: list[ToolCall]) -> str:
    return json.dumps(
        {
            "strategy_id": STRATEGY_ID,
            "execution_capabilities": calls[1].result["execution_capabilities"],
            "rejections": [
                {
                    "tool": call.name,
                    "next_tool": call.result["next_tool"],
                    "required_configuration": call.result["execution_guidance"][
                        "required_configuration"
                    ],
                }
                for call in calls[2:]
            ],
            "execution_started": False,
        }
    )


@pytest.fixture
def evidence() -> tuple[list[ToolCall], list[ModelTurn]]:
    capabilities = {
        "training_requirement": "required",
        "supported_execution_modes": ["walk_forward"],
    }
    strategy = {
        "strategy_id": STRATEGY_ID,
        "execution_capabilities": capabilities,
    }
    calls = [
        ToolCall(
            "list_strategies",
            {},
            {
                "items": [deepcopy(strategy)],
                "offset": 0,
                "limit": 20,
                "total": 1,
                "next_offset": None,
            },
        ),
        ToolCall(
            "get_strategy",
            guidance_arguments("get_strategy"),
            {
                **deepcopy(strategy),
                "success": True,
                "yaml_config": "validation:\n  n_splits: 2\n",
            },
        ),
    ]
    for tool in GUIDANCE_TOOLS[2:]:
        calls.append(
            ToolCall(
                tool,
                guidance_arguments(tool),
                {
                    "success": False,
                    "code": "strategy_requires_training",
                    "error": "This strategy requires training before evaluation.",
                    "execution_guidance": {
                        "strategy_id": STRATEGY_ID,
                        "capabilities": deepcopy(capabilities),
                        "recommended_mode": "walk_forward",
                        "required_configuration": ["validation"],
                    },
                    "next_tool": "prepare_walk_forward",
                },
            )
        )
    return calls, [ModelTurn(1, _report(calls), calls)]


def test_guidance_accepts_complete_evidence_without_mutating_it(
    evidence: tuple[list[ToolCall], list[ModelTurn]],
) -> None:
    original = deepcopy(evidence)
    assert_guidance_workflow(*evidence)
    assert evidence == original
    assert "prepare_walk_forward" not in GUIDANCE_PROMPT
    assert "validation" not in GUIDANCE_PROMPT


@pytest.mark.parametrize("wire_format", ["structured", "text"])
def test_guidance_accepts_mcp_wire_results(
    evidence: tuple[list[ToolCall], list[ModelTurn]], wire_format: str
) -> None:
    calls, turns = evidence
    for call in calls:
        if wire_format == "structured":
            call.result = {
                "structuredContent": {"result": call.result},
                "isError": False,
            }
        else:
            call.result = {
                "content": [{"type": "text", "text": json.dumps(call.result)}],
                "isError": False,
            }
    assert_guidance_workflow(calls, turns)


@pytest.mark.parametrize("index", range(4))
def test_guidance_requires_every_ordered_step(
    evidence: tuple[list[ToolCall], list[ModelTurn]], index: int
) -> None:
    calls, turns = evidence
    calls.pop(index)
    with pytest.raises(AssertionError, match="exactly discovery"):
        assert_guidance_workflow(calls, turns)


def test_guidance_rejects_preparation_before_discovery(
    evidence: tuple[list[ToolCall], list[ModelTurn]],
) -> None:
    calls, turns = evidence
    calls[1], calls[2] = calls[2], calls[1]
    with pytest.raises(AssertionError, match="exactly discovery"):
        assert_guidance_workflow(calls, turns)


def test_guidance_requires_the_target_in_the_strategy_page(
    evidence: tuple[list[ToolCall], list[ModelTurn]],
) -> None:
    calls, turns = evidence
    calls[0].result.update(items=[], total=0)
    with pytest.raises(AssertionError, match="Discovery must identify"):
        assert_guidance_workflow(calls, turns)


@pytest.mark.parametrize("index", [1, 2, 3])
def test_guidance_rejects_another_requested_strategy(
    evidence: tuple[list[ToolCall], list[ModelTurn]], index: int
) -> None:
    calls, turns = evidence
    calls[index].arguments["strategy_id"] = "kd_crossover"
    with pytest.raises(AssertionError):
        assert_guidance_workflow(calls, turns)


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_guidance_requires_consistent_capabilities(
    evidence: tuple[list[ToolCall], list[ModelTurn]], index: int
) -> None:
    calls, turns = evidence
    result = calls[index].result
    capabilities = (
        result["items"][0]["execution_capabilities"]
        if index == 0
        else result["execution_capabilities"]
        if index == 1
        else result["execution_guidance"]["capabilities"]
    )
    capabilities["training_requirement"] = "optional"
    with pytest.raises(AssertionError):
        assert_guidance_workflow(calls, turns)


@pytest.mark.parametrize("index", [2, 3])
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("code", "preflight_failed"),
        ("success", True),
        ("error", ""),
        ("execution_guidance", None),
        ("next_tool", "prepare_backtest"),
        ("plan_id", "unexpected-plan"),
        ("job_id", "unexpected-job"),
    ],
)
def test_guidance_rejects_incomplete_or_wrong_rejections(
    evidence: tuple[list[ToolCall], list[ModelTurn]], index: int, key: str, value: Any
) -> None:
    calls, turns = evidence
    calls[index].result[key] = value
    with pytest.raises(AssertionError):
        assert_guidance_workflow(calls, turns)


@pytest.mark.parametrize("key", ["strategy_id", "required_configuration"])
def test_guidance_rejects_wrong_structured_next_action(
    evidence: tuple[list[ToolCall], list[ModelTurn]], key: str
) -> None:
    calls, turns = evidence
    calls[2].result["execution_guidance"][key] = "unrelated"
    with pytest.raises(AssertionError, match="actionable guidance"):
        assert_guidance_workflow(calls, turns)


@pytest.mark.parametrize("tool", ["prepare_walk_forward", "submit_execution_decision"])
def test_guidance_does_not_allow_training_or_approval(
    evidence: tuple[list[ToolCall], list[ModelTurn]], tool: str
) -> None:
    calls, turns = evidence
    assert tool not in GUIDANCE_TOOLS
    calls.append(ToolCall(tool, {}, {"success": True}))
    with pytest.raises(AssertionError, match="exactly discovery"):
        assert_guidance_workflow(calls, turns)


def test_guidance_requires_one_unapproved_turn(
    evidence: tuple[list[ToolCall], list[ModelTurn]],
) -> None:
    calls, turns = evidence
    turns.append(ModelTurn(2, "approve", []))
    with pytest.raises(AssertionError, match="one unapproved model turn"):
        assert_guidance_workflow(calls, turns)


@pytest.mark.parametrize(
    "mutation", ["tool", "configuration", "started", "invalid_json"]
)
def test_final_report_must_retain_observed_next_actions(
    evidence: tuple[list[ToolCall], list[ModelTurn]], mutation: str
) -> None:
    calls, turns = evidence
    report = json.loads(turns[0].text)
    if mutation == "tool":
        report["rejections"][0]["next_tool"] = "prepare_backtest"
    elif mutation == "configuration":
        report["rejections"][1]["required_configuration"] = []
    elif mutation == "started":
        report["execution_started"] = True
    text = "complete" if mutation == "invalid_json" else json.dumps(report)
    with pytest.raises(AssertionError, match="Final guidance"):
        assert_guidance_workflow(calls, [ModelTurn(1, text, calls)])


@pytest.fixture
def empty_workspace(tmp_path: Path) -> Iterator[MCPWorkspace]:
    with TemporaryDirectory(dir=tmp_path) as directory:
        workspace = MCPWorkspace(Path(directory))
        workspace.workspace.mkdir()
        with closing(
            sqlite3.connect(workspace.workspace / "tradingdev.sqlite")
        ) as connection:
            for table in ("execution_plans", "jobs", "runs", "artifacts"):
                connection.execute(f"CREATE TABLE {table} (id TEXT)")
            connection.commit()
        yield workspace
    assert not workspace.root.exists()


def test_empty_diagnostic_workspace_is_accepted(empty_workspace: MCPWorkspace) -> None:
    assert_guidance_workspace(empty_workspace)


@pytest.mark.parametrize("table", ["execution_plans", "jobs", "runs", "artifacts"])
def test_guidance_rejects_persisted_execution_records(
    empty_workspace: MCPWorkspace, table: str
) -> None:
    with closing(
        sqlite3.connect(empty_workspace.workspace / "tradingdev.sqlite")
    ) as connection:
        connection.execute(f"INSERT INTO {table} VALUES ('unexpected')")
        connection.commit()
    with pytest.raises(AssertionError, match="persisted"):
        assert_guidance_workspace(empty_workspace)


@pytest.mark.parametrize(
    "relative_path",
    [
        "workspace/execution_plans/confirmation.html",
        "workspace/runs/result/manifest.json",
        "workspace/models/model.pkl",
        "workspace/data/raw/prices.parquet",
        "market-data/processed/prices.parquet",
    ],
)
def test_guidance_rejects_saved_execution_or_market_files(
    empty_workspace: MCPWorkspace, relative_path: str
) -> None:
    path = empty_workspace.root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("unexpected", encoding="utf-8")
    with pytest.raises(AssertionError, match="saved artifacts"):
        assert_guidance_workspace(empty_workspace)


@pytest.mark.integration
def test_guidance_live_fixture_without_model() -> None:
    """Real stdio MCP and rejection workers, without cached prices or substitutes."""
    with temporary_mcp_workspace() as workspace:

        async def run() -> list[ToolCall]:
            calls: list[ToolCall] = []
            async with workspace.connect() as client:
                for tool in GUIDANCE_TOOLS:
                    arguments = guidance_arguments(tool)
                    result = await client.call(tool, **arguments)
                    calls.append(ToolCall(tool, arguments, result))
            return calls

        calls = asyncio.run(run())
        assert_guidance_workflow(calls, [ModelTurn(1, _report(calls), calls)])
        assert_guidance_workspace(workspace)
    assert not workspace.root.exists()
