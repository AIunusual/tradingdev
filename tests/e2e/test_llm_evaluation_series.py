"""Natural discovery, continuation and failure recovery over saved ML series."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import pytest

from tests.e2e.codex_harness import run_codex
from tests.e2e.evaluation_series_evidence import (
    SERIES_CONTINUATION,
    SERIES_REQUEST,
    assert_overflow_recovery,
    assert_series_delivery,
    prepare_oversized_series,
    prepare_series_case,
)
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.storage.evaluation_series import EvaluationSeriesStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn, UserReply
    from tests.integration.mcp_harness import MCPWorkspace


async def _conversation(
    workspace: MCPWorkspace,
    pytestconfig: pytest.Config,
    request: str,
    continuation: str,
    user_reply: UserReply,
) -> set[str]:
    readonly_tools: set[str] = set()
    observed_calls: list[ToolCall] = []
    options: dict[str, Any] = {
        "timeout_seconds": pytestconfig.getoption("llm_timeout"),
        "model": pytestconfig.getoption("llm_model"),
        "reasoning_effort": pytestconfig.getoption("llm_reasoning_effort"),
        "user_reply": user_reply,
        "max_tool_calls": 24,
        "max_turns": 2,
        "on_tool_call": observed_calls.append,
    }
    try:
        async with workspace.connect() as client:
            product_tools = (await client.session.list_tools()).tools
            names = [tool.name for tool in product_tools]
            readonly_tools.update(
                tool.name
                for tool in product_tools
                if tool.annotations and tool.annotations.read_only_hint is True
            )
            assert all(name not in request + continuation for name in names)
            options["tool_names"] = names
            if pytestconfig.getoption("llm_provider") == "local":
                calls = await run_local_model(
                    client,
                    request,
                    base_url=pytestconfig.getoption("llm_base_url"),
                    temperature=pytestconfig.getoption("llm_temperature"),
                    **options,
                )
            else:
                calls = None
        if calls is None:
            calls = codex_calls(await run_codex(workspace.root, request, **options))
    except Exception as error:
        error.add_note(
            "Evaluation-series calls before model failure:\n"
            + json.dumps([asdict(call) for call in observed_calls], ensure_ascii=False)
        )
        raise
    print(json.dumps([asdict(call) for call in calls], ensure_ascii=False), flush=True)
    return readonly_tools


@pytest.mark.live_llm
def test_natural_request_reads_evaluation_series(pytestconfig: pytest.Config) -> None:
    with temporary_mcp_workspace() as workspace:
        case = asyncio.run(prepare_series_case(workspace))
        turns: list[ModelTurn] = []

        async def reply(turn: ModelTurn) -> str | None:
            turns.append(turn)
            print(turn.text, flush=True)
            return SERIES_CONTINUATION if len(turns) == 1 else None

        readonly_tools = asyncio.run(
            _conversation(
                workspace,
                pytestconfig,
                SERIES_REQUEST,
                SERIES_CONTINUATION,
                reply,
            )
        )
        assert_series_delivery(case, turns, readonly_tools)
    assert not workspace.root.exists()


@pytest.mark.live_llm
def test_natural_request_recovers_evaluation_series(
    pytestconfig: pytest.Config,
) -> None:
    with temporary_mcp_workspace() as workspace:
        case = asyncio.run(prepare_oversized_series(workspace))
        turns: list[ModelTurn] = []

        async def reply(turn: ModelTurn) -> str | None:
            turns.append(turn)
            print(turn.text, flush=True)
            if len(turns) != 1:
                return None
            store = EvaluationSeriesStore(
                WorkspacePaths(workspace.workspace).ml_evaluation_series
            )
            (store.path_for(case.artifact_id) / "series.parquet").unlink()
            return case.recheck_request

        readonly_tools = asyncio.run(
            _conversation(
                workspace,
                pytestconfig,
                case.request,
                case.recheck_request,
                reply,
            )
        )
        assert_overflow_recovery(case, turns, readonly_tools)
    assert not workspace.root.exists()
