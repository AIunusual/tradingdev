"""Opt-in natural intake, using the real tool surface and no private answer hints."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest

from tests.e2e.codex_harness import run_codex, verify_generated_strategy
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.e2e.novice_diagnostics import retain_novice_failure_evidence
from tests.e2e.novice_intake import (
    INTAKE_REQUEST,
    MISSING_REQUESTS,
    MissingInputUser,
    ObservedIntakeUser,
)
from tests.e2e.report_delivery import verify_report_delivery
from tests.e2e.strategy_scenarios import market_frame
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.run_service import RunService
from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.presentation.reports import REPORT_RECIPES

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.integration.mcp_harness import MCPWorkspace

pytestmark = pytest.mark.live_llm


def _run(
    pytestconfig: pytest.Config,
    workspace: MCPWorkspace,
    prompt: str,
    user: MissingInputUser | ObservedIntakeUser,
    *,
    on_tool_call: Callable[[ToolCall], None] | None = None,
) -> list[ToolCall]:
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}

    async def run() -> list[ToolCall]:
        options: dict[str, Any] = {
            "timeout_seconds": pytestconfig.getoption("llm_timeout"),
            "model": pytestconfig.getoption("llm_model"),
            "user_reply": user,
            "on_tool_call": on_tool_call,
        }
        async with workspace.connect() as client:
            tools = (await client.session.list_tools()).tools
            names = [tool.name for tool in tools]
            assert all(name not in prompt for name in names)
            user.read_only_tools = {
                tool.name
                for tool in tools
                if tool.annotations is not None and tool.annotations.read_only_hint
            }
            options["tool_names"] = names
            if provider == "local":
                return await run_local_model(
                    client,
                    prompt,
                    base_url=pytestconfig.getoption("llm_base_url"),
                    temperature=pytestconfig.getoption("llm_temperature"),
                    reasoning_effort=pytestconfig.getoption("llm_reasoning_effort"),
                    **options,
                )
        return codex_calls(await run_codex(workspace.root, prompt, **options))

    return asyncio.run(run())


@pytest.mark.parametrize("missing", MISSING_REQUESTS)
def test_novice_missing_input_asks_before_work(
    pytestconfig: pytest.Config, missing: str
) -> None:
    with temporary_mcp_workspace() as workspace:
        user = MissingInputUser(workspace, set())
        _run(pytestconfig, workspace, MISSING_REQUESTS[missing], user)
        assert len(user.turns) == 1
        assert not any(path.is_file() for path in workspace.data_root.rglob("*"))
        print(
            f"[intake] Missing {missing}; clarification for semantic review:\n"
            + user.turns[0].text,
            flush=True,
        )


def test_novice_clarification_defaults_and_report(pytestconfig: pytest.Config) -> None:
    with temporary_mcp_workspace() as workspace:
        workspace.seed_market(market_frame())
        user = ObservedIntakeUser(workspace, set())
        calls: list[ToolCall] = []
        with retain_novice_failure_evidence(workspace, user, calls):
            _run(
                pytestconfig, workspace, INTAKE_REQUEST, user, on_tool_call=calls.append
            )
            _verify_completed_intake(workspace, user, calls)


def _verify_completed_intake(
    workspace: MCPWorkspace, user: ObservedIntakeUser, calls: list[ToolCall]
) -> None:
    user.assert_complete()
    assert user.plan is not None
    assert isinstance(user.plan.manifest, ExecutionManifest)
    paths = WorkspacePaths(workspace.workspace)
    runs = SQLiteStore(paths).list_runs()
    assert len(runs) == 1, "The accepted single proposal must produce one saved result"
    run = runs[0]
    config = user.plan.manifest.config_copy()
    strategy_id = config["strategy"]["id"]
    assert strategy_id.startswith("strategy_")
    assert run["strategy_id"] == strategy_id
    assert run["revision_id"] == config["strategy"]["revision_id"]
    assert run["manifest_hash"] == user.plan.manifest.manifest_hash
    assert any(
        call.name == "save_strategy"
        and call.arguments.get("strategy_id") is None
        and call.result.get("strategy_id") == strategy_id
        for call in calls
    ), "The unnamed request must use backend naming"
    assert any(
        call.name == "get_job_status"
        and call.result.get("status") == "done"
        and call.result.get("run_id") == run["run_id"]
        for call in calls
    )
    verify_generated_strategy(workspace.root, scenario="novice_intake")
    verify_report_delivery(
        paths,
        user.turns[-1],
        run["run_id"],
        expected_sections=list(REPORT_RECIPES["standard"]),
        expected_commentary=None,
    )
    metrics = RunService(workspace=paths).get_run_metrics(
        run["run_id"],
        metric_ids=["total_return", "annual_return", "max_drawdown", "total_trades"],
    )
    assert metrics["success"] is True
    print(
        "[intake] Saved metric evidence for final-prose semantic review:\n"
        + json.dumps(
            {
                "run_id": run["run_id"],
                "scope": metrics["scope"],
                "metrics": metrics["metrics"],
                "display_metrics": metrics["display_metrics"],
                "units": {
                    key: definition["unit"]
                    for key, definition in metrics["definitions"].items()
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(
        "[intake] Accepted effective configuration for semantic review:\n"
        + str(config),
        flush=True,
    )
