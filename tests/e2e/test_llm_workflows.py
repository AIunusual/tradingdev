"""Explicit model tests: author/repair, run a real backtest, and read its result."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from pydantic import ValidationError

from tests.e2e.codex_harness import run_codex, verify_generated_strategy
from tests.e2e.conversation_user import ObservedPlanUser
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.e2e.report_delivery import verify_report_delivery
from tests.e2e.strategy_scenarios import SCENARIOS, Scenario, market_frame
from tests.e2e.workflow_arguments import effective_workflow_calls
from tests.e2e.workflow_diagnostics import workflow_diagnostics
from tests.e2e.workflow_research import assert_research_reused
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.domain.preflight import PreflightReceipt
from tradingdev.domain.presentation.reports import REPORT_RECIPES

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from tests.integration.mcp_harness import MCPWorkspace

pytestmark = pytest.mark.live_llm

DETAIL_METRIC_IDS = {"daily_pnl_mean", "total_volume", "n_days"}


async def seed_broken_draft(workspace: MCPWorkspace, scenario: Scenario) -> None:
    async with workspace.connect() as client:
        contract = (await client.call("get_strategy_contract", section="example"))[
            "content"
        ]
        if scenario.runtime_exit:
            code = contract["example_strategy_code"].replace(
                "        result = df.copy()",
                "        if len(df):\n            raise SystemExit(7)\n"
                "        result = df.copy()",
            )
            assert "raise SystemExit(7)" in code
        else:
            code = contract["example_strategy_code"].replace(
                "        return result",
                '        result["signal"] = np.nan\n        return result',
            )
            assert 'result["signal"] = np.nan' in code
        config = yaml.safe_load(contract["example_yaml_config"])
        config["strategy"]["parameters"] = scenario.parameters
        config["backtest"].update(
            start_date="2024-01-01",
            end_date="2024-01-08",
            init_cash=10000,
            fees=0,
            slippage=0,
            periods_per_year=365,
        )
        config["random_seed"] = 42
        saved = await client.call(
            "save_strategy",
            strategy_id=scenario.strategy_id,
            code=code,
            yaml_config=yaml.safe_dump(config),
            research=scenario.research,
        )
        assert saved["success"] and saved["status"] == "draft", saved


async def seed_legacy_strategy(
    workspace: MCPWorkspace, scenario: Scenario
) -> dict[Path, bytes]:
    """Create untouched pre-revision files solely inside the test workspace."""
    async with workspace.connect() as client:
        contract = (await client.call("get_strategy_contract", section="example"))[
            "content"
        ]
    generated = workspace.workspace / "generated_strategies"
    configs = workspace.workspace / "configs"
    generated.mkdir(parents=True, exist_ok=True)
    configs.mkdir(parents=True, exist_ok=True)
    source = generated / f"{scenario.legacy_strategy_id}.py"
    config_path = configs / f"{scenario.legacy_strategy_id}.yaml"
    config = yaml.safe_load(contract["example_yaml_config"])
    config["strategy"].update(
        id=scenario.legacy_strategy_id,
        parameters=scenario.parameters,
        source_path=str(source),
    )
    metadata = {
        "strategy_id": scenario.legacy_strategy_id,
        "class_name": config["strategy"]["class_name"],
        "status": "runnable",
        "source_path": str(source),
        "config_path": str(config_path),
        "validation": {"success": True},
        "dry_run": {"success": True},
    }
    originals = {
        source: contract["example_strategy_code"].encode("utf-8"),
        config_path: yaml.safe_dump(config).encode("utf-8"),
        generated / f"{scenario.legacy_strategy_id}.json": json.dumps(metadata).encode(
            "utf-8"
        ),
    }
    for path, content in originals.items():
        path.write_bytes(content)
    return originals


def assert_workflow(calls: list[ToolCall], scenario: Scenario) -> None:
    """Check successful evidence and necessary revision/execution ordering."""
    calls = effective_workflow_calls(calls)
    target = f"strategy_id={scenario.strategy_id!r}"

    def require_index(description: str, indices: Iterator[int]) -> int:
        index = next(indices, None)
        if index is None:
            raise AssertionError(f"Missing required MCP step: {description}")
        return index

    contract_index = require_index(
        f"get_strategy_contract before saving {target}",
        (
            index
            for index, call in enumerate(calls)
            if call.name == "get_strategy_contract"
        ),
    )
    save_index = require_index(
        f"save_strategy({target}) returning success=True",
        (
            index
            for index, call in enumerate(calls)
            if call.name == "save_strategy"
            and call.arguments.get("strategy_id") == scenario.strategy_id
            and call.result
            and call.result.get("success")
        ),
    )
    assert contract_index < save_index, (
        f"Invalid MCP step order for {target}: get_strategy_contract must precede "
        f"the first successful save_strategy (events {contract_index}, {save_index})"
    )
    if scenario.legacy:
        legacy_identity = {
            "strategy_id": scenario.legacy_strategy_id,
            "revision_id": None,
            "kind": "legacy",
            "status": "revision_required",
            "code": "strategy_revision_required",
        }
        list_index = require_index(
            f"list_strategies identifying {target} as legacy/revision_required",
            (
                index
                for index, call in enumerate(calls)
                if call.name == "list_strategies"
                and isinstance(call.result, dict)
                and any(
                    all(
                        item.get(key) == value for key, value in legacy_identity.items()
                    )
                    for item in call.result.get("items", [])
                )
            ),
        )
        read_index = require_index(
            f"get_strategy({target}) returning legacy source/config before resaving",
            (
                index
                for index, call in enumerate(calls)
                if call.name == "get_strategy"
                and call.arguments.get("strategy_id") == scenario.legacy_strategy_id
                and call.result
                and call.result.get("success") is True
                and all(
                    call.result.get(key) == value
                    for key, value in legacy_identity.items()
                )
                and call.result.get("source_code")
                and call.result.get("yaml_config")
            ),
        )
        assert list_index < read_index < save_index, (
            f"Invalid MCP legacy recovery order for {target}: "
            "discovery and source/config "
            "retrieval must precede successful save_strategy "
            f"(events {list_index}, {read_index}, {save_index})"
        )
        assert calls[save_index].result.get("status") == "draft", (
            "Resaving a legacy strategy must create a draft without inherited evidence"
        )
    previous = save_index
    for name, status in (
        ("validate_strategy", "validated"),
        ("dry_run_strategy", "runnable"),
    ):
        previous = require_index(
            f"{name}({target}) returning success=True, status={status!r} "
            f"after {calls[previous].name} (event {previous})",
            (
                index
                for index, call in enumerate(calls)
                if index > previous
                and call.name == name
                and call.result
                and call.result.get("success")
                and call.result.get("status") == status
                and call.arguments.get("strategy_id") == scenario.strategy_id
            ),
        )
    revision_id = calls[previous].result.get("revision_id")
    assert isinstance(revision_id, str) and revision_id, (
        "Dry-run must identify revision_id"
    )
    require_index(
        f"save_strategy returning revision_id={revision_id!r} before dry-run",
        (
            index
            for index, call in enumerate(calls)
            if index < previous
            and call.name == "save_strategy"
            and call.result
            and call.arguments.get("strategy_id") == scenario.strategy_id
            and call.result.get("success")
            and call.result.get("revision_id") == revision_id
        ),
    )
    checked: dict[str, ToolCall] = {}
    for name, status in (
        ("validate_strategy", "validated"),
        ("dry_run_strategy", "runnable"),
    ):
        checked_index = require_index(
            f"{name} explicitly checking revision_id={revision_id!r}",
            (
                index
                for index, call in enumerate(calls)
                if save_index < index <= previous
                and call.name == name
                and call.result
                and call.arguments.get("strategy_id") == scenario.strategy_id
                and call.arguments.get("revision_id") == revision_id
                and call.result.get("success")
                and call.result.get("status") == status
                and call.result.get("revision_id") == revision_id
            ),
        )
        checked[name] = calls[checked_index]
    prepare_index = require_index(
        f"prepare_backtest for {target} returning a ready plan_id "
        "after successful dry_run_strategy",
        (
            index
            for index, call in enumerate(calls)
            if index > previous
            and call.name == "prepare_backtest"
            and call.result
            and call.result.get("success") is True
            and call.result.get("status") == "ready"
            and call.result.get("plan_id")
        ),
    )
    prepared = calls[prepare_index]
    expected_arguments: dict[str, Any] = {
        "strategy_id": scenario.strategy_id,
        "revision_id": revision_id,
    }
    if scenario.experiment:
        expected_arguments["parameters"] = scenario.overrides
        assert not any(
            call.name == "save_strategy" for call in calls[previous + 1 :]
        ), "Parameter experiments must reuse the runnable revision"
    actual_arguments = dict(prepared.arguments)
    assert_research_reused(
        actual_arguments.pop("research", None),
        validated=checked["validate_strategy"],
        dry_run=checked["dry_run_strategy"],
    )
    from tests.e2e.research_evidence import assert_saved_research

    draft = assert_saved_research(calls, prepare_index)
    actual_arguments.pop("evidence_id")
    presentation = draft.presentation.model_dump(mode="json")
    assert presentation.get("title") and presentation.get("summary")
    assert set(presentation.get("parameter_descriptions", {})) == {
        f"/{name}" for name in scenario.parameters
    }, "Presentation must explain every captured parameter"
    for optional in ("parameters", "backtest_overrides"):
        if actual_arguments.get(optional) is None:
            actual_arguments.pop(optional, None)
    assert actual_arguments == expected_arguments, (
        f"prepare_backtest arguments differ for {target}: "
        f"expected {expected_arguments!r}; got {prepared.arguments!r}"
    )
    manifest_hash = prepared.result.get("manifest_hash")
    assert isinstance(manifest_hash, str) and len(manifest_hash) == 64
    plan_id = prepared.result["plan_id"]
    assert prepared.result.get("confirmation_summary")
    assert prepared.result.get("html_path") and prepared.result.get("artifact_id")
    assert prepared.result.get("preflight"), (
        "Preparation must retain preflight evidence"
    )
    try:
        receipt = PreflightReceipt.model_validate(prepared.result["preflight"])
    except ValidationError as exc:
        raise AssertionError("Preparation must retain valid offline evidence") from exc
    assert receipt.manifest_hash == manifest_hash
    assert receipt.signal_check == "synthetic_contract_passed"
    assert receipt.synthetic_fixture_rows == [80, 240]
    assert set(receipt.not_checked) == {
        "historical_data",
        "backtest_engine",
        "training",
        "performance",
    }
    assert not prepared.result.get("job_id"), "Preparing must not launch a formal job"
    start_index = require_index(
        f"submit_execution_decision(plan_id={plan_id!r}) returning a nonempty "
        "job_id after prepare_backtest",
        (
            index
            for index, call in enumerate(calls)
            if index > prepare_index
            and call.name == "submit_execution_decision"
            and call.arguments == {"plan_id": plan_id, "decision": "approve"}
            and call.result
            and call.result.get("success") is True
            and call.result.get("plan_id") == plan_id
            and call.result.get("job_id")
        ),
    )
    started = calls[start_index]
    assert started.result["manifest_hash"] == manifest_hash
    job_id = started.result["job_id"]
    done_index = require_index(
        f"get_job_status(job_id={job_id!r}) returning status='done' "
        f"after submit_execution_decision for {target}",
        (
            index
            for index, call in enumerate(calls)
            if index > start_index
            and call.name == "get_job_status"
            and call.result
            and call.arguments.get("job_id") == job_id
            and call.result.get("status") == "done"
        ),
    )
    done = calls[done_index]
    assert done.result["revision_id"] == revision_id
    assert done.result["manifest_hash"] == manifest_hash
    run_id = done.result["run_id"]
    queried_index = require_index(
        f"get_run(run_id={run_id!r}) returning a result after "
        f"get_job_status reported done for {target}",
        (
            index
            for index, call in enumerate(calls)
            if index > done_index
            and call.name == "get_run"
            and call.result
            and call.arguments.get("run_id") == run_id
        ),
    )
    queried = calls[queried_index]
    assert queried.result["run"]["revision_id"] == revision_id
    assert queried.result["run"]["manifest_hash"] == manifest_hash
    assert queried.result["success"], (
        f"get_run(run_id={run_id!r}) did not return success=True: {queried.result!r}"
    )
    assert queried.result["run"]["metrics"] == done.result["metrics"], (
        f"get_run(run_id={run_id!r}) metrics differ from "
        f"get_job_status(job_id={job_id!r}) metrics"
    )
    assert queried.result["run"]["strategy_id"] == scenario.strategy_id, (
        f"get_run(run_id={run_id!r}) belongs to "
        f"strategy_id={queried.result['run']['strategy_id']!r}, expected {target}"
    )
    if scenario.runtime_exit:
        # A natural request may obtain artifact IDs directly from the report
        # result. The outer verifier checks the delivered file, hash, sections
        # and saved analysis before the temporary workspace is removed.
        require_index(
            f"generate_report returning a report for run_id={run_id!r}",
            (
                index
                for index, call in enumerate(calls)
                if index > done_index
                and call.name == "generate_report"
                and call.result
                and call.result.get("success") is True
                and call.result.get("run_ids") == [run_id]
                and call.result.get("artifact_id")
            ),
        )
    else:
        require_index(
            f"list_artifacts(run_id={run_id!r}) returning a nonempty result "
            f"after get_job_status reported done for {target}",
            (
                index
                for index, call in enumerate(calls)
                if index > done_index
                and call.name == "list_artifacts"
                and isinstance(call.result, dict)
                and call.result.get("items")
                and call.arguments.get("run_id") == run_id
            ),
        )
    catalog_index = require_index(
        "get_metric_catalog returning the additional metrics",
        (
            index
            for index, call in enumerate(calls)
            if call.name == "get_metric_catalog"
            and call.result
            and call.result.get("success") is True
            and {definition["id"] for definition in call.result["definitions"]}
            >= DETAIL_METRIC_IDS
        ),
    )
    detail_index = require_index(
        f"get_run_metrics(run_id={run_id!r}) returning the requested extra metrics",
        (
            index
            for index, call in enumerate(calls)
            if index
            > (
                done_index
                if scenario.runtime_exit
                else max(queried_index, catalog_index)
            )
            and call.name == "get_run_metrics"
            and call.arguments.get("run_id") == run_id
            and call.result
            and call.result.get("success") is True
            and set(call.result.get("metrics", {})) >= DETAIL_METRIC_IDS
        ),
    )
    assert calls[detail_index].result["scope"] == "full"
    summary = queried.result["run"]
    assert set(summary["available_metric_ids"]) >= DETAIL_METRIC_IDS
    assert DETAIL_METRIC_IDS.isdisjoint(summary["metrics"])
    if scenario.history:
        for tool in (
            "find_runs",
            "get_run_trades",
            "get_run_equity",
            "get_run_executions",
            "get_run_account_history",
            "get_report_sections",
            "generate_report",
        ):
            require_index(
                f"{tool} returning saved history/report after completion",
                (
                    index
                    for index, call in enumerate(calls)
                    if index > done_index
                    and call.name == tool
                    and call.result
                    and call.result.get("success") is True
                    and (
                        tool in {"find_runs", "get_report_sections"}
                        or call.arguments.get("run_id") == run_id
                        or call.arguments.get("run_ids") == [run_id]
                    )
                ),
            )
        trade_calls = [
            call
            for call in calls
            if call.name == "get_run_trades" and call.result.get("success")
        ]
        assert trade_calls[0].arguments.get("limit") == 1
        assert trade_calls[0].result["next_offset"] == 1
        assert any(call.arguments.get("offset") == 1 for call in trade_calls[1:])
        found = next(call for call in calls if call.name == "find_runs")
        assert found.arguments.get("strategy_id") == scenario.strategy_id
        assert found.arguments.get("parameters") == scenario.parameters
        assert any(
            row["run_id"] == run_id and row["scope"] == "full"
            for row in found.result["runs"]
        )
        for name, count in (("get_run_executions", 1), ("get_run_account_history", 2)):
            queried_history = next(call for call in calls if call.name == name)
            assert queried_history.arguments.get("limit") == count
            assert queried_history.result["availability"] == "available"
        assert sum(call.name == "prepare_backtest" for call in calls) == 1
        assert sum(call.name == "submit_execution_decision" for call in calls) == 1
    if scenario.repair:
        failure_code = (
            "strategy_check_failed"
            if scenario.runtime_exit
            else "invalid_signal_values"
        )
        read_index = require_index(
            f"get_strategy({target}) returning success=True before repairing the draft",
            (
                index
                for index, call in enumerate(calls)
                if call.name == "get_strategy"
                and call.arguments.get("strategy_id") == scenario.strategy_id
                and call.result
                and call.result.get("success")
            ),
        )
        failure_index = require_index(
            f"validate_strategy({target}) returning success=False with "
            f"diagnostic code='{failure_code}' before repairing the draft",
            (
                index
                for index, call in enumerate(calls)
                if call.name == "validate_strategy"
                and call.arguments.get("strategy_id") == scenario.strategy_id
                and call.result
                and call.result.get("success") is False
                and any(
                    diagnostic.get("code") == failure_code
                    for diagnostic in call.result.get("diagnostics", [])
                )
            ),
        )
        assert read_index < failure_index < save_index, (
            f"Invalid MCP repair order for {target}: successful get_strategy must "
            f"precede validate_strategy with {failure_code}, which must "
            f"precede successful save_strategy (events {read_index}, "
            f"{failure_index}, {save_index})"
        )
        selected_revision = calls[read_index].result.get("revision_id")
        assert isinstance(selected_revision, str) and selected_revision, (
            "Repair retrieval must identify an immutable revision"
        )
        assert (
            calls[save_index].arguments.get("expected_revision_id") == selected_revision
        ), "Repair save_strategy must bind expected_revision_id to the retrieved draft"


@pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=SCENARIOS.keys())
def test_llm_authors_backtests_and_queries_results(
    pytestconfig: pytest.Config, scenario: Scenario
) -> None:
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}, "Use scripts/check-llm.sh codex|local"
    with (
        temporary_mcp_workspace() as workspace,
        workflow_diagnostics(workspace, scenario),
    ):
        workspace.seed_market(market_frame())
        legacy_files: dict[Path, bytes] = {}
        user = ObservedPlanUser(workspace)

        async def run() -> list[ToolCall]:
            if scenario.repair:
                await seed_broken_draft(workspace, scenario)
            if scenario.legacy:
                legacy_files.update(await seed_legacy_strategy(workspace, scenario))
            options: dict[str, Any] = {
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "model": pytestconfig.getoption("llm_model"),
            }
            if scenario.runtime_exit:
                async with workspace.connect() as client:
                    names = [
                        tool.name for tool in (await client.session.list_tools()).tools
                    ]
                    assert all(name not in scenario.prompt for name in names)
                    options["tool_names"] = names
            if provider == "codex":
                calls = codex_calls(
                    await run_codex(
                        workspace.root,
                        scenario.prompt,
                        user_reply=user,
                        **options,
                    )
                )
            else:
                async with workspace.connect() as client:
                    calls = await run_local_model(
                        client,
                        scenario.prompt,
                        base_url=pytestconfig.getoption("llm_base_url"),
                        temperature=pytestconfig.getoption("llm_temperature"),
                        reasoning_effort=pytestconfig.getoption("llm_reasoning_effort"),
                        user_reply=user,
                        **options,
                    )
            user.assert_complete()
            return calls

        calls = asyncio.run(run())
        try:
            assert_workflow(calls, scenario)
            detail = next(
                call for call in reversed(calls) if call.name == "get_run_metrics"
            )
            saved = json.loads(
                (
                    workspace.workspace
                    / "runs"
                    / detail.arguments["run_id"]
                    / "performance.json"
                ).read_text(encoding="utf-8")
            )
            persisted = saved["scopes"][detail.result["scope"]]["values"]
            for metric_id in DETAIL_METRIC_IDS:
                assert detail.result["metrics"][metric_id] == persisted[metric_id]
            if scenario.runtime_exit:
                verify_report_delivery(
                    WorkspacePaths(workspace.workspace),
                    user.turns[-1],
                    detail.arguments["run_id"],
                    expected_sections=list(REPORT_RECIPES["standard"]),
                    expected_commentary=None,
                    required_analysis_refs={(detail.arguments["run_id"], "full")},
                )
                print(
                    "[runtime-repair] Final model reply:\n" + user.turns[-1].text,
                    flush=True,
                )
            if scenario.history:
                print(
                    "[history] Final model reply (test evidence; temporary links "
                    "expire at cleanup):\n" + user.turns[-1].text,
                    flush=True,
                )
                report_html = verify_report_delivery(
                    WorkspacePaths(workspace.workspace),
                    user.turns[-1],
                    detail.arguments["run_id"],
                    expected_sections=["overview", "metrics", "ledger", "trades"],
                    expected_commentary=[
                        {"title": "研究評語", "text": "本次結果僅為歷史模擬。"}
                    ],
                )
                assert detail.arguments["run_id"] in report_html
                assert "<table" in report_html
                assert "本次結果僅為歷史模擬。" in report_html
                observations = json.loads(
                    (
                        workspace.workspace
                        / "runs"
                        / detail.arguments["run_id"]
                        / "observations.json"
                    ).read_text(encoding="utf-8")
                )["scopes"]["full"]
                for call in calls:
                    if call.name == "get_run_executions" and call.result.get("success"):
                        assert (
                            call.result["records"]
                            == observations["execution_records"][:1]
                        )
                    if call.name == "get_run_account_history" and call.result.get(
                        "success"
                    ):
                        assert (
                            call.result["states"] == observations["account_history"][:2]
                        )
                    if call.name == "get_run_trades" and call.result.get("success"):
                        for trade in call.result["trades"]:
                            assert (
                                trade["record"]
                                == observations["trades"][trade["trade_id"]]
                            )
            for path, original in legacy_files.items():
                assert path.read_bytes() == original, (
                    f"Legacy recovery modified the original file: {path.name}"
                )
        except (AssertionError, StopIteration, ValueError) as exc:
            pytest.fail(
                f"Incomplete {provider}/{scenario.name} workflow: {exc}\n{calls!r}"
            )
        verify_generated_strategy(workspace.root, scenario=scenario.name)
