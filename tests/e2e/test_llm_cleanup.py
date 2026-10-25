"""Explicit model cleanup of eligible drafts through a real MCP workspace."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from tests.e2e.codex_harness import run_codex
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.e2e.strategy_scenarios import Scenario, market_frame
from tests.e2e.workflow_arguments import effective_workflow_calls
from tests.e2e.workflow_diagnostics import workflow_diagnostics
from tests.integration.execution_fixtures import execution_options
from tests.integration.mcp_harness import temporary_mcp_workspace

if TYPE_CHECKING:
    from tests.integration.mcp_harness import MCPWorkspace

pytestmark = pytest.mark.live_llm

_SCENARIO = Scenario("cleanup", {}, {}, "")
STRATEGY_ID = _SCENARIO.strategy_id
MISSING_STRATEGY_ID = "llm_cleanup_missing"
_FORBIDDEN_MUTATIONS = {
    "save_strategy",
    "validate_strategy",
    "dry_run_strategy",
    "promote_strategy",
    "prepare_backtest",
    "prepare_walk_forward",
    "prepare_optimization",
    "submit_execution_decision",
    "cancel_job",
    "ensure_data",
    "record_feature_request",
}


@dataclass(frozen=True)
class CleanupSeed:
    eligible_id: str
    runnable_id: str
    current_id: str
    run_id: str
    protected_files: dict[Path, bytes]

    @property
    def prompt(self) -> str:
        return f"""請透過 TradingDev MCP 清理策略 {STRATEGY_ID} 的舊草稿。
先呼叫 cleanup_strategy_drafts(strategy_id={MISSING_STRATEGY_ID}) 預覽不存在的策略，
讀取 success=false、code=strategy_not_found 的錯誤後，再處理下面的既有策略。
不要建立這個不存在的策略，也不要對它 apply；不存在不代表成功清理。
我已明確授權刪除 cleanup_strategy_drafts 預覽中 outcome=eligible 的舊草稿。
必須先呼叫 cleanup_strategy_drafts(strategy_id={STRATEGY_ID}) 預覽全部版本，
檢查每個版本的 outcome/reasons，再以預覽回傳的明確 revision_ids 和 apply=true 執行。
current revision、已 runnable 的版本及歷史回測都必須保留。
此次只有一個可清理的舊草稿；不要建立、修改、驗證、promote 策略或重新回測。
刪除成功後必須呼叫 get_strategy(strategy_id={STRATEGY_ID})，確認目前版本仍是
{self.current_id} 且狀態為 draft。確認後結束，只回答完成與已刪除的 revision_id。
僅使用 MCP，不使用 shell、不直接讀寫檔案，不要再詢問已授權的清理。
"""


async def seed_cleanup_workspace(workspace: MCPWorkspace) -> CleanupSeed:
    """Create three revisions and a retained real backtest without model calls."""
    workspace.seed_market(market_frame())
    async with workspace.connect() as client:
        contract = (await client.call("get_strategy_contract", section="example"))[
            "content"
        ]
        code = contract["example_strategy_code"]
        config = yaml.safe_load(contract["example_yaml_config"])
        config["strategy"]["parameters"] = {"fast_period": 3, "slow_period": 8}
        config["backtest"].update(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-08",
            init_cash=10000,
            fees=0,
            slippage=0,
            periods_per_year=365,
        )
        config["random_seed"] = 42
        config["data"]["requirements"]["features"] = []
        config_text = yaml.safe_dump(config)

        async def save() -> dict[str, Any]:
            result: dict[str, Any] = await client.call(
                "save_strategy",
                strategy_id=STRATEGY_ID,
                code=code,
                yaml_config=config_text,
                request_summary="cleanup fixture",
            )
            assert result["success"] and result["status"] == "draft", result
            return result

        eligible = await save()
        runnable = await save()
        runnable_id = runnable["revision_id"]
        for tool in ("validate_strategy", "dry_run_strategy"):
            checked = await client.call(
                tool,
                strategy_id=STRATEGY_ID,
                revision_id=runnable_id,
            )
            assert checked["success"], checked
        prepared = await client.call(
            "prepare_backtest",
            strategy_id=STRATEGY_ID,
            revision_id=runnable_id,
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-08",
            **execution_options(config["strategy"]["parameters"]),
        )
        assert prepared["success"], prepared
        started = await client.call(
            "submit_execution_decision", plan_id=prepared["plan_id"], decision="approve"
        )
        assert started["job_id"], started
        done = await client.wait_for_job(started["job_id"])
        assert done["status"] == "done", done
        current = await save()
        preview = await client.call("cleanup_strategy_drafts", strategy_id=STRATEGY_ID)
        assert preview["success"] and preview["applied"] is False, preview
        outcomes = {
            item["revision_id"]: item["outcome"] for item in preview["revisions"]
        }
        assert outcomes == {
            eligible["revision_id"]: "eligible",
            runnable_id: "protected",
            current["revision_id"]: "protected",
        }, preview
        root = workspace.workspace / "generated_strategies" / STRATEGY_ID
        keep = [root / "current.json"]
        for saved in (runnable, current):
            keep.extend(Path(saved["py_path"]).parent.rglob("*"))
        keep.extend((workspace.workspace / "runs" / done["run_id"]).rglob("*"))
        return CleanupSeed(
            eligible["revision_id"],
            runnable_id,
            current["revision_id"],
            done["run_id"],
            {path: path.read_bytes() for path in keep if path.is_file()},
        )


def _cleanup_items(result: Any) -> dict[str, dict[str, Any]]:
    assert isinstance(result, dict) and result.get("success") is True, (
        "Cleanup tool must return success=True"
    )
    assert result.get("strategy_id") == STRATEGY_ID, "Cleanup result target mismatch"
    items = result.get("revisions")
    assert isinstance(items, list) and all(isinstance(item, dict) for item in items), (
        "Cleanup result must identify every selected revision"
    )
    revisions = {item.get("revision_id"): item for item in items}
    assert all(isinstance(key, str) and key for key in revisions), (
        "Cleanup result contains an invalid revision ID"
    )
    assert len(revisions) == len(items), (
        "Cleanup result contains duplicate revision IDs"
    )
    return revisions


def assert_cleanup_workflow(calls: list[ToolCall], seed: CleanupSeed) -> None:
    """Require ordered successful evidence; reject any unauthorized mutation."""
    calls = effective_workflow_calls(calls)
    assert not any(call.name in _FORBIDDEN_MUTATIONS for call in calls), (
        "Cleanup workflow must not create or modify strategies, jobs, or data"
    )
    missing_index: int | None = None
    preview_index: int | None = None
    apply_index: int | None = None
    for index, call in enumerate(calls):
        if call.name != "cleanup_strategy_drafts":
            continue
        if call.arguments.get("strategy_id") == MISSING_STRATEGY_ID:
            assert preview_index is None and apply_index is None, (
                "Missing-strategy preview must precede existing-strategy cleanup"
            )
            assert call.arguments.get("apply", False) is False, (
                "Missing-strategy check must be a preview without mutation"
            )
            assert call.arguments.get("revision_ids") is None, (
                "Missing-strategy preview must not select revisions"
            )
            assert isinstance(call.result, dict), "Missing-strategy error is absent"
            assert call.result.get("success") is False, (
                "Missing strategy must not be reported as successful cleanup"
            )
            assert call.result.get("code") == "strategy_not_found", (
                "Missing-strategy preview must return strategy_not_found"
            )
            missing_index = index
            continue
        assert call.arguments.get("strategy_id") == STRATEGY_ID, (
            "Cleanup request target mismatch"
        )
        assert missing_index is not None, (
            "Existing cleanup requires a prior missing-strategy error"
        )
        applied = call.arguments.get("apply", False)
        assert type(applied) is bool, "Cleanup apply must be a boolean"
        items = _cleanup_items(call.result)
        assert call.result.get("applied") is applied, "Cleanup apply result mismatch"
        if not applied:
            if apply_index is not None:
                continue
            assert call.arguments.get("revision_ids") is None, (
                "Preview must inspect all revisions before deletion"
            )
            assert set(items) == {
                seed.eligible_id,
                seed.runnable_id,
                seed.current_id,
            }, "Preview must identify eligible, runnable, and current revisions"
            assert items[seed.eligible_id]["outcome"] == "eligible", (
                "Expected old draft to be previewed as eligible"
            )
            assert items[seed.runnable_id]["outcome"] == "protected", (
                "Preview must protect the runnable revision"
            )
            assert "status:runnable" in items[seed.runnable_id]["reasons"], (
                "Preview must explain the runnable revision's protection"
            )
            assert items[seed.current_id]["outcome"] == "protected", (
                "Preview must protect the current revision"
            )
            assert "current_revision" in items[seed.current_id]["reasons"], (
                "Preview must explain the current revision's protection"
            )
            preview_index = index
        else:
            assert preview_index is not None, (
                "Apply requires a prior successful preview"
            )
            assert call.arguments.get("revision_ids") == [seed.eligible_id], (
                "Apply requires explicit revision_ids with only the eligible draft"
            )
            assert set(items) == {seed.eligible_id}, (
                "Apply must return only the explicitly selected draft"
            )
            assert items[seed.eligible_id]["outcome"] == "deleted", (
                "Apply must confirm actual deletion, not eligible/missing/protected"
            )
            assert apply_index is None, "Cleanup must not repeat apply after deletion"
            apply_index = index
    assert missing_index is not None, "Missing strategy_not_found preview"
    assert preview_index is not None, "Missing successful cleanup preview"
    assert apply_index is not None, "Missing successful cleanup apply"
    assert any(
        index > apply_index
        and call.name == "get_strategy"
        and call.arguments.get("strategy_id") == STRATEGY_ID
        and call.arguments.get("revision_id") is None
        and isinstance(call.result, dict)
        and call.result.get("success") is True
        and call.result.get("strategy_id") == STRATEGY_ID
        and call.result.get("revision_id") == seed.current_id
        and isinstance(call.result.get("metadata"), dict)
        and call.result["metadata"].get("status") == "draft"
        for index, call in enumerate(calls)
    ), "After deletion, get_strategy must verify the unchanged current draft"


def assert_cleanup_files(workspace: MCPWorkspace, seed: CleanupSeed) -> None:
    """Check actual deletion and byte-for-byte retention independently of replies."""
    generated = workspace.workspace / "generated_strategies"
    assert not (generated / MISSING_STRATEGY_ID).exists()
    assert not (generated / ".locks" / f"{MISSING_STRATEGY_ID}.lock").exists()
    revisions = generated / STRATEGY_ID / "revisions"
    assert {path.name for path in revisions.iterdir()} == {
        seed.runnable_id,
        seed.current_id,
    }
    assert {
        path: path.read_bytes() for path in seed.protected_files
    } == seed.protected_files


async def verify_cleanup_history(workspace: MCPWorkspace, seed: CleanupSeed) -> None:
    """Retained execution identity and current strategy must remain queryable."""
    async with workspace.connect() as client:
        current = await client.call("get_strategy", strategy_id=STRATEGY_ID)
        assert (
            current["revision_id"] == seed.current_id
            and current["metadata"]["status"] == "draft"
        )
        historical = await client.call("get_run", run_id=seed.run_id)
        assert historical["success"], historical
        assert historical["run"]["revision_id"] == seed.runnable_id
        jobs = await client.read_all_pages("list_jobs")
        assert len(jobs) == 1 and jobs[0]["job_id"] == seed.run_id


def test_llm_previews_then_cleans_only_authorized_drafts(
    pytestconfig: pytest.Config,
) -> None:
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}, "Use scripts/check-llm.sh codex|local"
    with (
        temporary_mcp_workspace() as workspace,
        workflow_diagnostics(workspace, _SCENARIO),
    ):
        seed = asyncio.run(seed_cleanup_workspace(workspace))

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "model": pytestconfig.getoption("llm_model"),
            }
            if provider == "codex":
                return codex_calls(
                    await run_codex(workspace.root, seed.prompt, **options)
                )
            async with workspace.connect() as client:
                return await run_local_model(
                    client,
                    seed.prompt,
                    base_url=pytestconfig.getoption("llm_base_url"),
                    temperature=pytestconfig.getoption("llm_temperature"),
                    reasoning_effort=pytestconfig.getoption("llm_reasoning_effort"),
                    **options,
                )

        calls = asyncio.run(run())
        try:
            assert_cleanup_workflow(calls, seed)
            assert_cleanup_files(workspace, seed)
            asyncio.run(verify_cleanup_history(workspace, seed))
        except AssertionError as error:
            pytest.fail(f"Incomplete {provider}/cleanup workflow: {error}\n{calls!r}")
