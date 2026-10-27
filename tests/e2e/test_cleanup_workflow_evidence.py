"""Offline cleanup evidence checks, plus a real MCP seed without model calls."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import pytest
import yaml

from tests.e2e.llm_client import ToolCall
from tests.e2e.test_llm_cleanup import (
    MISSING_STRATEGY_ID,
    STRATEGY_ID,
    CleanupSeed,
    assert_cleanup_files,
    assert_cleanup_workflow,
    seed_cleanup_workspace,
    verify_cleanup_history,
)
from tests.integration.mcp_harness import temporary_mcp_workspace


def test_successful_cleanup_accepts_sdk_encoded_selector_and_boolean(
    calls: list[ToolCall], seed: CleanupSeed
) -> None:
    calls[2].arguments["revision_ids"] = json.dumps([seed.eligible_id])
    calls[2].arguments["apply"] = "true"
    original = dict(calls[2].arguments)
    assert_cleanup_workflow(calls, seed)
    assert calls[2].arguments == original


@pytest.fixture
def seed() -> CleanupSeed:
    return CleanupSeed("old", "runnable", "current", "run", {})


@pytest.fixture
def calls(seed: CleanupSeed) -> list[ToolCall]:
    target = {"strategy_id": STRATEGY_ID}
    return [
        ToolCall(
            "cleanup_strategy_drafts",
            {"strategy_id": MISSING_STRATEGY_ID},
            {"success": False, "code": "strategy_not_found", "error": "not found"},
        ),
        ToolCall(
            "cleanup_strategy_drafts",
            target,
            {
                "success": True,
                **target,
                "applied": False,
                "revisions": [
                    {
                        "revision_id": seed.eligible_id,
                        "outcome": "eligible",
                        "reasons": [],
                    },
                    {
                        "revision_id": seed.runnable_id,
                        "outcome": "protected",
                        "reasons": ["status:runnable", "job:run"],
                    },
                    {
                        "revision_id": seed.current_id,
                        "outcome": "protected",
                        "reasons": ["current_revision"],
                    },
                ],
            },
        ),
        ToolCall(
            "cleanup_strategy_drafts",
            {
                **target,
                "revision_ids": [seed.eligible_id],
                "apply": True,
            },
            {
                "success": True,
                **target,
                "applied": True,
                "revisions": [
                    {
                        "revision_id": seed.eligible_id,
                        "outcome": "deleted",
                        "reasons": [],
                    }
                ],
            },
        ),
        ToolCall(
            "get_strategy",
            target,
            {
                "success": True,
                **target,
                "revision_id": seed.current_id,
                "metadata": {"status": "draft"},
            },
        ),
    ]


def test_complete_cleanup_evidence_is_accepted(
    calls: list[ToolCall], seed: CleanupSeed
) -> None:
    assert_cleanup_workflow(calls, seed)


@pytest.mark.parametrize(
    "event", [0, 1, 2, 3], ids=["missing", "preview", "apply", "current_read"]
)
def test_cleanup_requires_every_successful_step(
    calls: list[ToolCall], seed: CleanupSeed, event: int
) -> None:
    calls.pop(event)
    with pytest.raises(AssertionError):
        assert_cleanup_workflow(calls, seed)


@pytest.mark.parametrize("event", [1, 2, 3])
def test_cleanup_rejects_failed_tool_results(
    calls: list[ToolCall], seed: CleanupSeed, event: int
) -> None:
    calls[event].result["success"] = False
    with pytest.raises(AssertionError):
        assert_cleanup_workflow(calls, seed)


@pytest.mark.parametrize(
    "revision_ids",
    [None, [], ["current"], ["runnable"], ["old", "current"], ["old", "old"]],
)
def test_apply_requires_only_explicit_eligible_ids(
    calls: list[ToolCall], seed: CleanupSeed, revision_ids: Any
) -> None:
    calls[2].arguments["revision_ids"] = revision_ids
    with pytest.raises(AssertionError, match="explicit revision_ids"):
        assert_cleanup_workflow(calls, seed)


def test_apply_cannot_omit_revision_selector(
    calls: list[ToolCall], seed: CleanupSeed
) -> None:
    calls[2].arguments.pop("revision_ids")
    with pytest.raises(AssertionError, match="explicit revision_ids"):
        assert_cleanup_workflow(calls, seed)


@pytest.mark.parametrize("outcome", ["eligible", "protected", "missing", "failed"])
def test_apply_requires_confirmed_deletion(
    calls: list[ToolCall], seed: CleanupSeed, outcome: str
) -> None:
    calls[2].result["revisions"][0]["outcome"] = outcome
    with pytest.raises(AssertionError, match="confirm actual deletion"):
        assert_cleanup_workflow(calls, seed)


@pytest.mark.parametrize("event", [1, 2])
@pytest.mark.parametrize("side", ["arguments", "result"])
def test_cleanup_target_must_match_request(
    calls: list[ToolCall], seed: CleanupSeed, event: int, side: str
) -> None:
    value = calls[event].arguments if side == "arguments" else calls[event].result
    value["strategy_id"] = "different_strategy"
    with pytest.raises(AssertionError, match="target mismatch"):
        assert_cleanup_workflow(calls, seed)


@pytest.mark.parametrize("index", [1, 2], ids=["runnable", "current"])
def test_preview_must_protect_seeded_revisions(
    calls: list[ToolCall], seed: CleanupSeed, index: int
) -> None:
    calls[1].result["revisions"][index]["outcome"] = "eligible"
    with pytest.raises(AssertionError, match="must protect"):
        assert_cleanup_workflow(calls, seed)


@pytest.mark.parametrize(
    "change", ["out_of_order", "wrong_revision", "not_current_lookup", "wrong_status"]
)
def test_read_must_verify_current_after_apply(
    calls: list[ToolCall], seed: CleanupSeed, change: str
) -> None:
    if change == "out_of_order":
        calls[2], calls[3] = calls[3], calls[2]
    elif change == "wrong_revision":
        calls[3].result["revision_id"] = seed.runnable_id
    elif change == "not_current_lookup":
        calls[3].arguments["revision_id"] = seed.current_id
    else:
        calls[3].result["metadata"]["status"] = "runnable"
    with pytest.raises(AssertionError, match="unchanged current draft"):
        assert_cleanup_workflow(calls, seed)


def test_apply_cannot_precede_preview(calls: list[ToolCall], seed: CleanupSeed) -> None:
    calls[1], calls[2] = calls[2], calls[1]
    with pytest.raises(AssertionError, match="prior successful preview"):
        assert_cleanup_workflow(calls, seed)


def test_wrong_applied_flag_is_not_success(
    calls: list[ToolCall], seed: CleanupSeed
) -> None:
    calls[2].result["applied"] = False
    with pytest.raises(AssertionError, match="result mismatch"):
        assert_cleanup_workflow(calls, seed)


@pytest.mark.parametrize(
    "tool", ["save_strategy", "validate_strategy", "prepare_backtest", "ensure_data"]
)
def test_extra_mutations_are_rejected(
    calls: list[ToolCall], seed: CleanupSeed, tool: str
) -> None:
    calls.insert(0, ToolCall(tool, {"strategy_id": STRATEGY_ID}, {"success": True}))
    with pytest.raises(AssertionError, match="must not create or modify"):
        assert_cleanup_workflow(calls, seed)


@pytest.mark.integration
def test_cleanup_seed_and_evidence_use_real_mcp_and_preserve_history() -> None:
    """Exercise setup, deletion, and assertions independently of a language model."""
    with temporary_mcp_workspace() as workspace:
        root = workspace.root

        async def run() -> None:
            seeded = await seed_cleanup_workspace(workspace)
            observed: list[ToolCall] = []
            async with workspace.connect() as client:
                configs = []
                for revision_id in (
                    seeded.eligible_id,
                    seeded.runnable_id,
                    seeded.current_id,
                ):
                    saved = await client.call(
                        "get_strategy",
                        strategy_id=STRATEGY_ID,
                        revision_id=revision_id,
                    )
                    assert saved["success"] and saved["revision_id"] == revision_id
                    configs.append(yaml.safe_load(saved["yaml_config"]))
                manifest = json.loads(
                    (
                        workspace.workspace / "runs" / seeded.run_id / "manifest.json"
                    ).read_text()
                )
                configs.append(manifest["config"])
                for config in configs:
                    assert config["strategy"]["parameters"] == {
                        "fast_period": 3,
                        "slow_period": 8,
                    }
                    assert config["random_seed"] == 42
                    backtest = config["backtest"]
                    assert {
                        name: backtest[name]
                        for name in (
                            "symbol",
                            "timeframe",
                            "init_cash",
                            "fees",
                            "slippage",
                            "periods_per_year",
                        )
                    } == {
                        "symbol": "BTC/USDT",
                        "timeframe": "1h",
                        "init_cash": 10000,
                        "fees": 0,
                        "slippage": 0,
                        "periods_per_year": 365,
                    }
                    assert datetime.fromisoformat(backtest["start_date"]) == datetime(
                        2024, 1, 1, tzinfo=UTC
                    )
                    assert datetime.fromisoformat(backtest["end_date"]) == datetime(
                        2024, 1, 8, tzinfo=UTC
                    )
                assert (
                    manifest["config"]["research"]["selected_revision"]["revision_id"]
                    == seeded.runnable_id
                )
                for name, arguments in (
                    ("cleanup_strategy_drafts", {"strategy_id": MISSING_STRATEGY_ID}),
                    ("cleanup_strategy_drafts", {"strategy_id": STRATEGY_ID}),
                    (
                        "cleanup_strategy_drafts",
                        {
                            "strategy_id": STRATEGY_ID,
                            "revision_ids": [seeded.eligible_id],
                            "apply": True,
                        },
                    ),
                    ("get_strategy", {"strategy_id": STRATEGY_ID}),
                ):
                    result = await client.call(name, **arguments)
                    observed.append(ToolCall(name, arguments, result))
            assert_cleanup_workflow(observed, seeded)
            assert_cleanup_files(workspace, seeded)
            await verify_cleanup_history(workspace, seeded)

        asyncio.run(run())
    assert not root.exists()


@pytest.mark.parametrize(
    "response",
    [
        {"success": True, "code": "strategy_not_found"},
        {"success": False, "code": "strategy_cleanup_blocked"},
        {"success": False},
        None,
    ],
)
def test_missing_strategy_requires_actual_not_found_error(
    calls: list[ToolCall],
    seed: CleanupSeed,
    response: Any,
) -> None:
    calls[0].result = response
    with pytest.raises(AssertionError):
        assert_cleanup_workflow(calls, seed)


def test_missing_strategy_check_must_not_apply(
    calls: list[ToolCall],
    seed: CleanupSeed,
) -> None:
    calls[0].arguments["apply"] = True
    with pytest.raises(AssertionError, match="preview without mutation"):
        assert_cleanup_workflow(calls, seed)


def test_missing_strategy_error_must_precede_existing_cleanup(
    calls: list[ToolCall],
    seed: CleanupSeed,
) -> None:
    calls[0], calls[1] = calls[1], calls[0]
    with pytest.raises(AssertionError, match="prior missing-strategy error"):
        assert_cleanup_workflow(calls, seed)
