"""Legacy strategies remain discoverable and recoverable over real stdio MCP."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tests.integration.execution_fixtures import execution_options

if TYPE_CHECKING:
    from pathlib import Path

    from tests.integration.mcp_harness import MCPWorkspace

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

_LEGACY_ID = "legacy_strategy"
_LEGACY_SOURCE = (
    "raise RuntimeError('Legacy source must not execute during recovery')\n"
    "class LegacyStrategy: pass\n"
)
_LEGACY_CONFIG = "strategy:\n  class_name: LegacyStrategy\n"


def _write_legacy(
    workspace: MCPWorkspace,
    *,
    strategy_id: str = _LEGACY_ID,
    invalid_metadata: bool = False,
) -> tuple[Path, Path, Path]:
    generated = workspace.workspace / "generated_strategies"
    configs = workspace.workspace / "configs"
    generated.mkdir(parents=True, exist_ok=True)
    configs.mkdir(parents=True, exist_ok=True)
    source = generated / f"{strategy_id}.py"
    config = configs / f"{strategy_id}.yaml"
    metadata = generated / f"{strategy_id}.json"
    source.write_text(_LEGACY_SOURCE, encoding="utf-8")
    config.write_text(_LEGACY_CONFIG, encoding="utf-8")
    # Paths and approval claims in old metadata cannot select different source
    # or grant executable status to a newly saved revision.
    outside_source = workspace.root / "unrelated.py"
    outside_config = workspace.root / "unrelated.yaml"
    outside_source.write_text("UNRELATED SOURCE", encoding="utf-8")
    outside_config.write_text("UNRELATED CONFIG", encoding="utf-8")
    metadata.write_text(
        "{invalid legacy JSON"
        if invalid_metadata
        else json.dumps(
            {
                "strategy_id": "misleading_id",
                "status": "promoted",
                "source_path": str(outside_source),
                "config_path": str(outside_config),
                "validation": {"success": True},
                "dry_run": {"success": True},
            }
        ),
        encoding="utf-8",
    )
    return source, config, metadata


@pytest.mark.parametrize("invalid_metadata", [False, True])
async def test_legacy_strategy_discovery_and_explicit_resave(
    mcp_workspace: MCPWorkspace, invalid_metadata: bool
) -> None:
    legacy_paths = _write_legacy(mcp_workspace, invalid_metadata=invalid_metadata)
    original_bytes = {path: path.read_bytes() for path in legacy_paths}

    async with mcp_workspace.connect() as client:
        current = await client.call(
            "save_strategy",
            strategy_id="current_strategy",
            code="class CurrentStrategy: pass\n",
            yaml_config="strategy:\n  class_name: CurrentStrategy\n",
        )
        assert current["success"]
        # The harness validates every structured response against outputSchema
        # and rejects isError replies from the actual stdio server.
        listed = await client.read_all_pages("list_strategies")
        assert {"bundled", "generated", "legacy"} <= {item["kind"] for item in listed}
        legacy = next(item for item in listed if item["strategy_id"] == _LEGACY_ID)
        assert legacy == {
            "strategy_id": _LEGACY_ID,
            "revision_id": None,
            "kind": "legacy",
            "execution_capabilities": {
                "training_requirement": "unknown",
                "supported_execution_modes": None,
            },
            "status": "revision_required",
            "code": "strategy_revision_required",
            "message": legacy["message"],
        }
        assert legacy["message"]
        retrieved = await client.call("get_strategy", strategy_id=_LEGACY_ID)
        assert retrieved == {
            **legacy,
            "success": True,
            "source_code": _LEGACY_SOURCE,
            "yaml_config": _LEGACY_CONFIG,
        }
        for tool in ("validate_strategy", "dry_run_strategy", "promote_strategy"):
            rejected = await client.call(tool, strategy_id=_LEGACY_ID)
            assert rejected["success"] is False
            assert rejected["code"] == "strategy_revision_required"
        rejected_run = await client.call(
            "prepare_backtest",
            strategy_id=_LEGACY_ID,
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-02",
            **execution_options({}),
        )
        assert rejected_run["success"] is False
        assert rejected_run["code"] == "strategy_not_executable"
        assert (await client.call("list_jobs"))["items"] == []
        missing_revision = await client.call(
            "get_strategy",
            strategy_id=_LEGACY_ID,
            revision_id="0123456789ab4def8123456789abcdef",
        )
        assert missing_revision["success"] is False
        assert missing_revision["code"] == "strategy_not_found"

        saved = await client.call(
            "save_strategy",
            strategy_id=_LEGACY_ID,
            code=retrieved["source_code"],
            yaml_config=retrieved["yaml_config"],
        )
        assert saved["success"] and saved["revision_id"]
        assert saved["status"] == "draft"
        generated = await client.call("get_strategy", strategy_id=_LEGACY_ID)
        assert generated["kind"] == "generated"
        assert generated["revision_id"] == saved["revision_id"]
        assert generated["source_code"] == _LEGACY_SOURCE
        assert generated["metadata"]["status"] == "draft"
        assert generated["metadata"]["validation"] is None
        assert generated["metadata"]["dry_run"] is None
        replacement = [
            item
            for item in await client.read_all_pages("list_strategies")
            if item["strategy_id"] == _LEGACY_ID
        ]
        assert len(replacement) == 1
        assert replacement[0]["kind"] == "generated"
        assert replacement[0]["revision_id"] == saved["revision_id"]

    assert {path: path.read_bytes() for path in legacy_paths} == original_bytes


async def test_legacy_bundled_id_collision_can_be_recovered_under_another_id(
    mcp_workspace: MCPWorkspace,
) -> None:
    async with mcp_workspace.connect() as client:
        original_listing = await client.read_all_pages("list_strategies")
        bundled = next(item for item in original_listing if item["kind"] == "bundled")
        strategy_id = bundled["strategy_id"]
        original_bundle = await client.call("get_strategy", strategy_id=strategy_id)
        no_legacy = await client.call(
            "get_strategy", strategy_id=strategy_id, legacy=True
        )
        assert no_legacy["success"] is False
        assert no_legacy["code"] == "strategy_not_found"

        legacy_paths = _write_legacy(mcp_workspace, strategy_id=strategy_id)
        original_bytes = {path: path.read_bytes() for path in legacy_paths}
        collisions = [
            item
            for item in await client.read_all_pages("list_strategies")
            if item["strategy_id"] == strategy_id
        ]
        assert len(collisions) == 2
        assert {item["kind"] for item in collisions} == {"bundled", "legacy"}
        legacy = next(item for item in collisions if item["kind"] == "legacy")
        guidance = legacy["message"].casefold()
        assert "legacy" in guidance
        assert "strategy_id" in guidance
        assert await client.call("get_strategy", strategy_id=strategy_id) == (
            original_bundle
        )

        recovered = await client.call(
            "get_strategy", strategy_id=strategy_id, legacy=True
        )
        assert recovered["success"]
        assert recovered["kind"] == "legacy"
        assert recovered["source_code"] == _LEGACY_SOURCE
        assert recovered["yaml_config"] == _LEGACY_CONFIG
        conflicting_selection = await client.call(
            "get_strategy",
            strategy_id=strategy_id,
            revision_id="0123456789ab4def8123456789abcdef",
            legacy=True,
        )
        assert conflicting_selection["success"] is False
        assert conflicting_selection["code"] == "strategy_revision_invalid"

        rejected = await client.call(
            "save_strategy",
            strategy_id=strategy_id,
            code=recovered["source_code"],
            yaml_config=recovered["yaml_config"],
        )
        assert rejected["success"] is False
        assert rejected["code"] == "reserved_strategy_id"
        replacement_id = "recovered_legacy_strategy"
        saved = await client.call(
            "save_strategy",
            strategy_id=replacement_id,
            code=recovered["source_code"],
            yaml_config=recovered["yaml_config"],
        )
        assert saved["success"] and saved["revision_id"]
        assert saved["status"] == "draft"
        generated = await client.call("get_strategy", strategy_id=replacement_id)
        assert generated["kind"] == "generated"
        assert generated["metadata"]["validation"] is None
        assert generated["metadata"]["dry_run"] is None
        no_legacy = await client.call(
            "get_strategy", strategy_id=replacement_id, legacy=True
        )
        assert no_legacy["success"] is False
        assert no_legacy["code"] == "strategy_not_found"
        assert await client.call("get_strategy", strategy_id=strategy_id) == (
            original_bundle
        )

    assert {path: path.read_bytes() for path in legacy_paths} == original_bytes


@pytest.mark.parametrize("artifact", ["source", "config"])
@pytest.mark.parametrize("damage", ["missing", "non_utf8", "symlink"])
async def test_unreadable_legacy_source_keeps_discovery_available(
    mcp_workspace: MCPWorkspace, artifact: str, damage: str
) -> None:
    source, config, _metadata = _write_legacy(mcp_workspace)
    target = source if artifact == "source" else config
    if damage == "missing":
        target.unlink()
    elif damage == "non_utf8":
        target.write_bytes(b"\xff\xfe")
    else:
        outside = mcp_workspace.root / "symlink-target"
        outside.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(outside)

    async with mcp_workspace.connect() as client:
        listed = await client.read_all_pages("list_strategies")
        assert any(item["kind"] == "bundled" for item in listed)
        legacy = next(item for item in listed if item["strategy_id"] == _LEGACY_ID)
        assert legacy["kind"] == "legacy"
        assert legacy["code"] == "strategy_revision_required"
        retrieved = await client.call("get_strategy", strategy_id=_LEGACY_ID)
        assert retrieved["success"] is False
        assert retrieved["code"] == "strategy_revision_invalid"
        assert "source_code" not in retrieved
        assert "yaml_config" not in retrieved
        assert await client.read_all_pages("list_strategies") == listed
