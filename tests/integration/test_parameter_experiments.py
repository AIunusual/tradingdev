"""Parameter experiments reuse a revision through real MCP and workers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from tests.integration.execution_fixtures import (
    confirm_prepared,
    research_arguments,
    research_from_config,
    save_execution_evidence,
)
from tests.integration.test_mcp_protocol import (
    RUN_ARGUMENTS,
    STRATEGY_ID,
    make_runnable,
    save_example,
)

if TYPE_CHECKING:
    import pandas as pd

    from tests.integration.mcp_harness import MCPWorkspace

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.mark.parametrize("walk_forward", [False, True])
async def test_parameter_experiments_preserve_source_and_base_settings(
    mcp_workspace: MCPWorkspace,
    sample_ohlcv_with_kd: pd.DataFrame,
    walk_forward: bool,
) -> None:
    mcp_workspace.seed_market(sample_ohlcv_with_kd)
    async with mcp_workspace.connect() as client:
        code, yaml_config, saved = await save_example(client)
        if walk_forward:
            config = yaml.safe_load(yaml_config)
            config["validation"] = {
                "mode": "automatic",
                "n_splits": 2,
                "train_ratio": 0.6,
            }
            saved = await client.call(
                "save_strategy",
                strategy_id=STRATEGY_ID,
                code=code,
                yaml_config=yaml.safe_dump(config),
                research=research_from_config(config),
                expected_revision_id=saved["revision_id"],
            )
        revision_id = saved["revision_id"]
        await make_runnable(client, revision_id)
        directory = Path(saved["py_path"]).parent
        root = directory.parent.parent
        originals = {
            path: path.read_bytes()
            for path in [
                root / "current.json",
                directory / "strategy.py",
                directory / "config.yaml",
                directory / "metadata.json",
            ]
        }
        revisions = set((root / "revisions").iterdir())
        tool = "prepare_walk_forward" if walk_forward else "prepare_backtest"
        hashes = set()
        for fast_period in (2, 5):
            prepared = await client.call(
                tool,
                **research_arguments(RUN_ARGUMENTS),
                revision_id=revision_id,
                parameters={"fast_period": fast_period},
                **{
                    "evidence_id": await save_execution_evidence(
                        client, {"fast_period": fast_period, "slow_period": 8}
                    )
                },
            )
            started = await confirm_prepared(client, prepared)
            assert started["job_id"], started
            hashes.add(started["manifest_hash"])
            done = await client.wait_for_job(started["job_id"])
            assert done["status"] == "done", done
            run_dir = mcp_workspace.workspace / "runs" / done["run_id"]
            manifest = json.loads((run_dir / "manifest.json").read_text())
            expected = {"fast_period": fast_period, "slow_period": 8}
            assert manifest["config"]["strategy"]["parameters"] == expected
            assert manifest["config"]["strategy"]["revision_id"] == revision_id
            assert (run_dir / "strategy.py").read_bytes() == originals[
                directory / "strategy.py"
            ]
            metrics = await client.call(
                "get_run_metrics",
                run_id=done["run_id"],
                scope="fold/0/test" if walk_forward else "full",
            )
            assert metrics["parameters"] == expected
        assert len(hashes) == 2
        assert set((root / "revisions").iterdir()) == revisions
        assert {path: path.read_bytes() for path in originals} == originals

        jobs_before = await client.read_all_pages("list_jobs")
        rejected = await client.call(
            tool,
            **research_arguments(RUN_ARGUMENTS),
            revision_id=revision_id,
            parameters={"unknown_parameter": 1},
            **{
                "evidence_id": await save_execution_evidence(
                    client, {"fast_period": 3, "slow_period": 8}
                )
            },
        )
        assert rejected["success"] is False, rejected
        assert rejected["code"] == "invalid_execution_request"
        assert {job["job_id"] for job in await client.read_all_pages("list_jobs")} == {
            job["job_id"] for job in jobs_before
        }
        assert {path: path.read_bytes() for path in originals} == originals
