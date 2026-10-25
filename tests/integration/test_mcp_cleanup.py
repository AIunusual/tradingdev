"""Verify real active workers and temporary artifacts are cleaned on all exits."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import psutil
import pytest

from tests.integration.execution_fixtures import confirm_prepared, execution_options
from tests.integration.mcp_harness import (
    temporary_mcp_workspace,
    worker_is_alive,
)

if TYPE_CHECKING:
    import pandas as pd

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


class ForcedCleanupError(RuntimeError):
    """Deliberate failure while a real worker is paused by a test startup gate."""


def _exception_leaves(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in _exception_leaves(child)]
    return [error]


@pytest.mark.parametrize("exit_mode", ["success", "failure", "timeout"])
async def test_active_worker_and_files_are_removed_on_context_exit(
    sample_ohlcv_with_kd: pd.DataFrame,
    exit_mode: str,
) -> None:
    root: Path | None = None
    worker: psutil.Process | None = None
    server: psutil.Process | None = None
    caught: BaseException | None = None
    try:
        with temporary_mcp_workspace() as workspace:
            root = workspace.root
            workspace.seed_market(sample_ohlcv_with_kd)
            async with workspace.connect() as client:
                contract = (
                    await client.call("get_strategy_contract", section="example")
                )["content"]
                saved = await client.call(
                    "save_strategy",
                    strategy_id="cleanup_strategy",
                    code=contract["example_strategy_code"],
                    yaml_config=contract["example_yaml_config"],
                )
                assert saved["success"], saved
                for tool in ("validate_strategy", "dry_run_strategy"):
                    checked = await client.call(tool, strategy_id="cleanup_strategy")
                    assert checked["success"], checked
                prepared = await client.call(
                    "prepare_optimization",
                    strategy_id="cleanup_strategy",
                    symbol="BTC/USDT",
                    timeframe="1h",
                    param_ranges={"fast_period": [3], "slow_period": [8]},
                    optimization_metric="total_return",
                    train_start="2024-01-01",
                    train_end="2024-01-03",
                    test_start="2024-01-04",
                    test_end="2024-01-07",
                    **execution_options({"fast_period": 10, "slow_period": 30}),
                )
                assert prepared["success"], prepared
                # Keep the formal worker alive deterministically, independently
                # of grid-search speed. The supervisor, child process, MCP,
                # confirmation, identity checks and cleanup remain real.
                guard = workspace.root / "python-guard" / "sitecustomize.py"
                guard.write_text(
                    guard.read_text() + "\nif sys.orig_argv[1:3] == "
                    "['-m', 'tradingdev.mcp.workers.optimization']:\n"
                    + "    import time\n    time.sleep(120)\n"
                )
                started = await confirm_prepared(client, prepared)
                assert started["job_id"], started
                status = await client.call("get_job_status", job_id=started["job_id"])
                assert status["status"] == "queued", status
                with closing(
                    sqlite3.connect(workspace.workspace / "tradingdev.sqlite")
                ) as connection:
                    row = connection.execute(
                        "SELECT pid, payload FROM jobs WHERE job_id = ?",
                        (started["job_id"],),
                    ).fetchone()
                assert row is not None
                worker = psutil.Process(row[0])
                assert json.loads(row[1])["process_create_time"] == worker.create_time()
                server = psutil.Process(worker.ppid())
                assert worker_is_alive(worker)
                assert "tradingdev.mcp.server" in server.cmdline()
                assert Path(saved["yaml_path"]).is_file()
                assert Path(saved["py_path"]).is_file()
                if exit_mode == "failure":
                    raise ForcedCleanupError("Exercise failed-test teardown")
                if exit_mode == "timeout":
                    with anyio.fail_after(0):
                        await anyio.sleep_forever()
    except BaseException as error:
        caught = error

    assert root is not None and not root.exists()
    if worker is None and caught is not None:
        raise caught
    assert worker is not None and not worker_is_alive(worker)
    assert server is not None and not worker_is_alive(server)
    if exit_mode != "success":
        assert caught is not None
        leaves = _exception_leaves(caught)
        expected = ForcedCleanupError if exit_mode == "failure" else TimeoutError
        assert len(leaves) == 1 and isinstance(leaves[0], expected), caught
    elif caught is not None:
        raise caught
