"""Client orchestration must reap subprocesses even when cleanup reports errors."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import pytest
from tooling.process_guard import terminate_tree

from tests.e2e import codex_harness

if TYPE_CHECKING:
    import psutil
    from pytest import MonkeyPatch

_WAITING_CLIENT = """\
import json
import time
from pathlib import Path

Path('verifier.ready').touch()
print(json.dumps({'type': 'turn.completed'}), flush=True)
time.sleep(60)
"""


def _fail_after_cleanup(
    leader: psutil.Process | None,
    descendants: set[psutil.Process] | None = None,
) -> None:
    terminate_tree(leader, descendants)
    raise RuntimeError("Descendant cleanup could not be verified")


def test_codex_reaps_process_when_cleanup_verification_fails(
    monkeypatch: MonkeyPatch,
) -> None:
    reaped: list[int] = []
    original_wait = asyncio.subprocess.Process.wait

    async def wait(process: asyncio.subprocess.Process) -> int:
        returncode = await original_wait(process)
        reaped.append(process.pid)
        return returncode

    monkeypatch.setattr(asyncio.subprocess.Process, "wait", wait)
    monkeypatch.setattr(codex_harness, "terminate_tree", _fail_after_cleanup)
    monkeypatch.setattr(
        codex_harness,
        "_command",
        lambda root, model: [sys.executable, "-c", _WAITING_CLIENT],
    )
    with TemporaryDirectory(prefix="tradingdev-codex-reaping-") as temporary:
        root = Path(temporary)
        with pytest.raises(RuntimeError, match="cleanup could not be verified"):
            asyncio.run(codex_harness.run_codex(root, "prompt", timeout_seconds=0.2))
        assert len(reaped) == 1
        with pytest.raises(ProcessLookupError):
            os.kill(reaped[0], 0)
    assert not root.exists()


def test_verifier_reaps_process_when_cleanup_verification_fails(
    monkeypatch: MonkeyPatch,
) -> None:
    processes: list[subprocess.Popen[str]] = []
    original_popen = subprocess.Popen

    def spawn(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        process: subprocess.Popen[str] = original_popen(
            [sys.executable, "-c", _WAITING_CLIENT], **kwargs
        )
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(codex_harness, "terminate_tree", _fail_after_cleanup)
    with TemporaryDirectory(prefix="tradingdev-verifier-reaping-") as temporary:
        root = Path(temporary)
        try:
            with pytest.raises(RuntimeError, match="cleanup could not be verified"):
                codex_harness.verify_generated_strategy(root, timeout_seconds=0.2)
            assert len(processes) == 1
            assert processes[0].returncode is not None
            with pytest.raises(ProcessLookupError):
                os.kill(processes[0].pid, 0)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=3)
    assert not root.exists()
