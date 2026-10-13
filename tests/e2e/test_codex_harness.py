"""Client orchestration must reap subprocesses even when cleanup reports errors."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import psutil
import pytest
from tooling.process_guard import terminate_tree

from tests.e2e import codex_harness
from tests.integration.mcp_harness import SimulatedUserApproval

if TYPE_CHECKING:
    from pytest import MonkeyPatch

_WAITING_CLIENT = """\
import time
from pathlib import Path

Path('verifier.ready').touch()
time.sleep(60)
"""

_APP_SERVER = """\
import json
import os
import subprocess
import sys
import time
from pathlib import Path

child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                         start_new_session=True)
Path('client_pids.json').write_text(json.dumps([os.getpid(), child.pid]))
mode = sys.argv[1]
form_enabled = False
for line in sys.stdin:
    request = json.loads(line)
    method = request.get('method')
    if request.get('id') == 'form-1':
        Path('form_answer.json').write_text(json.dumps(request['result']))
        print(json.dumps({'method': 'turn/completed', 'params': {
            'threadId': 'thread', 'turn': {'id': 'turn', 'status': 'completed'}}}),
              flush=True)
        time.sleep(60)
    if method == 'initialized':
        continue
    if method == 'initialize':
        result = {}
    elif method == 'config/read':
        result = {'config': {'mcp_servers': {}}}
    elif method == 'thread/start':
        policy = request['params'].get('approvalPolicy')
        form_enabled = (isinstance(policy, dict)
                        and policy.get('granular', {}).get('mcp_elicitations'))
        result = {'thread': {'id': 'thread'}}
    elif method == 'turn/start':
        result = {'turn': {'id': 'turn'}}
    else:
        raise AssertionError(method)
    print(json.dumps({'id': request['id'], 'result': result}), flush=True)
    if method == 'turn/start':
        if mode == 'form':
            if form_enabled:
                print(json.dumps({'id': 'form-1',
                    'method': 'mcpServer/elicitation/request', 'params': {
                    'threadId': 'thread', 'turnId': 'turn',
                    'serverName': 'tradingdev', 'mode': 'form',
                    'message': 'Approve this test plan',
                    'requestedSchema': {'type': 'object', 'required': ['approved'],
                        'properties': {'approved': {'type': 'boolean'}}}}}), flush=True)
                continue
            # Codex's never policy declines internally without contacting the client.
            mode = 'completed'
        if mode != 'timeout':
            print(json.dumps({'method': 'turn/completed', 'params': {
                'threadId': 'thread', 'turn': {'id': 'turn', 'status': mode}}}),
                  flush=True)
        time.sleep(60)
"""


@pytest.mark.parametrize("status", ["completed", "failed", "timeout"])
def test_codex_stops_client_and_detached_child(
    monkeypatch: MonkeyPatch, status: str
) -> None:
    monkeypatch.setattr(
        codex_harness,
        "_command",
        lambda root, model: [sys.executable, "-c", _APP_SERVER, status],
    )
    with TemporaryDirectory(prefix="tradingdev-codex-lifecycle-") as temporary:
        root = Path(temporary)
        if status == "completed":
            events = asyncio.run(codex_harness.run_codex(root, "prompt"))
            assert events[-1]["method"] == "turn/completed"
        else:
            expected = "Codex turn failed" if status == "failed" else "Codex exceeded"
            with pytest.raises(AssertionError, match=expected):
                asyncio.run(codex_harness.run_codex(root, "prompt", timeout_seconds=1))
        pids = json.loads((root / "client_pids.json").read_text())
        for pid in pids:
            try:
                process = psutil.Process(pid)
            except psutil.NoSuchProcess:
                continue
            assert process.status() == psutil.STATUS_ZOMBIE
    assert not root.exists()


def test_codex_enables_forms_and_wires_the_explicit_callback(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        codex_harness,
        "_command",
        lambda root, model: [sys.executable, "-c", _APP_SERVER, "form"],
    )
    approval = SimulatedUserApproval(approved=False)
    with TemporaryDirectory(prefix="tradingdev-codex-form-") as temporary:
        root = Path(temporary)
        asyncio.run(
            codex_harness.run_codex(
                root, "prompt", elicitation_callback=approval.respond
            )
        )
        assert len(approval.requests) == 1
        answer = json.loads((root / "form_answer.json").read_text())
        assert answer["action"] == "accept" and answer["content"] == {"approved": False}
    assert not root.exists()


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
