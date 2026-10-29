"""Client orchestration must reap subprocesses even when cleanup reports errors."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import psutil
import pytest
from tooling.process_guard import terminate_tree

from tests.e2e import codex_harness
from tests.e2e.llm_client import ToolCall, codex_calls
from tests.integration.mcp_harness import SimulatedUserApproval

if TYPE_CHECKING:
    from pytest import MonkeyPatch

    from tests.e2e.conversation import ModelTurn

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
        Path('thread_request.json').write_text(json.dumps(request['params']))
        policy = request['params'].get('approvalPolicy')
        form_enabled = (isinstance(policy, dict)
                        and policy.get('granular', {}).get('mcp_elicitations'))
        result = {'thread': {'id': 'thread'},
                  'model': request['params'].get('model') or 'fixture-default',
                  'reasoningEffort': ('ultra' if mode == 'effort-mismatch'
                                      else 'xhigh')}
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
            for item in json.loads(sys.argv[2]) if len(sys.argv) > 2 else []:
                print(json.dumps({'method': 'item/completed', 'params': {
                    'threadId': 'thread', 'turnId': 'turn', 'item': item}}), flush=True)
            if mode != 'tool-timeout':
                print(json.dumps({'method': 'turn/completed', 'params': {
                    'threadId': 'thread', 'turn': {'id': 'turn', 'status': mode}}}),
                      flush=True)
        time.sleep(60)
"""


@pytest.fixture(params=[None, "gpt-5.6-luna"], ids=["model-unset", "model-env"])
def model_environment(
    request: pytest.FixtureRequest, monkeypatch: MonkeyPatch
) -> str | None:
    model: str | None = request.param
    if model is None:
        monkeypatch.delenv("TRADINGDEV_CODEX_MODEL", raising=False)
    else:
        monkeypatch.setenv("TRADINGDEV_CODEX_MODEL", model)
    return model


def test_codex_process_config_isolates_effort_and_host_capabilities(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(codex_harness, "resolve_codex_binary", lambda: "/test/codex")
    command = codex_harness._command(Path("/test/isolated"), "selected-model")
    assert command[:4] == ["/test/codex", "app-server", "--listen", "stdio://"]
    assert command[4::2] == ["-c"] * len(command[5::2])
    settings = {
        key: json.loads(value)
        for item in command[5::2]
        for key, _, value in [item.partition("=")]
    }
    assert settings["model"] == "selected-model"
    assert settings["model_reasoning_effort"] == "xhigh"
    assert settings["features.skip_host_skill_discovery"] is True
    assert "skip_host_skill_discovery" not in settings
    for feature in (
        "multi_agent",
        "shell_tool",
        "shell_snapshot",
        "plugins",
        "remote_plugin",
        "apps",
        "hooks",
        "memories",
        "code_mode",
        "js_repl",
    ):
        assert settings[f"features.{feature}"] is False
    assert settings["web_search"] == "disabled"
    assert settings["project_doc_max_bytes"] == 0
    assert settings["history.persistence"] == "none"
    assert settings["log_dir"] == "/test/isolated/codex_logs"
    assert settings["sqlite_home"] == "/test/isolated/codex_state"


@pytest.mark.parametrize("status", ["completed", "failed", "timeout"])
@pytest.mark.parametrize("tool_names", [None, ["get_ml_contract", "build_dataset"]])
def test_codex_stops_client_and_detached_child(
    monkeypatch: MonkeyPatch,
    status: str,
    model_environment: str | None,
    tool_names: list[str] | None,
) -> None:
    monkeypatch.setattr(
        codex_harness,
        "_command",
        lambda root, model: [sys.executable, "-c", _APP_SERVER, status],
    )
    with TemporaryDirectory(prefix="tradingdev-codex-lifecycle-") as temporary:
        root = Path(temporary)
        if status == "completed":
            events = asyncio.run(
                codex_harness.run_codex(root, "prompt", tool_names=tool_names)
            )
            assert events[-1]["method"] == "turn/completed"
        else:
            expected = "Codex turn failed" if status == "failed" else "Codex exceeded"
            with pytest.raises(AssertionError, match=expected):
                asyncio.run(
                    codex_harness.run_codex(
                        root, "prompt", timeout_seconds=1, tool_names=tool_names
                    )
                )
        requested = json.loads((root / "thread_request.json").read_text())
        assert requested["model"] == model_environment
        assert set(requested["config"]) == {"mcp_servers"}, (
            "A partial thread features table would erase process isolation flags"
        )
        servers = requested["config"]["mcp_servers"]
        assert len(servers) == 1
        assert next(iter(servers.values()))["enabled_tools"] == (
            codex_harness.LIFECYCLE_TOOLS if tool_names is None else tool_names
        )
        pids = json.loads((root / "client_pids.json").read_text())
        for pid in pids:
            try:
                process = psutil.Process(pid)
            except psutil.NoSuchProcess:
                continue
            assert process.status() == psutil.STATUS_ZOMBIE
    assert not root.exists()


@pytest.mark.parametrize("failure", ["effort-mismatch", "subagent"])
def test_codex_rejects_bypass_and_stops_client_and_detached_child(
    monkeypatch: MonkeyPatch, failure: str
) -> None:
    item = {"id": "delegated", "type": "subAgentActivity"}
    monkeypatch.setattr(
        codex_harness,
        "_command",
        lambda root, model: [
            sys.executable,
            "-c",
            _APP_SERVER,
            failure if failure == "effort-mismatch" else "completed",
            json.dumps([item]),
        ],
    )
    with TemporaryDirectory(prefix="tradingdev-codex-isolation-") as temporary:
        root = Path(temporary)
        expected = (
            "expected 'xhigh'" if failure == "effort-mismatch" else "bypassed MCP"
        )
        with pytest.raises(AssertionError, match=expected):
            asyncio.run(codex_harness.run_codex(root, "prompt"))
        for pid in json.loads((root / "client_pids.json").read_text()):
            assert not psutil.pid_exists(pid)
    assert not root.exists()


def test_codex_enables_forms_and_wires_the_explicit_callback(
    monkeypatch: MonkeyPatch, model_environment: str | None
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
        requested = json.loads((root / "thread_request.json").read_text())
        assert requested["model"] == model_environment
        answer = json.loads((root / "form_answer.json").read_text())
        assert answer["action"] == "accept" and answer["content"] == {"approved": False}
    assert not root.exists()


def test_codex_explicit_model_takes_precedence_over_environment(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADINGDEV_CODEX_MODEL", "environment-model")
    command_models: list[str | None] = []

    def command(root: Path, model: str | None) -> list[str]:
        command_models.append(model)
        return [sys.executable, "-c", _APP_SERVER, "completed"]

    monkeypatch.setattr(codex_harness, "_command", command)
    with TemporaryDirectory(prefix="tradingdev-codex-model-") as temporary:
        root = Path(temporary)
        events = asyncio.run(
            codex_harness.run_codex(root, "prompt", model="explicit-model")
        )
        assert events[-1]["method"] == "turn/completed"
        assert command_models == ["explicit-model"]
        requested = json.loads((root / "thread_request.json").read_text())
        assert requested["model"] == "explicit-model"
    assert not root.exists()


@pytest.mark.parametrize(
    ("phase", "final_text", "tools_after_message", "expected"),
    [
        ("final_answer", "完成 run", False, "完成 run"),
        (
            "final_answer",
            "完成 run\n[閱讀報告](</report.html>)",
            False,
            "完成 run\n[閱讀報告](</report.html>)",
        ),
        (None, "完成 run", False, "完成 run"),
        ("commentary", "[閱讀報告](</report.html>)", False, ""),
        ("partial_answer", "[閱讀報告](</report.html>)", False, ""),
        (None, "[閱讀報告](</report.html>)", True, ""),
        ("final_answer", "[閱讀報告](</report.html>)", True, ""),
    ],
)
def test_codex_user_callback_reads_only_the_terminal_answer(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    phase: str | None,
    final_text: str,
    tools_after_message: bool,
    expected: str,
) -> None:
    """An interim report link must not satisfy final-turn delivery evidence."""
    progress = {
        "id": "progress",
        "type": "agentMessage",
        "phase": "commentary",
        "text": "報告已生成：[閱讀報告](</report.html>)",
    }
    tool = {
        "id": "tool",
        "type": "mcpToolCall",
        "server": "tradingdev",
        "tool": "get_run_metrics",
        "arguments": {"run_id": "run"},
        "status": "completed",
        "result": {"structuredContent": {"success": True, "run_id": "run"}},
    }
    answer = {
        "id": "answer",
        "type": "agentMessage",
        "phase": phase,
        "text": final_text,
    }
    items = (
        [progress, answer, tool] if tools_after_message else [progress, tool, answer]
    )
    monkeypatch.setattr(
        codex_harness,
        "_command",
        lambda root, model: [
            sys.executable,
            "-c",
            _APP_SERVER,
            "completed",
            json.dumps(items),
        ],
    )
    turns: list[ModelTurn] = []

    async def observe(turn: ModelTurn) -> None:
        turns.append(turn)

    events = asyncio.run(
        codex_harness.run_codex(tmp_path, "prompt", user_reply=observe)
    )
    assert len(turns) == 1
    assert turns[0].number == 1
    assert turns[0].text == expected
    assert len(turns[0].calls) == 1
    assert turns[0].calls[0].name == "get_run_metrics"
    assert turns[0].calls[0].result == {"success": True, "run_id": "run"}
    assert any(event.get("params", {}).get("item") == progress for event in events)


@pytest.mark.parametrize("outcome", ["completed", "failed", "tool-timeout"])
def test_codex_completed_tool_evidence_survives_later_turn_failure(
    monkeypatch: MonkeyPatch, outcome: str
) -> None:
    arguments = {"run_id": "approved-run", "direction": "short", "limit": 1}
    result = {"success": False, "error": "saved history unavailable"}
    item = {
        "id": "trade-page",
        "type": "mcpToolCall",
        "server": "tradingdev",
        "tool": "get_run_trades",
        "arguments": json.dumps(arguments),
        "status": "completed",
        "result": {"structuredContent": result, "isError": True},
    }
    monkeypatch.setattr(
        codex_harness,
        "_command",
        lambda root, model: [
            sys.executable,
            "-c",
            _APP_SERVER,
            outcome,
            json.dumps([item]),
        ],
    )
    observed: list[ToolCall] = []
    turns: list[ModelTurn] = []

    def observe(call: ToolCall) -> None:
        observed.append(deepcopy(call))
        call.arguments["run_id"] = "wrong-run"
        call.result["error"] = "changed by observer"

    async def user_reply(turn: ModelTurn) -> None:
        turns.append(turn)

    with TemporaryDirectory(prefix="tradingdev-codex-tool-evidence-") as temporary:
        root = Path(temporary)

        async def run() -> list[dict[str, Any]]:
            return await codex_harness.run_codex(
                root,
                "Read trades",
                timeout_seconds=1,
                on_tool_call=observe,
                user_reply=user_reply,
            )

        if outcome == "completed":
            events = asyncio.run(run())
            assert codex_calls(events) == observed
            assert len(turns) == 1 and turns[0].calls == observed
        else:
            expected = "Codex turn failed" if outcome == "failed" else "Codex exceeded"
            with pytest.raises(AssertionError, match=expected):
                asyncio.run(run())
            assert turns == []
        assert observed == [ToolCall("get_run_trades", arguments, result)]
        for pid in json.loads((root / "client_pids.json").read_text()):
            try:
                process = psutil.Process(pid)
            except psutil.NoSuchProcess:
                continue
            assert not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
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


def test_codex_rejected_tool_response_can_be_repaired_without_losing_attempts(
    monkeypatch: MonkeyPatch,
) -> None:
    rejected: dict[str, Any] = {
        "id": "bad-draft",
        "type": "mcpToolCall",
        "server": "tradingdev",
        "tool": "save_research_evidence",
        "arguments": {"draft": {"parameter_provenance": ["invalid"]}},
        "status": "failed",
        "error": None,
        "result": {
            "content": [{"type": "text", "text": "Invalid target_ids: use search_1_1"}],
            "structuredContent": None,
        },
    }
    repaired: dict[str, Any] = {
        **rejected,
        "id": "corrected-draft",
        "arguments": {"draft": {"parameter_provenance": ["corrected"]}},
        "status": "completed",
        "result": {"structuredContent": {"success": True, "evidence_id": "e" * 32}},
    }
    monkeypatch.setattr(
        codex_harness,
        "_command",
        lambda root, model: [
            sys.executable,
            "-c",
            _APP_SERVER,
            "completed",
            json.dumps([rejected, repaired]),
        ],
    )
    observed: list[ToolCall] = []
    with TemporaryDirectory(prefix="tradingdev-codex-repair-") as temporary:
        root = Path(temporary)
        events = asyncio.run(
            codex_harness.run_codex(root, "prompt", on_tool_call=observed.append)
        )
        assert codex_calls(events) == observed
        assert len(observed) == 2
        assert observed[0].arguments == rejected["arguments"]
        assert observed[0].result["success"] is False
        assert observed[0].result["mcp_result"] == rejected["result"]
        assert observed[1].result == repaired["result"]["structuredContent"]
        for pid in json.loads((root / "client_pids.json").read_text()):
            assert not psutil.pid_exists(pid)
    assert not root.exists()


@pytest.mark.parametrize(
    ("result", "error"),
    [
        (None, {"message": "connection closed"}),
        ({}, None),
        (
            {"content": [{"type": "text", "text": "partial"}]},
            {"message": "transport failed"},
        ),
    ],
)
def test_codex_failed_transport_remains_a_failure(
    monkeypatch: MonkeyPatch,
    result: Any,
    error: Any,
) -> None:
    item = {
        "id": "lost",
        "type": "mcpToolCall",
        "server": "tradingdev",
        "tool": "save_research_evidence",
        "arguments": {},
        "status": "failed",
        "result": result,
        "error": error,
    }
    monkeypatch.setattr(
        codex_harness,
        "_command",
        lambda root, model: [
            sys.executable,
            "-c",
            _APP_SERVER,
            "completed",
            json.dumps([item]),
        ],
    )
    with TemporaryDirectory(prefix="tradingdev-codex-transport-") as temporary:
        root = Path(temporary)
        with pytest.raises(AssertionError, match="failed without a tool response"):
            asyncio.run(codex_harness.run_codex(root, "prompt"))
        for pid in json.loads((root / "client_pids.json").read_text()):
            assert not psutil.pid_exists(pid)
    assert not root.exists()
