"""Run an isolated Codex client and inspect its actual MCP tool events."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tooling.codex_binary import resolve_codex_binary
from tooling.process_guard import process_descendants, process_identity, terminate_tree

from tests.e2e.codex_app_server import CodexAppServer
from tests.integration.mcp_harness import MCPWorkspace

if TYPE_CHECKING:
    import psutil

    from tests.e2e.codex_app_server import ElicitationCallback
    from tests.e2e.conversation import UserReply

LIFECYCLE_TOOLS = [
    "list_strategies",
    "get_strategy_contract",
    "get_strategy",
    "cleanup_strategy_drafts",
    "save_strategy",
    "validate_strategy",
    "dry_run_strategy",
    "prepare_backtest",
    "prepare_walk_forward",
    "prepare_optimization",
    "get_execution_plan",
    "submit_execution_decision",
    "get_job_status",
    "list_runs",
    "get_run",
    "get_metric_catalog",
    "get_run_metrics",
    "find_runs",
    "get_run_trades",
    "get_run_equity",
    "get_run_executions",
    "get_run_account_history",
    "get_report_sections",
    "generate_report",
    "list_artifacts",
    "get_artifact",
]


def runtime_environment(root: Path) -> dict[str, str]:
    """Keep server, verifier and CLI runtime caches inside the temporary root."""
    repo = Path(__file__).resolve().parents[2]
    environment = MCPWorkspace(root).environment()
    environment["PYTHONPATH"] = os.pathsep.join((environment["PYTHONPATH"], str(repo)))
    return environment


def _command(root: Path, model: str | None = None) -> list[str]:
    command = [resolve_codex_binary(), "app-server", "--listen", "stdio://"]
    config: dict[str, Any] = {
        "features.shell_tool": False,
        "features.shell_snapshot": False,
        "features.plugins": False,
        "features.remote_plugin": False,
        "features.apps": False,
        "features.hooks": False,
        "features.memories": False,
        "features.multi_agent": False,
        "features.code_mode": False,
        "features.js_repl": False,
        "web_search": "disabled",
        "skip_host_skill_discovery": True,
        "notify": [],
        "developer_instructions": "",
        "project_doc_max_bytes": 0,
        "history.persistence": "none",
        "log_dir": str(root / "codex_logs"),
        "sqlite_home": str(root / "codex_state"),
    }
    if model := model or os.environ.get("TRADINGDEV_CODEX_MODEL"):
        config["model"] = model
    for key, value in config.items():
        command.extend(["-c", f"{key}={json.dumps(value)}"])
    return command


def _mcp_config(root: Path, tool_names: list[str]) -> dict[str, Any]:
    return {
        "command": sys.executable,
        "args": ["-m", "tradingdev.mcp.server", "--workspace", str(root / "workspace")],
        "cwd": str(root),
        "env": runtime_environment(root),
        "enabled": True,
        "required": True,
        "startup_timeout_sec": 30,
        "tool_timeout_sec": 60,
        "enabled_tools": tool_names,
        "default_tools_approval_mode": "approve",
    }


def _final_turn_text(events: list[dict[str, Any]]) -> str:
    """Read terminal assistant text without treating progress as delivery.

    App-server MessagePhase is nullable because providers may omit it. A missing
    phase is accepted only for the last message after all tool calls; explicitly
    nonterminal commentary/partial answers never establish a final response.
    """
    for event in reversed(events):
        if event.get("method") != "item/completed":
            continue
        item = event.get("params", {}).get("item", {})
        if item.get("type") == "mcpToolCall":
            return ""
        if item.get("type") == "agentMessage":
            text = item.get("text")
            if item.get("phase") in {None, "final_answer"} and isinstance(text, str):
                return text
            return ""
    return ""


async def run_codex(
    root: Path,
    prompt: str,
    *,
    timeout_seconds: float = 300,
    max_tool_calls: int = 64,
    model: str | None = None,
    elicitation_callback: ElicitationCallback | None = None,
    user_reply: UserReply | None = None,
    max_turns: int = 8,
    tool_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run an MCP-only conversation with one thread and reap its process tree."""
    from tests.e2e.conversation import ModelTurn
    from tests.e2e.llm_client import codex_calls

    enabled_tools = LIFECYCLE_TOOLS if tool_names is None else tool_names
    turn_number = 0

    async def respond_to_turn(events: list[dict[str, Any]]) -> str | None:
        nonlocal turn_number
        turn_number += 1
        assert user_reply is not None
        return await user_reply(
            ModelTurn(turn_number, _final_turn_text(events), codex_calls(events))
        )

    stderr_path = root / "codex_stderr.log"
    with stderr_path.open("w", encoding="utf-8") as stderr:
        process = await asyncio.create_subprocess_exec(
            *_command(root, model),
            cwd=root,
            env={**os.environ, **runtime_environment(root)},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr,
            start_new_session=True,
            limit=1024 * 1024,
        )
        identity = process_identity(process.pid)
        descendants: set[psutil.Process] = set()

        def observe_processes() -> None:
            descendants.update(process_descendants(identity))

        assert process.stdout is not None and process.stdin is not None
        client = CodexAppServer(
            process.stdout,
            process.stdin,
            tools=enabled_tools,
            max_tool_calls=max_tool_calls,
            elicitation_callback=elicitation_callback,
            observe_processes=observe_processes,
        )
        try:
            async with asyncio.timeout(timeout_seconds):
                return await client.run(
                    prompt,
                    {
                        "cwd": str(root),
                        "model": model or os.environ.get("TRADINGDEV_CODEX_MODEL"),
                        "ephemeral": True,
                        "sandbox": "read-only",
                        "approvalPolicy": {
                            "granular": {
                                "sandbox_approval": False,
                                "rules": False,
                                "mcp_elicitations": True,
                                "request_permissions": False,
                                "skill_approval": False,
                            }
                        },
                        "approvalsReviewer": "user",
                        "allowProviderModelFallback": False,
                        "developerInstructions": (
                            "Use only the configured tradingdev MCP tools."
                        ),
                    },
                    mcp_config=_mcp_config(root, enabled_tools),
                    user_reply=respond_to_turn if user_reply is not None else None,
                    max_turns=max_turns,
                )
        except TimeoutError as exc:
            msg = (
                f"Codex exceeded {timeout_seconds:g}s after {len(client.seen_calls)} "
                "MCP calls"
            )
            raise AssertionError(msg) from exc
        except AssertionError as exc:
            details = stderr_path.read_text(encoding="utf-8")[-4000:]
            raise AssertionError(f"{exc}\nCodex stderr:\n{details}") from exc
        finally:
            try:
                terminate_tree(identity, descendants)
            finally:
                await asyncio.wait_for(process.wait(), timeout=3)


def verify_generated_strategy(
    root: Path,
    *,
    timeout_seconds: float = 30,
    startup_timeout_seconds: float = 30,
    scenario: str | None = None,
) -> None:
    """Check generated code in a bounded subprocess and always reap it."""
    ready = root / "verifier.ready"
    ready.unlink(missing_ok=True)
    process = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "tests.e2e.verify_strategy",
            str(root),
            *([scenario] if scenario else []),
        ],
        cwd=root,
        env={**os.environ, **runtime_environment(root)},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    identity = process_identity(process.pid)
    try:
        deadline = time.monotonic() + startup_timeout_seconds
        while not ready.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                msg = (
                    "Generated strategy verifier startup exceeded "
                    f"{startup_timeout_seconds:g}s"
                )
                raise AssertionError(msg)
            time.sleep(0.02)
        output, _ = process.communicate(timeout=timeout_seconds)
        assert process.returncode == 0, (
            f"Generated strategy verification failed:\n{output[-4000:]}"
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"Generated strategy verification exceeded {timeout_seconds:g}s"
        raise AssertionError(msg) from exc
    finally:
        try:
            terminate_tree(identity)
        finally:
            process.communicate(timeout=3)


def successful_mcp_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract completed MCP invocations from native app-server notifications."""
    return [
        event["params"]["item"]
        for event in events
        if event.get("method") == "item/completed"
        and isinstance(event.get("params"), dict)
        and isinstance(event["params"].get("item"), dict)
        and event["params"]["item"].get("type") == "mcpToolCall"
        and event["params"]["item"].get("status") == "completed"
    ]
