"""Prerequisite failures must precede tests, formatters and installation work."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import psutil
import pytest
from tooling import environment, process_guard
from tooling.process_guard import run_captured

if TYPE_CHECKING:
    from pytest import CaptureFixture, MonkeyPatch

_PROJECT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("profile", ["full", "tooling"])
@pytest.mark.parametrize(
    "preflight_exit, fast_exit, pytest_exit, expected",
    [
        (17, 0, 0, ["environment"]),
        (0, 19, 0, ["environment", "fast"]),
        (0, 0, 23, ["environment", "fast", "pytest"]),
        (0, 0, 0, ["environment", "fast", "pytest"]),
    ],
)
def test_suite_launcher_checks_environment_before_quality_and_pytest(
    monkeypatch: MonkeyPatch,
    profile: str,
    preflight_exit: int,
    fast_exit: int,
    pytest_exit: int,
    expected: list[str],
) -> None:
    from tooling.git import suite

    events: list[str] = []

    def preflight(selected: str) -> None:
        assert selected == profile
        events.append("environment")
        if preflight_exit:
            raise subprocess.CalledProcessError(preflight_exit, ["environment"])

    def run(command: list[str], **kwargs: object) -> None:
        pytest_command = command[:3] == [sys.executable, "-m", "pytest"]
        events.append("pytest" if pytest_command else "fast")
        if pytest_command:
            assert command[3:] == [
                *(["tests/tooling"] if profile == "tooling" else []),
                "-n",
                "auto",
                "--dist",
                "worksteal",
                "--max-worker-restart=0",
            ]
        else:
            assert command == ["bash", "scripts/check-fast.sh"]
        exit_code = pytest_exit if pytest_command else fast_exit
        if exit_code:
            raise subprocess.CalledProcessError(exit_code, command)

    monkeypatch.setattr(suite, "check_environment", preflight)
    monkeypatch.setattr(suite, "run_checked", run)
    expected_exit = preflight_exit or fast_exit or pytest_exit
    if expected_exit:
        with pytest.raises(subprocess.CalledProcessError) as error:
            suite.run_suite(profile, 30)
        assert error.value.returncode == expected_exit
    else:
        suite.run_suite(profile, 30)
    assert events == expected


@pytest.mark.parametrize("profile", ["full", "tooling"])
def test_suite_shell_forwards_profile_and_deadline(
    tmp_path: Path, profile: str
) -> None:
    (tmp_path / "scripts").mkdir()
    shutil.copy2(_PROJECT / f"scripts/check-{profile}.sh", tmp_path / "scripts")
    interpreter = tmp_path / "python"
    interpreter.write_text(
        f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
        "Path('arguments.json').write_text(json.dumps(sys.argv[1:]))\n"
    )
    interpreter.chmod(0o755)
    result = run_captured(
        ["bash", f"scripts/check-{profile}.sh", "--timeout-seconds", "123"],
        cwd=tmp_path,
        env={**os.environ, "TRADINGDEV_CHECK_PYTHON": str(interpreter)},
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "arguments.json").read_text()) == [
        "-m",
        "tooling.git.suite",
        profile,
        "--timeout-seconds",
        "123",
    ]


def test_missing_packages_are_reported_together_without_runtime_checks(
    monkeypatch: MonkeyPatch,
) -> None:
    def missing(name: str) -> None:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    monkeypatch.setattr(
        environment, "require_strategy_runtime", lambda: pytest.fail("Docker called")
    )
    with pytest.raises(environment.EnvironmentCheckError) as failure:
        environment.check_environment("full")
    assert "pytest" in str(failure.value)
    assert "pytest-xdist" in str(failure.value)
    assert "mypy" in str(failure.value)
    assert "uv sync --locked --all-extras" in str(failure.value)


@pytest.mark.parametrize("profile", ["docs", "format", "fast", "tooling"])
def test_lightweight_profiles_do_not_require_docker_or_model_tools(
    monkeypatch: MonkeyPatch,
    profile: str,
) -> None:
    monkeypatch.setattr(environment, "require_packages", lambda *_: None)

    def command(name: str, *, help_text: str) -> str:
        assert name in {"git", "uv"}
        return name

    monkeypatch.setattr(environment, "require_command", command)
    monkeypatch.setattr(
        environment, "require_strategy_runtime", lambda: pytest.fail("Docker called")
    )
    environment.check_environment(profile)


@pytest.mark.parametrize(
    ("version", "supported"),
    [("v18.0.0", False), ("v22.16.9", False), ("invalid", False), ("v22.17.0", True)],
)
def test_node_version_is_checked_before_installation(
    monkeypatch: MonkeyPatch,
    version: str,
    supported: bool,
) -> None:
    monkeypatch.setattr(environment, "require_command", lambda name, **_: name)
    monkeypatch.setattr(
        process_guard,
        "run_captured",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, version, ""),
    )
    if supported:
        assert environment.require_node(npm=True) == "node"
    else:
        with pytest.raises(environment.EnvironmentCheckError, match="Node.js"):
            environment.require_node(npm=True)


def test_node_requirement_follows_the_compiler_manifest(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    manifest = tmp_path / "marp/package.json"
    manifest.parent.mkdir()
    monkeypatch.setattr(environment, "__file__", str(tmp_path / "environment.py"))
    monkeypatch.setattr(environment, "require_command", lambda name, **_: name)
    monkeypatch.setattr(
        process_guard,
        "run_captured",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "v24.0.0", ""),
    )
    manifest.write_text(json.dumps({"engines": {"node": ">=24.1.0"}}))
    with pytest.raises(environment.EnvironmentCheckError, match="24.1.0"):
        environment.require_node()
    manifest.write_text(json.dumps({"engines": {"node": ">=22.17.0"}}))
    assert environment.require_node() == "node"
    manifest.write_text(json.dumps({"engines": {"node": "unsupported range"}}))
    with pytest.raises(environment.EnvironmentCheckError, match="minimum version"):
        environment.require_node()


def test_cli_reports_prerequisite_failure_without_traceback(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    def fail(profile: str) -> None:
        raise environment.EnvironmentCheckError("missing image")

    monkeypatch.setattr(environment, "check_environment", fail)
    monkeypatch.setattr(sys, "argv", ["environment", "full"])
    assert environment.main() == 1
    assert "Environment check failed: missing image" in capsys.readouterr().err


@pytest.mark.parametrize("failure", [None, "image", "capability", "daemon"])
def test_runtime_probe_uses_product_policy_without_starting_containers(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    failure: str | None,
) -> None:
    binary = tmp_path / "docker"
    commands = tmp_path / "commands.jsonl"
    binary.write_text(
        f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
        f"with Path({str(commands)!r}).open('a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        f"failure = {failure!r}\n"
        "args = sys.argv[3:]\n"
        "if args[0] == 'info':\n"
        "    if failure == 'daemon': raise SystemExit(1)\n"
        "    print(json.dumps({'ID':'fixture', 'OSType':'linux', 'CgroupVersion':'2',\n"
        "        'MemoryLimit': True, 'CpuCfsQuota': failure != 'capability',\n"
        "        'PidsLimit': True}))\n"
        "elif args[:2] == ['image', 'inspect']:\n"
        "    assert args[-1] == 'fixture:custom'\n"
        "    if failure == 'image': raise SystemExit(1)\n"
        "    print('sha256:' + 'a' * 64)\n"
        "else: raise AssertionError(args)\n"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("DOCKER_HOST", "unix:///fixture.sock")
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setenv("TRADINGDEV_STRATEGY_IMAGE", "fixture:custom")
    monkeypatch.delenv("TRADINGDEV_RUNTIME_LIMITS", raising=False)
    if failure:
        with pytest.raises(
            environment.EnvironmentCheckError, match="Strategy runtime is unavailable"
        ):
            environment.require_strategy_runtime()
    else:
        assert environment.require_strategy_runtime() == "sha256:" + "a" * 64
    actual = [json.loads(line)[2:] for line in commands.read_text().splitlines()]
    assert actual[0] == ["info", "--format", "{{json .}}"]
    assert all(command[0] in {"info", "image"} for command in actual)


@pytest.mark.parametrize(
    "action, package", [("format", "pymarkdownlnt"), ("check", "jsonschema")]
)
def test_quality_checks_all_tools_before_first_operation(
    monkeypatch: MonkeyPatch,
    action: str,
    package: str,
) -> None:
    from tooling import quality

    original = importlib.metadata.distribution

    def distribution(name: str) -> importlib.metadata.Distribution:
        if name == package:
            raise importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(quality, "run", lambda *_: pytest.fail("Formatter started"))
    monkeypatch.setattr(sys, "argv", ["quality", action])
    with pytest.raises(SystemExit) as failure:
        quality.main()
    assert failure.value.code == 2


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
def test_node_probe_cleans_descendants_on_every_outcome(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    outcome: str,
) -> None:
    child_code = "import time; time.sleep(60)"
    marker = tmp_path / "child.pid"
    observed = tmp_path / "observed"
    binary = tmp_path / "node"
    binary.write_text(
        f"#!{sys.executable}\nimport subprocess, sys, time\nfrom pathlib import Path\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}],\n"
        "    start_new_session=True)\n"
        f"marker = Path({str(marker)!r})\n"
        "marker.with_suffix('.tmp').write_text(str(child.pid))\n"
        "marker.with_suffix('.tmp').replace(marker)\n"
        f"while not Path({str(observed)!r}).exists(): time.sleep(0.01)\n"
        "print('v22.17.0', flush=True)\n"
        + (
            "time.sleep(60)\n"
            if outcome == "timeout"
            else f"time.sleep(1)\nraise SystemExit({int(outcome == 'failure')})\n"
        )
    )
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    real_children = psutil.Process.children
    clock_offset = 0.0

    def children(
        process: psutil.Process, recursive: bool = False
    ) -> list[psutil.Process]:
        nonlocal clock_offset
        found: list[psutil.Process] = real_children(process, recursive=recursive)
        if marker.exists() and int(marker.read_text()) in {
            child.pid for child in found
        }:
            observed.touch()
            if outcome == "timeout":
                clock_offset = 10.0
        return found

    monkeypatch.setattr(psutil.Process, "children", children)
    if outcome == "timeout":
        # Expire only after the real descendant is known. A constant offset keeps
        # cleanup deadlines and sleeps on real time without patching global time.
        monkeypatch.setattr(
            process_guard,
            "time",
            SimpleNamespace(
                monotonic=lambda: time.monotonic() + clock_offset,
                sleep=time.sleep,
            ),
        )
    try:
        if outcome == "success":
            assert environment.require_node() == str(binary)
        else:
            with pytest.raises(
                environment.EnvironmentCheckError, match="Cannot verify Node.js"
            ) as error:
                environment.require_node()
            cause = error.value.__cause__
            if outcome == "failure":
                assert isinstance(cause, subprocess.CalledProcessError)
                assert cause.returncode == 1
            else:
                assert isinstance(cause, subprocess.TimeoutExpired)
                assert cause.timeout == 10
        assert observed.exists()
        with suppress(psutil.NoSuchProcess):
            assert (
                psutil.Process(int(marker.read_text())).status() == psutil.STATUS_ZOMBIE
            )
    finally:
        if marker.exists():
            with suppress(psutil.NoSuchProcess, psutil.ZombieProcess):
                child = psutil.Process(int(marker.read_text()))
                if child_code in child.cmdline():
                    child.kill()
                    child.wait(timeout=5)


@pytest.mark.parametrize("action", ["setup", "check", "watch", "package"])
def test_presentation_prerequisites_fail_before_installation_or_document_checks(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    action: str,
) -> None:
    from tooling import documentation, presentations

    def missing(**_: object) -> str:
        raise environment.EnvironmentCheckError("Node.js unavailable")

    monkeypatch.setattr(presentations, "require_node", missing)
    monkeypatch.setattr(presentations, "_toolchain", lambda root: root)
    monkeypatch.setattr(
        presentations, "_execute", lambda *_, **__: pytest.fail("npm started")
    )
    monkeypatch.setattr(
        documentation, "check_documents", lambda *_: pytest.fail("Documents started")
    )
    if action == "package":
        monkeypatch.setattr(
            presentations, "__file__", str(tmp_path / "tooling/presentations.py")
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["presentations", "package", "--output", str(tmp_path / "bundle")],
        )
        assert presentations.main() == 1
        assert "Node.js unavailable" in capsys.readouterr().err
        assert not list(tmp_path.iterdir())
        return
    with pytest.raises(RuntimeError, match="Node.js unavailable"):
        if action == "setup":
            presentations.setup(tmp_path)
        elif action == "watch":
            presentations.watch(tmp_path)
        else:
            presentations.check_presentations(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("exit_code", [0, 1])
def test_codex_probe_only_reads_login_and_cleans_its_workspace(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    exit_code: int,
) -> None:
    binary = tmp_path / "codex"
    record = tmp_path / "probe.json"
    binary.write_text(
        f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
        f"Path({str(record)!r}).write_text(json.dumps(\n"
        "    {'args': sys.argv[1:], 'cwd': str(Path.cwd()),\n"
        "     'cache': os.environ['XDG_CACHE_HOME']}))\n"
        "print('credential-like diagnostic must not be forwarded')\n"
        f"raise SystemExit({exit_code})\n"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("TRADINGDEV_CODEX_BIN", str(binary))
    if exit_code:
        with pytest.raises(
            environment.EnvironmentCheckError, match="not logged in"
        ) as failure:
            environment.require_codex()
        assert "credential-like" not in str(failure.value)
    else:
        assert environment.require_codex() == str(binary)
    observed = json.loads(record.read_text())
    assert observed["args"] == ["login", "status"]
    assert not Path(observed["cwd"]).exists()
    assert not Path(observed["cache"]).exists()


def test_local_probe_checks_connectivity_without_requesting_a_model() -> None:
    with socket.socket() as listener:
        listener.settimeout(2)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        environment.require_local_service(
            f"http://127.0.0.1:{listener.getsockname()[1]}/v1"
        )
        connection, _ = listener.accept()
        with connection:
            assert connection.recv(1) == b""


def test_unreachable_local_service_reports_a_bounded_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    def unreachable(address: tuple[str, int], *, timeout: float) -> None:
        assert address == ("::1", 1234)
        assert timeout == 5
        raise ConnectionRefusedError

    monkeypatch.setattr(socket, "create_connection", unreachable)
    with pytest.raises(
        environment.EnvironmentCheckError, match="Local model service is unreachable"
    ):
        environment.require_local_service("http://[::1]:1234/v1")
