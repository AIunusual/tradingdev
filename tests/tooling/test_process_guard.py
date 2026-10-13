"""Check-runner cleanup includes children that create independent sessions."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import psutil
import pytest
from tooling.process_guard import run_captured, run_checked

if TYPE_CHECKING:
    from pytest import MonkeyPatch

_CHILD = "import time; time.sleep(60)"
_SPAWN = """\
import signal
import subprocess
import sys
import time
import os
from pathlib import Path

child = subprocess.Popen([sys.executable, '-c', sys.argv[1]], start_new_session=True)
Path('child.pid').write_text(str(child.pid))
Path('leader.pid').write_text(str(os.getpid()))
"""


def _assert_stopped(root: Path) -> None:
    for name in ("leader.pid", "child.pid"):
        pid = int((root / name).read_text())
        try:
            process = psutil.Process(pid)
            assert process.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass


def _emergency_cleanup(root: Path) -> None:
    for name in ("leader.pid", "child.pid"):
        marker = root / name
        if marker.exists():
            with suppress(psutil.NoSuchProcess):
                process = psutil.Process(int(marker.read_text()))
                if _CHILD in process.cmdline():
                    process.kill()


@pytest.mark.parametrize("captured", [False, True])
def test_timeout_stops_independent_child_and_allows_leader_teardown(
    captured: bool,
) -> None:
    with TemporaryDirectory(prefix="tradingdev-process-test-") as directory:
        root = Path(directory)
        code = (
            _SPAWN
            + """\
print('started', flush=True)
print('diagnostic', file=sys.stderr, flush=True)
try:
    time.sleep(60)
except KeyboardInterrupt:
    Path('teardown.completed').write_text('done')
"""
        )
        try:
            with pytest.raises(subprocess.TimeoutExpired, match="timed out") as failure:
                runner = run_captured if captured else run_checked
                runner(
                    [sys.executable, "-c", code, _CHILD],
                    cwd=root,
                    env=dict(os.environ),
                    timeout=0.5,
                )
            if captured:
                assert failure.value.output == b"started\n"
                assert failure.value.stderr == b"diagnostic\n"
            assert (root / "teardown.completed").read_text() == "done"
            _assert_stopped(root)
        finally:
            _emergency_cleanup(root)
    assert not root.exists()


@pytest.mark.parametrize("returncode", [0, 7])
def test_captured_output_preserves_diagnostics_without_pipe_blocking(
    returncode: int,
) -> None:
    with TemporaryDirectory(prefix="tradingdev-process-test-") as directory:
        root = Path(directory)
        command = [
            sys.executable,
            "-c",
            "import sys; "
            "sys.stdout.write('out' * 100000); "
            "sys.stderr.write('err' * 100000); "
            f"sys.exit({returncode})",
        ]
        result = run_captured(command, cwd=root, env=dict(os.environ), timeout=5)
        assert result.args == command
        assert result.returncode == returncode
        assert result.stdout == "out" * 100000
        assert result.stderr == "err" * 100000
        assert list(root.iterdir()) == []
    assert not root.exists()


@pytest.mark.parametrize("returncode", [0, 7])
@pytest.mark.parametrize("captured", [False, True])
def test_leader_exit_cleans_child_without_signalling_exited_group(
    monkeypatch: MonkeyPatch,
    returncode: int,
    captured: bool,
) -> None:
    def reject_group(pid: int, sig: int) -> None:
        pytest.fail(f"Signalled exited leader group {pid} with {sig}")

    monkeypatch.setattr(os, "killpg", reject_group)
    with TemporaryDirectory(prefix="tradingdev-process-test-") as directory:
        root = Path(directory)
        original_children = psutil.Process.children

        def observe_children(
            process: psutil.Process, recursive: bool = False
        ) -> list[psutil.Process]:
            children: list[psutil.Process] = original_children(
                process, recursive=recursive
            )
            marker = root / "child.pid"
            if marker.exists() and int(marker.read_text()) in {
                child.pid for child in children
            }:
                (root / "observed").touch()
            return children

        monkeypatch.setattr(psutil.Process, "children", observe_children)
        command = [
            sys.executable,
            "-c",
            _SPAWN
            + "while not Path('observed').exists(): time.sleep(0.01)\n"
            + f"raise SystemExit({returncode})\n",
            _CHILD,
        ]
        try:
            if captured:
                result = run_captured(
                    command, cwd=root, env=dict(os.environ), timeout=5
                )
                assert result.returncode == returncode
            elif returncode:
                with pytest.raises(subprocess.CalledProcessError) as failure:
                    run_checked(command, cwd=root, env=dict(os.environ), timeout=5)
                assert failure.value.returncode == returncode
            else:
                run_checked(command, cwd=root, env=dict(os.environ), timeout=5)
            _assert_stopped(root)
        finally:
            _emergency_cleanup(root)
    assert not root.exists()


def test_group_permission_denied_falls_back_to_individual_signals(
    monkeypatch: MonkeyPatch,
) -> None:
    attempted: list[int] = []

    def deny_group(pid: int, sig: int) -> None:
        attempted.append(sig)
        raise PermissionError("group signalling denied")

    monkeypatch.setattr(os, "killpg", deny_group)
    with TemporaryDirectory(prefix="tradingdev-process-test-") as directory:
        root = Path(directory)
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                run_checked(
                    [sys.executable, "-c", _SPAWN + "time.sleep(60)", _CHILD],
                    cwd=root,
                    env=dict(os.environ),
                    timeout=0.5,
                )
            assert signal.SIGINT in attempted
            _assert_stopped(root)
        finally:
            _emergency_cleanup(root)
    assert not root.exists()


def test_process_inspection_denied_prevents_launch(monkeypatch: MonkeyPatch) -> None:
    def deny_inspection(process: psutil.Process, recursive: bool = False) -> None:
        raise psutil.AccessDenied(process.pid)

    def reject_launch(*args: object, **kwargs: object) -> None:
        pytest.fail("A check was launched without access to track its children")

    monkeypatch.setattr(psutil.Process, "children", deny_inspection)
    monkeypatch.setattr(subprocess, "Popen", reject_launch)
    with TemporaryDirectory(prefix="tradingdev-process-test-") as directory:
        root = Path(directory)
        with pytest.raises(RuntimeError, match="inspection access.*before"):
            run_checked(
                [sys.executable, "-c", "raise SystemExit(0)"],
                cwd=root,
                env=dict(os.environ),
                timeout=1,
            )
    assert not root.exists()
