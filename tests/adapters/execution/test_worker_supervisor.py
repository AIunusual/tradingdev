"""Regressions for the owned, unreaped process-group anchor and protocol."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock
from uuid import uuid4

import psutil
import pytest

from tradingdev.adapters.execution import worker_supervisor as supervisor

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def owned_child(monkeypatch: pytest.MonkeyPatch) -> Mock:
    child = Mock(spec=subprocess.Popen)
    child.pid = 4321
    child.returncode = None
    child.wait.return_value = -signal.SIGTERM
    monkeypatch.setattr(os, "waitid", lambda *_args: None)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(supervisor, "_group_is_alive", lambda *_args: False)
    return child


def test_cleanup_holds_leader_until_all_descendants_exit(
    monkeypatch: pytest.MonkeyPatch, owned_child: Mock
) -> None:
    events: list[str] = []

    def wait_for_group(
        _group_id: int, _timeout: float, _observed: dict[int, psutil.Process]
    ) -> bool:
        owned_child.wait.assert_not_called()
        events.append("group-observed")
        return events.count("group-observed") == 2

    def signal_group(pid: int, sig: signal.Signals) -> None:
        assert pid == owned_child.pid
        owned_child.poll.assert_not_called()
        owned_child.wait.assert_not_called()
        events.append(sig.name)

    monkeypatch.setattr(os, "killpg", signal_group)
    monkeypatch.setattr(supervisor, "_wait_for_group", wait_for_group)

    def reap(**_kwargs: object) -> int:
        events.append("reaped")
        return 0

    owned_child.wait.side_effect = reap

    assert supervisor.terminate_owned_group(owned_child) == 0
    assert events == [
        "SIGTERM",
        "group-observed",
        "SIGKILL",
        "group-observed",
        "reaped",
    ]


def test_natural_leader_exit_still_cleans_descendants_before_reaping(
    monkeypatch: pytest.MonkeyPatch, owned_child: Mock
) -> None:
    waitid = Mock(return_value=object())
    monkeypatch.setattr(os, "waitid", waitid)
    killpg = Mock()
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(supervisor, "_wait_for_group", lambda *_args: True)

    supervisor.terminate_owned_group(owned_child)

    waitid.assert_called_once_with(os.P_PID, 4321, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    assert [call.args for call in killpg.call_args_list] == [
        (4321, signal.SIGTERM),
        (4321, signal.SIGKILL),
    ]
    owned_child.wait.assert_called_once()
    owned_child.poll.assert_not_called()


def test_cleanup_accepts_an_already_exited_group(
    monkeypatch: pytest.MonkeyPatch, owned_child: Mock
) -> None:
    monkeypatch.setattr(os, "killpg", Mock(side_effect=ProcessLookupError))
    monkeypatch.setattr(supervisor, "_wait_for_group", lambda *_args: True)

    assert supervisor.terminate_owned_group(owned_child) == -signal.SIGTERM
    owned_child.wait.assert_called_once()


@pytest.mark.parametrize("pending", ["leader", "descendant"])
def test_permission_error_waits_for_verified_exit_before_reaping(
    monkeypatch: pytest.MonkeyPatch, owned_child: Mock, pending: str
) -> None:
    elapsed = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(
        time, "sleep", lambda delay: elapsed.__setitem__(0, elapsed[0] + delay)
    )
    monkeypatch.setattr(
        supervisor,
        "_child_exited",
        lambda _proc: pending != "leader" or elapsed[0] >= 0.1,
    )
    monkeypatch.setattr(
        supervisor,
        "_group_is_alive",
        lambda *_args: pending == "descendant" and elapsed[0] < 0.1,
    )
    monkeypatch.setattr(os, "killpg", Mock(side_effect=PermissionError("exiting")))

    def reap(**_kwargs: object) -> int:
        assert elapsed[0] >= 0.1
        return 0

    owned_child.wait.side_effect = reap

    assert supervisor.terminate_owned_group(owned_child) == 0
    owned_child.wait.assert_called_once()
    owned_child.poll.assert_not_called()


@pytest.mark.parametrize(
    ("leader_exited", "group_alive", "accepted"),
    [(True, False, True), (True, True, False), (False, False, False)],
)
def test_permission_error_only_means_exited_when_entire_owned_group_has_stopped(
    monkeypatch: pytest.MonkeyPatch,
    owned_child: Mock,
    leader_exited: bool,
    group_alive: bool,
    accepted: bool,
) -> None:
    elapsed = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(
        time, "sleep", lambda delay: elapsed.__setitem__(0, elapsed[0] + delay)
    )
    monkeypatch.setattr(supervisor, "_child_exited", lambda _proc: leader_exited)
    monkeypatch.setattr(supervisor, "_group_is_alive", lambda *_args: group_alive)
    monkeypatch.setattr(supervisor, "_wait_for_group", lambda *_args: True)
    monkeypatch.setattr(os, "killpg", Mock(side_effect=PermissionError("denied")))

    if accepted:
        assert supervisor.terminate_owned_group(owned_child) == -signal.SIGTERM
        owned_child.wait.assert_called_once()
    else:
        with pytest.raises(BaseExceptionGroup):
            supervisor.terminate_owned_group(owned_child)
        owned_child.wait.assert_not_called()
        assert elapsed[0] <= supervisor._STOP_TIMEOUT_SECONDS


@pytest.mark.parametrize("reaped_by_popen", [True, False])
def test_reaped_group_is_never_signalled(
    monkeypatch: pytest.MonkeyPatch, owned_child: Mock, reaped_by_popen: bool
) -> None:
    if reaped_by_popen:
        owned_child.returncode = 0
    else:
        monkeypatch.setattr(os, "waitid", Mock(side_effect=ChildProcessError))
    killpg = Mock()
    monkeypatch.setattr(os, "killpg", killpg)

    with pytest.raises((RuntimeError, ChildProcessError)):
        supervisor.terminate_owned_group(owned_child)

    killpg.assert_not_called()
    owned_child.wait.assert_not_called()


@pytest.mark.parametrize(
    "error", [PermissionError("inspection denied"), TimeoutError()]
)
def test_failed_cleanup_does_not_reap_or_claim_success(
    monkeypatch: pytest.MonkeyPatch, owned_child: Mock, error: Exception
) -> None:
    killpg = Mock()
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(supervisor, "_wait_for_group", Mock(side_effect=error))

    with pytest.raises(type(error)):
        supervisor.terminate_owned_group(owned_child)

    assert killpg.call_args_list[0].args == (4321, signal.SIGTERM)
    assert killpg.call_args_list[-1].args == (4321, signal.SIGKILL)
    owned_child.wait.assert_not_called()


def test_group_inspection_ignores_other_groups_and_exited_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unrelated = Mock(pid=22)
    unrelated.status.side_effect = psutil.AccessDenied(22)
    zombie = Mock(pid=33)
    zombie.status.return_value = psutil.STATUS_ZOMBIE
    exited = Mock(pid=44)
    exited.status.side_effect = psutil.NoSuchProcess(44)
    monkeypatch.setattr(psutil, "process_iter", lambda: [unrelated, zombie, exited])
    monkeypatch.setattr(os, "getpgid", lambda pid: 99 if pid == 22 else 4321)

    assert supervisor._group_is_alive(4321) is False
    unrelated.status.assert_not_called()


def test_group_inspection_fails_when_process_inventory_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(psutil, "process_iter", Mock(side_effect=PermissionError))
    with pytest.raises(PermissionError):
        supervisor._group_is_alive(4321)


@pytest.fixture
def control(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / uuid4().hex
    directory.mkdir()
    monkeypatch.setattr(psutil, "Process", lambda _pid: Mock(create_time=lambda: 100.0))
    return directory


def _start(control: Path, **changes: Any) -> dict[str, Any]:
    identity = {
        "pid": os.getpid(),
        "process_create_time": 100.0,
        "worker_control_id": control.name,
        **changes,
    }
    (control / "start.json").write_text(json.dumps(identity), encoding="utf-8")
    return identity


def test_start_handshake_passes_supervisor_identity_to_worker(
    control: Path, monkeypatch: pytest.MonkeyPatch, owned_child: Mock
) -> None:
    identity = _start(control)
    popen = Mock(return_value=owned_child)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(supervisor, "_child_exited", lambda _child: True)
    cleanup = Mock(return_value=0)
    monkeypatch.setattr(supervisor, "terminate_owned_group", cleanup)
    with monkeypatch.context() as context:
        context.setattr(signal, "signal", Mock())
        supervisor.supervise(control, "test.worker", ["job-1"])

    assert popen.call_args.args[0][-3:] == ["-m", "test.worker", "job-1"]
    assert popen.call_args.kwargs["start_new_session"] is True
    assert (
        json.loads(popen.call_args.kwargs["env"]["TRADINGDEV_WORKER_IDENTITY"])
        == identity
    )
    assert json.loads((control / "ready.json").read_text()) == {"worker_pid": 4321}
    assert json.loads((control / "finished.json").read_text()) == {
        "cleaned": True,
        "returncode": 0,
        "error": None,
    }
    cleanup.assert_called_once_with(owned_child)


@pytest.mark.parametrize(
    "changes",
    [
        {"pid": -1},
        {"process_create_time": 200.0},
        {"worker_control_id": "different-launch"},
    ],
)
def test_invalid_handshake_never_starts_worker(
    control: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, Any]
) -> None:
    _start(control, **changes)
    popen = Mock()
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(signal, "signal", Mock())

    supervisor.supervise(control, "test.worker", [])

    popen.assert_not_called()
    assert (
        "Invalid worker startup identity"
        in json.loads((control / "ready.json").read_text())["error"]
    )


@pytest.mark.parametrize("cancelled", [False, True])
def test_missing_start_handshake_cannot_leave_a_worker_running(
    control: Path, monkeypatch: pytest.MonkeyPatch, cancelled: bool
) -> None:
    if cancelled:
        (control / "stop").touch()
    monkeypatch.setattr(supervisor, "_START_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(signal, "signal", Mock())
    popen = Mock()
    monkeypatch.setattr(subprocess, "Popen", popen)

    supervisor.supervise(control, "test.worker", [])

    popen.assert_not_called()
    finished = json.loads((control / "finished.json").read_text())
    assert finished["cleaned"] is True
    assert ("cancelled" if cancelled else "timed out") in finished["error"]


def test_supervisor_reports_cleanup_failure_without_success_ack(
    control: Path, monkeypatch: pytest.MonkeyPatch, owned_child: Mock
) -> None:
    _start(control)
    monkeypatch.setattr(subprocess, "Popen", Mock(return_value=owned_child))
    monkeypatch.setattr(signal, "signal", Mock())
    monkeypatch.setattr(supervisor, "_child_exited", lambda _child: True)
    monkeypatch.setattr(
        supervisor, "terminate_owned_group", Mock(side_effect=PermissionError("denied"))
    )

    supervisor.supervise(control, "test.worker", [])

    finished = json.loads((control / "finished.json").read_text())
    assert finished == {
        "cleaned": False,
        "returncode": None,
        "error": "PermissionError: denied",
    }


def test_supervisor_preserves_nested_cleanup_failure_reasons(
    control: Path, monkeypatch: pytest.MonkeyPatch, owned_child: Mock
) -> None:
    _start(control)
    monkeypatch.setattr(subprocess, "Popen", Mock(return_value=owned_child))
    monkeypatch.setattr(signal, "signal", Mock())
    monkeypatch.setattr(supervisor, "_child_exited", lambda _child: True)
    failure = ExceptionGroup(
        "Worker process group cleanup failed",
        [
            PermissionError("process inventory denied"),
            ExceptionGroup("kill failed", [TimeoutError("descendant still active")]),
        ],
    )
    monkeypatch.setattr(supervisor, "terminate_owned_group", Mock(side_effect=failure))

    supervisor.supervise(control, "test.worker", [])

    finished = json.loads((control / "finished.json").read_text())
    assert finished["cleaned"] is False
    assert finished["returncode"] is None
    assert "PermissionError: process inventory denied" in finished["error"]
    assert "TimeoutError: descendant still active" in finished["error"]


@pytest.mark.parametrize("leader_exits", [False, True])
def test_real_owned_group_cleanup_waits_for_descendants(
    tmp_path: Path, leader_exits: bool
) -> None:
    descendant_marker = tmp_path / "descendant.pid"
    ready = tmp_path / "ready"
    descendant_script = (
        "import os, pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(descendant_marker)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    leader_script = (
        "import os, pathlib, signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"subprocess.Popen([sys.executable, '-c', {descendant_script!r}])\n"
        f"while not pathlib.Path({str(descendant_marker)!r}).exists():\n"
        "    time.sleep(0.01)\n"
        f"pathlib.Path({str(ready)!r}).touch()\n"
        + ("os._exit(0)\n" if leader_exits else "time.sleep(60)\n")
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", leader_script], start_new_session=True
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() or (
            leader_exits and not supervisor._child_exited(leader)
        ):
            if time.monotonic() >= deadline:
                pytest.fail("Worker descendant did not start")
            time.sleep(0.01)
        descendant = psutil.Process(int(descendant_marker.read_text()))

        returncode = supervisor.terminate_owned_group(leader)

        assert returncode == (0 if leader_exits else -signal.SIGKILL)
        assert leader.returncode is not None
        with suppress(psutil.NoSuchProcess):
            assert descendant.status() in {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD}
    finally:
        if leader.returncode is None:
            supervisor.terminate_owned_group(leader)
