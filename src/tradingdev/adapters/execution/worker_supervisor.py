"""Own a worker's process group until every active member has stopped.

Only the direct parent signals the group. It keeps the worker unreaped, including
after natural exit, so the worker PID cannot become another process group's ID.
External clients request cancellation through a launch-specific control folder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

if TYPE_CHECKING:
    from types import FrameType

_POLL_SECONDS = 0.05
_START_TIMEOUT_SECONDS = 10.0
_STOP_GRACE_SECONDS = 0.25
_STOP_TIMEOUT_SECONDS = 5.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish complete protocol messages, never partially written JSON."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def _describe_failure(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        return "".join(traceback.format_exception(error)).rstrip()
    return f"{type(error).__name__}: {error}"


def _child_exited(proc: subprocess.Popen[bytes]) -> bool:
    """Observe the owned child without freeing its PID for reuse."""
    return (
        os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    )


def _group_is_alive(
    group_id: int, observed: dict[int, psutil.Process] | None = None
) -> bool:
    """Observe group members; never send signals to enumerated numeric PIDs."""
    members = observed if observed is not None else {}
    for process in psutil.process_iter():
        try:
            if os.getpgid(process.pid) == group_id:
                members[process.pid] = process
        except ProcessLookupError, psutil.NoSuchProcess:
            continue
    alive = False
    for pid, process in list(members.items()):
        try:
            active = process.is_running() and process.status() not in {
                psutil.STATUS_ZOMBIE,
                psutil.STATUS_DEAD,
            }
        except psutil.NoSuchProcess:
            active = False
        if active:
            alive = True
        else:
            members.pop(pid)
    return alive


def _signal_owned_group(
    proc: subprocess.Popen[bytes],
    sig: signal.Signals,
    observed: dict[int, psutil.Process],
    *,
    deadline: float,
) -> None:
    # No Popen.poll(), wait(), or psutil.wait(): the unreaped child is the anchor.
    # The leader and every descendant may already have exited.
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Darwin can stop exposing exiting members to killpg before the leader
        # becomes waitable. Keep the unreaped anchor and allow that transition
        # to finish; permission denial alone never proves cleanup succeeded.
        while True:
            if _child_exited(proc) and not _group_is_alive(proc.pid, observed):
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(_POLL_SECONDS, remaining))


def _wait_for_group(
    group_id: int, timeout: float, observed: dict[int, psutil.Process]
) -> bool:
    deadline = time.monotonic() + timeout
    while _group_is_alive(group_id, observed):
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_SECONDS)
    return True


def terminate_owned_group(proc: subprocess.Popen[bytes]) -> int:
    """Stop and reap a direct session-leader child that has never been reaped.

    The caller must be the sole waiter and must not previously call poll/wait.
    Keeping that child unreaped binds killpg to this launch, even if its leader
    has already exited. Acknowledgement requires every group member to be gone
    or a zombie, rather than merely observing the leader exit.
    """
    if proc.returncode is not None:
        raise RuntimeError("Cannot signal a process group after its child was reaped")
    _child_exited(proc)  # Also rejects children already reaped by another waiter.
    # macOS hides exited leaders from getpgid before wait/reap. The successful
    # waitid above still proves ownership of the unreaped child in that case.
    with suppress(ProcessLookupError):
        if os.getpgid(proc.pid) != proc.pid:
            raise RuntimeError("Owned child is not its process group leader")

    failure: BaseException | None = None
    observed: dict[int, psutil.Process] = {}
    signal_deadline = time.monotonic() + _STOP_TIMEOUT_SECONDS
    try:
        _group_is_alive(proc.pid, observed)
        _signal_owned_group(proc, signal.SIGTERM, observed, deadline=signal_deadline)
        _wait_for_group(proc.pid, _STOP_GRACE_SECONDS, observed)
        # An enumerated parent can exit just after creating a new child. Signal
        # the still-owned entire group even when the first inventory was empty.
        _signal_owned_group(proc, signal.SIGKILL, observed, deadline=signal_deadline)
        if not _wait_for_group(proc.pid, _STOP_TIMEOUT_SECONDS, observed):
            raise TimeoutError("Worker process group did not stop")
    except BaseException as error:
        failure = error
        # Make the strongest safe attempt, but retain the anchor on failure.
        # Failure to inspect descendants must not leave them running silently.
        try:
            _signal_owned_group(
                proc, signal.SIGKILL, observed, deadline=signal_deadline
            )
        except BaseException as kill_error:
            failure = BaseExceptionGroup(
                "Worker process group cleanup failed", [error, kill_error]
            )
    if failure is not None:
        raise failure
    return proc.wait(timeout=_STOP_TIMEOUT_SECONDS)


class _StopRequest:
    requested = False

    def receive_signal(self, _signal: int, _frame: FrameType | None) -> None:
        self.requested = True


def _wait_for_start(control: Path, stop: _StopRequest) -> dict[str, Any]:
    deadline = time.monotonic() + _START_TIMEOUT_SECONDS
    while not (control / "start.json").exists():
        if stop.requested or (control / "stop").exists():
            raise RuntimeError("Worker startup cancelled")
        if time.monotonic() >= deadline:
            raise TimeoutError("Worker startup handshake timed out")
        time.sleep(_POLL_SECONDS)
    payload: Any = json.loads((control / "start.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Invalid worker startup identity")
    creation_time = payload.get("process_create_time")
    if (
        payload.get("pid") != os.getpid()
        or isinstance(payload.get("pid"), bool)
        or not isinstance(creation_time, int | float)
        or isinstance(creation_time, bool)
        or not math.isfinite(creation_time)
        or creation_time != psutil.Process(os.getpid()).create_time()
        or payload.get("worker_control_id") != control.name
        or re.fullmatch(r"[0-9a-f]{32}", control.name) is None
    ):
        raise ValueError("Invalid worker startup identity")
    return payload


def supervise(control: Path, module: str, args: list[str]) -> None:
    """Run one module, acknowledge startup, and publish verified cleanup."""
    stop = _StopRequest()
    signal.signal(signal.SIGTERM, stop.receive_signal)
    signal.signal(signal.SIGINT, stop.receive_signal)
    child: subprocess.Popen[bytes] | None = None
    ready = False
    failure: str | None = None
    returncode: int | None = None
    cleaned = True
    try:
        identity = _wait_for_start(control, stop)
        if stop.requested or (control / "stop").exists():
            raise RuntimeError("Worker startup cancelled")
        environment = os.environ.copy()
        environment["TRADINGDEV_WORKER_IDENTITY"] = json.dumps(identity)
        environment["TRADINGDEV_RUNTIME_CONTROL_ROOT"] = str(control / "runtime")
        child = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", module, *args],
            env=environment,
            start_new_session=True,
        )
        _write_json(control / "ready.json", {"worker_pid": child.pid})
        ready = True
        while not stop.requested and not (control / "stop").exists():
            if _child_exited(child):
                break
            time.sleep(_POLL_SECONDS)
    except BaseException as error:
        failure = _describe_failure(error)
    finally:
        if child is not None:
            try:
                returncode = terminate_owned_group(child)
            except BaseException as error:
                cleaned = False
                prefix = f"{failure}; " if failure else ""
                failure = f"{prefix}{_describe_failure(error)}"
        runtime_root = control / "runtime"
        if any(runtime_root.glob("*.json")):
            try:
                # The host worker and attach clients are stopped before leases
                # are recovered. Container IDs remain valid across MCP sessions.
                from tradingdev.adapters.execution.container_runtime import (
                    DockerRuntime,
                )

                DockerRuntime(runtime_root).recover_abandoned()
                if any(runtime_root.glob("*.json")):
                    raise RuntimeError(
                        "Execution-unit cleanup has active ownership leases"
                    )
            except BaseException as error:
                cleaned = False
                prefix = f"{failure}; " if failure else ""
                failure = f"{prefix}{_describe_failure(error)}"
        if not ready:
            _write_json(control / "ready.json", {"error": failure})
        _write_json(
            control / "finished.json",
            {"cleaned": cleaned, "returncode": returncode, "error": failure},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control_dir", type=Path)
    parser.add_argument("module")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    supervise(arguments.control_dir, arguments.module, arguments.args)


if __name__ == "__main__":
    main()
