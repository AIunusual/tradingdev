"""Subprocess execution adapter."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import psutil

from tradingdev.adapters.execution.worker_supervisor import terminate_owned_group
from tradingdev.adapters.storage.filesystem import WorkspacePaths

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class ProcessIdentity:
    """Persisted identity of one process, distinct from a reusable PID."""

    pid: int
    create_time: float

    @classmethod
    def capture(cls, pid: int) -> ProcessIdentity:
        """Capture creation time while the caller still owns the process."""
        return cls(pid, float(psutil.Process(pid).create_time()))

    @classmethod
    def from_values(cls, pid: object, create_time: object) -> ProcessIdentity | None:
        """Reject absent or invalid identities without guessing from a PID."""
        if (
            not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 0
            or not isinstance(create_time, int | float)
            or isinstance(create_time, bool)
            or not math.isfinite(create_time)
            or create_time <= 0
        ):
            return None
        return cls(pid, float(create_time))

    def get_process(self) -> psutil.Process | None:
        """Resolve the original live process; permission failures propagate."""
        try:
            process = psutil.Process(self.pid)
            if (
                process.create_time() != self.create_time
                or not process.is_running()
                or process.status() == psutil.STATUS_ZOMBIE
            ):
                return None
            return process
        except psutil.NoSuchProcess:
            return None


@dataclass(frozen=True)
class WorkerHandle(ProcessIdentity):
    """Supervisor identity and a unique, non-reusable launch control channel."""

    control_id: str

    def job_fields(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "process_create_time": self.create_time,
            "worker_control_id": self.control_id,
        }

    @classmethod
    def from_job(cls, job: Mapping[str, Any]) -> WorkerHandle | None:
        identity = ProcessIdentity.from_values(
            job.get("pid"), job.get("process_create_time")
        )
        control_id = job.get("worker_control_id")
        if (
            identity is None
            or not isinstance(control_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", control_id) is None
        ):
            return None
        return cls(identity.pid, identity.create_time, control_id)

    @classmethod
    def from_environment(cls) -> WorkerHandle:
        handle = cls.from_job(json.loads(os.environ["TRADINGDEV_WORKER_IDENTITY"]))
        if handle is None:
            raise ValueError("Worker must be launched by its supervisor")
        return handle


class BoundedWorkerError(RuntimeError):
    """A bounded worker failed, with explicit evidence about process cleanup."""

    def __init__(self, code: str, message: str, *, cleanup_verified: bool) -> None:
        super().__init__(message)
        self.code = code
        self.cleanup_verified = cleanup_verified


def _read_control(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Invalid worker control record: {path}")
    return value


def _wait_for_control(path: Path, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while (value := _read_control(path)) is None:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Worker supervisor did not respond: {path}")
        time.sleep(0.05)
    return value


def request_worker_stop(
    workspace_root: Path, handle: WorkerHandle, *, timeout: float = 10.0
) -> bool:
    """Request cleanup through a launch token, never signal a numeric PID.

    Finished records are retained so later sessions can verify cleanup even if
    the supervisor PID has since been reused. Missing evidence fails closed.
    """
    if WorkerHandle.from_job(handle.job_fields()) != handle:
        raise ValueError("Invalid worker control identity")
    deadline = time.monotonic() + timeout
    directory = workspace_root / ".workers" / handle.control_id
    if _read_control(directory / "start.json") != handle.job_fields():
        raise RuntimeError(
            f"Worker control identity missing or mismatched: {directory}"
        )
    result = _read_control(directory / "finished.json")
    requested = result is None
    if requested:
        # Do not mkdir here: a removed launch must never be resurrected.
        (directory / "stop").touch(exist_ok=True)
        result = _wait_for_control(directory / "finished.json", timeout)
    assert result is not None
    if result.get("cleaned") is not True:
        raise RuntimeError(f"Worker cleanup failed: {result.get('error', result)}")
    # The acknowledgement proves worker-group cleanup, but is published just
    # before the supervisor exits. Observe that identity until it is inactive
    # before callers delete its workspace. Reused PIDs are never signalled.
    try:
        while handle.get_process() is not None:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Worker supervisor did not exit after cleanup: {directory}"
                )
            time.sleep(0.05)
    except psutil.Error as error:
        raise RuntimeError(f"Cannot verify supervisor exit: {error}") from error
    return requested


class ProcessRunner:
    """Spawn detached Python module subprocesses."""

    def __init__(
        self,
        project_root: Path | None = None,
        *,
        workspace: WorkspacePaths | None = None,
        env_overrides: Mapping[str, str] | None = None,
    ) -> None:
        configured_root = os.environ.get("TRADINGDEV_PROJECT_ROOT")
        self._project_root = (
            (project_root or (Path(configured_root) if configured_root else Path.cwd()))
            .expanduser()
            .resolve()
        )
        self._env = os.environ.copy()
        if env_overrides is not None:
            self._env.update(env_overrides)
        self._workspace = workspace or WorkspacePaths()
        self._env["TRADINGDEV_WORKSPACE"] = str(self._workspace.root)
        if data_root := self._env.get("TRADINGDEV_DATA_ROOT"):
            self._env["TRADINGDEV_DATA_ROOT"] = str(
                Path(data_root).expanduser().resolve()
            )

    def spawn_module(self, module: str, *args: str) -> WorkerHandle:
        """Start a supervisor; release the worker only after identity capture."""
        if os.name != "posix" or not hasattr(os, "WNOWAIT"):
            raise RuntimeError("Background workers require POSIX waitid/WNOWAIT")
        control_id = uuid4().hex
        directory = self._workspace.root / ".workers" / control_id
        directory.mkdir(parents=True)
        try:
            proc = subprocess.Popen(  # noqa: S603
                [
                    sys.executable,
                    "-m",
                    "tradingdev.adapters.execution.worker_supervisor",
                    str(directory),
                    module,
                    *args,
                ],
                cwd=str(self._project_root),
                env=self._env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except BaseException:
            shutil.rmtree(directory)
            raise

        try:
            identity = ProcessIdentity.capture(proc.pid)
            handle = WorkerHandle(identity.pid, identity.create_time, control_id)
            temporary = directory / "start.tmp"
            temporary.write_text(json.dumps(handle.job_fields()), encoding="utf-8")
            temporary.replace(directory / "start.json")
            ready = _wait_for_control(directory / "ready.json", 10)
            if "error" in ready:
                raise RuntimeError(f"Worker failed to launch: {ready['error']}")
            return handle
        except BaseException as startup_error:
            try:
                if (directory / "start.json").exists():
                    request_worker_stop(self._workspace.root, handle, timeout=15)
                    proc.wait(timeout=5)
                else:
                    # This is still our unreaped direct child. The supervisor
                    # has not received start, so it cannot have detached workers.
                    terminate_owned_group(proc)
                shutil.rmtree(directory)
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    f"Worker startup and cleanup failed; retained {directory}",
                    [startup_error, cleanup_error],
                ) from None
            raise

    def run_module(
        self,
        module: str,
        *args: str,
        timeout_seconds: float,
        operation: str = "preflight",
    ) -> None:
        """Wait for one supervised module and verify cleanup on every outcome."""
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Worker timeout must be finite and positive")
        deadline = time.monotonic() + timeout_seconds
        try:
            handle = self.spawn_module(module, *args)
        except BaseException as error:
            raise BoundedWorkerError(
                f"{operation}_start_failed",
                f"{operation} worker could not start: {error}",
                cleanup_verified=not isinstance(error, BaseExceptionGroup),
            ) from error
        directory = self._workspace.root / ".workers" / handle.control_id
        failure: BaseException | None = None
        completion: dict[str, Any] | None = None
        try:
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{operation} exceeded its total time budget")
                completion = _read_control(directory / "finished.json")
                if completion is not None:
                    break
                time.sleep(0.02)
        except BaseException as error:
            failure = error
        try:
            request_worker_stop(self._workspace.root, handle, timeout=15)
        except BaseException as error:
            raise BoundedWorkerError(
                f"{operation}_cleanup_failed",
                f"{operation} cleanup could not be verified: {error}",
                cleanup_verified=False,
            ) from error
        if failure is not None:
            raise BoundedWorkerError(
                f"{operation}_timeout"
                if isinstance(failure, TimeoutError)
                else f"{operation}_failed",
                str(failure),
                cleanup_verified=True,
            ) from failure
        if (
            completion is None
            or completion.get("returncode") != 0
            or completion.get("error")
        ):
            raise BoundedWorkerError(
                f"{operation}_failed",
                f"{operation} worker failed: {completion}",
                cleanup_verified=True,
            )
