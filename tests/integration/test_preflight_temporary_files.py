"""Preflight owns library temporary files through supervised worker cleanup."""

from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

import pytest

from tests.preflight_fixtures import make_preflight_fixture
from tradingdev.adapters.execution.process_runner import (
    ProcessIdentity,
    ProcessRunner,
    WorkerHandle,
    request_worker_stop,
)
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.preflight_service import PreflightError, PreflightService

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tradingdev.domain.preflight import PreflightRequest

pytestmark = pytest.mark.integration


_TEMPFILE_PROBE = '''
import json, os, runpy, signal, subprocess, sys, tempfile, time, types
from pathlib import Path
import psutil

root = Path(os.environ["PREFLIGHT_TEST_ROOT"])
outcome = os.environ["PREFLIGHT_TEST_OUTCOME"]

def write_record(name, record):
    temporary = root / (name + ".tmp")
    temporary.write_text(json.dumps(record))
    temporary.replace(root / name)

signal.signal(signal.SIGTERM, signal.SIG_IGN)
directory = Path(tempfile.mkdtemp(prefix="preflight-probe-"))
(directory / "artifact.bin").write_bytes(b"partial library output")
with tempfile.NamedTemporaryFile(delete=False) as handle:
    handle.write(b"library temporary file")
    temporary_file = handle.name
record = {
    "pid": os.getpid(),
    "created": psutil.Process().create_time(),
    "paths": [str(directory), temporary_file],
    "temporary_environment": {
        name: os.environ[name] for name in ("TMPDIR", "TMP", "TEMP")
    },
}
write_record("worker.json", record)
child = subprocess.Popen([sys.executable, "-c", """
import json, os, signal, tempfile, time
from pathlib import Path
import psutil
signal.signal(signal.SIGTERM, signal.SIG_IGN)
directory = Path(tempfile.mkdtemp(prefix="preflight-descendant-"))
(directory / "artifact.bin").write_bytes(b"child library output")
record = {
    "pid": os.getpid(),
    "created": psutil.Process().create_time(),
    "paths": [str(directory)],
}
root = Path(os.environ["PREFLIGHT_TEST_ROOT"])
(root / "descendant.tmp").write_text(json.dumps(record))
(root / "descendant.tmp").replace(root / "descendant.json")
time.sleep(60)
"""])
while not (root / "descendant.json").exists():
    time.sleep(0.01)
if outcome == "failure":
    raise RuntimeError("temporary artifact failure")
if outcome == "timeout":
    time.sleep(60)
if outcome == "ml_failure":
    import pandas as pd
    from tradingdev.domain.ml.models.flaml_model import FLAMLAutoMLDirectionModel

    class FailingAutoML:
        # Mimics a third-party learner that writes to TMPDIR before dying.
        # The spill outlives the failed search and is left for the supervisor
        # to clean up, even when the wrapper disables its known learners'
        # diagnostic output.
        classes_ = (0, 1)
        model = None

        def fit(self, X_train, y_train, **kwargs):
            scratch = Path(tempfile.mkdtemp(prefix="flaml_direction_"))
            spill = scratch / "catboost_1700000000.0" / "learn"
            spill.mkdir(parents=True)
            (spill / "partial-model.bin").write_bytes(b"unfinished model")
            record["paths"].append(str(spill.parent))
            record["training_rows"] = len(X_train)
            write_record("worker.json", record)
            raise RuntimeError("substitute AutoML search failed")

    flaml = types.ModuleType("flaml")
    flaml.AutoML = FailingAutoML
    sys.modules["flaml"] = flaml
    FLAMLAutoMLDirectionModel(random_seed=None).train(
        pd.DataFrame({"feature": [0.0, 1.0, 2.0], "target": [0, 1, 0]})
    )
module = sys.argv.pop(1)
runpy.run_module(module, run_name="__main__")
'''


def _assert_processes_stopped(root: Path) -> list[Path]:
    artifacts: list[Path] = []
    for name in ("worker.json", "descendant.json"):
        record = json.loads((root / name).read_text())
        identity = ProcessIdentity.from_values(record["pid"], record["created"])
        assert identity is not None and identity.get_process() is None
        artifacts.extend(Path(path) for path in record["paths"])
    return artifacts


def _cleanup_probe_workspace(root: Path) -> None:
    try:
        try:
            launches = list((root / "preflight" / ".workers").iterdir())
        except FileNotFoundError:
            launches = []
        for launch in launches:
            start = launch / "start.json"
            record = json.loads(start.read_text())
            handle = WorkerHandle.from_job(record) if isinstance(record, dict) else None
            if handle is None or handle.control_id != start.parent.name:
                raise RuntimeError(f"Invalid test worker identity: {start}")
            request_worker_stop(root / "preflight", handle)
            if handle.get_process() is not None:
                raise RuntimeError(f"Test supervisor exit is not verified: {start}")
        for name in ("worker.json", "descendant.json"):
            path = root / name
            try:
                record = json.loads(path.read_text())
            except FileNotFoundError:
                continue
            identity = ProcessIdentity.from_values(record["pid"], record["created"])
            if identity is None or identity.get_process() is not None:
                raise RuntimeError(f"Test probe exit is not verified: {path}")
    except Exception as error:
        raise RuntimeError(
            f"Test worker cleanup failed; temporary directory retained: {root}"
        ) from error
    shutil.rmtree(root)


@contextmanager
def _temporary_probe_workspace() -> Iterator[Path]:
    # Do not nest under tmp_path: its unconditional teardown could erase live files.
    root = Path(mkdtemp(prefix="tradingdev-preflight-test-")).resolve()
    try:
        yield root
    finally:
        _cleanup_probe_workspace(root)


def _configure_probe(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> tuple[WorkspacePaths, PreflightRequest]:
    workspace, request = make_preflight_fixture(root, monkeypatch)
    directory = root / "preflight"
    directory.mkdir()
    external_temp = root / "external-temp"
    external_temp.mkdir()
    # Make an old implementation's leaks observable and safe to remove at teardown.
    for variable in ("TMPDIR", "TMP", "TEMP"):
        monkeypatch.setenv(variable, str(external_temp))
    monkeypatch.setenv("PREFLIGHT_TEST_ROOT", str(root))
    monkeypatch.setenv("PREFLIGHT_TEST_OUTCOME", outcome)
    (root / "preflight_temp_probe.py").write_text(_TEMPFILE_PROBE)
    source = Path(__file__).resolve().parents[2] / "src"
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(root), str(source))))
    monkeypatch.setattr(
        "tradingdev.app.preflight_service.mkdtemp", lambda **_kwargs: str(directory)
    )
    run_module = ProcessRunner.run_module

    def run_probe(
        runner: ProcessRunner, module: str, *args: str, timeout_seconds: float
    ) -> None:
        run_module(
            runner,
            "preflight_temp_probe",
            module,
            *args,
            timeout_seconds=timeout_seconds,
        )

    monkeypatch.setattr(ProcessRunner, "run_module", run_probe)
    return workspace, request


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout", "ml_failure"])
def test_preflight_removes_worker_library_temporary_files_after_process_exit(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    with _temporary_probe_workspace() as root:
        _check_preflight_cleanup(root, monkeypatch, outcome)
    assert not root.exists()


def _check_preflight_cleanup(
    root: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    workspace, request = _configure_probe(root, monkeypatch, outcome)
    directory = root / "preflight"
    rmtree = shutil.rmtree
    removal_verified = False

    def verify_before_removal(path: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal removal_verified
        if path == directory:
            assert all(path.exists() for path in _assert_processes_stopped(root))
            removal_verified = True
        rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", verify_before_removal)
    parent_environment = dict(os.environ)
    service = PreflightService(
        workspace, timeout_seconds=3 if outcome == "timeout" else 60
    )
    if outcome == "success":
        assert service.prepare(request).receipt.sample_bars_used == 128
    else:
        with pytest.raises(PreflightError) as caught:
            service.prepare(request)
        assert caught.value.code == (
            "preflight_timeout" if outcome == "timeout" else "preflight_failed"
        )
    assert os.environ == parent_environment
    assert removal_verified
    artifacts = _assert_processes_stopped(root)
    assert not directory.exists()
    assert all(not path.exists() for path in artifacts)
    assert all(path.is_relative_to(directory) for path in artifacts)
    assert list((root / "external-temp").iterdir()) == []
    assert list(workspace.runs.iterdir()) == []
    record = json.loads((root / "worker.json").read_text())
    assert record["temporary_environment"] == {
        name: str(directory / "tmp") for name in ("TMPDIR", "TMP", "TEMP")
    }
    if outcome == "ml_failure":
        assert record["training_rows"] == 3
        # The learner's own temporary directory must sit under the
        # supervised temp tree.
        assert any(path.name.startswith("catboost_") for path in artifacts)
        assert any(
            part.startswith("flaml_direction_")
            for path in artifacts
            for part in path.parts
        )


def _live_probe_identities(root: Path) -> list[ProcessIdentity]:
    deadline = time.monotonic() + 5
    while not (root / "descendant.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Test probe did not start")
        time.sleep(0.01)
    identities = []
    for name in ("worker.json", "descendant.json"):
        record = json.loads((root / name).read_text())
        identity = ProcessIdentity.from_values(record["pid"], record["created"])
        assert identity is not None and identity.get_process() is not None
        identities.append(identity)
    return identities


def _fail_worker_stop(*_args: Any, **_kwargs: Any) -> bool:
    raise RuntimeError("injected worker stop failure")


def test_probe_teardown_stops_processes_after_assertion_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities: list[ProcessIdentity] = []
    with (
        pytest.raises(AssertionError, match="injected assertion"),
        _temporary_probe_workspace() as root,
    ):
        _configure_probe(root, monkeypatch, "timeout")
        runner = ProcessRunner(workspace=WorkspacePaths(root / "preflight"))
        runner.spawn_module("preflight_temp_probe", "unused")
        identities = _live_probe_identities(root)
        raise AssertionError("injected assertion")
    assert all(identity.get_process() is None for identity in identities)
    assert not root.exists()


def test_probe_teardown_stops_processes_when_preflight_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities: list[ProcessIdentity] = []
    monkeypatch.setattr(
        "tradingdev.adapters.execution.process_runner.request_worker_stop",
        _fail_worker_stop,
    )
    with pytest.raises(PreflightError) as caught, _temporary_probe_workspace() as root:
        workspace, request = _configure_probe(root, monkeypatch, "timeout")
        try:
            PreflightService(workspace, timeout_seconds=3).prepare(request)
        except PreflightError:
            identities = _live_probe_identities(root)
            raise
    assert caught.value.code == "preflight_cleanup_failed"
    assert all(identity.get_process() is None for identity in identities)
    assert not root.exists()


@pytest.mark.parametrize("failure", ["exception", "unverified_return"])
def test_probe_teardown_retains_files_when_its_cleanup_cannot_be_verified(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    root: Path | None = None
    identities: list[ProcessIdentity] = []
    try:
        with (
            pytest.raises(RuntimeError, match="temporary directory retained") as caught,
            monkeypatch.context() as fault,
        ):
            fault.setattr(
                "tests.integration.test_preflight_temporary_files.request_worker_stop",
                _fail_worker_stop if failure == "exception" else lambda *_args: False,
            )
            with _temporary_probe_workspace() as root:
                _configure_probe(root, monkeypatch, "timeout")
                runner = ProcessRunner(workspace=WorkspacePaths(root / "preflight"))
                runner.spawn_module("preflight_temp_probe", "unused")
                identities = _live_probe_identities(root)
        assert root is not None and root.is_dir()
        assert str(root) in str(caught.value)
        assert (
            "injected worker stop failure"
            if failure == "exception"
            else "Test supervisor exit is not verified"
        ) in str(caught.value.__cause__)
        assert all(identity.get_process() is not None for identity in identities)
        assert list((root / "external-temp").iterdir())
    finally:
        if root is not None and root.exists():
            _cleanup_probe_workspace(root)
    assert all(identity.get_process() is None for identity in identities)
    assert root is not None and not root.exists()
