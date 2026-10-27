"""Real bounded workers clean owned descendants before temporary files disappear."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tests.preflight_fixtures import make_preflight_fixture
from tradingdev.adapters.execution.process_runner import (
    BoundedWorkerError,
    ProcessIdentity,
    ProcessRunner,
)
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.preflight_service import PreflightError, PreflightService

pytestmark = pytest.mark.integration


_PROBE = '''
import json, os, signal, subprocess, sys, time
from pathlib import Path
import psutil
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', """
import json, os, signal, time
from pathlib import Path
import psutil
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path('descendant.json').write_text(json.dumps({
    'pid': os.getpid(), 'created': psutil.Process().create_time()
}))
time.sleep(60)
"""])
while not Path('descendant.json').exists():
    time.sleep(0.01)
Path('leader.json').write_text(json.dumps({
    'pid': os.getpid(), 'created': psutil.Process().create_time()
}))
if sys.argv[1] == 'success':
    sys.exit(0)
if sys.argv[1] == 'failure':
    sys.exit(7)
time.sleep(60)
'''


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
def test_bounded_runner_stops_group_on_success_failure_and_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    with TemporaryDirectory(dir=tmp_path, prefix="bounded-worker-") as name:
        root = Path(name)
        (root / "preflight_probe.py").write_text(_PROBE)
        source = Path(__file__).resolve().parents[2] / "src"
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(root), str(source))))
        workspace = WorkspacePaths(root / "runtime")
        runner = ProcessRunner(root, workspace=workspace)
        if outcome == "success":
            runner.run_module("preflight_probe", outcome, timeout_seconds=3)
        else:
            with pytest.raises(BoundedWorkerError) as caught:
                runner.run_module("preflight_probe", outcome, timeout_seconds=3)
            assert caught.value.cleanup_verified
            assert caught.value.code == (
                "preflight_timeout" if outcome == "timeout" else "preflight_failed"
            )
        for path in (root / "leader.json", root / "descendant.json"):
            record = json.loads(path.read_text())
            identity = ProcessIdentity.from_values(record["pid"], record["created"])
            assert identity is not None and identity.get_process() is None
        controls = list((workspace.root / ".workers").iterdir())
        assert len(controls) == 1
        finished = json.loads((controls[0] / "finished.json").read_text())
        assert finished["cleaned"] is True
    assert not root.exists()


@pytest.mark.parametrize(
    ("outcome", "kind"),
    [
        ("success", "backtest"),
        ("success", "walk_forward"),
        ("success", "optimization"),
        ("exception", "backtest"),
        ("timeout", "backtest"),
    ],
)
def test_real_preflight_parent_removes_validation_workspace_after_worker_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    kind: str,
) -> None:
    workspace, request = make_preflight_fixture(
        tmp_path,
        monkeypatch,
        kind=kind,
        failure=None if outcome == "success" else outcome,
    )
    temporary = tmp_path / "preflight"
    temporary.mkdir()
    monkeypatch.setattr(
        "tradingdev.app.preflight_service.mkdtemp", lambda **_kwargs: str(temporary)
    )
    service = PreflightService(
        workspace, timeout_seconds=3 if outcome == "timeout" else 60
    )
    if outcome == "success":
        result = service.prepare(request)
        assert result.receipt.synthetic_fixture_rows == [80, 240]
        assert result.prepared.manifest.manifest_hash == result.receipt.manifest_hash
    else:
        with pytest.raises(PreflightError) as caught:
            service.prepare(request)
        assert caught.value.code == (
            "preflight_timeout" if outcome == "timeout" else "invalid_execution_request"
        )
    assert not temporary.exists()
    assert list(workspace.runs.iterdir()) == []
