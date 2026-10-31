"""Verify claim boundaries with fake GitHub responses and real local processes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager, suppress
from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING

import psutil
import pytest
from tooling.git import claim_issue
from tooling.process_guard import ProcessCleanupError, run_captured

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_URL = "https://github.com/owner/project/issues/123"
_ENDPOINT = "repos/owner/project/issues/123"
_ISSUE: dict[str, object] = {
    "id": 42,
    "number": 123,
    "html_url": _URL,
    "state": "open",
    "assignees": [{"login": "bob"}],
}
_ACCOUNT: dict[str, object] = {"login": "Alice", "type": "User"}
type Responses = list[dict[str, object] | Exception]
type Calls = list[tuple[str, str, str | None]]


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Callable[[Responses], Calls]:
    def install(responses: Responses) -> Calls:
        calls: Calls = []

        def fake(
            method: str, endpoint: str, *, user: str | None = None
        ) -> dict[str, object]:
            calls.append((method, endpoint, user))
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return deepcopy(response)

        monkeypatch.setattr(claim_issue, "_api", fake)
        return calls

    return install


def test_claim_adds_only_self_and_verifies_existing_assignees(
    api: Callable[[Responses], Calls],
) -> None:
    assigned = {**_ISSUE, "assignees": [{"login": "bob"}, {"login": "alice"}]}
    calls = api([_ACCOUNT, _ISSUE, assigned, assigned])
    assert "Assigned and verified" in claim_issue.claim_issue(
        123, repository="owner/project", user="alice"
    )
    assert calls == [
        ("GET", "user", None),
        ("GET", _ENDPOINT, None),
        ("POST", f"{_ENDPOINT}/assignees", "alice"),
        ("GET", _ENDPOINT, None),
    ]
    calls = api([_ACCOUNT, assigned])
    assert "no changes made" in claim_issue.claim_issue(
        123, repository="owner/project", user="ALICE"
    )
    assert all(method == "GET" for method, _, _ in calls)


@pytest.mark.parametrize(
    ("number", "repository", "user"),
    [
        (0, "owner/project", "alice"),
        (123, "owner/../project", "alice"),
        (123, "owner/..", "alice"),
        (123, "owner/project;touch marker", "alice"),
        (123, "https://github.com/owner/project", "alice"),
        (123, "owner/project", "@me"),
        (123, "owner/project", "--help"),
        (123, "owner/project", "alice$(touch marker)"),
    ],
)
def test_invalid_inputs_never_call_github(
    api: Callable[[Responses], Calls], number: int, repository: str, user: str
) -> None:
    calls = api([])
    with pytest.raises(RuntimeError):
        claim_issue.claim_issue(number, repository=repository, user=user)
    assert not calls


@pytest.mark.parametrize(
    "responses",
    [
        [{"login": "other", "type": "User"}],
        [{"login": "alice", "type": "Bot"}],
        [_ACCOUNT, {**_ISSUE, "state": "closed"}],
        [_ACCOUNT, {**_ISSUE, "pull_request": {}}],
        [_ACCOUNT, {**_ISSUE, "number": 124}],
        [_ACCOUNT, {**_ISSUE, "html_url": _URL.replace("owner", "other")}],
        [_ACCOUNT, {**_ISSUE, "id": True}],
        [_ACCOUNT, {**_ISSUE, "assignees": None}],
        [_ACCOUNT, {**_ISSUE, "assignees": [{}]}],
        [RuntimeError("permission denied")],
    ],
)
def test_unverified_account_or_issue_never_writes(
    api: Callable[[Responses], Calls], responses: Responses
) -> None:
    calls = api(responses)
    with pytest.raises(RuntimeError, match="no assignment request sent"):
        claim_issue.claim_issue(123, repository="owner/project", user="alice")
    assert all(method == "GET" for method, _, _ in calls)


@pytest.mark.parametrize(
    "outcome",
    [
        _ISSUE,
        {**_ISSUE, "assignees": [{"login": "alice"}]},
        {**_ISSUE, "id": 43, "assignees": [{"login": "bob"}, {"login": "alice"}]},
        RuntimeError("read-back unavailable"),
    ],
)
def test_incomplete_readback_is_not_success_and_does_not_retry(
    api: Callable[[Responses], Calls], outcome: dict[str, object] | Exception
) -> None:
    calls = api([_ACCOUNT, _ISSUE, {}, outcome])
    with pytest.raises(RuntimeError, match="success is unverified"):
        claim_issue.claim_issue(123, repository="owner/project", user="alice")
    assert sum(method == "POST" for method, _, _ in calls) == 1


def test_cleanup_failure_keeps_distinct_exit_status(
    api: Callable[[Responses], Calls], capsys: pytest.CaptureFixture[str]
) -> None:
    api([_ACCOUNT, _ISSUE, ProcessCleanupError("process still alive")])
    assert claim_issue.main(["123", "--repo", "owner/project", "--user", "alice"]) == 86
    assert "assignment may have changed" in capsys.readouterr().err


_FAKE_GH = """
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import psutil
root = Path(os.environ["CLAIM_TEST_ROOT"])
def record(process):
    (root / f"process-{process.pid}.json").write_text(json.dumps({
        "pid": process.pid, "created": process.create_time()
    }))
record(psutil.Process())
args = sys.argv[1:]
with (root / "calls.jsonl").open("a") as stream:
    stream.write(json.dumps(args) + "\\n")
assert args[:4] == ["api", "--hostname", "github.com", "--method"]
method, endpoint = args[4:6]
if endpoint == "user":
    print(json.dumps({"login": "alice", "type": "User"}))
    sys.exit(0)
assert endpoint in (
    "repos/owner/project/issues/123", "repos/owner/project/issues/123/assignees"
)
if method == "POST":
    assert args[6:] == ["--raw-field", "assignees[]=alice"]
    (root / "assigned").touch()
    mode = os.environ.get("CLAIM_TEST_MODE", "success")
    if mode == "failure":
        print("secret credential must not be echoed", file=sys.stderr)
        sys.exit(1)
    if mode == "timeout":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        record(psutil.Process(child.pid))
        (root / "child.pid").write_text(str(child.pid))
        time.sleep(30)
else:
    assert method == "GET" and not args[6:]
assignees = [{"login": "bob"}]
if (root / "assigned").exists():
    assignees.append({"login": "alice"})
print(json.dumps({"id": 42, "number": 123, "state": "open", "assignees": assignees,
                  "html_url": "https://github.com/owner/project/issues/123"}))
"""


@contextmanager
def _process_workspace() -> Iterator[Path]:
    root = Path(mkdtemp(prefix="tradingdev-claim-test-"))
    try:
        yield root
    finally:
        try:
            for marker in root.glob("process-*.json"):
                identity = json.loads(marker.read_text())
                with suppress(psutil.NoSuchProcess):
                    process = psutil.Process(identity["pid"])
                    if process.create_time() != identity["created"]:
                        continue
                    if process.status() != psutil.STATUS_ZOMBIE:
                        process.kill()
                        try:
                            process.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            if process.status() != psutil.STATUS_ZOMBIE:
                                raise
        except (OSError, ValueError, psutil.Error) as error:
            raise RuntimeError(
                f"Test process cleanup failed; retained {root}"
            ) from error
        shutil.rmtree(root)
        assert not root.exists()


@pytest.mark.parametrize("mode", ["success", "failure", "timeout", "snapshot"])
def test_real_process_execution_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    with _process_workspace() as root:
        executable = root / "gh"
        executable.write_text(f"#!{sys.executable}\n{_FAKE_GH}")
        executable.chmod(0o700)
        monkeypatch.setenv("PATH", f"{root}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setenv("CLAIM_TEST_ROOT", str(root))
        monkeypatch.setenv("CLAIM_TEST_MODE", mode)
        if mode == "timeout":

            def bounded(
                command: list[str], *, cwd: Path, env: dict[str, str], timeout: float
            ) -> subprocess.CompletedProcess[str]:
                return run_captured(command, cwd=cwd, env=env, timeout=1)

            monkeypatch.setattr(claim_issue, "run_captured", bounded)
            with pytest.raises(RuntimeError, match="success is unverified"):
                claim_issue.claim_issue(123, repository="owner/project", user="alice")
            child = int((root / "child.pid").read_text())
            assert not psutil.pid_exists(child) or (
                psutil.Process(child).status() == psutil.STATUS_ZOMBIE
            )
        else:
            project = Path(__file__).resolve().parents[2]
            script = project / "scripts/claim-issue.sh"
            if mode == "snapshot":
                snapshot = root / "snapshot"
                for name in (
                    "scripts/claim-issue.sh",
                    "tooling/__init__.py",
                    "tooling/process_guard.py",
                    "tooling/git/__init__.py",
                    "tooling/git/claim_issue.py",
                ):
                    target = snapshot / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(project / name, target)
                script = snapshot / "scripts/claim-issue.sh"
                monkeypatch.setenv("TRADINGDEV_CHECK_PYTHON", sys.executable)
                assert not (snapshot / ".venv").exists()
            result = run_captured(
                [str(script), "123", "--repo", "owner/project", "--user", "alice"],
                cwd=root,
                env=dict(os.environ),
                timeout=30,
            )
            assert result.returncode == (1 if mode == "failure" else 0)
            if mode != "failure":
                assert "Assigned and verified" in result.stdout
            else:
                assert "success is unverified" in result.stderr
                assert "secret credential" not in result.stderr
        calls = [
            json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()
        ]
        assert sum(call[4] == "POST" for call in calls) == 1
    assert not root.exists()


def test_failed_assertion_stops_recorded_child_before_removing_workspace() -> None:
    child: subprocess.Popen[bytes] | None = None
    try:
        with (
            pytest.raises(AssertionError, match="injected assertion failure"),
            _process_workspace() as root,
        ):
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"]
            )
            (root / f"process-{child.pid}.json").write_text(
                json.dumps(
                    {
                        "pid": child.pid,
                        "created": psutil.Process(child.pid).create_time(),
                    }
                )
            )
            raise AssertionError("injected assertion failure")
        assert child is not None
        assert child.poll() is not None
        assert not root.exists()
    finally:
        if child is not None:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=3)


@pytest.mark.parametrize("extra", [["--close"], ["--", "close"], ["--us", "alice"]])
def test_cli_rejects_other_operations_before_github(
    api: Callable[[Responses], Calls], extra: list[str]
) -> None:
    calls = api([])
    with pytest.raises(SystemExit) as error:
        claim_issue.main(["123", "--repo", "owner/project", "--user", "alice", *extra])
    assert error.value.code == 2
    assert not calls
