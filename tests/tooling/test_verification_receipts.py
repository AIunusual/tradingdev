"""Reuse only successful test evidence whose executable inputs still match."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from filelock import FileLock
from tests.tooling.git_check_scenarios import _commit, _git
from tooling.git import candidates, check_pr, gate, receipts, suite
from tooling.process_guard import ProcessCleanupError

if TYPE_CHECKING:
    from pytest import CaptureFixture, MonkeyPatch
    from tests.tooling.git_check_scenarios import Scenario

from tests.tooling.git_check_scenarios import scenario as scenario


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    _git(tmp_path, "init", "--quiet", "--initial-branch=main", "--template=")
    _git(tmp_path, "config", "user.name", "Receipt tests")
    _git(tmp_path, "config", "user.email", "tests@example.invalid")
    _commit(tmp_path, "app.py", "VALUE = 1\n")
    _commit(tmp_path, "README.md", "# Guide\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(receipts, "environment_fingerprint", lambda *_: "environment")
    return tmp_path


@pytest.fixture
def checks(monkeypatch: MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def check(tree: str, profile: str, **kwargs: object) -> None:
        calls.append(profile)

    monkeypatch.setattr(candidates, "check_snapshot", check)
    return calls


def _check(root: Path, profile: str = "full", *, reuse: bool = True) -> None:
    candidates.check_test_snapshot(
        _git(root, "rev-parse", "HEAD^{tree}"), profile, reuse=reuse
    )


def test_documentation_refresh_reuses_success_without_relabelling_tested_tree(
    repository: Path,
    checks: list[str],
    capsys: CaptureFixture[str],
) -> None:
    _check(repository)
    tested = _git(repository, "rev-parse", "HEAD^{tree}")
    _commit(repository, "README.md", "# Corrected instructions\n")
    _check(repository)
    assert checks == ["full", "docs"]
    record = json.loads(receipts.receipt_path(repository, "full").read_text())
    assert record["tree"] == tested
    assert record["state"] == "passed"
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output[-1]["code"] == "test_evidence_reused"
    assert output[-1]["tested_tree"] == tested
    assert output[-1]["candidate_tree"] != tested


@pytest.mark.parametrize(
    "change",
    [
        "code",
        "tests",
        "dependency",
        "policy",
        "tooling_helper",
        "wait_skill",
        "pr_review_instructions",
        "prompt",
        "mode",
        "symlink",
        "rename",
        "presentation",
    ],
)
def test_executable_or_unknown_changes_invalidate_even_after_a_docs_commit(
    repository: Path,
    checks: list[str],
    change: str,
) -> None:
    _check(repository)
    if change == "mode":
        _git(repository, "update-index", "--chmod=+x", "README.md")
    elif change == "symlink":
        (repository / "README.md").unlink()
        (repository / "README.md").symlink_to("app.py")
        _git(repository, "add", "README.md")
    elif change == "rename":
        _git(repository, "mv", "app.py", "code.md")
    else:
        name = {
            "code": "app.py",
            "tests": "tests/conftest.py",
            "dependency": "uv.lock",
            "policy": "AGENTS.md",
            "tooling_helper": "tests/tooling/git_check_scenarios.py",
            "wait_skill": ".agents/skills/wait-for-check/SKILL.md",
            "pr_review_instructions": "docs/pr_review.md",
            "prompt": "src/prompt.md",
            "presentation": "docs/talks/tradingdev-introduction/slides.md",
        }[change]
        (repository / name).parent.mkdir(parents=True, exist_ok=True)
        (repository / name).write_text("changed\n")
        _git(repository, "add", name)
    _git(repository, "commit", "--quiet", "-m", "test: executable change")
    (repository / "docs").mkdir(exist_ok=True)
    _commit(repository, "docs/guide.md", "# Last commit is documentation\n")
    _check(repository)
    assert checks == ["full", "full"]


@pytest.mark.parametrize("change", ["add", "delete", "rename"])
def test_ordinary_documentation_structural_changes_need_only_docs(
    repository: Path,
    checks: list[str],
    change: str,
) -> None:
    _check(repository)
    if change == "add":
        _commit(repository, "ARCHITECTURE.md", "# Architecture\n")
    else:
        if change == "delete":
            _git(repository, "rm", "README.md")
        else:
            (repository / "docs").mkdir()
            _git(repository, "mv", "README.md", "docs/guide.md")
        _git(repository, "commit", "--quiet", "-m", "docs: restructure")
    _check(repository)
    assert checks == ["full", "docs"]


@pytest.mark.parametrize(
    "failure", ["test", "timeout", "interrupted", "cleanup", "mutation"]
)
def test_failed_explicit_run_replaces_older_success(
    repository: Path,
    checks: list[str],
    monkeypatch: MonkeyPatch,
    failure: str,
) -> None:
    _check(repository)
    errors: dict[str, BaseException] = {
        "test": subprocess.CalledProcessError(1, ["check"]),
        "timeout": subprocess.TimeoutExpired(["check"], 1),
        "interrupted": KeyboardInterrupt(),
        "cleanup": ProcessCleanupError("retained processes"),
        "mutation": RuntimeError("snapshot modified"),
    }
    real_check = candidates.check_snapshot

    def fail(*args: object, **kwargs: object) -> None:
        raise errors[failure]

    monkeypatch.setattr(candidates, "check_snapshot", fail)
    with pytest.raises(type(errors[failure])):
        _check(repository, reuse=False)
    record = json.loads(receipts.receipt_path(repository, "full").read_text())
    assert record["state"] == "failed"
    monkeypatch.setattr(candidates, "check_snapshot", real_check)
    _check(repository)
    assert checks == ["full", "full"]


def test_environment_changes_invalidate_and_midrun_changes_fail(
    repository: Path,
    checks: list[str],
    monkeypatch: MonkeyPatch,
) -> None:
    _check(repository)
    monkeypatch.setattr(
        receipts, "environment_fingerprint", lambda *_: "new environment"
    )
    _check(repository)
    assert checks == ["full", "full"]
    values = iter(["before", "after"])
    monkeypatch.setattr(receipts, "environment_fingerprint", lambda *_: next(values))
    with pytest.raises(RuntimeError, match="environment changed during"):
        _check(repository)
    assert (
        json.loads(receipts.receipt_path(repository, "full").read_text())["state"]
        == "failed"
    )


@pytest.mark.parametrize(
    "invalid", ["{", "[]", '{"state":"passed"}', "running", "old-version"]
)
def test_incomplete_or_unknown_records_cannot_skip_checks(
    repository: Path,
    checks: list[str],
    invalid: str,
) -> None:
    _check(repository)
    path = receipts.receipt_path(repository, "full")
    value = json.loads(path.read_text())
    if invalid == "running":
        value["state"] = "running"
        invalid = json.dumps(value)
    elif invalid == "old-version":
        value["version"] = 0
        invalid = json.dumps(value)
    path.write_text(invalid)
    _check(repository)
    assert checks == ["full", "full"]


def test_missing_environment_proof_runs_but_cannot_record_success(
    repository: Path,
    checks: list[str],
    monkeypatch: MonkeyPatch,
) -> None:
    def unavailable(*args: object) -> str:
        raise OSError("missing browser")

    monkeypatch.setattr(receipts, "environment_fingerprint", unavailable)
    _check(repository)
    _check(repository)
    assert checks == ["full", "full"]
    assert (
        json.loads(receipts.receipt_path(repository, "full").read_text())["state"]
        == "unavailable"
    )


def test_profiles_are_separate_and_concurrent_checks_cannot_replace_evidence(
    repository: Path,
    checks: list[str],
) -> None:
    _check(repository, "tooling")
    _check(repository, "full")
    _check(repository, "full")
    assert checks == ["tooling", "full"]
    path = receipts.receipt_path(repository, "full")
    before = path.read_bytes()
    with (
        FileLock(str(path.with_suffix(".lock"))),
        pytest.raises(RuntimeError, match="Another full check"),
    ):
        _check(repository, reuse=False)
    assert path.read_bytes() == before


def test_manual_clean_suite_runs_again_and_dirty_failure_invalidates(
    repository: Path,
    checks: list[str],
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRADINGDEV_CHECK_SNAPSHOT", raising=False)
    monkeypatch.setattr(sys, "argv", ["suite", "full"])
    assert suite.main() == 0
    assert suite.main() == 0
    assert checks == ["full", "full"]
    (repository / "app.py").write_text("uncommitted\n")
    path = receipts.receipt_path(repository, "full")

    def fail(*args: object) -> None:
        assert not path.exists()
        raise subprocess.CalledProcessError(1, ["pytest"])

    monkeypatch.setattr(suite, "run_suite", fail)
    with pytest.raises(subprocess.CalledProcessError):
        suite.main()
    assert not path.exists()


def test_failed_docs_refresh_is_not_reported_as_a_pass(
    repository: Path,
    checks: list[str],
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    _check(repository)
    tested_tree = _git(repository, "rev-parse", "HEAD^{tree}")
    normal_check = candidates.check_snapshot
    capsys.readouterr()
    _commit(repository, "README.md", "# Broken link\n")

    def fail(tree: str, profile: str, **kwargs: object) -> None:
        assert profile == "docs"
        raise subprocess.CalledProcessError(1, ["bash", "scripts/check-docs.sh"])

    monkeypatch.setattr(candidates, "check_snapshot", fail)
    with pytest.raises(subprocess.CalledProcessError):
        _check(repository)
    assert "test_evidence_reused" not in capsys.readouterr().out
    record = json.loads(receipts.receipt_path(repository, "full").read_text())
    assert record["state"] == "passed" and record["tree"] == tested_tree
    _commit(repository, "README.md", "# Repaired documentation\n")
    monkeypatch.setattr(candidates, "check_snapshot", normal_check)
    _check(repository)
    assert checks == ["full", "docs"]


@pytest.mark.parametrize("entry", ["probe", "docs"])
def test_cleanup_and_recording_failure_cannot_leave_prior_success(
    repository: Path,
    checks: list[str],
    monkeypatch: MonkeyPatch,
    entry: str,
) -> None:
    _check(repository)
    tested_tree = _git(repository, "rev-parse", "HEAD^{tree}")
    _commit(repository, "README.md", "# Updated documentation\n")
    write = receipts.write_receipt
    original_probe = receipts.environment_fingerprint
    original_check = candidates.check_snapshot
    failure = ProcessCleanupError(
        "cleanup unverified", retained_workspaces=[repository]
    )

    def record(path: Path, **kwargs: object) -> None:
        if kwargs["state"] == "failed":
            raise OSError("disk full")
        write(path, **kwargs)  # type: ignore[arg-type]

    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(receipts, "write_receipt", record)
    if entry == "probe":
        monkeypatch.setattr(receipts, "environment_fingerprint", fail)
    else:
        monkeypatch.setattr(candidates, "check_snapshot", fail)
    with pytest.raises(ProcessCleanupError) as error:
        _check(repository)
    assert error.value is failure
    assert error.value.retained_workspaces == (repository.resolve(),)
    assert (
        json.loads(receipts.receipt_path(repository, "full").read_text())["state"]
        == "running"
    )
    monkeypatch.setattr(receipts, "write_receipt", write)
    monkeypatch.setattr(receipts, "environment_fingerprint", original_probe)
    monkeypatch.setattr(candidates, "check_snapshot", original_check)
    candidates.check_test_snapshot(tested_tree, "full")
    assert checks == ["full", "full"]


def test_running_record_must_be_durable_before_any_probe(
    repository: Path,
    checks: list[str],
    monkeypatch: MonkeyPatch,
) -> None:
    _check(repository)
    path = receipts.receipt_path(repository, "full")
    previous = path.read_bytes()

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("cannot persist running state")

    monkeypatch.setattr(receipts, "write_receipt", fail)
    monkeypatch.setattr(
        receipts, "environment_fingerprint", lambda *_: pytest.fail("probe started")
    )
    with pytest.raises(OSError, match="running state"):
        _check(repository)
    assert path.read_bytes() == previous
    assert checks == ["full"]


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired(["docs"], 1),
        KeyboardInterrupt(),
        subprocess.CalledProcessError(-15, ["bash", "scripts/check-docs.sh"]),
        subprocess.CalledProcessError(130, ["bash", "scripts/check-docs.sh"]),
        subprocess.CalledProcessError(143, ["bash", "scripts/check-docs.sh"]),
        subprocess.CalledProcessError(2, ["bash", "scripts/check-docs.sh"]),
        subprocess.CalledProcessError(1, ["git", "archive"]),
    ],
)
def test_interrupted_docs_refresh_does_not_restore_prior_evidence(
    repository: Path,
    checks: list[str],
    monkeypatch: MonkeyPatch,
    error: BaseException,
) -> None:
    _check(repository)
    _commit(repository, "README.md", "# Updated documentation\n")

    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(candidates, "check_snapshot", fail)
    with pytest.raises(type(error)):
        _check(repository)
    assert (
        json.loads(receipts.receipt_path(repository, "full").read_text())["state"]
        == "failed"
    )


def test_receipt_write_failure_removes_partial_file(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(json, "dump", fail)
    with pytest.raises(OSError, match="disk full"):
        receipts.write_receipt(
            tmp_path / "full.json",
            state="passed",
            tree="a" * 40,
            profile="full",
            inputs="inputs",
            environment="environment",
        )
    assert not list(tmp_path.iterdir())


def test_failed_receipt_write_preserves_cleanup_error_and_blocks_reuse(
    repository: Path,
    checks: list[str],
    monkeypatch: MonkeyPatch,
) -> None:
    _check(repository)
    write = receipts.write_receipt

    def fail_record(path: Path, **kwargs: object) -> None:
        if kwargs["state"] == "failed":
            raise OSError("disk full")
        write(path, **kwargs)  # type: ignore[arg-type]

    failure = ProcessCleanupError("child is alive", retained_workspaces=[repository])

    def fail_check(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(receipts, "write_receipt", fail_record)
    monkeypatch.setattr(candidates, "check_snapshot", fail_check)
    with pytest.raises(ProcessCleanupError) as caught:
        _check(repository, reuse=False)
    assert caught.value is failure
    assert caught.value.retained_workspaces == (repository.resolve(),)
    assert (
        json.loads(receipts.receipt_path(repository, "full").read_text())["state"]
        == "running"
    )


def test_environment_cleanup_failure_invalidates_previous_success(
    repository: Path,
    checks: list[str],
    monkeypatch: MonkeyPatch,
) -> None:
    _check(repository)

    def fail(*args: object) -> str:
        raise ProcessCleanupError("inspection process is alive")

    monkeypatch.setattr(receipts, "environment_fingerprint", fail)
    with pytest.raises(ProcessCleanupError):
        _check(repository)
    assert (
        json.loads(receipts.receipt_path(repository, "full").read_text())["state"]
        == "failed"
    )


def test_docs_cleanup_failure_prevents_reuse_of_even_the_original_tree(
    repository: Path,
    checks: list[str],
    monkeypatch: MonkeyPatch,
) -> None:
    _check(repository)
    original = _git(repository, "rev-parse", "HEAD^{tree}")
    _commit(repository, "README.md", "# Updated docs\n")
    real_check = candidates.check_snapshot

    def fail(*args: object, **kwargs: object) -> None:
        raise ProcessCleanupError(
            "docs child is alive", retained_workspaces=[repository]
        )

    monkeypatch.setattr(candidates, "check_snapshot", fail)
    with pytest.raises(ProcessCleanupError) as error:
        _check(repository)
    assert error.value.retained_workspaces == (repository.resolve(),)
    monkeypatch.setattr(candidates, "check_snapshot", real_check)
    candidates.check_test_snapshot(original, "full")
    assert checks == ["full", "full"]


def test_snapshot_deadline_reaches_inner_suite_runner(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[2]
    code = (
        "import json, sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(project)!r})\n"
        "from tooling.git import suite\n"
        "suite.run_suite = lambda profile, timeout: "
        "Path('deadline.json').write_text(json.dumps([profile, timeout]))\n"
        "sys.argv = ['suite', 'full']\n"
        "raise SystemExit(suite.main())\n"
    )
    candidates.run([sys.executable, "-c", code], cwd=tmp_path, timeout=7200)
    assert json.loads((tmp_path / "deadline.json").read_text()) == ["full", 7200]


@pytest.mark.parametrize("runner", ["pr", "local"])
def test_candidate_reuses_development_snapshot_but_reviews_new_docs(
    scenario: Scenario,
    monkeypatch: MonkeyPatch,
    runner: str,
) -> None:
    root = scenario.source
    calls: list[str] = []
    monkeypatch.setattr(receipts, "environment_fingerprint", lambda *_: "environment")

    def run(command: list[str], **kwargs: object) -> None:
        calls.append("review" if "tooling.review_docs" in command else command[1])

    monkeypatch.setattr(candidates, "run", run)
    # The tested merge tree includes main's base.txt, not just the branch head.
    with candidates.merge_candidate(str(root), "main", "feature/work") as candidate:
        candidates.check_test_snapshot(
            candidate.tree, "full", repo=candidate.repository
        )
    assert calls == ["scripts/check-full.sh"]
    head = _commit(root, "README.md", "# Corrected documentation\n")
    _git(root, "update-ref", "refs/pull/7/head", head)
    request = replace(scenario.request, head_sha=head)
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: request)
    if runner == "pr":
        check_pr.check_pr(7)
    else:
        gate.check_full("main", "feature/work")
    assert calls == ["scripts/check-full.sh", "review", "scripts/check-docs.sh"]
    assert all(not path.exists() for path in scenario.temporary_paths)


@pytest.mark.parametrize(
    "change",
    [
        "settings",
        "package",
        "binary",
        "browser",
        "image",
        "cpu",
        "docker_cli",
        "daemon",
        "node_file",
        "python_file",
        "permissions",
    ],
)
def test_environment_fingerprint_tracks_actual_external_inputs(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    change: str,
) -> None:
    binary = tmp_path / "binary"
    browser = tmp_path / "browser"
    docker = tmp_path / "docker"
    binary.write_text("tool version 1")
    browser.write_text("browser version 1")
    docker.write_text("docker version 1")
    modules = tmp_path / "node_modules"
    modules.mkdir()
    (modules / ".package-lock.json").write_text("{}")
    (modules / "module.js").write_text("module version 1")
    (tmp_path / "pytest.py").write_text("package version 1")
    version = "1"
    image = "sha256:one"
    cpu = 8
    engine = dict.fromkeys(
        (
            "ID",
            "ServerVersion",
            "OSType",
            "Architecture",
            "KernelVersion",
            "Driver",
            "CgroupDriver",
            "CgroupVersion",
            "SecurityOptions",
            "NCPU",
            "MemTotal",
        ),
        "value",
    )
    env = {"PATH": str(tmp_path), "TRADINGDEV_PRESENTATION_TOOLCHAIN": str(tmp_path)}
    monkeypatch.setattr(
        shutil, "which", lambda name, **_: str(docker if name == "docker" else binary)
    )
    monkeypatch.setattr(receipts, "_toolchain", lambda _: tmp_path)
    monkeypatch.setattr(receipts, "require_strategy_runtime", lambda: image)
    monkeypatch.setattr(os, "process_cpu_count", lambda: cpu)
    monkeypatch.setattr(
        importlib.metadata,
        "distributions",
        lambda: [
            SimpleNamespace(
                metadata={"Name": "pytest"},
                version=version,
                files=[importlib.metadata.PackagePath("pytest.py")],
                locate_file=lambda name: tmp_path / name,
            )
        ],
    )

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        stdout = str(browser) if "--input-type=module" in command else "version 1"
        if command[:2] == ["docker", "info"]:
            stdout = json.dumps(engine)
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(receipts, "run_captured", run)
    before = receipts.environment_fingerprint(tmp_path, "full", env)
    # Managed snapshot paths may differ; actual test options may not.
    env["TMPDIR"] = "/different/snapshot/runtime"
    env["PWD"] = "/different/snapshot"
    for key in ("CODEX_TOOL_CALL_ID", "CODEX_SESSION_ID", "CODEX_THREAD_ID"):
        env[key] = "different-invocation"
    assert receipts.environment_fingerprint(tmp_path, "full", env) == before
    if change == "settings":
        env["PYTEST_XDIST_AUTO_NUM_WORKERS"] = "10"
    elif change == "permissions":
        env["CODEX_PERMISSION_PROFILE"] = "different-permissions"
    elif change == "package":
        version = "2"
    elif change == "binary":
        binary.write_text("tool version 2 changed size")
    elif change == "browser":
        browser.write_text("browser version 2 changed size")
    elif change == "image":
        image = "sha256:two"
    elif change == "docker_cli":
        docker.write_text("docker version 2 with a different size")
    elif change == "daemon":
        engine["ID"] = "another daemon"
    elif change == "node_file":
        (modules / "module.js").unlink()
    elif change == "python_file":
        (tmp_path / "pytest.py").write_text(
            "edited package code with unchanged version"
        )
    else:
        cpu = 4
    assert receipts.environment_fingerprint(tmp_path, "full", env) != before
