"""Candidate failures stop owned processes and clean or retain their workspaces."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
import pytest
from tooling import presentations, process_guard, review_docs
from tooling.git import candidates, check_pr, gate
from tooling.process_guard import ProcessCleanupError, terminate_tree

if TYPE_CHECKING:
    from typing import Any

    from pytest import CaptureFixture, MonkeyPatch
    from tests.tooling.git_check_scenarios import Scenario

from tests.tooling.git_check_scenarios import (
    _assert_cleaned,
    _commit,
    _git,
    _source_state,
)
from tests.tooling.git_check_scenarios import (
    scenario as scenario,
)


def test_conflicting_candidate_fails_before_verification_and_cleans_up(
    scenario: Scenario, monkeypatch: MonkeyPatch
) -> None:
    _git(scenario.source, "switch", "--quiet", "main")
    base = _commit(scenario.source, "shared.txt", "main changed\n")
    _git(scenario.source, "switch", "--quiet", "feature/work")
    head = _commit(scenario.source, "shared.txt", "PR changed\n")
    _git(scenario.source, "update-ref", "refs/pull/7/head", head)
    request = replace(scenario.request, base_sha=base, head_sha=head)
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: request)
    monkeypatch.setattr(
        check_pr,
        "verify_candidate",
        lambda _, **__: pytest.fail("conflict reached checks"),
    )
    before = _source_state(scenario.source)
    with pytest.raises(RuntimeError, match="Merge conflicts"):
        check_pr.check_pr(7)
    assert _source_state(scenario.source) == before
    _assert_cleaned(scenario)


@pytest.mark.parametrize("runner", ["pr", "full"])
@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(1, "docs"),
        KeyboardInterrupt(),
        subprocess.TimeoutExpired("docs", 900),
    ],
)
def test_reviewer_failure_skips_snapshot_and_cleans_candidate(
    scenario: Scenario,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    runner: str,
    failure: BaseException,
) -> None:
    before = _source_state(scenario.source)

    def fail(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 900,
        temporary_root: Path | None = None,
    ) -> None:
        assert command[:3] == [sys.executable, "-m", "tooling.review_docs"]
        repository = Path(command[command.index("--repo") + 1])
        (repository / "partial-model-output").write_text("temporary result")
        raise failure

    monkeypatch.setattr(candidates, "run", fail)
    monkeypatch.setattr(
        candidates,
        "check_snapshot",
        lambda *_, **__: pytest.fail("Snapshot ran after reviewer failure"),
    )
    with pytest.raises(type(failure)):
        if runner == "pr":
            check_pr.check_pr(7, documentation_timeout_seconds=600)
        else:
            gate.check_full("main", "feature/work", documentation_timeout_seconds=600)
    assert "checks passed" not in capsys.readouterr().out
    assert _source_state(scenario.source) == before
    _assert_cleaned(scenario)


def test_full_snapshot_failure_cleans_both_temporary_repositories(
    scenario: Scenario, monkeypatch: MonkeyPatch
) -> None:
    before = _source_state(scenario.source)

    def run(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 900,
        temporary_root: Path | None = None,
    ) -> None:
        if command[0] == "bash":
            (cwd / "partial-test-output").write_text("temporary")
            raise RuntimeError("full suite failed")

    monkeypatch.setattr(candidates, "run", run)
    with pytest.raises(RuntimeError, match="full suite failed"):
        check_pr.check_pr(7)
    assert _source_state(scenario.source) == before
    assert len(scenario.temporary_paths) == 2
    _assert_cleaned(scenario)


def test_fetch_failure_removes_partial_repository_without_touching_source(
    scenario: Scenario, monkeypatch: MonkeyPatch
) -> None:
    _git(scenario.source, "update-ref", "-d", "refs/pull/7/head")
    before = _source_state(scenario.source)
    monkeypatch.setattr(
        check_pr,
        "verify_candidate",
        lambda _, **__: pytest.fail("failed fetch reached checks"),
    )
    with pytest.raises(RuntimeError, match="Candidate git fetch failed"):
        check_pr.check_pr(7)
    assert _source_state(scenario.source) == before
    _assert_cleaned(scenario)


@pytest.mark.parametrize("profile", ["full", "tooling", "docs", "presentations"])
def test_snapshot_timeout_stops_children_and_removes_snapshot(
    scenario: Scenario, monkeypatch: MonkeyPatch, profile: str
) -> None:
    marker = scenario.source.parent / "timeout-processes.json"
    teardown = scenario.source.parent / "teardown.completed"
    (scenario.source / "scripts").mkdir()
    _commit(
        scenario.source,
        f"scripts/check-{profile}.sh",
        'exec "$TRADINGDEV_CHECK_PYTHON" timeout_probe.py\n',
    )
    _commit(
        scenario.source,
        "timeout_probe.py",
        """\
import json, os, subprocess, sys, time
from pathlib import Path
import psutil
child = subprocess.Popen(
    [sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True
)
processes = [psutil.Process(), psutil.Process(child.pid)]
Path(os.environ['TIMEOUT_PROBE_MARKER']).write_text(json.dumps([
    {'pid': process.pid, 'created': process.create_time()} for process in processes
]))
try:
    time.sleep(60)
except KeyboardInterrupt:
    Path(os.environ['TIMEOUT_PROBE_TEARDOWN']).write_text('complete')
""",
    )
    monkeypatch.setenv("TIMEOUT_PROBE_MARKER", str(marker))
    monkeypatch.setenv("TIMEOUT_PROBE_TEARDOWN", str(teardown))
    before = _source_state(scenario.source)
    tree = _git(scenario.source, "rev-parse", "HEAD^{tree}")
    try:
        with pytest.raises(subprocess.TimeoutExpired) as failure:
            candidates.check_snapshot(
                tree, profile, repo=scenario.source, timeout_seconds=1
            )
        assert failure.value.timeout == 1
        assert teardown.read_text() == "complete"
        records = json.loads(marker.read_text())
        assert len(records) == 2
        for record in records:
            try:
                process = psutil.Process(record["pid"])
                assert (
                    process.create_time() != record["created"]
                    or process.status() == psutil.STATUS_ZOMBIE
                )
            except psutil.NoSuchProcess:
                pass
        assert _source_state(scenario.source) == before
        _assert_cleaned(scenario)
    finally:
        survivors: set[psutil.Process] = set()
        if marker.exists():
            for record in json.loads(marker.read_text()):
                try:
                    process = psutil.Process(record["pid"])
                    if process.create_time() == record["created"]:
                        survivors.add(process)
                except psutil.NoSuchProcess:
                    pass
        terminate_tree(None, survivors)


def test_snapshot_environment_keeps_checkout_toolchain_and_explicit_override(
    scenario: Scenario, monkeypatch: MonkeyPatch
) -> None:
    candidate = scenario.source.parent / "candidate"
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    expected = str(Path(candidates.__file__).resolve().parents[2] / "tooling/marp")
    assert (
        candidates.execution_environment(candidate)["TRADINGDEV_PRESENTATION_TOOLCHAIN"]
        == expected
    )
    explicit = str(scenario.source.parent / "installed-toolchain")
    monkeypatch.setenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", explicit)
    assert (
        candidates.execution_environment(candidate)["TRADINGDEV_PRESENTATION_TOOLCHAIN"]
        == explicit
    )


def test_snapshot_retains_workspace_when_real_presentation_cli_reports_cleanup_error(
    scenario: Scenario,
) -> None:
    source = scenario.source
    (source / "tooling").mkdir()
    (source / "scripts").mkdir()
    project = Path(presentations.__file__).resolve().parent
    for name in (
        "__init__.py",
        "environment.py",
        "presentations.py",
        "presentation_errors.py",
        "presentation_html.py",
        "presentation_bundle.py",
        "process_guard.py",
    ):
        _commit(source, f"tooling/{name}", (project / name).read_text())
    _commit(
        source,
        "scripts/check-presentations.sh",
        'exec "$TRADINGDEV_CHECK_PYTHON" cleanup_probe.py\n',
    )
    _commit(
        source,
        "cleanup_probe.py",
        """\
import sys
from pathlib import Path
from tooling import presentations
from tooling.process_guard import ProcessCleanupError

def unverified(*args, **kwargs):
    Path(kwargs['cwd'], 'retained-marker').write_text('unverified cleanup evidence')
    raise ProcessCleanupError('Injected cleanup uncertainty; no compiler was started')

presentations._toolchain = lambda root: root
presentations._command = lambda *args: ['unused-compiler']
presentations.run_captured = unverified
sys.argv = ['presentations', 'build']
raise SystemExit(presentations.main())
""",
    )
    before = _source_state(source)
    tree = _git(source, "rev-parse", "HEAD^{tree}")
    try:
        with pytest.raises(
            ProcessCleanupError, match="Snapshot retained at"
        ) as failure:
            candidates.check_snapshot(
                tree, "presentations", repo=source, timeout_seconds=10
            )
        assert len(scenario.temporary_paths) == 1
        retained = scenario.temporary_paths[0]
        assert retained.is_dir()
        assert str(retained) in str(failure.value)
        assert (retained / "candidate/cleanup_probe.py").is_file()
        markers = list((retained / "runtime").glob("*/retained-marker"))
        assert len(markers) == 1
        assert markers[0].read_text() == "unverified cleanup evidence"
        assert _source_state(source) == before
    finally:
        # The injected uncertainty starts no compiler. The real CLI has exited,
        # and the outer runner verified its descendants before decoding its code.
        for retained in scenario.temporary_paths:
            if retained.exists():
                shutil.rmtree(retained)
    _assert_cleaned(scenario)


@pytest.mark.parametrize("during_setup", [False, True])
def test_merge_candidate_retains_partial_or_ready_repository_on_cleanup_failure(
    scenario: Scenario, monkeypatch: MonkeyPatch, during_setup: bool
) -> None:
    before = _source_state(scenario.source)
    if during_setup:

        def unverified(*args: Any, **kwargs: Any) -> None:
            (kwargs["cwd"] / "diagnostic.txt").write_text("cleanup unverified")
            raise ProcessCleanupError("injected setup cleanup failure")

        monkeypatch.setattr(candidates, "run_captured", unverified)
    try:
        with (
            pytest.raises(
                ProcessCleanupError, match="Merge candidate retained at"
            ) as error,
            candidates.merge_candidate(
                str(scenario.source), "refs/heads/main", "refs/pull/7/head"
            ) as candidate,
        ):
            (candidate.repository / "diagnostic.txt").write_text("cleanup unverified")
            raise ProcessCleanupError("injected check cleanup failure")
        assert len(scenario.temporary_paths) == 1
        root = scenario.temporary_paths[0]
        assert error.value.retained_workspaces == (root.resolve(),)
        assert (root / "diagnostic.txt").read_text() == "cleanup unverified"
        assert _source_state(scenario.source) == before
    finally:
        # The injected failure starts no process; real Git commands have returned
        # through the supervised runner before the body failure is raised.
        for root in scenario.temporary_paths:
            if root.exists():
                shutil.rmtree(root)
    _assert_cleaned(scenario)


def test_documentation_cli_cleanup_failure_retains_outer_candidate(
    scenario: Scenario,
    capfd: CaptureFixture[str],
) -> None:
    code = """\
import sys
from pathlib import Path
from types import SimpleNamespace
from tooling import review_docs
from tooling.process_guard import ProcessCleanupError
candidate = Path.cwd()
plan = SimpleNamespace(prompts=('evidence',), verify=lambda: None, manifest={
    'plan_hash': 'test', 'path_owners': {}, 'selected_scopes': ['tooling'],
    'evidence_utf8_bytes': 8, 'source_pool_utf8_bytes': 8,
    'total_prompt_utf8_bytes': 8})
review_docs.prepare_review = lambda *args, **kwargs: plan
review_docs.require_codex = lambda: 'unused-codex'
def unverified(root, *args, **kwargs):
    (candidate / 'review-path').write_text(str(root))
    (root / 'diagnostic.txt').write_text('unverified; no reviewer launched')
    raise ProcessCleanupError('injected cleanup failure')
review_docs._run = unverified
sys.argv = ['review_docs', '--base', 'base', '--tree', 'tree']
raise SystemExit(review_docs.main())
"""
    try:
        with (
            pytest.raises(ProcessCleanupError, match="Merge candidate retained at"),
            candidates.merge_candidate(
                str(scenario.source), "refs/heads/main", "refs/pull/7/head"
            ) as candidate,
        ):
            process_guard.run_checked(
                [sys.executable, "-c", code],
                cwd=candidate.repository,
                env={
                    **os.environ,
                    "PYTHONPATH": str(
                        Path(review_docs.__file__).resolve().parent.parent
                    ),
                },
                timeout=10,
            )
        root = scenario.temporary_paths[0]
        review_root = Path((root / "review-path").read_text())
        event = json.loads(capfd.readouterr().err)
        assert event["code"] == "process_cleanup_unverified"
        assert event["retained_workspaces"] == [str(review_root.resolve())]
        assert (
            review_root / "diagnostic.txt"
        ).read_text() == "unverified; no reviewer launched"
    finally:
        # The real review CLI has stopped and its descendants were verified by
        # run_checked. Only its injected cleanup status deliberately fails.
        for root in scenario.temporary_paths:
            marker = root / "review-path"
            if marker.exists():
                review_root = Path(marker.read_text())
                if review_root.exists():
                    shutil.rmtree(review_root)
            if root.exists():
                shutil.rmtree(root)
    _assert_cleaned(scenario)
