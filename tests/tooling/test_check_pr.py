"""PR checks validate fetched integration content and discard all temporary state."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import psutil
import pytest
from tooling import review_docs
from tooling.git import candidates, check_pr, gate
from tooling.git.pr_context import PullRequest
from tooling.process_guard import terminate_tree

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from pytest import CaptureFixture, MonkeyPatch


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            f"core.hooksPath={os.devnull}",
            "-c",
            "commit.gpgSign=false",
            *arguments,
        ],
        text=True,
    ).strip()


def _commit(repo: Path, path: str, text: str) -> str:
    (repo / path).write_text(text, encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "--quiet", "-m", f"test: change {path}")
    return _git(repo, "rev-parse", "HEAD")


def _source_state(repo: Path) -> tuple[str, str, str, bytes]:
    return (
        _git(repo, "rev-parse", "HEAD"),
        _git(repo, "show-ref"),
        _git(repo, "status", "--porcelain", "--untracked-files=all"),
        (repo / ".git/config").read_bytes(),
    )


@dataclass(frozen=True)
class Scenario:
    source: Path
    request: PullRequest
    temporary_paths: list[Path]


@pytest.fixture
def scenario(monkeypatch: MonkeyPatch) -> Iterator[Scenario]:
    with TemporaryDirectory(prefix="tradingdev-check-pr-test-") as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        _git(source, "init", "--quiet", "--initial-branch=main", "--template=")
        _git(source, "config", "user.name", "PR tests")
        _git(source, "config", "user.email", "checks@example.invalid")
        _commit(source, "shared.txt", "common\n")
        _git(source, "branch", "feature/work")
        base = _commit(source, "base.txt", "new main behavior\n")
        _git(source, "switch", "--quiet", "feature/work")
        head = _commit(source, "head.txt", "new PR behavior\n")
        _git(source, "update-ref", "refs/pull/7/head", head)
        request = PullRequest(
            repository="owner/project",
            number=7,
            base_ref="main",
            base_sha=base,
            head_repository="contributor/project",
            head_ref="feature/work",
            head_sha=head,
            merged=False,
            state="open",
            merge_commit_sha=None,
            url="https://github.com/owner/project/pull/7",
        )
        monkeypatch.chdir(source)
        monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: request)
        monkeypatch.setattr(check_pr, "remote_url", lambda *_: str(source))
        temporary_paths: list[Path] = []

        def directory(*, prefix: str) -> TemporaryDirectory[str]:
            result = TemporaryDirectory(prefix=prefix, dir=root)
            temporary_paths.append(Path(result.name))
            return result

        monkeypatch.setattr(candidates, "TemporaryDirectory", directory)
        yield Scenario(source, request, temporary_paths)


def _assert_cleaned(scenario: Scenario) -> None:
    assert scenario.temporary_paths
    assert all(not path.exists() for path in scenario.temporary_paths)


@pytest.fixture
def documentation_scenario(scenario: Scenario, monkeypatch: MonkeyPatch) -> Scenario:
    _git(
        scenario.source,
        "switch",
        "--quiet",
        "-c",
        "feature/docs",
        f"{scenario.request.base_sha}^",
    )
    head = _commit(scenario.source, "README.md", "# Documentation change\n")
    _git(scenario.source, "update-ref", "refs/pull/7/head", head)
    request = replace(scenario.request, head_sha=head, head_ref="feature/docs")
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: request)
    return replace(scenario, request=request)


@pytest.mark.parametrize("runner", ["pr", "full"])
@pytest.mark.parametrize("failure", [None, "rejection", "timeout", "interrupt"])
def test_documentation_candidate_uses_merged_tree_and_only_document_checks(
    documentation_scenario: Scenario,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    runner: str,
    failure: str | None,
) -> None:
    scenario = documentation_scenario
    before = _source_state(scenario.source)
    calls: list[str] = []

    def run(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 900,
        temporary_root: Path | None = None,
    ) -> None:
        assert command == ["bash", "scripts/check-docs.sh"]
        assert timeout == 30
        calls.append(command[1])
        # The PR branched before this code change on main. Classification must
        # compare main with the merge tree, not main with the divergent PR head.
        assert (cwd / "base.txt").read_text() == "new main behavior\n"
        assert (cwd / "README.md").read_text() == "# Documentation change\n"
        assert temporary_root is not None
        if failure:
            temporary_root.mkdir()
            (temporary_root / "partial.txt").write_text("partial check output")
        if failure == "rejection":
            raise subprocess.CalledProcessError(1, command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, timeout)
        if failure == "interrupt":
            raise KeyboardInterrupt

    monkeypatch.setattr(candidates, "run", run)

    def check() -> None:
        if runner == "pr":
            check_pr.check_pr(7, snapshot_timeout_seconds=30)
        else:
            gate.check_full("main", "feature/docs", snapshot_timeout_seconds=30)

    if failure:
        error = {
            "rejection": subprocess.CalledProcessError,
            "timeout": subprocess.TimeoutExpired,
            "interrupt": KeyboardInterrupt,
        }[failure]
        with pytest.raises(error):
            check()
    else:
        check()
    assert calls == ["scripts/check-docs.sh"]
    output = capsys.readouterr().out
    assert "Codex documentation review and full pytest passed" not in output
    assert ("checks passed" in output) is (failure is None)
    assert _source_state(scenario.source) == before
    assert len(scenario.temporary_paths) == 2
    _assert_cleaned(scenario)


@pytest.mark.parametrize("mutation", ["remote_ref", "checkout"])
def test_documentation_pr_still_rejects_version_changes(
    documentation_scenario: Scenario,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    mutation: str,
) -> None:
    scenario = documentation_scenario

    def run(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 900,
        temporary_root: Path | None = None,
    ) -> None:
        assert command == ["bash", "scripts/check-docs.sh"]
        if mutation == "remote_ref":
            _git(
                scenario.source,
                "update-ref",
                "refs/heads/main",
                scenario.request.head_sha,
            )
        else:
            (scenario.source / "user-work.txt").write_text("keep this")

    monkeypatch.setattr(candidates, "run", run)
    with pytest.raises(RuntimeError, match="refs changed|checkout changed"):
        check_pr.check_pr(7)
    assert "PR checks passed" not in capsys.readouterr().out
    _assert_cleaned(scenario)


def _forbid_check_side_effects(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        check_pr, "get_pull_request", lambda *_: pytest.fail("PR API called")
    )
    for module in (check_pr, gate, candidates):
        monkeypatch.setattr(module, "git", lambda *_: pytest.fail("Git called"))
    monkeypatch.setattr(
        candidates, "run", lambda *_, **__: pytest.fail("Review called")
    )
    monkeypatch.setattr(
        candidates,
        "TemporaryDirectory",
        lambda **_: pytest.fail("Temporary files created"),
    )


@pytest.mark.parametrize("runner", ["pr", "full"])
@pytest.mark.parametrize(
    "seconds, outer_timeout", [(1, 900), (600, 900), (3600, 3660), (86_400, 86_460)]
)
def test_documentation_timeout_reaches_reviewer_and_preserves_snapshot_budget(
    scenario: Scenario,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    runner: str,
    seconds: int,
    outer_timeout: int,
) -> None:
    before = _source_state(scenario.source)
    calls: list[str] = []

    def run(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 900,
        temporary_root: Path | None = None,
    ) -> None:
        if command[0] == "bash":
            assert timeout == 3600
            calls.append("full")
        else:
            assert timeout == outer_timeout
            assert command[command.index("--timeout-seconds") + 1] == str(seconds)
            calls.append("docs")

    monkeypatch.setattr(candidates, "run", run)
    if runner == "pr":
        check_pr.check_pr(
            7, snapshot_timeout_seconds=3600, documentation_timeout_seconds=seconds
        )
    else:
        gate.check_full(
            "main",
            "feature/work",
            snapshot_timeout_seconds=3600,
            documentation_timeout_seconds=seconds,
        )
    assert calls == ["docs", "full"]
    assert f"Documentation timeout: {seconds} seconds" in capsys.readouterr().out
    assert _source_state(scenario.source) == before
    _assert_cleaned(scenario)


@pytest.mark.parametrize("runner", ["pr", "full"])
@pytest.mark.parametrize("budget", [None, 1_000_000, 2_000_000])
@pytest.mark.parametrize("snapshot_timeout", [900, 3600])
def test_check_validates_both_sides_and_removes_candidate_and_snapshot(
    scenario: Scenario,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    runner: str,
    budget: int | None,
    snapshot_timeout: int,
) -> None:
    before = _source_state(scenario.source)
    calls: list[str] = []

    def run(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 900,
        temporary_root: Path | None = None,
    ) -> None:
        if command[0] == "bash":
            assert timeout == snapshot_timeout
            calls.append("full")
            assert command[1] == "scripts/check-full.sh"
            assert (cwd / "base.txt").read_text() == "new main behavior\n"
            assert (cwd / "head.txt").read_text() == "new PR behavior\n"
            assert temporary_root is not None
            temporary_root.mkdir()
            (temporary_root / "model-output.txt").write_text("temporary output")
        else:
            assert timeout == 900
            assert command[command.index("--timeout-seconds") + 1] == "180"
            calls.append("docs")
            assert command[:3] == [sys.executable, "-m", "tooling.review_docs"]
            assert cwd == Path(review_docs.__file__).resolve().parents[1]
            assert command[command.index("--base") + 1] == scenario.request.base_sha
            assert Path(command[command.index("--repo") + 1]).exists()
            if budget is None:
                assert "--max-evidence-bytes" not in command
                assert review_docs.MAX_EVIDENCE_BYTES == 600_000
            else:
                assert command[command.index("--max-evidence-bytes") + 1] == str(budget)

    monkeypatch.setattr(candidates, "run", run)
    if runner == "pr":
        check_pr.check_pr(
            7, max_evidence_bytes=budget, snapshot_timeout_seconds=snapshot_timeout
        )
    else:
        gate.check_full(
            "main",
            "feature/work",
            max_evidence_bytes=budget,
            snapshot_timeout_seconds=snapshot_timeout,
        )
    assert calls == ["docs", "full"]
    assert _source_state(scenario.source) == before
    assert len(scenario.temporary_paths) == 2
    _assert_cleaned(scenario)
    output = capsys.readouterr().out
    assert f"Snapshot timeout: {snapshot_timeout} seconds" in output
    assert (
        "PR checks passed" if runner == "pr" else "Candidate checks passed"
    ) in output
    assert scenario.request.base_sha in output and scenario.request.head_sha in output


@pytest.mark.parametrize("runner", ["pr", "full"])
@pytest.mark.parametrize(
    "budget, snapshot_timeout, documentation_timeout",
    [
        (None, None, None),
        (1_000_000, None, None),
        (2_000_000, 3600, None),
        (None, 3600, 600),
        (1_000_000, None, 600),
    ],
)
def test_check_cli_forwards_review_options(
    monkeypatch: MonkeyPatch,
    runner: str,
    budget: int | None,
    snapshot_timeout: int | None,
    documentation_timeout: int | None,
) -> None:
    calls: list[tuple[int | None, int, int]] = []

    def check(
        first: int | str,
        second: str,
        *,
        max_evidence_bytes: int | None,
        snapshot_timeout_seconds: int,
        documentation_timeout_seconds: int,
    ) -> None:
        assert (first, second) == (
            (7, "upstream") if runner == "pr" else ("main", "HEAD")
        )
        calls.append(
            (
                max_evidence_bytes,
                snapshot_timeout_seconds,
                documentation_timeout_seconds,
            )
        )

    module = check_pr if runner == "pr" else gate
    monkeypatch.setattr(module, "git", lambda *_: str(Path.cwd()))
    monkeypatch.setattr(module, "check_pr" if runner == "pr" else "check_full", check)
    arguments = (
        ["7", "--remote", "upstream"] if runner == "pr" else ["full", "--base", "main"]
    )
    if budget is not None:
        arguments.extend(["--max-evidence-bytes", str(budget)])
    if snapshot_timeout is not None:
        arguments.extend(["--snapshot-timeout-seconds", str(snapshot_timeout)])
    if documentation_timeout is not None:
        arguments.extend(
            ["--documentation-timeout-seconds", str(documentation_timeout)]
        )
    monkeypatch.setattr(sys, "argv", ["check", *arguments])
    module.main()
    assert calls == [(budget, snapshot_timeout or 900, documentation_timeout or 180)]


@pytest.mark.parametrize("budget", [0, -1])
@pytest.mark.parametrize("entry", ["pr", "full", "verify", "pr_cli", "full_cli"])
def test_invalid_budget_fails_before_git_or_review(
    monkeypatch: MonkeyPatch, budget: int, entry: str
) -> None:
    _forbid_check_side_effects(monkeypatch)
    with pytest.raises(RuntimeError, match="budget must be positive"):
        if entry == "pr":
            check_pr.check_pr(7, max_evidence_bytes=budget)
        elif entry == "full":
            gate.check_full("main", "HEAD", max_evidence_bytes=budget)
        elif entry == "verify":
            candidate = candidates.MergeCandidate(Path.cwd(), "base", "head", "tree")
            candidates.verify_candidate(candidate, max_evidence_bytes=budget)
        else:
            arguments = ["7"] if entry == "pr_cli" else ["full", "--base", "main"]
            monkeypatch.setattr(
                sys, "argv", ["check", *arguments, "--max-evidence-bytes", str(budget)]
            )
            (check_pr if entry == "pr_cli" else gate).main()


@pytest.mark.parametrize("action", ["commit", "push"])
def test_budget_option_is_not_silently_ignored_by_other_git_actions(
    monkeypatch: MonkeyPatch, action: str, capsys: CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gate, "git", lambda *_: pytest.fail("Git called"))
    arguments = ["gate", action, "--max-evidence-bytes", "1000000"]
    if action == "push":
        arguments.extend(["--destination", "git@github.com:owner/project.git"])
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as failure:
        gate.main()
    assert failure.value.code == 2
    assert "only valid for full checks" in capsys.readouterr().err


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


def test_fetched_head_must_match_api_before_running_checks(
    scenario: Scenario, monkeypatch: MonkeyPatch
) -> None:
    request = replace(scenario.request, head_sha="a" * 40)
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: request)
    monkeypatch.setattr(
        check_pr,
        "verify_candidate",
        lambda _, **__: pytest.fail("mismatch reached checks"),
    )
    with pytest.raises(RuntimeError, match="moved while fetching"):
        check_pr.check_pr(7)
    _assert_cleaned(scenario)


@pytest.mark.parametrize("field", ["head_sha", "state", "base_ref"])
def test_pr_movement_during_checks_invalidates_result(
    scenario: Scenario,
    monkeypatch: MonkeyPatch,
    field: str,
    capsys: CaptureFixture[str],
) -> None:
    changed = {
        "state": replace(scenario.request, state="closed"),
        "base_ref": replace(scenario.request, base_ref="release"),
        "head_sha": replace(scenario.request, head_sha="a" * 40),
    }[field]
    responses = iter((scenario.request, changed))
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: next(responses))
    monkeypatch.setattr(check_pr, "verify_candidate", lambda _, **__: None)
    with pytest.raises(RuntimeError, match="changed during checks"):
        check_pr.check_pr(7)
    assert "PR checks passed" not in capsys.readouterr().out
    _assert_cleaned(scenario)


def test_github_provisional_merge_sha_update_does_not_invalidate_same_content(
    scenario: Scenario, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    responses = iter(
        (scenario.request, replace(scenario.request, merge_commit_sha="a" * 40))
    )
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: next(responses))
    monkeypatch.setattr(check_pr, "verify_candidate", lambda _, **__: None)
    check_pr.check_pr(7)
    assert "PR checks passed" in capsys.readouterr().out
    _assert_cleaned(scenario)


@pytest.mark.parametrize("mode", ["closed", "merged", "wrong_base", "deleted_fork"])
def test_ineligible_pr_never_fetches_or_checks(
    scenario: Scenario, monkeypatch: MonkeyPatch, mode: str
) -> None:
    request = {
        "closed": replace(scenario.request, state="closed"),
        "merged": replace(scenario.request, state="closed", merged=True),
        "wrong_base": replace(scenario.request, base_ref="release"),
        "deleted_fork": replace(scenario.request, head_repository=None),
    }[mode]
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: request)
    monkeypatch.setattr(
        check_pr,
        "verify_candidate",
        lambda _, **__: pytest.fail("ineligible PR reached checks"),
    )
    with pytest.raises(RuntimeError):
        check_pr.check_pr(7)
    assert not scenario.temporary_paths


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


@pytest.mark.parametrize("runner", ["pr", "full"])
def test_dirty_runner_is_rejected_before_fetch(
    scenario: Scenario, monkeypatch: MonkeyPatch, runner: str
) -> None:
    (scenario.source / "uncommitted.txt").write_text("user work")
    if runner == "pr":
        with pytest.raises(RuntimeError, match="clean, committed"):
            check_pr.check_pr(7)
    else:
        with pytest.raises(RuntimeError, match="Commit the task stages"):
            gate.check_full("main", "HEAD")
    assert not scenario.temporary_paths
    assert (scenario.source / "uncommitted.txt").read_text() == "user work"


@pytest.mark.parametrize("mutation", ["new_file", "new_commit"])
def test_runner_changes_during_pr_check_are_detected(
    scenario: Scenario, monkeypatch: MonkeyPatch, mutation: str
) -> None:
    def mutate(
        _: candidates.MergeCandidate,
        *,
        max_evidence_bytes: int | None,
        snapshot_timeout_seconds: int,
        documentation_timeout_seconds: int,
    ) -> None:
        if mutation == "new_file":
            (scenario.source / "user-work.txt").write_text("keep this")
        else:
            _commit(scenario.source, "user-work.txt", "keep this")

    monkeypatch.setattr(check_pr, "verify_candidate", mutate)
    with pytest.raises(RuntimeError, match="runner checkout changed"):
        check_pr.check_pr(7)
    assert (scenario.source / "user-work.txt").read_text() == "keep this"
    _assert_cleaned(scenario)


def test_manual_full_uses_explicit_base_and_preserves_source(
    scenario: Scenario, monkeypatch: MonkeyPatch
) -> None:
    before = _source_state(scenario.source)
    observed: list[candidates.MergeCandidate] = []
    monkeypatch.setattr(
        gate, "verify_candidate", lambda candidate, **_: observed.append(candidate)
    )
    gate.check_full("main", "feature/work")
    assert len(observed) == 1
    assert observed[0].base == scenario.request.base_sha
    assert observed[0].head == scenario.request.head_sha
    assert _source_state(scenario.source) == before
    _assert_cleaned(scenario)


@pytest.mark.parametrize("ref", ["main", "feature/work"])
def test_manual_full_rejects_ref_movement(
    scenario: Scenario, monkeypatch: MonkeyPatch, ref: str
) -> None:
    def mutate(
        _: candidates.MergeCandidate,
        *,
        max_evidence_bytes: int | None,
        snapshot_timeout_seconds: int,
        documentation_timeout_seconds: int,
    ) -> None:
        replacement = (
            scenario.request.head_sha if ref == "main" else scenario.request.base_sha
        )
        _git(scenario.source, "update-ref", f"refs/heads/{ref}", replacement)

    monkeypatch.setattr(gate, "verify_candidate", mutate)
    with pytest.raises(RuntimeError, match="compared refs changed"):
        gate.check_full("main", "feature/work")
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


@pytest.mark.parametrize("lag", ["initial", "final", "both"])
def test_api_base_lag_does_not_override_live_remote_base(
    scenario: Scenario, monkeypatch: MonkeyPatch, lag: str, capsys: CaptureFixture[str]
) -> None:
    first = (
        replace(scenario.request, base_sha="a" * 40)
        if lag in ("initial", "both")
        else scenario.request
    )
    final = (
        replace(scenario.request, base_sha="b" * 40)
        if lag in ("final", "both")
        else scenario.request
    )
    responses = iter((first, final))
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: next(responses))
    checked: list[candidates.MergeCandidate] = []
    monkeypatch.setattr(
        check_pr, "verify_candidate", lambda candidate, **_: checked.append(candidate)
    )
    remote_calls: list[str] = []

    def source(remote: str) -> str:
        remote_calls.append(remote)
        return str(scenario.source)

    monkeypatch.setattr(check_pr, "remote_url", source)
    check_pr.check_pr(7)
    assert checked[0].base == scenario.request.base_sha
    assert remote_calls == ["origin"]
    assert "PR checks passed" in capsys.readouterr().out
    _assert_cleaned(scenario)


@pytest.mark.parametrize("ref", ["refs/heads/main", "refs/pull/7/head"])
@pytest.mark.parametrize("change", ["move", "delete"])
def test_actual_remote_ref_movement_invalidates_unchanged_api_response(
    scenario: Scenario,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    ref: str,
    change: str,
) -> None:
    def mutate(
        _: candidates.MergeCandidate,
        *,
        max_evidence_bytes: int | None,
        snapshot_timeout_seconds: int,
        documentation_timeout_seconds: int,
    ) -> None:
        if change == "delete":
            _git(scenario.source, "update-ref", "-d", ref)
        else:
            replacement = (
                scenario.request.head_sha
                if ref == "refs/heads/main"
                else scenario.request.base_sha
            )
            _git(scenario.source, "update-ref", ref, replacement)

    monkeypatch.setattr(check_pr, "verify_candidate", mutate)
    with pytest.raises(RuntimeError, match="changed during checks|disappeared"):
        check_pr.check_pr(7)
    assert "PR checks passed" not in capsys.readouterr().out
    _assert_cleaned(scenario)


@pytest.mark.parametrize(
    "output",
    [
        "",
        "a" * 40 + "\trefs/heads/main\n",
        "not-a-sha\trefs/heads/main\n",
        "a" * 40 + "\trefs/heads/main\n" + "b" * 40 + "\trefs/heads/main\n",
        "a" * 40 + "\trefs/unexpected/main\n",
        "a" * 40 + " refs/heads/main\n",
    ],
)
def test_advertised_ref_response_must_contain_two_valid_unique_refs(
    scenario: Scenario, monkeypatch: MonkeyPatch, output: str
) -> None:
    def captured(
        command: list[str], *, cwd: Path, env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        assert command == [
            "git",
            "ls-remote",
            "--refs",
            "--",
            str(scenario.source),
            "refs/heads/main",
            "refs/pull/7/head",
        ]
        assert cwd.resolve() == scenario.source.resolve()
        assert timeout == 120
        assert "GIT_DIR" not in env
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(check_pr, "run_captured", captured)
    with pytest.raises(RuntimeError, match="invalid Git data|disappeared"):
        check_pr._remote_refs(
            str(scenario.source), "refs/heads/main", "refs/pull/7/head"
        )


def test_remote_ref_verification_failure_hides_credential_bearing_command(
    scenario: Scenario, monkeypatch: MonkeyPatch
) -> None:
    def captured(
        command: list[str], *, cwd: Path, env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, timeout, stderr="private-token")

    monkeypatch.setattr(check_pr, "run_captured", captured)
    with pytest.raises(RuntimeError, match="Remote ref verification failed") as failure:
        check_pr._remote_refs(
            "https://private-token@github.com/owner/project",
            "refs/heads/main",
            "refs/pull/7/head",
        )
    assert "private-token" not in str(failure.value)


@pytest.mark.parametrize("timeout", [0, -1])
@pytest.mark.parametrize(
    "entry", ["pr", "full", "verify", "snapshot", "pr_cli", "full_cli"]
)
def test_invalid_snapshot_timeout_fails_before_git_or_review(
    monkeypatch: MonkeyPatch, timeout: int, entry: str
) -> None:
    _forbid_check_side_effects(monkeypatch)
    with pytest.raises(
        RuntimeError, match="Snapshot timeout must be a positive integer"
    ):
        if entry == "pr":
            check_pr.check_pr(7, snapshot_timeout_seconds=timeout)
        elif entry == "full":
            gate.check_full("main", "HEAD", snapshot_timeout_seconds=timeout)
        elif entry == "verify":
            candidate = candidates.MergeCandidate(Path.cwd(), "base", "head", "tree")
            candidates.verify_candidate(candidate, snapshot_timeout_seconds=timeout)
        elif entry == "snapshot":
            candidates.check_snapshot("tree", "full", timeout_seconds=timeout)
        else:
            arguments = ["7"] if entry == "pr_cli" else ["full", "--base", "main"]
            monkeypatch.setattr(
                sys,
                "argv",
                ["check", *arguments, "--snapshot-timeout-seconds", str(timeout)],
            )
            (check_pr if entry == "pr_cli" else gate).main()


@pytest.mark.parametrize(
    "timeout", [0, -1, 86_401, True, 1.5, float("nan"), float("inf")]
)
@pytest.mark.parametrize("entry", ["pr", "full", "verify"])
def test_invalid_documentation_timeout_fails_before_git_or_review(
    monkeypatch: MonkeyPatch, timeout: Any, entry: str
) -> None:
    _forbid_check_side_effects(monkeypatch)
    with pytest.raises(RuntimeError, match="Documentation timeout must be"):
        if entry == "pr":
            check_pr.check_pr(7, documentation_timeout_seconds=timeout)
        elif entry == "full":
            gate.check_full("main", "HEAD", documentation_timeout_seconds=timeout)
        else:
            candidate = candidates.MergeCandidate(Path.cwd(), "base", "head", "tree")
            candidates.verify_candidate(
                candidate, documentation_timeout_seconds=timeout
            )


@pytest.mark.parametrize("timeout", [0, -1, 86_401])
@pytest.mark.parametrize("entry", ["pr", "full"])
def test_documentation_timeout_cli_rejects_out_of_range_before_git(
    monkeypatch: MonkeyPatch, timeout: int, entry: str
) -> None:
    _forbid_check_side_effects(monkeypatch)
    arguments = ["7"] if entry == "pr" else ["full", "--base", "main"]
    monkeypatch.setattr(
        sys,
        "argv",
        ["check", *arguments, "--documentation-timeout-seconds", str(timeout)],
    )
    with pytest.raises(RuntimeError, match="Documentation timeout must be"):
        (check_pr if entry == "pr" else gate).main()


@pytest.mark.parametrize("timeout", ["nan", "inf", "-inf", "1.5", "text"])
@pytest.mark.parametrize("entry", ["pr", "full"])
@pytest.mark.parametrize("option", ["snapshot", "documentation"])
def test_timeout_cli_rejects_noninteger_before_git(
    monkeypatch: MonkeyPatch, timeout: str, entry: str, option: str
) -> None:
    module = check_pr if entry == "pr" else gate
    _forbid_check_side_effects(monkeypatch)
    arguments = ["7"] if entry == "pr" else ["full", "--base", "main"]
    monkeypatch.setattr(
        sys, "argv", ["check", *arguments, f"--{option}-timeout-seconds={timeout}"]
    )
    with pytest.raises(SystemExit) as failure:
        module.main()
    assert failure.value.code == 2


@pytest.mark.parametrize("action", ["commit", "push"])
@pytest.mark.parametrize("option", ["snapshot", "documentation"])
def test_timeout_option_is_rejected_for_other_git_actions(
    monkeypatch: MonkeyPatch,
    action: str,
    option: str,
    capsys: CaptureFixture[str],
) -> None:
    monkeypatch.setattr(gate, "git", lambda *_: pytest.fail("Git called"))
    arguments = ["gate", action, f"--{option}-timeout-seconds", "3600"]
    if action == "push":
        arguments.extend(["--destination", "git@github.com:owner/project.git"])
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as failure:
        gate.main()
    assert failure.value.code == 2
    assert "only valid for full checks" in capsys.readouterr().err


@pytest.mark.parametrize("profile", ["full", "docs"])
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
