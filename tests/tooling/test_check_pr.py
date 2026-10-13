"""PR checks validate fetched integration content and discard all temporary state."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest
from tooling import review_docs
from tooling.git import candidates, check_pr, gate
from tooling.git.pr_context import PullRequest

if TYPE_CHECKING:
    from collections.abc import Iterator

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


@pytest.mark.parametrize("runner", ["pr", "full"])
@pytest.mark.parametrize("budget", [None, 1_000_000, 2_000_000])
def test_check_validates_both_sides_and_removes_candidate_and_snapshot(
    scenario: Scenario,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    runner: str,
    budget: int | None,
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
        assert timeout == 900
        if command[0] == "bash":
            calls.append("full")
            assert command[1] == "scripts/check-full.sh"
            assert (cwd / "base.txt").read_text() == "new main behavior\n"
            assert (cwd / "head.txt").read_text() == "new PR behavior\n"
            assert temporary_root is not None
            temporary_root.mkdir()
            (temporary_root / "model-output.txt").write_text("temporary output")
        else:
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
        check_pr.check_pr(7, max_evidence_bytes=budget)
    else:
        gate.check_full("main", "feature/work", max_evidence_bytes=budget)
    assert calls == ["docs", "full"]
    assert _source_state(scenario.source) == before
    assert len(scenario.temporary_paths) == 2
    _assert_cleaned(scenario)
    output = capsys.readouterr().out
    assert ("PR checks passed" if runner == "pr" else "Full checks passed") in output
    assert scenario.request.base_sha in output and scenario.request.head_sha in output


@pytest.mark.parametrize("runner", ["pr", "full"])
@pytest.mark.parametrize("budget", [None, 1_000_000, 2_000_000])
def test_check_cli_forwards_evidence_budget(
    monkeypatch: MonkeyPatch, runner: str, budget: int | None
) -> None:
    calls: list[int | None] = []

    def check(first: int | str, second: str, *, max_evidence_bytes: int | None) -> None:
        assert (first, second) == (
            (7, "upstream") if runner == "pr" else ("main", "HEAD")
        )
        calls.append(max_evidence_bytes)

    module = check_pr if runner == "pr" else gate
    monkeypatch.setattr(module, "git", lambda *_: str(Path.cwd()))
    monkeypatch.setattr(module, "check_pr" if runner == "pr" else "check_full", check)
    arguments = (
        ["7", "--remote", "upstream"] if runner == "pr" else ["full", "--base", "main"]
    )
    if budget is not None:
        arguments.extend(["--max-evidence-bytes", str(budget)])
    monkeypatch.setattr(sys, "argv", ["check", *arguments])
    module.main()
    assert calls == [budget]


@pytest.mark.parametrize("budget", [0, -1])
@pytest.mark.parametrize("entry", ["pr", "full", "verify", "pr_cli", "full_cli"])
def test_invalid_budget_fails_before_git_or_review(
    monkeypatch: MonkeyPatch, budget: int, entry: str
) -> None:
    monkeypatch.setattr(
        check_pr, "get_pull_request", lambda *_: pytest.fail("PR API called")
    )
    monkeypatch.setattr(check_pr, "git", lambda *_: pytest.fail("Git called"))
    monkeypatch.setattr(gate, "git", lambda *_: pytest.fail("Git called"))
    monkeypatch.setattr(
        candidates, "run", lambda *_, **__: pytest.fail("Review called")
    )
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
    monkeypatch.setattr(
        sys, "argv", ["gate", action, "--max-evidence-bytes", "1000000"]
    )
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


@pytest.mark.parametrize("failure", [RuntimeError("failed"), KeyboardInterrupt()])
def test_verifier_failure_or_interrupt_cleans_up_and_preserves_source(
    scenario: Scenario, monkeypatch: MonkeyPatch, failure: BaseException
) -> None:
    before = _source_state(scenario.source)

    def fail(
        candidate: candidates.MergeCandidate, *, max_evidence_bytes: int | None
    ) -> None:
        (candidate.repository / "partial-model-output").write_text("temporary result")
        raise failure

    monkeypatch.setattr(check_pr, "verify_candidate", fail)
    with pytest.raises(type(failure)):
        check_pr.check_pr(7)
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
    def mutate(_: candidates.MergeCandidate, *, max_evidence_bytes: int | None) -> None:
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
    def mutate(_: candidates.MergeCandidate, *, max_evidence_bytes: int | None) -> None:
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
    def mutate(_: candidates.MergeCandidate, *, max_evidence_bytes: int | None) -> None:
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
