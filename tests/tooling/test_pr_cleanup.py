"""Exercise deterministic cleanup against real local Git remotes and fake PR data."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest
from tooling.git import cleanup_pr
from tooling.git.pr_context import PullRequest

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class Repository:
    root: Path
    pr: PullRequest

    def git(self, *arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout.strip()

    def head(self, branch: str = "feature/work") -> str:
        return self.git("rev-parse", f"refs/heads/{branch}")

    def commit(self, filename: str, content: str) -> str:
        (self.root / filename).write_text(content, encoding="utf-8")
        self.git("add", filename)
        self.git("commit", "-qm", "feat: fixture change")
        return self.git("rev-parse", "HEAD")


@pytest.fixture
def repository(monkeypatch: pytest.MonkeyPatch) -> Iterator[Repository]:
    for key in os.environ:
        if key.startswith("GIT_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    with TemporaryDirectory(prefix="tradingdev-pr-cleanup-test-") as temporary:
        directory = Path(temporary)
        root = directory / "repository"
        root.mkdir()
        monkeypatch.chdir(root)
        placeholder = PullRequest(
            repository="owner/project",
            number=123,
            base_ref="main",
            base_sha="",
            head_repository="owner/project",
            head_ref="feature/work",
            head_sha="",
            merged=True,
            state="closed",
            merge_commit_sha=None,
            url="https://github.com/owner/project/pull/123",
        )
        repo = Repository(root, placeholder)
        repo.git("init", "--quiet", "--template=", "--initial-branch=main")
        repo.git("config", "user.name", "Cleanup Test")
        repo.git("config", "user.email", "cleanup@example.invalid")
        repo.git("config", "core.hooksPath", os.devnull)
        repo.git("config", "commit.gpgsign", "false")
        repo.git("init", "--quiet", "--bare", "--template=", str(directory / "remote"))
        repo.git("remote", "add", "origin", str(directory / "remote"))
        repo.commit("base.txt", "base\n")
        repo.git("push", "--set-upstream", "origin", "main")
        repo.git("switch", "-c", "feature/work")
        head = repo.commit("feature.txt", "feature\n")
        repo.git("push", "--set-upstream", "origin", "feature/work")
        repo.git("switch", "main")
        repo.git("merge", "--no-ff", "--no-edit", "feature/work")
        merge = repo.git("rev-parse", "HEAD")
        repo.git("push", "origin", "main")
        repo.pr = replace(
            placeholder, base_sha=merge, head_sha=head, merge_commit_sha=merge
        )

        def get_pr(number: int, remote: str = "origin") -> PullRequest:
            assert number == 123
            assert remote == "origin"
            return repo.pr

        def repository_for_remote(remote: str) -> str:
            return "other/fork" if remote == "fork" else "owner/project"

        monkeypatch.setattr(cleanup_pr, "get_pull_request", get_pr)
        monkeypatch.setattr(cleanup_pr, "repository_for_remote", repository_for_remote)
        yield repo
        monkeypatch.chdir(directory.parent)
    assert not directory.exists()


def test_cleanup_fetches_main_and_deletes_only_the_explicit_local_branch(
    repository: Repository,
) -> None:
    repo = repository
    original_head = repo.git("rev-parse", "HEAD")
    remote_refs = repo.git("ls-remote", "origin")
    repo.git("branch", "feature/unrelated", repo.pr.head_sha)
    repo.git("update-ref", "refs/remotes/origin/main", repo.git("rev-parse", "HEAD^1"))
    (repo.root / "untracked-user-file").write_text("preserve", encoding="utf-8")

    result = cleanup_pr.cleanup_pull_request(123, branch="feature/work")

    assert "Deleted local branch" in result
    assert repo.git("for-each-ref", "refs/heads/feature/work") == ""
    assert repo.head("feature/unrelated") == repo.pr.head_sha
    assert repo.git("rev-parse", "HEAD") == original_head
    assert repo.git("rev-parse", "origin/main") == repo.pr.merge_commit_sha
    assert repo.git("ls-remote", "origin") == remote_refs
    assert (repo.root / "untracked-user-file").read_text() == "preserve"


@pytest.mark.parametrize("branch", ["main", "master", "production"])
def test_primary_branches_are_preserved(repository: Repository, branch: str) -> None:
    if branch != "main":
        repository.git("branch", branch, repository.pr.head_sha)
        repository.git("config", "tradingdev.primaryBranch", branch)
    expected = repository.head(branch)
    with pytest.raises(RuntimeError, match="Protected branch"):
        cleanup_pr.cleanup_pull_request(123, branch=branch)
    assert repository.head(branch) == expected


def test_configured_long_lived_branch_is_preserved(repository: Repository) -> None:
    repository.git("config", "--add", "tradingdev.protectedBranch", "feature/work")
    with pytest.raises(RuntimeError, match="Protected branch"):
        cleanup_pr.cleanup_pull_request(123, branch="feature/work")
    assert repository.head() == repository.pr.head_sha


@pytest.mark.parametrize("branch", ["feature/missing", "feature", "FEATURE/work"])
def test_cleanup_requires_the_exact_existing_local_branch(
    repository: Repository, branch: str
) -> None:
    with pytest.raises(RuntimeError, match="exact local branch"):
        cleanup_pr.cleanup_pull_request(123, branch=branch)
    assert repository.head() == repository.pr.head_sha


@pytest.mark.parametrize("case", ["closed", "open", "missing_source", "wrong_target"])
def test_unverifiable_or_unmerged_pr_is_preserved(
    repository: Repository, case: str
) -> None:
    repo = repository
    if case == "closed":
        repo.pr = replace(repo.pr, merged=False)
    elif case == "open":
        repo.pr = replace(repo.pr, state="open", merged=False)
    elif case == "missing_source":
        repo.pr = replace(repo.pr, head_repository=None)
    else:
        repo.pr = replace(repo.pr, base_ref="develop")
    with pytest.raises(RuntimeError):
        cleanup_pr.cleanup_pull_request(123, branch="feature/work")
    assert repo.head() == repo.pr.head_sha


def test_api_failure_preserves_branch(
    repository: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing_pr(number: int, remote: str = "origin") -> PullRequest:
        raise RuntimeError("PR not found or authentication failed")

    monkeypatch.setattr(cleanup_pr, "get_pull_request", missing_pr)
    with pytest.raises(RuntimeError, match="PR not found"):
        cleanup_pr.cleanup_pull_request(123, branch="feature/work")
    assert repository.head() == repository.pr.head_sha


def test_failed_fetch_preserves_branch(repository: Repository) -> None:
    repository.git("remote", "set-url", "origin", str(repository.root / "missing"))
    with pytest.raises(RuntimeError, match="does not appear to be a git repository"):
        cleanup_pr.cleanup_pull_request(123, branch="feature/work")
    assert repository.head() == repository.pr.head_sha


@pytest.mark.parametrize(
    "upstream", ["missing", "other_branch", "fork", "multiple_remotes", "multiple_refs"]
)
def test_upstream_must_match_repository_and_branch(
    repository: Repository, upstream: str
) -> None:
    repo = repository
    if upstream == "missing":
        repo.git("config", "--unset", "branch.feature/work.remote")
    elif upstream == "other_branch":
        repo.git("config", "branch.feature/work.merge", "refs/heads/other")
    elif upstream == "fork":
        repo.git("remote", "add", "fork", repo.git("remote", "get-url", "origin"))
        repo.git("config", "branch.feature/work.remote", "fork")
    elif upstream == "multiple_remotes":
        repo.git("config", "--add", "branch.feature/work.remote", "other")
    else:
        repo.git("config", "--add", "branch.feature/work.merge", "refs/heads/other")
    with pytest.raises(RuntimeError, match="upstream"):
        cleanup_pr.cleanup_pull_request(123, branch="feature/work")
    assert repo.head() == repo.pr.head_sha


def test_unambiguously_mapped_fork_branch_can_be_cleaned(
    repository: Repository,
) -> None:
    repo = repository
    repo.git("remote", "add", "fork", repo.git("remote", "get-url", "origin"))
    repo.git("config", "branch.feature/work.remote", "fork")
    repo.pr = replace(repo.pr, head_repository="other/fork")
    cleanup_pr.cleanup_pull_request(123, branch="feature/work")
    assert repo.git("for-each-ref", "refs/heads/feature/work") == ""


@pytest.mark.parametrize("dirty", [False, True])
def test_checked_out_worktree_is_preserved(repository: Repository, dirty: bool) -> None:
    repo = repository
    worktree = repo.root.parent / "task-worktree"
    repo.git("worktree", "add", str(worktree), "feature/work")
    marker = worktree / "feature.txt"
    if dirty:
        marker.write_text("unfinished work\n", encoding="utf-8")
    expected = marker.read_bytes()
    with pytest.raises(RuntimeError, match="checked out in a worktree"):
        cleanup_pr.cleanup_pull_request(123, branch="feature/work")
    assert repo.head() == repo.pr.head_sha
    assert marker.read_bytes() == expected


def test_advanced_local_branch_is_preserved(repository: Repository) -> None:
    repo = repository
    repo.git("switch", "feature/work")
    advanced = repo.commit("next.txt", "new unmerged work\n")
    repo.git("switch", "main")
    with pytest.raises(RuntimeError, match="does not match the merged PR head"):
        cleanup_pr.cleanup_pull_request(123, branch="feature/work")
    assert repo.head() == advanced


@pytest.mark.parametrize("method", ["squash", "rebase"])
def test_rewritten_merge_without_source_ancestry_is_preserved(
    repository: Repository, method: str
) -> None:
    repo = repository
    repo.git("switch", "-c", "feature/squashed")
    head = repo.commit("squashed.txt", "new feature\n")
    repo.git("push", "--set-upstream", "origin", "feature/squashed")
    repo.git("switch", "main")
    if method == "squash":
        repo.git("merge", "--squash", "feature/squashed")
        repo.git("commit", "-qm", "feat: squash merge")
    else:
        repo.commit("meanwhile.txt", "main advanced\n")
        repo.git("cherry-pick", head)
    merge = repo.git("rev-parse", "HEAD")
    repo.git("push", "origin", "main")
    repo.pr = replace(
        repo.pr,
        base_sha=merge,
        head_ref="feature/squashed",
        head_sha=head,
        merge_commit_sha=merge,
    )
    with pytest.raises(RuntimeError, match="Squash/rebase"):
        cleanup_pr.cleanup_pull_request(123, branch="feature/squashed")
    assert repo.head("feature/squashed") == head


def test_merge_commit_missing_from_primary_is_preserved(repository: Repository) -> None:
    repo = repository
    repo.git("switch", "-c", "feature/unmerged")
    unrelated_merge = repo.commit("unmerged.txt", "not on main\n")
    repo.git("switch", "main")
    repo.pr = replace(repo.pr, merge_commit_sha=unrelated_merge)
    with pytest.raises(RuntimeError, match="must both be contained"):
        cleanup_pr.cleanup_pull_request(123, branch="feature/work")
    assert repo.head() == repo.pr.head_sha


def test_symbolic_branch_cannot_delete_its_target(repository: Repository) -> None:
    repo = repository
    expected = repo.head("main")
    repo.git("symbolic-ref", "refs/heads/feature/alias", "refs/heads/main")
    with pytest.raises(RuntimeError, match="Symbolic branch"):
        cleanup_pr.cleanup_pull_request(123, branch="feature/alias")
    assert repo.head("main") == expected
    assert repo.git("symbolic-ref", "refs/heads/feature/alias") == "refs/heads/main"


def test_pr_changes_during_fetch_preserve_branch(
    repository: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def changing_pr(number: int, remote: str = "origin") -> PullRequest:
        nonlocal calls
        calls += 1
        return (
            repository.pr if calls == 1 else replace(repository.pr, head_ref="new-name")
        )

    monkeypatch.setattr(cleanup_pr, "get_pull_request", changing_pr)
    with pytest.raises(RuntimeError, match="metadata changed"):
        cleanup_pr.cleanup_pull_request(123, branch="feature/work")
    assert repository.head() == repository.pr.head_sha


def test_atomic_delete_preserves_a_concurrently_advanced_branch(
    repository: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_git = cleanup_pr._git
    replacement = repository.git("rev-parse", "main")

    def concurrent_git(*arguments: str, allow_missing: bool = False) -> str:
        if arguments[:3] == ("update-ref", "--no-deref", "-d"):
            repository.git("update-ref", "refs/heads/feature/work", replacement)
        return original_git(*arguments, allow_missing=allow_missing)

    monkeypatch.setattr(cleanup_pr, "_git", concurrent_git)
    with pytest.raises(RuntimeError, match="cannot lock ref"):
        cleanup_pr.cleanup_pull_request(123, branch="feature/work")
    assert repository.head() == replacement


def test_cli_requires_explicit_branch_designation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["cleanup_pr.py", "123"])
    with pytest.raises(SystemExit) as error:
        cleanup_pr.main()
    assert error.value.code == 2
