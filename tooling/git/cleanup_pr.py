"""Delete only an explicitly designated, verifiably merged local PR branch."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from tooling.git.pr_context import (
    PullRequest,
    get_pull_request,
    repository_for_remote,
)
from tooling.process_guard import run_captured


def _git(*arguments: str, allow_missing: bool = False) -> str:
    result = run_captured(
        ["git", *arguments], cwd=Path.cwd(), env=dict(os.environ), timeout=120
    )
    if result.returncode and not (allow_missing and result.returncode == 1):
        diagnostic = "\n".join(
            value.strip() for value in (result.stdout, result.stderr) if value.strip()
        )
        raise RuntimeError(diagnostic or f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def _primary_branch() -> str:
    result = subprocess.run(
        [
            "bash",
            str(
                Path(__file__).resolve().parents[2] / "scripts/detect-primary-branch.sh"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Cannot determine primary branch.")
    return result.stdout.strip()


def _assert_candidate(pr: PullRequest, *, repository: str, primary: str) -> None:
    if pr.repository.casefold() != repository.casefold() or pr.base_ref != primary:
        raise RuntimeError(
            "PR repository or target does not match this primary branch."
        )
    if not pr.merged or pr.state != "closed" or pr.merge_commit_sha is None:
        raise RuntimeError("PR has not been merged; preserving the local branch.")
    if pr.head_repository is None:
        raise RuntimeError("PR source repository is unavailable; preserving branch.")


def _assert_upstream(branch: str, pr: PullRequest) -> None:
    remotes = _git(
        "config", "--get-all", f"branch.{branch}.remote", allow_missing=True
    ).splitlines()
    merge_refs = _git(
        "config", "--get-all", f"branch.{branch}.merge", allow_missing=True
    ).splitlines()
    if (
        len(remotes) != 1
        or remotes[0] == "."
        or merge_refs != [f"refs/heads/{pr.head_ref}"]
    ):
        raise RuntimeError("Branch upstream does not identify this PR source branch.")
    if (
        repository_for_remote(remotes[0]).casefold()
        != str(pr.head_repository).casefold()
    ):
        raise RuntimeError("Branch upstream repository does not match this PR source.")


def _assert_not_checked_out(ref: str) -> None:
    for field in _git("worktree", "list", "--porcelain", "-z").split("\0"):
        if field == f"branch {ref}":
            raise RuntimeError(
                "Branch is checked out in a worktree; preserve it and switch away "
                "or explicitly remove that worktree before retrying."
            )


def _assert_ancestor(commit: str, base: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, base],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode:
        raise RuntimeError(
            "The PR source and merge commits must both be contained in the fetched "
            "primary branch. Squash/rebase or rewritten history requires manual review."
        )


def cleanup_pull_request(number: int, *, branch: str, remote: str = "origin") -> str:
    """Fetch remote state and conditionally delete exactly one local branch ref."""
    if number <= 0:
        raise RuntimeError("PR number must be positive.")
    _git("check-ref-format", "--branch", branch)
    primary = _primary_branch()
    protected = set(
        _git(
            "config", "--get-all", "tradingdev.protectedBranch", allow_missing=True
        ).splitlines()
    )
    if branch in {primary, *protected}:
        raise RuntimeError(f"Protected branch {branch!r} cannot be cleaned up.")
    ref = f"refs/heads/{branch}"
    if _git("for-each-ref", "--format=%(refname)", ref).splitlines() != [ref]:
        raise RuntimeError("The exact local branch name does not exist.")
    if _git("symbolic-ref", "--quiet", ref, allow_missing=True):
        raise RuntimeError("Symbolic branch references require manual review.")
    expected_head = _git("rev-parse", "--verify", f"{ref}^{{commit}}")
    _assert_not_checked_out(ref)
    repository = repository_for_remote(remote)
    pr = get_pull_request(number, remote)
    _assert_candidate(pr, repository=repository, primary=primary)
    _assert_upstream(branch, pr)

    target_ref = f"refs/remotes/{remote}/{primary}"
    _git("check-ref-format", target_ref)
    _git(
        "fetch",
        "--no-tags",
        "--no-prune",
        "--no-recurse-submodules",
        "--",
        remote,
        f"refs/heads/{primary}:{target_ref}",
    )
    base = _git("rev-parse", "--verify", f"{target_ref}^{{commit}}")
    current_pr = get_pull_request(number, remote)
    _assert_candidate(current_pr, repository=repository, primary=primary)
    if current_pr != pr:
        raise RuntimeError("PR metadata changed during cleanup; preserving branch.")
    if expected_head != current_pr.head_sha:
        raise RuntimeError(
            "Local branch does not match the merged PR head; preserving it."
        )
    assert current_pr.merge_commit_sha is not None
    _assert_ancestor(current_pr.head_sha, base)
    _assert_ancestor(current_pr.merge_commit_sha, base)
    _assert_upstream(branch, current_pr)
    _assert_not_checked_out(ref)

    # Compare-and-delete refuses a concurrently advanced/replaced branch. Keep the
    # branch config: deleting a separate config section could affect a new branch
    # recreated after this atomic ref deletion. Never delete remote refs/worktrees.
    _git("update-ref", "--no-deref", "-d", ref, expected_head)
    return (
        f"Deleted local branch {branch!r} at {expected_head[:12]} for "
        f"{repository}#{number}. Remote branches and worktrees were preserved; "
        "local branch configuration was retained."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, help="Merged PR number")
    parser.add_argument("--branch", required=True, help="Explicit local task branch")
    parser.add_argument("--remote", default="origin", help="PR target remote")
    arguments = parser.parse_args()
    try:
        print(
            cleanup_pull_request(
                arguments.number, branch=arguments.branch, remote=arguments.remote
            )
        )
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print(f"PR cleanup failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
