"""Check immutable Git candidates without changing the developer checkout."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tooling.process_guard import (  # noqa: E402 - standalone entry point
    run_captured,
    run_checked,
)


def git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], text=True, capture_output=True, check=False
    )
    if result.returncode:
        diagnostic = "\n".join(
            part.strip() for part in (result.stdout, result.stderr) if part.strip()
        )
        raise RuntimeError(diagnostic or f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def execution_environment(cwd: Path) -> dict[str, str]:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join([str(cwd / "src"), str(cwd)]),
        "TRADINGDEV_CHECK_PYTHON": sys.executable,
    }
    for name in git("rev-parse", "--local-env-vars").splitlines():
        environment.pop(name, None)
    environment.pop("PYTEST_ADDOPTS", None)
    return environment


def run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 900,
    temporary_root: Path | None = None,
) -> None:
    environment = execution_environment(cwd)
    if temporary_root is not None:
        temporary_root.mkdir()
        environment.update(
            {name: str(temporary_root) for name in ("TMPDIR", "TMP", "TEMP")}
        )
    run_checked(command, cwd=cwd, env=environment, timeout=timeout)


def check_snapshot(tree: str, profile: str, *, repo: Path | None = None) -> None:
    with TemporaryDirectory(prefix="tradingdev-git-check-") as temporary:
        root = Path(temporary)
        archive = root / "candidate.tar"
        with archive.open("wb") as output:
            subprocess.run(
                ["git", "-C", str(repo or Path.cwd()), "archive", tree],
                stdout=output,
                check=True,
            )
        candidate = root / "candidate"
        candidate.mkdir()
        with tarfile.open(archive) as contents:
            contents.extractall(candidate, filter="data")
        environment = {
            **execution_environment(candidate),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            **{
                f"GIT_{role}_{field}": value
                for role in ("AUTHOR", "COMMITTER")
                for field, value in {
                    "NAME": "TradingDev checks",
                    "EMAIL": "checks@tradingdev.invalid",
                    "DATE": "2000-01-01T00:00:00+00:00",
                }.items()
            },
        }

        def snapshot_git(*arguments: str) -> str:
            return subprocess.check_output(
                [
                    "git",
                    "-c",
                    f"core.hooksPath={os.devnull}",
                    "-c",
                    "commit.gpgSign=false",
                    *arguments,
                ],
                cwd=candidate,
                env=environment,
                text=True,
            ).strip()

        snapshot_git("init", "--quiet", "--template=")
        snapshot_git("add", "--force", "--all")
        if snapshot_git("write-tree") != tree:
            raise RuntimeError("The exported files differ from the Git candidate.")
        commit = snapshot_git("commit-tree", tree, "-m", "Check candidate snapshot")
        snapshot_git("update-ref", "HEAD", commit)
        run(
            ["bash", f"scripts/check-{profile}.sh"],
            cwd=candidate,
            temporary_root=root / "runtime",
        )
        if (
            snapshot_git("write-tree") != tree
            or snapshot_git("diff", "--name-only")
            or snapshot_git("ls-files", "--others", "--exclude-standard")
        ):
            raise RuntimeError("A check modified the candidate snapshot.")


def assert_index_unchanged(tree: str) -> None:
    if git("write-tree") != tree:
        raise RuntimeError("The staged content changed during checks; stage and retry.")


def check_commit() -> None:
    git("diff", "--cached", "--check")
    tree = git("write-tree")
    check_snapshot(tree, "fast")
    assert_index_unchanged(tree)


@dataclass(frozen=True)
class MergeCandidate:
    repository: Path
    base: str
    head: str
    tree: str


@contextmanager
def merge_candidate(
    source: str, base_ref: str, head_ref: str
) -> Generator[MergeCandidate, None, None]:
    """Fetch and merge in a disposable repository, leaving the caller untouched."""
    with TemporaryDirectory(prefix="tradingdev-pr-candidate-") as temporary:
        repository = Path(temporary)
        environment = execution_environment(repository)

        def candidate_git(*args: str) -> str:
            result = run_captured(
                [
                    "git",
                    "-C",
                    str(repository),
                    "-c",
                    f"core.hooksPath={os.devnull}",
                    *args,
                ],
                # Fetch needs authentication helpers; merge uses project rules.
                env=environment
                if args[0] == "fetch"
                else {
                    **environment,
                    "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_CONFIG_SYSTEM": os.devnull,
                    "GIT_ATTR_NOSYSTEM": "1",
                },
                cwd=repository,
                timeout=120,
            )
            if result.returncode:
                # Fetch diagnostics can include credential-bearing remote URLs.
                detail = (
                    " Merge conflicts must be resolved on the PR branch."
                    if args[0] == "merge-tree"
                    else ""
                )
                raise RuntimeError(f"Candidate git {args[0]} failed.{detail}")
            return result.stdout.strip()

        candidate_git("init", "--quiet", "--template=")
        candidate_git(
            "fetch",
            "--quiet",
            "--no-tags",
            "--no-prune",
            "--no-recurse-submodules",
            "--no-write-fetch-head",
            "--",
            source,
            f"{base_ref}:refs/check/base",
            f"{head_ref}:refs/check/head",
        )
        base = candidate_git("rev-parse", "refs/check/base^{commit}")
        head = candidate_git("rev-parse", "refs/check/head^{commit}")
        tree = candidate_git("merge-tree", "--write-tree", base, head)
        yield MergeCandidate(repository, base, head, tree)


def validate_evidence_budget(max_evidence_bytes: int | None) -> None:
    if max_evidence_bytes is not None and max_evidence_bytes <= 0:
        raise RuntimeError("Documentation evidence budget must be positive.")


def verify_candidate(
    candidate: MergeCandidate, *, max_evidence_bytes: int | None = None
) -> None:
    """Always run a fresh review and full suite for an explicit candidate."""
    validate_evidence_budget(max_evidence_bytes)
    print("Reviewing code/documentation consistency with Codex...", flush=True)
    command = [
        sys.executable,
        str(Path(__file__).with_name("review_docs.py")),
        "--repo",
        str(candidate.repository),
        "--base",
        candidate.base,
        "--tree",
        candidate.tree,
    ]
    if max_evidence_bytes is not None:
        print(f"Documentation evidence budget: {max_evidence_bytes} bytes", flush=True)
        command.extend(["--max-evidence-bytes", str(max_evidence_bytes)])
    run(command, cwd=Path(__file__).resolve().parents[1])
    check_snapshot(candidate.tree, "full", repo=candidate.repository)


def check_full(
    base_ref: str, head_ref: str, *, max_evidence_bytes: int | None = None
) -> None:
    validate_evidence_budget(max_evidence_bytes)
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Commit the task stages before running full checks.")
    base = git("rev-parse", "--verify", "--end-of-options", f"{base_ref}^{{commit}}")
    head = git("rev-parse", "--verify", "--end-of-options", f"{head_ref}^{{commit}}")
    with merge_candidate(str(Path.cwd()), base, head) as candidate:
        verify_candidate(candidate, max_evidence_bytes=max_evidence_bytes)
        if (
            git("status", "--porcelain", "--untracked-files=all")
            or git(
                "rev-parse", "--verify", "--end-of-options", f"{base_ref}^{{commit}}"
            )
            != base
            or git(
                "rev-parse", "--verify", "--end-of-options", f"{head_ref}^{{commit}}"
            )
            != head
        ):
            raise RuntimeError("The checkout or compared refs changed during checks.")
        print(f"Full checks passed. Base: {base} Head: {head} Tree: {candidate.tree}")


def check_push() -> None:
    primary = subprocess.check_output(
        ["bash", "scripts/detect-primary-branch.sh"], text=True
    ).strip()
    for line in sys.stdin:
        _, _, remote_ref, _ = line.split()
        if remote_ref == f"refs/heads/{primary}":
            raise RuntimeError(
                f"Direct pushes/deletions of {primary} are blocked; use a GitHub PR."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["commit", "full", "push"])
    parser.add_argument("--base", help="Explicit local comparison base for full checks")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument(
        "--max-evidence-bytes",
        type=int,
        help="Override the documentation reviewer byte limit for this full check",
    )
    args = parser.parse_args()
    action = args.action
    if action == "full" and not args.base:
        parser.error("full requires --base; use scripts/check-pr.sh for a remote PR")
    if args.max_evidence_bytes is not None and action != "full":
        parser.error("--max-evidence-bytes is only valid for full checks")
    validate_evidence_budget(args.max_evidence_bytes)
    os.chdir(git("rev-parse", "--show-toplevel"))
    if action == "commit":
        check_commit()
    elif action == "push":
        check_push()
    else:
        check_full(args.base, args.head, max_evidence_bytes=args.max_evidence_bytes)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.SubprocessError, OSError) as error:
        print(f"Git checks failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
