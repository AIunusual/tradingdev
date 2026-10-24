"""Build and verify immutable Git candidates in disposable repositories."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from tooling.git.change_scope import documentation_only
from tooling.process_guard import run_captured, run_checked

if TYPE_CHECKING:
    from collections.abc import Generator

DEFAULT_SNAPSHOT_TIMEOUT_SECONDS = 900
DEFAULT_DOCUMENTATION_TIMEOUT_SECONDS = 180
MAX_DOCUMENTATION_TIMEOUT_SECONDS = 86_400
_DOCUMENTATION_CLEANUP_SECONDS = 60


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


def check_snapshot(
    tree: str,
    profile: str,
    *,
    repo: Path | None = None,
    timeout_seconds: int = DEFAULT_SNAPSHOT_TIMEOUT_SECONDS,
) -> None:
    validate_snapshot_timeout(timeout_seconds)
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
            timeout=timeout_seconds,
            temporary_root=root / "runtime",
        )
        if (
            snapshot_git("write-tree") != tree
            or snapshot_git("diff", "--name-only")
            or snapshot_git("ls-files", "--others", "--exclude-standard")
        ):
            raise RuntimeError("A check modified the candidate snapshot.")


@dataclass(frozen=True)
class MergeCandidate:
    repository: Path
    base: str
    head: str
    tree: str


@contextmanager
def merge_candidate(
    source: str, base_ref: str, head_ref: str
) -> Generator[MergeCandidate]:
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


def validate_snapshot_timeout(timeout_seconds: int) -> None:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
    ):
        raise RuntimeError(
            "Snapshot timeout must be a positive integer number of seconds."
        )


def validate_documentation_timeout(timeout_seconds: int) -> None:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or not 1 <= timeout_seconds <= MAX_DOCUMENTATION_TIMEOUT_SECONDS
    ):
        raise RuntimeError(
            "Documentation timeout must be a positive integer number of seconds "
            f"no greater than {MAX_DOCUMENTATION_TIMEOUT_SECONDS}."
        )


def verify_candidate(
    candidate: MergeCandidate,
    *,
    max_evidence_bytes: int | None = None,
    snapshot_timeout_seconds: int = DEFAULT_SNAPSHOT_TIMEOUT_SECONDS,
    documentation_timeout_seconds: int = DEFAULT_DOCUMENTATION_TIMEOUT_SECONDS,
) -> None:
    """Check each immutable candidate with the profile required by its changes."""
    validate_evidence_budget(max_evidence_bytes)
    validate_snapshot_timeout(snapshot_timeout_seconds)
    validate_documentation_timeout(documentation_timeout_seconds)
    print(f"Snapshot timeout: {snapshot_timeout_seconds} seconds", flush=True)
    if documentation_only(candidate.repository, candidate.base, candidate.tree):
        print(
            "Documentation-only candidate: checking Markdown and local links. "
            "Codex, Python quality checks and pytest are not run.",
            flush=True,
        )
        check_snapshot(
            candidate.tree,
            "docs",
            repo=candidate.repository,
            timeout_seconds=snapshot_timeout_seconds,
        )
        print("Documentation-only checks passed.", flush=True)
        return
    print(f"Documentation timeout: {documentation_timeout_seconds} seconds", flush=True)
    print("Reviewing code/documentation consistency with Codex...", flush=True)
    command = [
        sys.executable,
        "-m",
        "tooling.review_docs",
        "--repo",
        str(candidate.repository),
        "--base",
        candidate.base,
        "--tree",
        candidate.tree,
        "--timeout-seconds",
        str(documentation_timeout_seconds),
    ]
    if max_evidence_bytes is not None:
        print(f"Documentation evidence budget: {max_evidence_bytes} bytes", flush=True)
        command.extend(["--max-evidence-bytes", str(max_evidence_bytes)])
    # Preserve setup time and allow the review's own deadline/cleanup to finish.
    run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        timeout=max(
            900, documentation_timeout_seconds + _DOCUMENTATION_CLEANUP_SECONDS
        ),
    )
    check_snapshot(
        candidate.tree,
        "full",
        repo=candidate.repository,
        timeout_seconds=snapshot_timeout_seconds,
    )
    print("Codex documentation review and full pytest passed.", flush=True)
