"""Build and verify immutable Git candidates in disposable repositories."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock, Timeout

from tooling.environment import check_environment
from tooling.git import receipts
from tooling.git.change_scope import classify_changes, test_input_fingerprint
from tooling.process_guard import (
    ProcessCleanupError,
    process_workspace,
    run_captured,
    run_checked,
)
from tooling.review_options import review_arguments, validate_review_options

if TYPE_CHECKING:
    from collections.abc import Generator

DEFAULT_SNAPSHOT_TIMEOUT_SECONDS = 1800
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
        "TRADINGDEV_PRESENTATION_TOOLCHAIN": os.environ.get(
            "TRADINGDEV_PRESENTATION_TOOLCHAIN",
            str(Path(__file__).resolve().parents[2] / "tooling/marp"),
        ),
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
    environment["TRADINGDEV_CHECK_SNAPSHOT"] = "1"
    environment["TRADINGDEV_CHECK_SNAPSHOT_TIMEOUT"] = str(timeout)
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
    with process_workspace(prefix="tradingdev-git-check-", label="Snapshot") as root:
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


def check_test_snapshot(
    tree: str,
    profile: str,
    *,
    repo: Path | None = None,
    timeout_seconds: int = DEFAULT_SNAPSHOT_TIMEOUT_SECONDS,
    reuse: bool = True,
) -> None:
    """Verify test evidence, refreshing ordinary docs without rerunning pytest."""
    source = repo or Path.cwd()
    owner = Path.cwd()
    path = receipts.receipt_path(owner, profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    inputs = test_input_fingerprint(source, tree)

    def environment() -> str | None:
        try:
            return receipts.environment_fingerprint(
                owner, profile, execution_environment(owner)
            )
        except ProcessCleanupError:
            raise
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            print(
                f"Test evidence reuse unavailable: {type(error).__name__}", flush=True
            )
            return None

    def record(state: str, tested_tree: str, fingerprint: str | None) -> None:
        receipts.write_receipt(
            path,
            state=state,
            tree=tested_tree,
            profile=profile,
            inputs=inputs,
            environment=fingerprint,
        )

    try:
        with FileLock(str(path.with_suffix(".lock")), timeout=0):
            previous = receipts.read_receipt(path) if reuse else None
            # Invalidate durably before any probe or child check can fail. If
            # writing fails here, no check starts and the old proof is untouched.
            record("running", tree, None)
            before = None
            restored_prior = False
            try:
                before = environment()
                prior = (
                    receipts.matching_receipt(
                        previous,
                        profile=profile,
                        inputs=inputs,
                        environment=before,
                    )
                    if before is not None
                    else None
                )
                tested_tree = str(prior["tree"]) if prior is not None else tree
                if prior is not None:
                    if tested_tree != tree:
                        try:
                            check_snapshot(
                                tree,
                                "docs",
                                repo=source,
                                timeout_seconds=timeout_seconds,
                            )
                        except subprocess.CalledProcessError as error:
                            # A normal docs-quality failure does not disprove
                            # the original pytest result. Restore only after
                            # verified cleanup and a stable environment; the
                            # candidate still fails with the original exception.
                            if (
                                error.returncode == 1
                                and error.cmd == ["bash", "scripts/check-docs.sh"]
                                and environment() == before
                            ):
                                record("passed", tested_tree, before)
                                restored_prior = True
                            raise
                else:
                    check_snapshot(
                        tree, profile, repo=source, timeout_seconds=timeout_seconds
                    )
                after = environment()
                if before is not None and after != before:
                    raise RuntimeError(
                        "The test environment changed during checks; "
                        "no successful evidence was saved."
                    )
                record(
                    "passed" if before is not None else "unavailable",
                    tested_tree,
                    before,
                )
            except BaseException as error:
                if not restored_prior:
                    try:
                        record("failed", tree, before)
                    except OSError:
                        # The durable running record still blocks reuse.
                        error.add_note(
                            f"Could not record failed test evidence at {path}"
                        )
                raise
            if prior is not None:
                payload = {
                    "code": "test_evidence_reused",
                    "profile": profile,
                    "tested_tree": tested_tree,
                    "candidate_tree": tree,
                    "receipt": str(path),
                }
            else:
                payload = {
                    "code": "test_evidence_recorded"
                    if before
                    else "test_evidence_unavailable",
                    "profile": profile,
                    "tree": tree,
                    "receipt": str(path),
                }
            print(json.dumps(payload), flush=True)
    except Timeout as error:
        raise RuntimeError(
            f"Another {profile} check owns {path}; wait for it to finish."
        ) from error


@dataclass(frozen=True)
class MergeCandidate:
    repository: Path
    base: str
    head: str
    tree: str


def _candidate_git(repository: Path, *args: str) -> str:
    environment = execution_environment(repository)
    result = run_captured(
        ["git", "-C", str(repository), "-c", f"core.hooksPath={os.devnull}", *args],
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


def fetch_merge_candidate(
    repository: Path, source: str, base_ref: str, head_ref: str
) -> MergeCandidate:
    """Fetch a candidate in an owned repository, retaining earlier Git objects."""
    _candidate_git(
        repository,
        "fetch",
        "--quiet",
        "--no-tags",
        "--no-prune",
        "--no-recurse-submodules",
        "--no-write-fetch-head",
        "--",
        source,
        # Owned temporary refs must reflect rewrites so callers can reject them
        # using the fetched identities and ancestry, without altering user refs.
        f"+{base_ref}:refs/check/base",
        f"+{head_ref}:refs/check/head",
    )
    base = _candidate_git(repository, "rev-parse", "refs/check/base^{commit}")
    head = _candidate_git(repository, "rev-parse", "refs/check/head^{commit}")
    tree = _candidate_git(repository, "merge-tree", "--write-tree", base, head)
    return MergeCandidate(repository, base, head, tree)


def base_advanced(previous: MergeCandidate, updated: MergeCandidate) -> bool:
    """Verify ancestry using actual fetched commits, not PR API base metadata."""
    if previous.base == updated.base:
        return False
    result = run_captured(
        [
            "git",
            "-C",
            str(updated.repository),
            "merge-base",
            "--is-ancestor",
            previous.base,
            updated.base,
        ],
        cwd=updated.repository,
        env=execution_environment(updated.repository),
        timeout=120,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError("Cannot verify the updated primary branch ancestry.")
    return result.returncode == 0


@contextmanager
def merge_candidate(
    source: str, base_ref: str, head_ref: str
) -> Generator[MergeCandidate]:
    """Fetch and merge in a disposable repository, leaving the caller untouched."""
    with process_workspace(
        prefix="tradingdev-pr-candidate-", label="Merge candidate"
    ) as repository:
        _candidate_git(repository, "init", "--quiet", "--template=")
        yield fetch_merge_candidate(repository, source, base_ref, head_ref)


def validate_evidence_budget(max_evidence_bytes: int | None) -> None:
    if max_evidence_bytes is not None and (
        isinstance(max_evidence_bytes, bool)
        or not isinstance(max_evidence_bytes, int)
        or max_evidence_bytes <= 0
    ):
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
    review_scopes: tuple[str, ...] | None = None,
    review_context: tuple[str, ...] = (),
    review_context_for: tuple[tuple[str, str], ...] = (),
    max_prompt_bytes: int | None = None,
    max_total_prompt_bytes: int | None = None,
    snapshot_timeout_seconds: int = DEFAULT_SNAPSHOT_TIMEOUT_SECONDS,
    documentation_timeout_seconds: int = DEFAULT_DOCUMENTATION_TIMEOUT_SECONDS,
) -> None:
    """Check each immutable candidate with the profile required by its changes."""
    validate_evidence_budget(max_evidence_bytes)
    validate_review_options(
        review_scopes=review_scopes,
        review_context=review_context,
        review_context_for=review_context_for,
        max_prompt_bytes=max_prompt_bytes,
        max_total_prompt_bytes=max_total_prompt_bytes,
    )
    validate_snapshot_timeout(snapshot_timeout_seconds)
    validate_documentation_timeout(documentation_timeout_seconds)
    print(f"Snapshot timeout: {snapshot_timeout_seconds} seconds", flush=True)
    scope = classify_changes(candidate.repository, candidate.base, candidate.tree)
    checks = scope.candidate_checks
    if scope.kind in {"tooling", "code"}:
        check_environment(scope.delivery_profile)
    if "documentation_review" not in checks:
        label = "Documentation-only" if scope.kind == "docs" else "Presentation"
        print(
            f"{label} candidate: running {scope.kind} snapshot checks. "
            "Codex, Python quality checks and pytest are not run.",
            flush=True,
        )
        check_snapshot(
            candidate.tree,
            scope.delivery_profile,
            repo=candidate.repository,
            timeout_seconds=snapshot_timeout_seconds,
        )
        print(f"{label} checks passed.", flush=True)
        return
    for check in checks:
        if check != "documentation_review":
            print(f"Candidate test profile: {check}", flush=True)
            checker = (
                check_test_snapshot if check in {"full", "tooling"} else check_snapshot
            )
            checker(
                candidate.tree,
                check,
                repo=candidate.repository,
                timeout_seconds=snapshot_timeout_seconds,
            )
            continue
        print(
            f"Documentation timeout: {documentation_timeout_seconds} seconds",
            flush=True,
        )
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
            print(
                f"Documentation evidence budget: {max_evidence_bytes} bytes", flush=True
            )
            command.extend(["--max-evidence-bytes", str(max_evidence_bytes)])
        command.extend(
            review_arguments(
                review_scopes=review_scopes,
                review_context=review_context,
                review_context_for=review_context_for,
                max_prompt_bytes=max_prompt_bytes,
                max_total_prompt_bytes=max_total_prompt_bytes,
            )
        )
        # Preserve setup time and allow the review's deadline/cleanup to finish.
        run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            timeout=max(
                900, documentation_timeout_seconds + _DOCUMENTATION_CLEANUP_SECONDS
            ),
        )
    completed = (
        f"{scope.delivery_profile} snapshot checks"
        if scope.kind in {"docs", "presentations"}
        else f"{scope.delivery_profile} pytest"
    )
    print(
        f"Codex documentation review and {completed} passed.",
        flush=True,
    )
