"""Check a specified GitHub PR locally; publishing and merging remain manual."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from tooling.git.candidates import (
    DEFAULT_DOCUMENTATION_TIMEOUT_SECONDS,
    DEFAULT_SNAPSHOT_TIMEOUT_SECONDS,
    execution_environment,
    git,
    merge_candidate,
    validate_documentation_timeout,
    validate_evidence_budget,
    validate_snapshot_timeout,
    verify_candidate,
)
from tooling.git.pr_context import get_pull_request, remote_url
from tooling.process_guard import run_captured


def _remote_refs(source: str, base_ref: str, head_ref: str) -> tuple[str, str]:
    """Recheck advertised Git refs without trusting cached API base metadata."""
    try:
        result = run_captured(
            ["git", "ls-remote", "--refs", "--", source, base_ref, head_ref],
            cwd=Path.cwd(),
            env=execution_environment(Path.cwd()),
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        # Git commands and their diagnostics can contain credential-bearing URLs.
        raise RuntimeError(
            "Remote ref verification failed; rerun the PR check."
        ) from error
    if result.returncode:
        raise RuntimeError("Remote ref verification failed; check repository access.")
    expected = {base_ref, head_ref}
    refs: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if (
            len(fields) != 2
            or re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", fields[0]) is None
            or fields[1] not in expected
            or fields[1] in refs
        ):
            raise RuntimeError("Remote ref verification returned invalid Git data.")
        refs[fields[1]] = fields[0].lower()
    if refs.keys() != expected:
        raise RuntimeError(
            "The PR/base ref disappeared during checks; rerun the check."
        )
    return refs[base_ref], refs[head_ref]


def check_pr(
    number: int,
    remote: str = "origin",
    *,
    max_evidence_bytes: int | None = None,
    snapshot_timeout_seconds: int = DEFAULT_SNAPSHOT_TIMEOUT_SECONDS,
    documentation_timeout_seconds: int = DEFAULT_DOCUMENTATION_TIMEOUT_SECONDS,
) -> None:
    validate_evidence_budget(max_evidence_bytes)
    validate_snapshot_timeout(snapshot_timeout_seconds)
    validate_documentation_timeout(documentation_timeout_seconds)
    request = get_pull_request(number, remote)
    primary = subprocess.check_output(
        [
            "bash",
            str(
                Path(__file__).resolve().parents[2] / "scripts/detect-primary-branch.sh"
            ),
        ],
        text=True,
    ).strip()
    if request.merged or request.state != "open":
        raise RuntimeError("Choose an open, unmerged PR.")
    if request.base_ref != primary or request.head_repository is None:
        raise RuntimeError(
            "The PR must target the primary branch and have a source repository."
        )
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Use a clean, committed checkout for the PR check runner.")
    starting_head = git("rev-parse", "HEAD")
    source = remote_url(remote)
    base_ref = f"refs/heads/{request.base_ref}"
    head_ref = f"refs/pull/{number}/head"
    with merge_candidate(source, base_ref, head_ref) as candidate:
        if candidate.head != request.head_sha:
            raise RuntimeError("The PR head moved while fetching; run the check again.")
        print(
            f"Checking {request.url}\nBase: {candidate.base}\n"
            f"Head: {candidate.head}\nTree: {candidate.tree}",
            flush=True,
        )
        verify_candidate(
            candidate,
            max_evidence_bytes=max_evidence_bytes,
            snapshot_timeout_seconds=snapshot_timeout_seconds,
            documentation_timeout_seconds=documentation_timeout_seconds,
        )
        current = get_pull_request(number, remote)
        # API base metadata can lag the advertised main ref, and GitHub computes
        # provisional merge SHAs asynchronously. Validate actual Git tips below.
        if (
            replace(
                current,
                base_sha=request.base_sha,
                merge_commit_sha=request.merge_commit_sha,
            )
            != request
        ):
            raise RuntimeError(
                "The PR/base changed during checks; run the check again."
            )
        if _remote_refs(source, base_ref, head_ref) != (candidate.base, candidate.head):
            raise RuntimeError(
                "The PR/base refs changed during checks; rerun the check."
            )
        if git("rev-parse", "HEAD") != starting_head or git(
            "status", "--porcelain", "--untracked-files=all"
        ):
            raise RuntimeError("The check runner checkout changed during checks.")
        print(
            f"PR checks passed: {request.url}\n"
            f"Base: {candidate.base}\nHead: {candidate.head}\nTree: {candidate.tree}\n"
            "Before merging on GitHub, compare the current base/head with these SHAs. "
            "If either changed, rerun. This local result does not lock GitHub merging.",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int)
    parser.add_argument("--remote", default="origin")
    parser.add_argument(
        "--max-evidence-bytes",
        type=int,
        help="Override the documentation reviewer byte limit for this PR check",
    )
    parser.add_argument(
        "--documentation-timeout-seconds",
        type=int,
        help="Set this documentation review deadline (1–86400 seconds; default: 180)",
    )
    parser.add_argument(
        "--snapshot-timeout-seconds",
        type=int,
        help="Set a positive timeout for this full snapshot check (default: 900)",
    )
    args = parser.parse_args()
    validate_evidence_budget(args.max_evidence_bytes)
    snapshot_timeout_seconds = (
        DEFAULT_SNAPSHOT_TIMEOUT_SECONDS
        if args.snapshot_timeout_seconds is None
        else args.snapshot_timeout_seconds
    )
    validate_snapshot_timeout(snapshot_timeout_seconds)
    documentation_timeout_seconds = (
        DEFAULT_DOCUMENTATION_TIMEOUT_SECONDS
        if args.documentation_timeout_seconds is None
        else args.documentation_timeout_seconds
    )
    validate_documentation_timeout(documentation_timeout_seconds)
    os.chdir(git("rev-parse", "--show-toplevel"))
    check_pr(
        args.number,
        args.remote,
        max_evidence_bytes=args.max_evidence_bytes,
        snapshot_timeout_seconds=snapshot_timeout_seconds,
        documentation_timeout_seconds=documentation_timeout_seconds,
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"PR checks failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
