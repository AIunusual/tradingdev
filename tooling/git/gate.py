"""Run commit, push and explicit local integration checks."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from tooling.git.candidates import (
    DEFAULT_DOCUMENTATION_TIMEOUT_SECONDS,
    DEFAULT_SNAPSHOT_TIMEOUT_SECONDS,
    check_snapshot,
    git,
    merge_candidate,
    validate_documentation_timeout,
    validate_evidence_budget,
    validate_snapshot_timeout,
    verify_candidate,
)


def assert_index_unchanged(tree: str) -> None:
    if git("write-tree") != tree:
        raise RuntimeError("The staged content changed during checks; stage and retry.")


def check_commit() -> None:
    git("diff", "--cached", "--check")
    tree = git("write-tree")
    check_snapshot(tree, "fast")
    assert_index_unchanged(tree)


def check_full(
    base_ref: str,
    head_ref: str,
    *,
    max_evidence_bytes: int | None = None,
    snapshot_timeout_seconds: int = DEFAULT_SNAPSHOT_TIMEOUT_SECONDS,
    documentation_timeout_seconds: int = DEFAULT_DOCUMENTATION_TIMEOUT_SECONDS,
) -> None:
    validate_evidence_budget(max_evidence_bytes)
    validate_snapshot_timeout(snapshot_timeout_seconds)
    validate_documentation_timeout(documentation_timeout_seconds)
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Commit the task stages before running full checks.")
    base = git("rev-parse", "--verify", "--end-of-options", f"{base_ref}^{{commit}}")
    head = git("rev-parse", "--verify", "--end-of-options", f"{head_ref}^{{commit}}")
    with merge_candidate(str(Path.cwd()), base, head) as candidate:
        verify_candidate(
            candidate,
            max_evidence_bytes=max_evidence_bytes,
            snapshot_timeout_seconds=snapshot_timeout_seconds,
            documentation_timeout_seconds=documentation_timeout_seconds,
        )
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


def _is_github_destination(destination: str) -> bool:
    if "://" in destination:
        try:
            parsed = urlsplit(destination)
            if parsed.scheme in {"ssh", "git", "git+ssh", "ssh+git"}:
                # Git decodes native transport URLs before splitting user/host/port.
                parsed = urlsplit(unquote(destination))
                host = parsed.hostname
            else:
                host = unquote(parsed.hostname) if parsed.hostname else None
        except ValueError as error:
            raise RuntimeError("Cannot determine the push destination host.") from error
        if parsed.scheme not in {"http", "https", "ssh", "git", "git+ssh", "ssh+git"}:
            return False
    else:
        # Git also accepts SCP-style SSH URLs, with an optional username.
        match = re.match(r"(?:[^/@:]+@)?([^/:]+):", destination)
        host = match[1] if match else None
    return host is not None and host.rstrip(".").casefold() in {
        "github.com",
        "ssh.github.com",
    }


def check_push(destination: str) -> None:
    if not _is_github_destination(destination):
        # Drain Git's ref updates before exiting, including large multi-ref pushes.
        for _ in sys.stdin:
            pass
        return
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
    parser.add_argument("--destination", help="Actual push URL supplied by Git")
    parser.add_argument(
        "--max-evidence-bytes",
        type=int,
        help="Override the documentation reviewer byte limit for this full check",
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
    action = args.action
    if action == "full" and not args.base:
        parser.error("full requires --base; use scripts/check-pr.sh for a remote PR")
    if action == "push" and not args.destination:
        parser.error("push requires --destination from the pre-push hook")
    if args.destination is not None and action != "push":
        parser.error("--destination is only valid for push checks")
    if args.max_evidence_bytes is not None and action != "full":
        parser.error("--max-evidence-bytes is only valid for full checks")
    if args.snapshot_timeout_seconds is not None and action != "full":
        parser.error("--snapshot-timeout-seconds is only valid for full checks")
    if args.documentation_timeout_seconds is not None and action != "full":
        parser.error("--documentation-timeout-seconds is only valid for full checks")
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
    if action == "commit":
        check_commit()
    elif action == "push":
        check_push(args.destination)
    else:
        check_full(
            args.base,
            args.head,
            max_evidence_bytes=args.max_evidence_bytes,
            snapshot_timeout_seconds=snapshot_timeout_seconds,
            documentation_timeout_seconds=documentation_timeout_seconds,
        )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.SubprocessError, OSError) as error:
        print(f"Git checks failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
