"""Run commit, push and explicit local integration checks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from tooling.git.candidates import (
    check_snapshot,
    git,
    merge_candidate,
    validate_evidence_budget,
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
