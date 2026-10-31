"""Add only the verified CLI user to one explicitly selected GitHub issue."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tooling.process_guard import (
    CLEANUP_UNVERIFIED_EXIT_CODE,
    ProcessCleanupError,
    run_captured,
)

_LOGIN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")
_REPOSITORY = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?/[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class Issue:
    """Only the fields needed to bind and verify an additive assignment."""

    identity: int
    assignees: frozenset[str]


def _api(method: str, endpoint: str, *, user: str | None = None) -> dict[str, object]:
    command = ["gh", "api", "--hostname", "github.com", "--method", method, endpoint]
    if user is not None:
        command.extend(["--raw-field", f"assignees[]={user}"])
    try:
        result = run_captured(command, cwd=Path.cwd(), env=dict(os.environ), timeout=30)
    except FileNotFoundError as error:
        raise RuntimeError("GitHub CLI (gh) is not installed.") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("GitHub request timed out after 30 seconds.") from error
    except (OSError, UnicodeError) as error:
        raise RuntimeError("GitHub request could not be completed.") from error
    if result.returncode:
        # Do not expose credentials or arbitrary server text from diagnostics.
        raise RuntimeError(
            f"GitHub request failed (gh exit {result.returncode}); "
            "check GitHub CLI authentication and issue permissions."
        )
    try:
        data: object = json.loads(result.stdout)
    except ValueError as error:
        raise RuntimeError("GitHub returned invalid JSON.") from error
    if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
        raise RuntimeError("GitHub returned an invalid response object.")
    return data


def _issue(data: dict[str, object], *, number: int, url: str) -> Issue:
    identity, actual_url = data.get("id"), data.get("html_url")
    if (
        type(identity) is not int
        or identity <= 0
        or type(data.get("number")) is not int
        or data["number"] != number
        or not isinstance(actual_url, str)
        or actual_url.casefold() != url.casefold()
    ):
        raise RuntimeError("GitHub issue identity does not match the requested target.")
    if "pull_request" in data or data.get("state") != "open":
        raise RuntimeError(
            "Only an open issue can be claimed; pull requests are refused."
        )
    raw_assignees = data.get("assignees")
    if not isinstance(raw_assignees, list):
        raise RuntimeError("GitHub issue assignees are missing or invalid.")
    logins: set[str] = set()
    for assignee in raw_assignees:
        if not isinstance(assignee, dict):
            raise RuntimeError("GitHub issue assignees are invalid.")
        login = assignee.get("login")
        if not isinstance(login, str) or not login:
            raise RuntimeError("GitHub issue assignee login is missing or invalid.")
        logins.add(login.casefold())
    return Issue(identity=identity, assignees=frozenset(logins))


def claim_issue(number: int, *, repository: str, user: str) -> str:
    """Validate the target and identity before sending at most one additive write."""
    if type(number) is not int or number <= 0:
        raise RuntimeError("Issue number must be a positive integer.")
    if _REPOSITORY.fullmatch(repository) is None or repository.split("/", 1)[1] in {
        ".",
        "..",
    }:
        raise RuntimeError("Repository must be OWNER/REPO on github.com.")
    if _LOGIN.fullmatch(user) is None:
        raise RuntimeError("User must be an explicit GitHub login, not @me or a bot.")
    url = f"https://github.com/{repository}/issues/{number}"
    endpoint = f"repos/{repository}/issues/{number}"
    try:
        account = _api("GET", "user")
        login = account.get("login")
        if (
            account.get("type") != "User"
            or not isinstance(login, str)
            or login.casefold() != user.casefold()
        ):
            raise RuntimeError("Authenticated GitHub user does not match --user.")
        before = _issue(_api("GET", endpoint), number=number, url=url)
    except ProcessCleanupError:
        raise
    except RuntimeError as error:
        raise RuntimeError(f"{url}: no assignment request sent. {error}") from error
    if user.casefold() in before.assignees:
        return f"Already assigned: {url} -> {user}; no changes made."
    try:
        _api("POST", f"{endpoint}/assignees", user=user)
        after = _issue(_api("GET", endpoint), number=number, url=url)
        if after.identity != before.identity:
            raise RuntimeError("Issue identity changed during assignment.")
        if not (before.assignees | {user.casefold()}) <= after.assignees:
            raise RuntimeError(
                "Read-back did not confirm the user and prior assignees."
            )
    except ProcessCleanupError as error:
        raise ProcessCleanupError(
            f"{url}: assignment may have changed; cleanup and success are unverified. "
            f"{error} Inspect Assignees on the issue page."
        ) from error
    except RuntimeError as error:
        raise RuntimeError(
            f"{url}: assignment may have changed; success is unverified. "
            f"{error} Inspect Assignees on the issue page before retrying."
        ) from error
    return f"Assigned and verified: {url} -> {user}; prior assignees preserved."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Claim one open github.com issue as the authenticated user.",
        allow_abbrev=False,
    )
    parser.add_argument("number", type=int, help="Positive issue number (not a PR)")
    parser.add_argument(
        "--repo", required=True, help="Explicit OWNER/REPO on github.com"
    )
    parser.add_argument("--user", required=True, help="Confirmed user's GitHub login")
    args = parser.parse_args(argv)
    try:
        print(claim_issue(args.number, repository=args.repo, user=args.user))
    except ProcessCleanupError as error:
        print(f"Issue claim cleanup unverified: {error}", file=sys.stderr)
        return CLEANUP_UNVERIFIED_EXIT_CODE
    except RuntimeError as error:
        print(f"Issue claim failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
