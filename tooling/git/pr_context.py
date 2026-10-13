"""Read validated GitHub PR identity without changing Git or remote state."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from tooling.process_guard import run_captured

_SLUG = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?/[A-Za-z0-9_.-]+")
_SHA = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")
_REMOTE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")


@dataclass(frozen=True)
class PullRequest:
    """The server-reported identities used for checks and conservative cleanup."""

    repository: str
    number: int
    base_ref: str
    base_sha: str
    head_repository: str | None
    head_ref: str
    head_sha: str
    merged: bool
    state: str
    merge_commit_sha: str | None
    url: str


def _read(command: list[str], description: str) -> str:
    try:
        result = run_captured(command, cwd=Path.cwd(), env=dict(os.environ), timeout=30)
    except FileNotFoundError as error:
        raise RuntimeError(f"{description}: {command[0]} is not installed.") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{description} timed out after 30 seconds.") from error
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"{description} could not be read.") from error
    if result.returncode:
        hint = (
            " Check gh auth status --hostname github.com and repository access."
            if command[0] == "gh"
            else " Check the repository and remote configuration."
        )
        # URLs and diagnostics may contain authentication credentials.
        raise RuntimeError(
            f"{description} failed ({command[0]} exit {result.returncode}).{hint}"
        )
    return result.stdout.strip()


def _repository(value: object) -> str:
    if (
        not isinstance(value, str)
        or _SLUG.fullmatch(value) is None
        or value.split("/", 1)[1] in {".", ".."}
    ):
        raise RuntimeError("Invalid GitHub repository identity.")
    return value


def remote_url(remote: str = "origin") -> str:
    """Return the configured fetch URL; callers must not print credentials."""
    if _REMOTE.fullmatch(remote) is None or ".." in remote:
        raise RuntimeError("Invalid Git remote name.")
    return _read(["git", "remote", "get-url", "--", remote], "Reading Git remote")


def repository_for_remote(remote: str = "origin") -> str:
    """Accept GitHub HTTPS and SSH remotes, rejecting ambiguous destinations."""
    value = remote_url(remote)
    if value.startswith("git@github.com:"):
        path = value.removeprefix("git@github.com:")
    else:
        try:
            parsed = urlsplit(value)
            valid = (
                parsed.hostname == "github.com"
                and not parsed.query
                and not parsed.fragment
                and parsed.password is None
                and (
                    (
                        parsed.scheme == "https"
                        and parsed.username is None
                        and parsed.port in (None, 443)
                    )
                    or (
                        parsed.scheme == "ssh"
                        and parsed.username == "git"
                        and parsed.port in (None, 22)
                    )
                )
            )
        except ValueError:
            valid = False
        if not valid:
            raise RuntimeError(
                "The selected remote must be a github.com HTTPS/SSH URL."
            )
        path = parsed.path.removeprefix("/")
    return _repository(path.removesuffix(".git"))


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeError(f"Invalid GitHub PR response: {field} must be an object.")
    return value


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise RuntimeError(f"Invalid GitHub PR response: {field} must be a commit SHA.")
    return value.lower()


def _ref(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("-"):
        raise RuntimeError(
            f"Invalid GitHub PR response: {field} must be a branch name."
        )
    _read(["git", "check-ref-format", f"refs/heads/{value}"], f"Validating PR {field}")
    return value


def get_pull_request(number: int, remote: str = "origin") -> PullRequest:
    """Read one PR using GitHub's API; all identity fields are validated."""
    if type(number) is not int or number <= 0:
        raise RuntimeError("A positive integer PR number is required.")
    repository = repository_for_remote(remote)
    raw = _read(
        [
            "gh",
            "api",
            "--hostname",
            "github.com",
            "--method",
            "GET",
            f"repos/{repository}/pulls/{number}",
        ],
        "Reading GitHub PR",
    )
    try:
        data = _object(json.loads(raw), "response")
    except ValueError as error:
        raise RuntimeError("GitHub returned an invalid JSON PR response.") from error
    if type(data.get("number")) is not int or data["number"] != number:
        raise RuntimeError("GitHub returned a different PR number.")
    base = _object(data.get("base"), "base")
    base_repository = _repository(
        _object(base.get("repo"), "base.repo").get("full_name")
    )
    if base_repository.casefold() != repository.casefold():
        raise RuntimeError("GitHub returned a PR for a different repository.")
    head = _object(data.get("head"), "head")
    if "repo" not in head or "merge_commit_sha" not in data:
        raise RuntimeError("Invalid GitHub PR response: required metadata is missing.")
    head_repository = (
        None
        if head.get("repo") is None
        else _repository(_object(head["repo"], "head.repo").get("full_name"))
    )
    merged, state = data.get("merged"), data.get("state")
    if (
        type(merged) is not bool
        or not isinstance(state, str)
        or state not in ("open", "closed")
    ):
        raise RuntimeError(
            "Invalid GitHub PR response: merged/state is missing or invalid."
        )
    if merged and state != "closed":
        raise RuntimeError("Invalid GitHub PR response: a merged PR must be closed.")
    merge_sha = data.get("merge_commit_sha")
    if merged or merge_sha is not None:
        merge_sha = _sha(merge_sha, "merge_commit_sha")
    url = data.get("html_url")
    expected_url = f"https://github.com/{base_repository}/pull/{number}"
    if not isinstance(url, str) or url != expected_url:
        raise RuntimeError(
            "Invalid GitHub PR response: PR URL does not match its identity."
        )
    return PullRequest(
        repository=base_repository,
        number=number,
        base_ref=_ref(base.get("ref"), "base.ref"),
        base_sha=_sha(base.get("sha"), "base.sha"),
        head_repository=head_repository,
        head_ref=_ref(head.get("ref"), "head.ref"),
        head_sha=_sha(head.get("sha"), "head.sha"),
        merged=merged,
        state=state,
        merge_commit_sha=merge_sha,
        url=url,
    )
