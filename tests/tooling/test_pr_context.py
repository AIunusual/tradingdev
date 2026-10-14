"""PR identity comes from validated, read-only Git and GitHub responses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest
from tooling.git import pr_context

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest import MonkeyPatch


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def repository(monkeypatch: MonkeyPatch) -> Iterator[Path]:
    with TemporaryDirectory(prefix="tradingdev-pr-context-test-") as temporary:
        root = Path(temporary)
        _git(root, "init", "-q")
        _git(root, "remote", "add", "origin", "git@github.com:owner/project.git")
        monkeypatch.chdir(root)
        yield root


@pytest.fixture
def fake_gh(repository: Path, monkeypatch: MonkeyPatch) -> tuple[Path, Path]:
    binary = repository / "gh"
    response = repository / "response.json"
    record = repository / "arguments.json"
    response.write_text(
        json.dumps(
            {
                "number": 7,
                "base": {
                    "repo": {"full_name": "owner/project"},
                    "ref": "main",
                    "sha": "a" * 40,
                },
                "head": {
                    "repo": {"full_name": "contributor/project"},
                    "ref": "feature/work",
                    "sha": "b" * 40,
                },
                "merged": False,
                "state": "open",
                "merge_commit_sha": "c" * 40,
                "html_url": "https://github.com/owner/project/pull/7",
            }
        ),
        encoding="utf-8",
    )
    binary.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        f"pathlib.Path({str(record)!r}).write_text(__import__('json').dumps(sys.argv[1:]))\n"
        "if os.environ.get('PR_FAKE_GH_FAIL'):\n"
        "    print('ghp_private_token', file=sys.stderr)\n"
        "    sys.exit(4)\n"
        f"print(pathlib.Path({str(response)!r}).read_text())\n",
        encoding="utf-8",
    )
    binary.chmod(0o700)
    monkeypatch.setenv("PATH", str(repository) + os.pathsep + os.environ["PATH"])
    return response, record


@pytest.mark.parametrize(
    "url",
    [
        "git@github.com:owner/project.git",
        "https://github.com/owner/project.git",
        "https://github.com:443/owner/project",
        "ssh://git@github.com/owner/project.git",
        "ssh://git@github.com:22/owner/project",
    ],
)
def test_repository_accepts_github_remote_forms(repository: Path, url: str) -> None:
    _git(repository, "remote", "set-url", "origin", url)
    assert pr_context.repository_for_remote() == "owner/project"


@pytest.mark.parametrize(
    "url",
    [
        "git@example.com:owner/project.git",
        "https://github.com.evil.test/owner/project.git",
        "https://private-token@github.com/owner/project.git",
        "https://github.com/owner/project?token=private-token",
        "https://github.com/owner/project#fragment",
        "https://github.com/owner/project/extra",
        "https://github.com/owner/..",
        "ssh://git@github.com:1234/owner/project",
        "https://github.com:invalid/owner/project",
        "ssh://other@github.com/owner/project",
        "/local/project.git",
    ],
)
def test_repository_rejects_ambiguous_remotes(repository: Path, url: str) -> None:
    _git(repository, "remote", "set-url", "origin", url)
    with pytest.raises(RuntimeError) as failure:
        pr_context.repository_for_remote()
    assert "private-token" not in str(failure.value)


def test_get_pr_records_only_get_and_preserves_remote_configuration(
    repository: Path, fake_gh: tuple[Path, Path]
) -> None:
    _, record = fake_gh
    config = (repository / ".git/config").read_bytes()
    pr = pr_context.get_pull_request(7)
    assert pr == pr_context.PullRequest(
        repository="owner/project",
        number=7,
        base_ref="main",
        base_sha="a" * 40,
        head_repository="contributor/project",
        head_ref="feature/work",
        head_sha="b" * 40,
        merged=False,
        state="open",
        merge_commit_sha="c" * 40,
        url="https://github.com/owner/project/pull/7",
    )
    assert json.loads(record.read_text()) == [
        "api",
        "--hostname",
        "github.com",
        "--method",
        "GET",
        "repos/owner/project/pulls/7",
    ]
    assert (repository / ".git/config").read_bytes() == config


def test_merged_pr_supports_deleted_fork_and_sha256(fake_gh: tuple[Path, Path]) -> None:
    response, _ = fake_gh
    data = json.loads(response.read_text())
    data.update(merged=True, state="closed", merge_commit_sha="C" * 64)
    data["head"]["repo"] = None
    data["head"]["sha"] = "B" * 64
    response.write_text(json.dumps(data))
    pr = pr_context.get_pull_request(7)
    assert pr.merged and pr.head_repository is None
    assert pr.merge_commit_sha == "c" * 64
    assert pr.head_sha == "b" * 64


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("number", 8),
        ("number", True),
        ("base.repo.full_name", "other/project"),
        ("base.ref", "bad..branch"),
        ("head.ref", "-option"),
        ("head.ref", "bad\nbranch"),
        ("head.sha", "HEAD"),
        ("base.sha", "a" * 39),
        ("head.repo.full_name", "owner/project/extra"),
        ("merged", "false"),
        ("state", "merged"),
        ("html_url", "https://evil.test/owner/project/pull/7"),
        ("html_url", "https://github.com/owner/project/pull/8"),
        ("html_url", "https://github.com/owner/project/pull/7?other"),
    ],
)
def test_invalid_pr_identity_is_rejected(
    fake_gh: tuple[Path, Path], path: str, value: object
) -> None:
    response, _ = fake_gh
    data = json.loads(response.read_text())
    target = data
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    response.write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        pr_context.get_pull_request(7)


@pytest.mark.parametrize(
    "mode",
    ["missing_merge_sha", "open_merged", "malformed", "missing_repo", "missing_field"],
)
def test_inconsistent_or_unreadable_response_fails(
    fake_gh: tuple[Path, Path], mode: str
) -> None:
    response, _ = fake_gh
    data = json.loads(response.read_text())
    if mode == "missing_merge_sha":
        data.update(merged=True, state="closed", merge_commit_sha=None)
    elif mode == "open_merged":
        data["merged"] = True
    elif mode == "missing_repo":
        del data["head"]["repo"]
    elif mode == "missing_field":
        del data["merge_commit_sha"]
    response.write_text("not-json" if mode == "malformed" else json.dumps(data))
    with pytest.raises(RuntimeError):
        pr_context.get_pull_request(7)


def test_gh_failure_does_not_expose_cli_diagnostics(
    fake_gh: tuple[Path, Path], monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("PR_FAKE_GH_FAIL", "1")
    with pytest.raises(RuntimeError, match="gh auth status") as failure:
        pr_context.get_pull_request(7)
    assert "ghp_private_token" not in str(failure.value)


@pytest.mark.parametrize("number", [0, -1, True])
def test_invalid_pr_number_never_calls_gh(
    fake_gh: tuple[Path, Path], number: int
) -> None:
    with pytest.raises(RuntimeError, match="positive integer"):
        pr_context.get_pull_request(number)
    assert not fake_gh[1].exists()


def test_command_timeout_is_bounded_and_reported(monkeypatch: MonkeyPatch) -> None:
    def timeout(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == 30
        raise subprocess.TimeoutExpired(command, 30, stderr="private-token")

    monkeypatch.setattr(pr_context, "run_captured", timeout)
    with pytest.raises(RuntimeError, match="30 seconds") as failure:
        pr_context.repository_for_remote()
    assert "private-token" not in str(failure.value)
