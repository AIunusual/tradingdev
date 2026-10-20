"""Exercise the installed hooks and task scripts against isolated Git repositories."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

_PROJECT = Path(__file__).resolve().parents[2]
_CHECKER = """import json
import os
import subprocess
import sys
from pathlib import Path

mode = sys.argv[1]
root = Path.cwd()
if mode == 'docs':
    base = sys.argv[sys.argv.index('--base') + 1]
    tree = sys.argv[sys.argv.index('--tree') + 1]
    repository = sys.argv[sys.argv.index('--repo') + 1]
    content = subprocess.check_output(
        ['git', '-C', repository, 'show', f'{tree}:src/app.py'], text=True
    )
else:
    base = tree = None
    content = (root / 'src/app.py').read_text()
record = {'mode': mode, 'root': str(root), 'content': content,
          'base': base, 'tree': tree}
if mode != 'docs':
    for field, args in {
        'head': ['rev-parse', 'HEAD'],
        'head_tree': ['rev-parse', 'HEAD^{tree}'],
        'head_content': ['show', 'HEAD:src/app.py'],
    }.items():
        result = subprocess.run(['git', *args], capture_output=True, text=True)
        record[field] = result.stdout if result.returncode == 0 else None
with Path(os.environ['HOOK_TEST_RECORD']).open('a') as stream:
    stream.write(json.dumps(record) + '\\n')
failure = os.environ.get('HOOK_TEST_FAIL')
if mode == 'full' and failure in {'mutation', 'staged_mutation'}:
    (root / 'src/app.py').write_text("VALUE = 'modified by checker'\\n")
    if os.environ['HOOK_TEST_FAIL'] == 'staged_mutation':
        subprocess.run(['git', 'add', 'src/app.py'], check=True)
if os.environ.get('HOOK_TEST_FAIL') == mode or 'REJECT_FAST' in content:
    print(f'{mode} rejected candidate', file=sys.stderr)
    sys.exit(1)
"""


@dataclass
class GitRepository:
    root: Path
    environment: dict[str, str]
    record: Path

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            args,
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if check:
            assert result.returncode == 0, result.stdout + result.stderr
        return result

    def git(self, *args: str) -> str:
        return self.run("git", *args).stdout.strip()

    def write(self, name: str, content: str) -> None:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def events(self, mode: str) -> list[dict[str, str | None]]:
        if not self.record.exists():
            return []
        events = [json.loads(line) for line in self.record.read_text().splitlines()]
        return [event for event in events if event["mode"] == mode]

    def start_task(self) -> None:
        self.git("switch", "-c", "feature/test-hooks")

    def commit_task(self) -> str:
        self.start_task()
        self.write("src/app.py", "VALUE = 'feature'\n")
        self.run("scripts/git-flow-commit.sh", "feat: update behavior", "src/app.py")
        return self.git("rev-parse", "HEAD")


@contextmanager
def _repository(primary: str = "main") -> Generator[GitRepository]:
    with TemporaryDirectory(prefix="tradingdev-git-hook-test-") as temporary:
        root = Path(temporary) / "repository"
        root.mkdir()
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        }
        record = Path(temporary) / "checks.jsonl"
        environment.update(
            {
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_TERMINAL_PROMPT": "0",
                "TRADINGDEV_CHECK_PYTHON": sys.executable,
                "HOOK_TEST_RECORD": str(record),
                "HOOK_TEST_FAIL": "",
            }
        )
        repo = GitRepository(root, environment, record)
        repo.git("init", "-q", "-b", primary)
        repo.git("config", "user.name", "Hook Test")
        repo.git("config", "user.email", "hook-test@example.invalid")
        if primary not in {"main", "master"}:
            repo.git("config", "tradingdev.primaryBranch", primary)
        shutil.copytree(_PROJECT / ".githooks", root / ".githooks")
        (root / "scripts").mkdir()
        for name in (
            "detect-primary-branch.sh",
            "git-flow-commit.sh",
            "install-git-hooks.sh",
        ):
            shutil.copy2(_PROJECT / "scripts" / name, root / "scripts" / name)
        for name in (
            "__init__.py",
            "process_guard.py",
            "git/__init__.py",
            "git/gate.py",
            "git/candidates.py",
        ):
            destination = root / "tooling" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_PROJECT / "tooling" / name, destination)
        repo.write("scripts/test_checker.py", _CHECKER)
        for mode in ("fast", "full"):
            repo.write(
                f"scripts/check-{mode}.sh",
                "#!/usr/bin/env bash\n"
                f"exec {shlex.quote(sys.executable)} scripts/test_checker.py {mode}\n",
            )
        repo.write(
            "tooling/review_docs.py",
            "import sys\n" + _CHECKER.replace("mode = sys.argv[1]", "mode = 'docs'"),
        )
        repo.write("src/app.py", "VALUE = 'initial'\n")
        repo.write("README.md", "# Test repository\n")
        repo.write(".gitignore", ".venv/\n__pycache__/\n")
        repo.git("add", ".")
        repo.git("commit", "-qm", "chore: initialize fixture")
        repo.git("config", "core.hooksPath", ".githooks")
        yield repo


def test_primary_branch_rejects_direct_commit_without_losing_staged_changes() -> None:
    with _repository() as repo:
        before = repo.git("rev-parse", "HEAD")
        repo.write("src/app.py", "VALUE = 'direct'\n")
        repo.git("add", "src/app.py")
        result = repo.run("git", "commit", "-m", "feat: direct commit", check=False)
        assert result.returncode != 0
        assert repo.git("rev-parse", "HEAD") == before
        assert repo.git("diff", "--cached", "--name-only") == "src/app.py"
        assert not repo.events("full")
        assert not repo.events("docs")


def test_whitespace_failure_reports_the_file_and_preserves_the_index() -> None:
    with _repository() as repo:
        repo.start_task()
        before = repo.git("rev-parse", "HEAD")
        repo.write("src/app.py", "VALUE = 'feature'  \n")
        repo.git("add", "src/app.py")
        result = repo.run("git", "commit", "-m", "feat: candidate", check=False)
        assert result.returncode != 0
        assert "src/app.py:1: trailing whitespace" in result.stdout + result.stderr
        assert repo.git("rev-parse", "HEAD") == before
        assert repo.git("diff", "--cached", "--name-only") == "src/app.py"


def test_commit_message_hook_rejects_invalid_message_and_accepts_conventional() -> None:
    with _repository() as repo:
        repo.start_task()
        repo.write("src/app.py", "VALUE = 'feature'\n")
        repo.git("add", "src/app.py")
        before = repo.git("rev-parse", "HEAD")
        result = repo.run("git", "commit", "-m", "miscellaneous work", check=False)
        assert result.returncode != 0
        assert repo.git("rev-parse", "HEAD") == before
        repo.run("scripts/git-flow-commit.sh", "feat(mcp)!: revise contract")
        assert repo.git("log", "-1", "--format=%s") == "feat(mcp)!: revise contract"
        assert repo.git("branch", "--show-current") == "feature/test-hooks"
        assert not repo.events("full")
        assert not repo.events("docs")


def test_fast_check_uses_staged_content_and_leaves_unstaged_edits_alone() -> None:
    with _repository() as repo:
        repo.start_task()
        before = repo.git("rev-parse", "HEAD")
        repo.write("src/app.py", "VALUE = 'REJECT_FAST'\n")
        repo.git("add", "src/app.py")
        repo.write("src/app.py", "VALUE = 'working tree passes'\n")
        rejected = repo.run(
            "scripts/git-flow-commit.sh", "feat: rejected snapshot", check=False
        )
        assert rejected.returncode != 0
        assert repo.git("rev-parse", "HEAD") == before
        assert "REJECT_FAST" in str(repo.events("fast")[-1]["content"])

        repo.write("src/app.py", "VALUE = 'staged candidate'\n")
        repo.git("add", "src/app.py")
        repo.write("src/app.py", "VALUE = 'REJECT_FAST unstaged'\n")
        repo.run("scripts/git-flow-commit.sh", "feat: accepted snapshot")
        assert repo.git("show", "HEAD:src/app.py") == "VALUE = 'staged candidate'"
        assert "REJECT_FAST unstaged" in (repo.root / "src/app.py").read_text()
        events = repo.events("fast")
        assert len(events) == 2
        assert events[-1]["content"] == "VALUE = 'staged candidate'\n"
        assert all(not Path(str(event["root"])).exists() for event in events)


def test_stage_commit_script_keeps_unrelated_edits_and_branch() -> None:
    with _repository() as repo:
        primary = repo.git("rev-parse", "main")
        repo.start_task()
        repo.write("src/app.py", "VALUE = 'stage one'\n")
        repo.write("README.md", "# Unfinished documentation\n")
        repo.run("scripts/git-flow-commit.sh", "feat: complete stage one", "src/app.py")
        assert repo.git("show", "--format=", "--name-only", "HEAD") == "src/app.py"
        assert repo.git("status", "--porcelain") == "M README.md"
        assert repo.git("rev-parse", "main") == primary
        assert repo.git("branch", "--show-current") == "feature/test-hooks"
        repo.run("scripts/git-flow-commit.sh", "docs: complete stage two", "README.md")
        assert repo.git("rev-list", "--count", "main..HEAD") == "2"
        assert repo.git("status", "--porcelain") == ""
        assert not repo.events("full")
        assert not repo.events("docs")


def test_snapshot_head_contains_staged_candidate_and_is_reproducible() -> None:
    with _repository() as repo:
        repo.start_task()
        previous = repo.git("rev-parse", "HEAD")
        repo.write("src/app.py", "VALUE = 'staged candidate'\n")
        repo.git("add", "src/app.py")
        candidate = repo.git("write-tree")
        repo.write("src/app.py", "VALUE = 'unstaged changes'\n")
        repo.git("config", "commit.gpgSign", "true")

        for label, year in (("First", 2020), ("Second", 2025)):
            for role in ("AUTHOR", "COMMITTER"):
                repo.environment.update(
                    {
                        f"GIT_{role}_NAME": f"{label} User",
                        f"GIT_{role}_EMAIL": f"{label.lower()}@example.invalid",
                        f"GIT_{role}_DATE": f"{year}-01-01T00:00:00+00:00",
                    }
                )
            repo.run(sys.executable, "-m", "tooling.git.gate", "commit")

        events = repo.events("fast")
        assert len(events) == 2
        assert events[0]["head"] == events[1]["head"]
        assert events[0]["head"] not in {None, previous + "\n"}
        for event in events:
            assert event["head_tree"] == candidate + "\n"
            assert event["head_content"] == "VALUE = 'staged candidate'\n"
            assert not Path(str(event["root"])).exists()
        assert repo.git("rev-parse", "HEAD") == previous
        assert repo.git("write-tree") == candidate
        assert (repo.root / "src/app.py").read_text() == "VALUE = 'unstaged changes'\n"


@pytest.mark.parametrize("failure", ["full", "mutation", "staged_mutation"])
def test_full_snapshot_failure_cleans_temporary_files_and_preserves_checkout(
    failure: str,
) -> None:
    with _repository() as repo:
        before = repo.git("rev-parse", "HEAD")
        tree = repo.git("write-tree")
        repo.environment["HOOK_TEST_FAIL"] = failure
        result = repo.run(
            sys.executable,
            "-B",
            "-c",
            "import sys; from tooling.git.candidates import check_snapshot; "
            "check_snapshot(sys.argv[1], 'full')",
            tree,
            check=False,
        )
        assert result.returncode != 0
        diagnostic = (
            "full rejected candidate"
            if failure == "full"
            else "A check modified the candidate snapshot."
        )
        assert diagnostic in result.stdout + result.stderr
        events = repo.events("full")
        assert len(events) == 1
        assert events[0]["head_tree"] == tree + "\n"
        assert events[0]["content"] == "VALUE = 'initial'\n"
        assert not Path(str(events[0]["root"])).parent.exists()
        assert repo.git("rev-parse", "HEAD") == before
        assert repo.git("write-tree") == tree
        assert repo.git("status", "--porcelain") == ""


def _advance_primary(repo: GitRepository, path: str, content: str) -> str:
    """Model a remote PR merged into main, followed by a local fast-forward."""
    current = repo.git("branch", "--show-current")
    before = repo.git("rev-parse", "main")
    repo.git("switch", "-c", "feature/upstream", "main")
    repo.write(path, content)
    repo.run("scripts/git-flow-commit.sh", "feat: update upstream", path)
    upstream = repo.git("rev-parse", "HEAD")
    repo.git("update-ref", "refs/heads/main", upstream, before)
    repo.git("switch", current)
    return upstream


def test_local_full_uses_runner_reviewer_and_candidate_checks() -> None:
    with _repository() as repo:
        repo.commit_task()
        _advance_primary(
            repo,
            "tooling/review_docs.py",
            "raise RuntimeError('candidate reviewer must not be executed')\n",
        )
        before = repo.git("rev-parse", "HEAD")
        result = repo.run(
            sys.executable, "-m", "tooling.git.gate", "full", "--base", "main"
        )
        assert "Full checks passed" in result.stdout
        reviews = repo.events("docs")
        checks = repo.events("full")
        assert len(reviews) == len(checks) == 1
        assert Path(str(reviews[0]["root"])).samefile(repo.root)
        assert checks[0]["content"] == "VALUE = 'feature'\n"
        assert checks[0]["head_tree"] == str(reviews[0]["tree"]) + "\n"
        assert not Path(str(checks[0]["root"])).parent.exists()
        assert repo.git("rev-parse", "HEAD") == before
        assert repo.git("status", "--porcelain") == ""


@pytest.mark.parametrize("primary", ["main", "master", "trunk"])
def test_local_primary_merge_commit_is_blocked(primary: str) -> None:
    with _repository(primary) as repo:
        before = repo.git("rev-parse", primary)
        task = repo.commit_task()
        repo.git("switch", primary)
        result = repo.run("git", "merge", "--no-ff", "feature/test-hooks", check=False)
        assert result.returncode != 0
        assert "integrate through a GitHub PR" in result.stdout + result.stderr
        assert repo.git("rev-parse", primary) == before
        assert repo.git("rev-parse", "feature/test-hooks") == task
        assert not repo.events("full")
        assert not repo.events("docs")
        manual = repo.run("git", "commit", "--no-edit", check=False)
        assert manual.returncode != 0
        assert repo.git("rev-parse", primary) == before


def test_task_branch_can_merge_primary_with_only_fast_checks() -> None:
    with _repository() as repo:
        task = repo.commit_task()
        upstream = _advance_primary(repo, "README.md", "# Updated upstream\n")
        previous_checks = len(repo.events("fast"))
        repo.run("git", "merge", "--no-edit", "main")
        assert repo.git("show", "-s", "--format=%P", "HEAD") == f"{task} {upstream}"
        assert repo.git("rev-parse", "main") == upstream
        assert repo.git("status", "--porcelain") == ""
        assert len(repo.events("fast")) == previous_checks + 1
        assert not repo.events("full")
        assert not repo.events("docs")


def test_task_branch_merge_failure_preserves_commits_and_candidate() -> None:
    with _repository() as repo:
        task = repo.commit_task()
        upstream = _advance_primary(repo, "README.md", "# Updated upstream\n")
        repo.environment["HOOK_TEST_FAIL"] = "fast"
        result = repo.run("git", "merge", "--no-edit", "main", check=False)
        assert result.returncode != 0
        assert "fast rejected candidate" in result.stdout + result.stderr
        assert repo.git("rev-parse", "HEAD") == task
        assert repo.git("rev-parse", "MERGE_HEAD") == upstream
        assert repo.git("show", ":README.md") == "# Updated upstream"
        assert not repo.events("full")
        assert not repo.events("docs")
        assert all(
            not Path(str(event["root"])).exists() for event in repo.events("fast")
        )


def test_task_branch_checks_resolved_conflicts_before_committing_merge() -> None:
    with _repository() as repo:
        task = repo.commit_task()
        upstream = _advance_primary(repo, "src/app.py", "VALUE = 'upstream'\n")
        conflict = repo.run("git", "merge", "main", check=False)
        assert conflict.returncode != 0
        assert repo.git("diff", "--name-only", "--diff-filter=U") == "src/app.py"
        repo.write("src/app.py", "VALUE = 'REJECT_FAST'\n")
        repo.git("add", "src/app.py")
        rejected = repo.run("git", "commit", "--no-edit", check=False)
        assert rejected.returncode != 0
        assert repo.git("rev-parse", "HEAD") == task
        assert repo.git("rev-parse", "MERGE_HEAD") == upstream
        repo.write("src/app.py", "VALUE = 'resolved'\n")
        repo.git("add", "src/app.py")
        repo.run("git", "commit", "--no-edit")
        assert repo.git("show", "-s", "--format=%P", "HEAD") == f"{task} {upstream}"
        assert repo.events("fast")[-1]["content"] == "VALUE = 'resolved'\n"
        assert repo.git("status", "--porcelain") == ""
        assert not repo.events("full")
        assert not repo.events("docs")


def test_merge_message_without_an_actual_merge_is_not_exempt() -> None:
    with _repository() as repo:
        repo.start_task()
        before = repo.git("rev-parse", "HEAD")
        repo.write("src/app.py", "VALUE = 'feature'\n")
        repo.git("add", "src/app.py")
        result = repo.run("git", "commit", "-m", "Merge pretend branch", check=False)
        assert result.returncode != 0
        assert "Conventional Commits" in result.stdout + result.stderr
        assert repo.git("rev-parse", "HEAD") == before


def test_primary_rebase_is_rejected_without_moving_primary() -> None:
    with _repository() as repo:
        primary = repo.git("rev-parse", "main")
        task = repo.commit_task()
        repo.git("switch", "main")
        result = repo.run("git", "rebase", "feature/test-hooks", check=False)
        assert result.returncode != 0
        assert repo.git("rev-parse", "main") == primary
        assert repo.git("rev-parse", "feature/test-hooks") == task
        assert repo.git("status", "--porcelain") == ""


def test_task_branch_rebase_can_synchronize_primary() -> None:
    with _repository() as repo:
        repo.commit_task()
        upstream = _advance_primary(repo, "README.md", "# Updated upstream\n")
        repo.run("git", "rebase", "main")
        assert repo.git("rev-parse", "HEAD^") == upstream
        assert repo.git("branch", "--show-current") == "feature/test-hooks"
        assert not repo.events("full")
        assert not repo.events("docs")


def _add_remote(repo: GitRepository) -> Path:
    remote = repo.root.parent / "remote.git"
    repo.run("git", "init", "--bare", "--initial-branch=main", str(remote))
    repo.run("git", "-C", str(remote), "fetch", str(repo.root), "main:refs/heads/main")
    repo.run("git", "-C", str(remote), "config", "receive.denyDeleteCurrent", "ignore")
    repo.git("remote", "add", "origin", str(remote))
    return remote


def test_task_push_does_not_run_checks_or_require_a_receipt() -> None:
    with _repository() as repo:
        _add_remote(repo)
        task = repo.commit_task()
        checks = repo.record.read_text()
        repo.environment["HOOK_TEST_FAIL"] = "full"
        repo.run("git", "push", "-u", "origin", "feature/test-hooks")
        assert repo.git("ls-remote", "origin", "refs/heads/feature/test-hooks") == (
            f"{task}\trefs/heads/feature/test-hooks"
        )
        assert repo.record.read_text() == checks
        repo.run("git", "push", "origin", "--delete", "feature/test-hooks")
        assert repo.git("ls-remote", "origin", "refs/heads/feature/test-hooks") == ""
        assert repo.record.read_text() == checks


@pytest.mark.parametrize("refspec", ["HEAD:main", ":main"])
def test_primary_push_or_deletion_is_blocked_by_destination(refspec: str) -> None:
    with _repository() as repo:
        remote = _add_remote(repo)
        before = repo.git("rev-parse", "main")
        repo.commit_task()
        checks = repo.record.read_text()
        result = repo.run("git", "push", "origin", refspec, check=False)
        assert result.returncode != 0
        assert (
            "Direct pushes/deletions of main are blocked"
            in result.stdout + result.stderr
        )
        assert (
            repo.run("git", "-C", str(remote), "rev-parse", "main").stdout.strip()
            == before
        )
        assert repo.record.read_text() == checks


def test_initial_primary_push_is_also_blocked() -> None:
    with _repository() as repo:
        remote = repo.root.parent / "empty-remote.git"
        repo.run("git", "init", "--bare", str(remote))
        repo.git("remote", "add", "origin", str(remote))
        repo.commit_task()
        checks = repo.record.read_text()
        result = repo.run("git", "push", "origin", "HEAD:main", check=False)
        assert result.returncode != 0
        assert (
            "Direct pushes/deletions of main are blocked"
            in result.stdout + result.stderr
        )
        assert repo.git("ls-remote", "origin", "refs/heads/main") == ""
        assert repo.record.read_text() == checks


def test_installer_migrates_only_the_old_main_merge_option() -> None:
    with _repository() as repo:
        repo.git("config", "core.hooksPath", "scripts/hooks")
        repo.git("config", "branch.main.mergeOptions", "--no-ff")
        repo.git("config", "pull.rebase", "true")
        repo.git("config", "branch.feature/test-hooks.mergeOptions", "--log")
        repo.run("scripts/install-git-hooks.sh")
        assert repo.git("config", "core.hooksPath") == ".githooks"
        assert (
            repo.run(
                "git", "config", "--get", "branch.main.mergeOptions", check=False
            ).returncode
            == 1
        )
        assert repo.git("config", "branch.main.rebase") == "false"
        assert repo.git("config", "pull.ff") == "only"
        assert repo.git("config", "pull.rebase") == "true"
        assert repo.git("config", "branch.feature/test-hooks.mergeOptions") == "--log"


@pytest.mark.parametrize("options", [["--ff-only"], ["--no-ff", "--log"]])
def test_installer_preserves_custom_main_merge_options(options: list[str]) -> None:
    with _repository() as repo:
        for option in options:
            repo.git("config", "--add", "branch.main.mergeOptions", option)
        repo.run("scripts/install-git-hooks.sh")
        assert (
            repo.git("config", "--get-all", "branch.main.mergeOptions").splitlines()
            == options
        )


def test_installer_refuses_to_replace_unrelated_hooks() -> None:
    with _repository() as repo:
        repo.git("config", "core.hooksPath", "custom-hooks")
        result = repo.run("scripts/install-git-hooks.sh", check=False)
        assert result.returncode != 0
        assert repo.git("config", "core.hooksPath") == "custom-hooks"
        repo.git("config", "--unset", "core.hooksPath")
        existing = repo.root / ".git/hooks/pre-commit"
        existing.write_text("#!/bin/sh\nexit 0\n")
        existing.chmod(0o755)
        result = repo.run("scripts/install-git-hooks.sh", check=False)
        assert result.returncode != 0
        assert "Refusing to disable existing hook" in result.stderr
        assert existing.read_text() == "#!/bin/sh\nexit 0\n"


@pytest.mark.parametrize("source", ["config", "remote-head", "remote-main"])
def test_primary_detection_does_not_require_local_primary(source: str) -> None:
    with _repository() as repo:
        primary = repo.git("rev-parse", "main")
        repo.start_task()
        repo.git("branch", "-D", "main")
        if source == "config":
            repo.git("config", "tradingdev.primaryBranch", "main")
        else:
            repo.git("update-ref", "refs/remotes/origin/main", primary)
            if source == "remote-head":
                repo.git(
                    "symbolic-ref",
                    "refs/remotes/origin/HEAD",
                    "refs/remotes/origin/main",
                )
        assert repo.run("scripts/detect-primary-branch.sh").stdout.strip() == "main"
        repo.run("scripts/install-git-hooks.sh")
        repo.write("src/app.py", "VALUE = 'remote-only primary'\n")
        repo.run(
            "scripts/git-flow-commit.sh", "feat: work without local main", "src/app.py"
        )


def test_primary_detection_rejects_ambiguous_main_and_master() -> None:
    with _repository() as repo:
        repo.git("branch", "master")
        result = repo.run("scripts/detect-primary-branch.sh", check=False)
        assert result.returncode != 0
        assert "both main and master exist" in result.stderr
