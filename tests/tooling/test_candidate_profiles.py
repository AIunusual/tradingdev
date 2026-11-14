"""Candidate checks select and execute the required verification profiles."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from tooling import review_docs, review_evidence
from tooling.environment import EnvironmentCheckError
from tooling.git import candidates, check_pr, gate

if TYPE_CHECKING:
    from pytest import CaptureFixture, MonkeyPatch
    from tests.tooling.git_check_scenarios import Scenario

from tests.tooling.git_check_scenarios import (
    _assert_cleaned,
    _commit,
    _git,
    _source_state,
)
from tests.tooling.git_check_scenarios import (
    lightweight_scenario as lightweight_scenario,
)
from tests.tooling.git_check_scenarios import (
    scenario as scenario,
)


@pytest.mark.parametrize("runner", ["pr", "local"])
def test_candidate_missing_environment_stops_before_review_and_tests(
    scenario: Scenario,
    monkeypatch: MonkeyPatch,
    runner: str,
) -> None:
    before = _source_state(scenario.source)

    def unavailable(profile: str) -> None:
        assert profile == "full"
        raise EnvironmentCheckError("Missing strategy runtime image")

    monkeypatch.setattr(candidates, "check_environment", unavailable)
    monkeypatch.setattr(
        candidates, "run", lambda *_, **__: pytest.fail("Review or tests started")
    )
    with pytest.raises(EnvironmentCheckError, match="Missing strategy runtime image"):
        if runner == "pr":
            check_pr.check_pr(7)
        else:
            gate.check_full("main", "feature/work")
    assert _source_state(scenario.source) == before
    _assert_cleaned(scenario)


@pytest.mark.parametrize("runner", ["pr", "full"])
@pytest.mark.parametrize("failure", [None, "rejection", "timeout", "interrupt"])
def test_lightweight_candidate_uses_merged_tree_and_only_selected_checks(
    lightweight_scenario: Scenario,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    runner: str,
    failure: str | None,
) -> None:
    scenario = lightweight_scenario
    profile = scenario.request.head_ref.removeprefix("feature/")
    before = _source_state(scenario.source)
    calls: list[str] = []

    def run(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 900,
        temporary_root: Path | None = None,
    ) -> None:
        assert command == ["bash", f"scripts/check-{profile}.sh"]
        assert timeout == 30
        calls.append(command[1])
        # The PR branched before this code change on main. Classification must
        # compare main with the merge tree, not main with the divergent PR head.
        assert (cwd / "base.txt").read_text() == "new main behavior\n"
        name = (
            "docs/talks/tradingdev-introduction/slides.md"
            if profile == "presentations"
            else "README.md"
        )
        assert (cwd / name).read_text() == "# Documentation change\n"
        assert temporary_root is not None
        if failure:
            temporary_root.mkdir()
            (temporary_root / "partial.txt").write_text("partial check output")
        if failure == "rejection":
            raise subprocess.CalledProcessError(1, command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, timeout)
        if failure == "interrupt":
            raise KeyboardInterrupt

    monkeypatch.setattr(candidates, "run", run)

    def check() -> None:
        if runner == "pr":
            check_pr.check_pr(
                7,
                snapshot_timeout_seconds=30,
                review_scopes=("tooling",),
                review_context=("AGENTS.md",),
                max_prompt_bytes=123,
                max_total_prompt_bytes=456,
            )
        else:
            gate.check_full(
                "main",
                f"feature/{profile}",
                snapshot_timeout_seconds=30,
                review_scopes=("tooling",),
                review_context=("AGENTS.md",),
                max_prompt_bytes=123,
                max_total_prompt_bytes=456,
            )

    if failure:
        error = {
            "rejection": subprocess.CalledProcessError,
            "timeout": subprocess.TimeoutExpired,
            "interrupt": KeyboardInterrupt,
        }[failure]
        with pytest.raises(error):
            check()
    else:
        check()
    assert calls == [f"scripts/check-{profile}.sh"]
    output = capsys.readouterr().out
    assert "Codex documentation review and full pytest passed" not in output
    assert ("checks passed" in output) is (failure is None)
    assert _source_state(scenario.source) == before
    assert len(scenario.temporary_paths) == 2
    _assert_cleaned(scenario)


@pytest.mark.parametrize("runner", ["pr", "full"])
@pytest.mark.parametrize(
    ("name", "failure", "slides"),
    [
        ("AGENTS.md", None, False),
        ("AGENTS.md", "rejection", False),
        ("AGENTS.md", "mutation", False),
        ("AGENTS.md", "refs", False),
        ("AGENTS.md", None, True),
        ("AGENTS.md", "slides", True),
        ("tests/tooling/git_check_scenarios.py", None, False),
        (".agents/skills/wait-for-check/SKILL.md", None, False),
    ],
)
def test_tooling_candidate_selects_its_suite_without_losing_candidate_guards(
    scenario: Scenario,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    runner: str,
    name: str,
    failure: str | None,
    slides: bool,
) -> None:
    _git(
        scenario.source,
        "switch",
        "--quiet",
        "-c",
        "feature/tooling",
        f"{scenario.request.base_sha}^",
    )
    (scenario.source / name).parent.mkdir(parents=True, exist_ok=True)
    head = _commit(scenario.source, name, "# Development tool or policy\n")
    if slides:
        slides_name = "docs/talks/tradingdev-introduction/slides.md"
        (scenario.source / slides_name).parent.mkdir(parents=True)
        head = _commit(scenario.source, slides_name, "# Slides\n")
    _git(scenario.source, "update-ref", "refs/pull/7/head", head)
    request = replace(scenario.request, head_sha=head, head_ref="feature/tooling")
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: request)
    before = _source_state(scenario.source)
    calls: list[str] = []

    def run(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 900,
        temporary_root: Path | None = None,
    ) -> None:
        if command[0] != "bash":
            assert command[:3] == [sys.executable, "-m", "tooling.review_docs"]
            calls.append("review")
            return
        profile = command[1]
        calls.append(profile)
        assert profile in {"scripts/check-tooling.sh", "scripts/check-presentations.sh"}
        assert timeout == 30
        assert (cwd / "base.txt").read_text() == "new main behavior\n"
        assert (cwd / name).read_text() == "# Development tool or policy\n"
        if profile == "scripts/check-presentations.sh":
            if failure == "slides":
                raise subprocess.CalledProcessError(1, command)
            return
        if failure == "rejection":
            raise subprocess.CalledProcessError(1, command)
        if failure == "mutation":
            (cwd / name).write_text("checker changed candidate\n")
        if failure == "refs":
            _git(scenario.source, "update-ref", "refs/heads/main", head)

    monkeypatch.setattr(candidates, "run", run)

    def check() -> None:
        if runner == "pr":
            check_pr.check_pr(7, snapshot_timeout_seconds=30)
        else:
            gate.check_full("main", "feature/tooling", snapshot_timeout_seconds=30)

    if failure in {"rejection", "slides"}:
        with pytest.raises(subprocess.CalledProcessError):
            check()
    elif failure:
        with pytest.raises(RuntimeError, match="modified.*snapshot|refs changed"):
            check()
    else:
        check()
    expected = ["scripts/check-presentations.sh"] if slides else []
    if failure != "slides":
        expected.extend(["review", "scripts/check-tooling.sh"])
    assert calls == expected
    output = capsys.readouterr().out
    assert "full pytest passed" not in output
    assert ("PR checks passed:" in output or "Candidate checks passed." in output) is (
        failure is None
    )
    if failure != "refs":
        assert _source_state(scenario.source) == before
    _assert_cleaned(scenario)


@pytest.mark.parametrize("runner", ["pr", "full"])
@pytest.mark.parametrize("budget", [None, 1_000_000, 2_000_000])
@pytest.mark.parametrize("snapshot_timeout", [900, 3600])
def test_check_validates_both_sides_and_removes_candidate_and_snapshot(
    scenario: Scenario,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    runner: str,
    budget: int | None,
    snapshot_timeout: int,
) -> None:
    before = _source_state(scenario.source)
    calls: list[str] = []

    def run(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 900,
        temporary_root: Path | None = None,
    ) -> None:
        if command[0] == "bash":
            assert timeout == snapshot_timeout
            calls.append("full")
            assert command[1] == "scripts/check-full.sh"
            assert (cwd / "base.txt").read_text() == "new main behavior\n"
            assert (cwd / "head.txt").read_text() == "new PR behavior\n"
            assert temporary_root is not None
            temporary_root.mkdir()
            (temporary_root / "model-output.txt").write_text("temporary output")
        else:
            assert timeout == 900
            assert command[command.index("--timeout-seconds") + 1] == "180"
            calls.append("docs")
            assert command[:3] == [sys.executable, "-m", "tooling.review_docs"]
            assert cwd == Path(review_docs.__file__).resolve().parents[1]
            for flag in (
                "--review-scope",
                "--review-context",
                "--max-prompt-bytes",
                "--max-total-prompt-bytes",
            ):
                assert flag not in command
            assert command[command.index("--base") + 1] == scenario.request.base_sha
            assert Path(command[command.index("--repo") + 1]).exists()
            if budget is None:
                assert "--max-evidence-bytes" not in command
                assert review_evidence.MAX_EVIDENCE_BYTES == 600_000
            else:
                assert command[command.index("--max-evidence-bytes") + 1] == str(budget)

    monkeypatch.setattr(candidates, "run", run)
    if runner == "pr":
        check_pr.check_pr(
            7, max_evidence_bytes=budget, snapshot_timeout_seconds=snapshot_timeout
        )
    else:
        gate.check_full(
            "main",
            "feature/work",
            max_evidence_bytes=budget,
            snapshot_timeout_seconds=snapshot_timeout,
        )
    assert calls == ["docs", "full"]
    assert _source_state(scenario.source) == before
    assert len(scenario.temporary_paths) == 2
    _assert_cleaned(scenario)
    output = capsys.readouterr().out
    assert f"Snapshot timeout: {snapshot_timeout} seconds" in output
    assert (
        "PR checks passed" if runner == "pr" else "Candidate checks passed"
    ) in output
    assert scenario.request.base_sha in output and scenario.request.head_sha in output


@pytest.mark.parametrize("runner", ["pr", "full"])
@pytest.mark.parametrize("failure", [False, True])
def test_code_and_presentation_changes_run_both_check_profiles(
    scenario: Scenario, monkeypatch: MonkeyPatch, runner: str, failure: bool
) -> None:
    name = "docs/talks/tradingdev-introduction/slides.md"
    (scenario.source / name).parent.mkdir(parents=True)
    head = _commit(scenario.source, name, "# Changed slides\n")
    _git(scenario.source, "update-ref", "refs/pull/7/head", head)
    request = replace(scenario.request, head_sha=head)
    monkeypatch.setattr(check_pr, "get_pull_request", lambda *_: request)
    before = _source_state(scenario.source)
    calls: list[str] = []

    def run(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 900,
        temporary_root: Path | None = None,
    ) -> None:
        if command[0] != "bash":
            assert command[:3] == [sys.executable, "-m", "tooling.review_docs"]
            assert command[command.index("--review-scope") + 1] == "presentation"
            assert command[command.index("--review-context") + 1] == name
            calls.append("review")
            return
        assert timeout == 45
        assert (cwd / name).read_text() == "# Changed slides\n"
        assert (cwd / "base.txt").read_text() == "new main behavior\n"
        assert (cwd / "head.txt").read_text() == "new PR behavior\n"
        calls.append(command[1])
        if failure and command[1] == "scripts/check-presentations.sh":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(candidates, "run", run)

    def check() -> None:
        if runner == "pr":
            check_pr.check_pr(
                7,
                snapshot_timeout_seconds=45,
                review_scopes=("presentation",),
                review_context=(name,),
            )
        else:
            gate.check_full(
                "main",
                "feature/work",
                snapshot_timeout_seconds=45,
                review_scopes=("presentation",),
                review_context=(name,),
            )

    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            check()
        assert calls == ["scripts/check-presentations.sh"]
    else:
        check()
        assert calls == [
            "scripts/check-presentations.sh",
            "review",
            "scripts/check-full.sh",
        ]
    assert _source_state(scenario.source) == before
    _assert_cleaned(scenario)


@pytest.mark.parametrize("failure", [None, "docs", "review"])
def test_srs_candidate_runs_snapshot_then_semantic_review_without_pytest(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    failure: str | None,
) -> None:
    _git(tmp_path, "init", "--quiet", "--template=")
    (tmp_path / "README.md").write_text("# Product\n")
    _git(tmp_path, "add", ".")
    base = _git(tmp_path, "write-tree")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/requirements.md").write_text("# SRS\n")
    _git(tmp_path, "add", ".")
    tree = _git(tmp_path, "write-tree")
    observed: list[str] = []

    def snapshot(actual: str, profile: str, **kwargs: object) -> None:
        assert actual == tree and kwargs["repo"] == tmp_path
        observed.append(profile)
        if failure == "docs":
            raise RuntimeError("snapshot rejected")

    def review(command: list[str], **kwargs: object) -> None:
        assert "tooling.review_docs" in command
        assert command[command.index("--tree") + 1] == tree
        assert command[command.index("--base") + 1] == base
        observed.append("review")
        if failure == "review":
            raise RuntimeError("review rejected")

    monkeypatch.setattr(candidates, "check_snapshot", snapshot)
    monkeypatch.setattr(candidates, "run", review)
    candidate = candidates.MergeCandidate(tmp_path, base, tree, tree)
    if failure:
        with pytest.raises(RuntimeError, match="rejected"):
            candidates.verify_candidate(candidate)
        assert "passed" not in capsys.readouterr().out
    else:
        candidates.verify_candidate(candidate)
        output = capsys.readouterr().out
        assert "docs snapshot checks passed" in output
        assert "pytest passed" not in output
    assert observed == (["docs"] if failure == "docs" else ["docs", "review"])
