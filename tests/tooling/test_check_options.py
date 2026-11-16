"""Review controls are validated and forwarded through every check entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from tooling import process_guard
from tooling.git import candidates, check_pr, gate
from tooling.review_options import add_review_arguments, review_arguments

if TYPE_CHECKING:
    from typing import Any

    from pytest import CaptureFixture, MonkeyPatch
    from tests.tooling.git_check_scenarios import Scenario

from tests.tooling.git_check_scenarios import (
    _assert_cleaned,
    _source_state,
)
from tests.tooling.git_check_scenarios import (
    scenario as scenario,
)


def _forbid_check_side_effects(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        check_pr, "get_pull_request", lambda *_: pytest.fail("PR API called")
    )
    for module in (check_pr, gate, candidates):
        monkeypatch.setattr(module, "git", lambda *_: pytest.fail("Git called"))
    monkeypatch.setattr(
        candidates, "run", lambda *_, **__: pytest.fail("Review called")
    )
    monkeypatch.setattr(
        process_guard,
        "mkdtemp",
        lambda **_: pytest.fail("Temporary files created"),
    )


@pytest.mark.parametrize("runner", ["pr", "full"])
@pytest.mark.parametrize(
    "seconds, outer_timeout", [(1, 900), (600, 900), (3600, 3660), (86_400, 86_460)]
)
def test_documentation_timeout_reaches_reviewer_and_preserves_snapshot_budget(
    scenario: Scenario,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    runner: str,
    seconds: int,
    outer_timeout: int,
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
            assert timeout == 3600
            calls.append("full")
        else:
            assert timeout == outer_timeout
            assert command[command.index("--timeout-seconds") + 1] == str(seconds)
            calls.append("docs")

    monkeypatch.setattr(candidates, "run", run)
    if runner == "pr":
        check_pr.check_pr(
            7, snapshot_timeout_seconds=3600, documentation_timeout_seconds=seconds
        )
    else:
        gate.check_full(
            "main",
            "feature/work",
            snapshot_timeout_seconds=3600,
            documentation_timeout_seconds=seconds,
        )
    assert calls == ["docs", "full"]
    assert f"Documentation timeout: {seconds} seconds" in capsys.readouterr().out
    assert _source_state(scenario.source) == before
    _assert_cleaned(scenario)


@pytest.mark.parametrize("runner", ["pr", "full"])
@pytest.mark.parametrize(
    "budget, snapshot_timeout, documentation_timeout",
    [
        (None, None, None),
        (1_000_000, None, None),
        (2_000_000, 3600, None),
        (None, 3600, 600),
        (1_000_000, None, 600),
    ],
)
def test_check_cli_forwards_review_options(
    monkeypatch: MonkeyPatch,
    runner: str,
    budget: int | None,
    snapshot_timeout: int | None,
    documentation_timeout: int | None,
) -> None:
    calls: list[tuple[int | None, int, int]] = []

    def check(
        first: int | str,
        second: str,
        *,
        max_evidence_bytes: int | None,
        snapshot_timeout_seconds: int,
        documentation_timeout_seconds: int,
        review_scopes: tuple[str, ...] | None,
        review_context: tuple[str, ...],
        review_context_for: tuple[tuple[str, str], ...],
        max_prompt_bytes: int | None,
        max_total_prompt_bytes: int | None,
    ) -> None:
        assert review_scopes is None
        assert review_context == ()
        assert review_context_for == ()
        assert max_prompt_bytes is max_total_prompt_bytes is None
        assert (first, second) == (
            (7, "upstream") if runner == "pr" else ("main", "HEAD")
        )
        calls.append(
            (
                max_evidence_bytes,
                snapshot_timeout_seconds,
                documentation_timeout_seconds,
            )
        )

    module = check_pr if runner == "pr" else gate
    monkeypatch.setattr(module, "git", lambda *_: str(Path.cwd()))
    monkeypatch.setattr(module, "check_pr" if runner == "pr" else "check_full", check)
    arguments = (
        ["7", "--remote", "upstream"] if runner == "pr" else ["full", "--base", "main"]
    )
    if budget is not None:
        arguments.extend(["--max-evidence-bytes", str(budget)])
    if snapshot_timeout is not None:
        arguments.extend(["--snapshot-timeout-seconds", str(snapshot_timeout)])
    if documentation_timeout is not None:
        arguments.extend(
            ["--documentation-timeout-seconds", str(documentation_timeout)]
        )
    monkeypatch.setattr(sys, "argv", ["check", *arguments])
    module.main()
    assert calls == [(budget, snapshot_timeout or 1800, documentation_timeout or 180)]


@pytest.mark.parametrize("runner", ["pr", "full"])
def test_scoped_review_options_reach_reviewer_without_skipping_full_checks(
    scenario: Scenario, monkeypatch: MonkeyPatch, runner: str
) -> None:
    before = _source_state(scenario.source)
    calls: list[str] = []
    contexts = ("docs/中文 $(literal); space%.md", "AGENTS.md")
    expected = [
        "--review-scope",
        "tooling",
        "--review-scope",
        "execution",
        "--review-context",
        contexts[0],
        "--review-context",
        contexts[1],
        "--review-context-for=tooling:docs/中文: $(literal).md",
        "--review-context-for=cross_scope:-name.md",
        "--max-prompt-bytes",
        "123456",
        "--max-total-prompt-bytes",
        "456789",
    ]

    def run(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 900,
        temporary_root: Path | None = None,
    ) -> None:
        if command[0] == "bash":
            assert command == ["bash", "scripts/check-full.sh"]
            calls.append("full")
        else:
            assert command[:3] == [sys.executable, "-m", "tooling.review_docs"]
            assert command[command.index("--review-scope") :] == expected
            assert command[command.index("--max-evidence-bytes") + 1] == "600123"
            calls.append("review")

    monkeypatch.setattr(candidates, "run", run)
    options: dict[str, Any] = {
        "review_scopes": ("tooling", "execution"),
        "review_context": contexts,
        "review_context_for": (
            ("tooling", "docs/中文: $(literal).md"),
            ("cross_scope", "-name.md"),
        ),
        "max_prompt_bytes": 123456,
        "max_total_prompt_bytes": 456789,
        "max_evidence_bytes": 600123,
    }
    # Every value retains its own type through both public entrypoints.
    if runner == "pr":
        check_pr.check_pr(7, **options)
    else:
        gate.check_full("main", "feature/work", **options)
    assert calls == ["review", "full"]
    assert _source_state(scenario.source) == before
    _assert_cleaned(scenario)


@pytest.mark.parametrize("runner", ["pr", "full"])
def test_cli_forwards_repeated_scope_and_literal_context_paths(
    monkeypatch: MonkeyPatch, runner: str
) -> None:
    calls: list[dict[str, object]] = []
    module = check_pr if runner == "pr" else gate
    monkeypatch.setattr(module, "git", lambda *_: str(Path.cwd()))
    monkeypatch.setattr(
        module,
        "check_pr" if runner == "pr" else "check_full",
        lambda *_, **options: calls.append(options),
    )
    entry = ["7"] if runner == "pr" else ["full", "--base", "main"]
    contexts = ("docs/中文 $(literal); %.md", "AGENTS.md")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check",
            *entry,
            "--review-scope",
            "tooling",
            "--review-scope",
            "execution",
            "--review-context",
            contexts[0],
            "--review-context",
            contexts[1],
            "--review-context-for=tooling:docs/中文: $(literal).md",
            "--review-context-for=cross_scope:-name.md",
            "--max-prompt-bytes",
            "123",
            "--max-total-prompt-bytes",
            "456",
            "--max-evidence-bytes",
            "789",
        ],
    )
    module.main()
    assert calls == [
        {
            "review_scopes": ("tooling", "execution"),
            "review_context": contexts,
            "review_context_for": (
                ("tooling", "docs/中文: $(literal).md"),
                ("cross_scope", "-name.md"),
            ),
            "max_prompt_bytes": 123,
            "max_total_prompt_bytes": 456,
            "max_evidence_bytes": 789,
            "snapshot_timeout_seconds": 1800,
            "documentation_timeout_seconds": 180,
        }
    ]


def test_forwarded_option_like_context_paths_remain_literal_files() -> None:
    parser = argparse.ArgumentParser()
    add_review_arguments(parser)
    contexts = ("--max-prompt-bytes", "-研究 $(literal); %.md", "docs/plain.md")
    parsed = parser.parse_args(review_arguments(review_context=contexts))
    assert parsed.review_context == list(contexts)
    scoped = (
        ("tooling", "--max-prompt-bytes"),
        ("cross_scope", "-研究: $(literal); %.md"),
    )
    argv = review_arguments(review_context_for=scoped)
    assert argv == [f"--review-context-for={scope}:{path}" for scope, path in scoped]
    assert parser.parse_args(argv).review_context_for == list(scoped)
    assert parsed.max_prompt_bytes is None
    assert parsed.max_total_prompt_bytes is None
    assert parsed.review_scopes is None


@pytest.mark.parametrize("entry", ["pr", "full", "verify"])
@pytest.mark.parametrize(
    "options",
    [
        {name: value}
        for name in (
            "max_evidence_bytes",
            "max_prompt_bytes",
            "max_total_prompt_bytes",
        )
        for value in (False, True, 0, -1, 1.5, "10")
    ]
    + [{"review_scopes": scopes} for scopes in ((), ("",), ("unknown",), "tooling")]
    + [{"review_context": ("",)}]
    + [
        {"review_context_for": value}
        for value in (
            (("unknown", "file.py"),),
            (("tooling", ""),),
            (("tooling",),),
            (("tooling", 1),),
            [("tooling", "file.py")],
        )
    ],
)
def test_invalid_model_review_options_reject_before_side_effects(
    monkeypatch: MonkeyPatch, entry: str, options: dict[str, Any]
) -> None:
    _forbid_check_side_effects(monkeypatch)
    with pytest.raises(RuntimeError):
        if entry == "pr":
            check_pr.check_pr(7, **options)
        elif entry == "full":
            gate.check_full("main", "HEAD", **options)
        else:
            candidates.verify_candidate(
                candidates.MergeCandidate(Path.cwd(), "base", "head", "tree"),
                **options,
            )


@pytest.mark.parametrize("runner", ["pr", "full"])
@pytest.mark.parametrize(
    "flag, value",
    [
        ("--review-scope", ""),
        ("--review-scope", "unknown"),
        ("--review-context-for", "unknown:file.py"),
        ("--review-context-for", "tooling:"),
        ("--review-context-for", "tooling"),
        *[
            (flag, value)
            for flag in ("--max-prompt-bytes", "--max-total-prompt-bytes")
            for value in ("0", "-1", "1.5", "true")
        ],
    ],
)
def test_cli_rejects_invalid_review_controls_at_parse_boundary(
    monkeypatch: MonkeyPatch, runner: str, flag: str, value: str
) -> None:
    _forbid_check_side_effects(monkeypatch)
    entry = ["7"] if runner == "pr" else ["full", "--base", "main"]
    monkeypatch.setattr(sys, "argv", ["check", *entry, flag, value])
    with pytest.raises(SystemExit) as failure:
        (check_pr if runner == "pr" else gate).main()
    assert failure.value.code == 2


@pytest.mark.parametrize("action", ["commit", "push"])
@pytest.mark.parametrize(
    "option, value",
    [
        ("--review-scope", "tooling"),
        ("--review-context", "AGENTS.md"),
        ("--review-context-for", "tooling:AGENTS.md"),
        ("--max-prompt-bytes", "123"),
        ("--max-total-prompt-bytes", "456"),
    ],
)
def test_review_controls_are_not_ignored_by_other_gate_actions(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    action: str,
    option: str,
    value: str,
) -> None:
    _forbid_check_side_effects(monkeypatch)
    arguments = ["gate", action, option, value]
    if action == "push":
        arguments.extend(["--destination", "git@github.com:owner/project.git"])
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as failure:
        gate.main()
    assert failure.value.code == 2
    assert f"{option} is only valid for full checks" in capsys.readouterr().err


@pytest.mark.parametrize("budget", [0, -1])
@pytest.mark.parametrize("entry", ["pr", "full", "verify", "pr_cli", "full_cli"])
def test_invalid_budget_fails_before_git_or_review(
    monkeypatch: MonkeyPatch, budget: int, entry: str
) -> None:
    _forbid_check_side_effects(monkeypatch)
    with pytest.raises(RuntimeError, match="budget must be positive"):
        if entry == "pr":
            check_pr.check_pr(7, max_evidence_bytes=budget)
        elif entry == "full":
            gate.check_full("main", "HEAD", max_evidence_bytes=budget)
        elif entry == "verify":
            candidate = candidates.MergeCandidate(Path.cwd(), "base", "head", "tree")
            candidates.verify_candidate(candidate, max_evidence_bytes=budget)
        else:
            arguments = ["7"] if entry == "pr_cli" else ["full", "--base", "main"]
            monkeypatch.setattr(
                sys, "argv", ["check", *arguments, "--max-evidence-bytes", str(budget)]
            )
            (check_pr if entry == "pr_cli" else gate).main()


@pytest.mark.parametrize("action", ["commit", "push"])
def test_budget_option_is_not_silently_ignored_by_other_git_actions(
    monkeypatch: MonkeyPatch, action: str, capsys: CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gate, "git", lambda *_: pytest.fail("Git called"))
    arguments = ["gate", action, "--max-evidence-bytes", "1000000"]
    if action == "push":
        arguments.extend(["--destination", "git@github.com:owner/project.git"])
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as failure:
        gate.main()
    assert failure.value.code == 2
    assert "only valid for full checks" in capsys.readouterr().err


@pytest.mark.parametrize("timeout", [0, -1])
@pytest.mark.parametrize(
    "entry", ["pr", "full", "verify", "snapshot", "pr_cli", "full_cli"]
)
def test_invalid_snapshot_timeout_fails_before_git_or_review(
    monkeypatch: MonkeyPatch, timeout: int, entry: str
) -> None:
    _forbid_check_side_effects(monkeypatch)
    with pytest.raises(
        RuntimeError, match="Snapshot timeout must be a positive integer"
    ):
        if entry == "pr":
            check_pr.check_pr(7, snapshot_timeout_seconds=timeout)
        elif entry == "full":
            gate.check_full("main", "HEAD", snapshot_timeout_seconds=timeout)
        elif entry == "verify":
            candidate = candidates.MergeCandidate(Path.cwd(), "base", "head", "tree")
            candidates.verify_candidate(candidate, snapshot_timeout_seconds=timeout)
        elif entry == "snapshot":
            candidates.check_snapshot("tree", "full", timeout_seconds=timeout)
        else:
            arguments = ["7"] if entry == "pr_cli" else ["full", "--base", "main"]
            monkeypatch.setattr(
                sys,
                "argv",
                ["check", *arguments, "--snapshot-timeout-seconds", str(timeout)],
            )
            (check_pr if entry == "pr_cli" else gate).main()


@pytest.mark.parametrize(
    "timeout", [0, -1, 86_401, True, 1.5, float("nan"), float("inf")]
)
@pytest.mark.parametrize("entry", ["pr", "full", "verify"])
def test_invalid_documentation_timeout_fails_before_git_or_review(
    monkeypatch: MonkeyPatch, timeout: Any, entry: str
) -> None:
    _forbid_check_side_effects(monkeypatch)
    with pytest.raises(RuntimeError, match="Documentation timeout must be"):
        if entry == "pr":
            check_pr.check_pr(7, documentation_timeout_seconds=timeout)
        elif entry == "full":
            gate.check_full("main", "HEAD", documentation_timeout_seconds=timeout)
        else:
            candidate = candidates.MergeCandidate(Path.cwd(), "base", "head", "tree")
            candidates.verify_candidate(
                candidate, documentation_timeout_seconds=timeout
            )


@pytest.mark.parametrize("timeout", [0, -1, 86_401])
@pytest.mark.parametrize("entry", ["pr", "full"])
def test_documentation_timeout_cli_rejects_out_of_range_before_git(
    monkeypatch: MonkeyPatch, timeout: int, entry: str
) -> None:
    _forbid_check_side_effects(monkeypatch)
    arguments = ["7"] if entry == "pr" else ["full", "--base", "main"]
    monkeypatch.setattr(
        sys,
        "argv",
        ["check", *arguments, "--documentation-timeout-seconds", str(timeout)],
    )
    with pytest.raises(RuntimeError, match="Documentation timeout must be"):
        (check_pr if entry == "pr" else gate).main()


@pytest.mark.parametrize("timeout", ["nan", "inf", "-inf", "1.5", "text"])
@pytest.mark.parametrize("entry", ["pr", "full"])
@pytest.mark.parametrize("option", ["snapshot", "documentation"])
def test_timeout_cli_rejects_noninteger_before_git(
    monkeypatch: MonkeyPatch, timeout: str, entry: str, option: str
) -> None:
    module = check_pr if entry == "pr" else gate
    _forbid_check_side_effects(monkeypatch)
    arguments = ["7"] if entry == "pr" else ["full", "--base", "main"]
    monkeypatch.setattr(
        sys, "argv", ["check", *arguments, f"--{option}-timeout-seconds={timeout}"]
    )
    with pytest.raises(SystemExit) as failure:
        module.main()
    assert failure.value.code == 2


@pytest.mark.parametrize("action", ["commit", "push"])
@pytest.mark.parametrize("option", ["snapshot", "documentation"])
def test_timeout_option_is_rejected_for_other_git_actions(
    monkeypatch: MonkeyPatch,
    action: str,
    option: str,
    capsys: CaptureFixture[str],
) -> None:
    monkeypatch.setattr(gate, "git", lambda *_: pytest.fail("Git called"))
    arguments = ["gate", action, f"--{option}-timeout-seconds", "3600"]
    if action == "push":
        arguments.extend(["--destination", "git@github.com:owner/project.git"])
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as failure:
        gate.main()
    assert failure.value.code == 2
    assert "only valid for full checks" in capsys.readouterr().err
