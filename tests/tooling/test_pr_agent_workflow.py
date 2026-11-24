"""Exercise the fork PR author check without GitHub or model credentials."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

_WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/pr-agent.yml"


@pytest.mark.parametrize(
    ("http_status", "expected_exit", "eligible"),
    [("204", 0, True), ("404", 0, False), ("403", 1, False), ("503", 1, False)],
)
def test_only_confirmed_pr_authors_reach_paid_review(
    tmp_path: Path, http_status: str, expected_exit: int, eligible: bool
) -> None:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["review"]["steps"]
    author = next(step for step in steps if step.get("id") == "author")
    review = next(step for step in steps if "OPENAI_KEY" in step.get("env", {}))
    assert steps.index(author) < steps.index(review)
    assert author["env"]["PR_AUTHOR"] == "${{ github.event.pull_request.user.login }}"
    assert review["if"] == "steps.author.outputs.eligible == 'true'"
    assert review["env"]["OPENAI_KEY"] == "${{ secrets.OPENAI_API_KEY }}"

    # Check the membership in the target repository, even for a fork author.
    curl = tmp_path / "curl"
    curl.write_text(
        "#!/bin/bash\n"
        '[[ "${!#}" == "https://api.github.test/repos/base/repo/collaborators/alice" ]]'
        ' || exit 2\nprintf %s "$HTTP_STATUS"\n',
        encoding="utf-8",
    )
    curl.chmod(0o700)
    output = tmp_path / "output"
    output.touch()
    result = subprocess.run(
        ["/bin/bash", "-e", "-c", author["run"]],
        cwd=tmp_path,
        env={
            "PATH": str(tmp_path),
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
            "GITHUB_API_URL": "https://api.github.test",
            "GITHUB_REPOSITORY": "base/repo",
            "PR_AUTHOR": "alice",
            "GH_TOKEN": "test-token",
            "HTTP_STATUS": http_status,
        },
        check=False,
        timeout=5,
    )
    assert result.returncode == expected_exit
    assert ("eligible=true" in output.read_text().splitlines()) is eligible


def test_collaborator_pr_with_auto_title_can_be_reviewed_after_updates() -> None:
    workflow = yaml.load(_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    events = workflow["on"]["pull_request_target"]["types"]
    steps = workflow["jobs"]["review"]["steps"]
    review = next(step for step in steps if step.get("id") == "review")
    settings = review["env"]
    agent_events = json.loads(settings["GITHUB_ACTION_CONFIG__PR_ACTIONS"])
    assert "synchronize" in events and set(events) <= set(agent_events)
    # v0.50.0 silently skips [Auto] titles unless the workflow overrides its default.
    title_filters = json.loads(settings["CONFIG__IGNORE_PR_TITLE"])
    assert not any(re.search(pattern, "[Auto] 更新清單") for pattern in title_filters)
