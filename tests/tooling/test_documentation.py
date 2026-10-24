"""Documentation checks stay offline and inspect rendered Markdown links."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from tooling import documentation

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import CaptureFixture, MonkeyPatch


def _write(root: Path, name: str, text: str) -> Path:
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def test_local_links_resolve_headings_paths_images_and_reference_links(
    tmp_path: Path,
) -> None:
    source = _write(
        tmp_path,
        "README.md",
        "# Start\n\n"
        "[guide](docs/My%20Guide.md#設定與連線)\n"
        "[root](/docs/My%20Guide.md#format-code)\n"
        "[repeat](docs/My%20Guide.md#repeat-1)\n"
        "[collision](docs/My%20Guide.md#repeat-1-1)\n"
        "[encoded](docs/My%20Guide.md#%E8%A8%AD%E5%AE%9A%E8%88%87%E9%80%A3%E7%B7%9A)\n"
        "[source](src/app.py#L1)\n"
        "[directory](docs/)\n"
        "[self](#start)\n"
        "![chart](docs/chart.svg)\n"
        "[reference][guide]\n\n"
        "[guide]: docs/My%20Guide.md#format-code\n",
    )
    _write(
        tmp_path,
        "docs/My Guide.md",
        "# 設定與連線\n\n"
        "## **Format** `code`!\n\n"
        "## Repeat\n\n"
        "## Repeat\n\n"
        "## Repeat-1\n",
    )
    _write(tmp_path, "docs/chart.svg", "<svg/>\n")
    _write(tmp_path, "src/app.py", "VALUE = 1\n")
    documentation.check_local_links(tmp_path, [source])


@pytest.mark.parametrize(
    "destination, reason",
    [
        ("missing.md", "target does not exist"),
        ("docs/guide.md#removed", "Markdown heading does not exist"),
        ("#removed", "Markdown heading does not exist"),
        ("../outside.md", "target is outside the repository"),
        ("docs/%2e%2e/%2e%2e/outside.md", "target is outside the repository"),
    ],
)
def test_invalid_local_links_report_source_line_and_destination(
    tmp_path: Path, destination: str, reason: str
) -> None:
    source = _write(tmp_path, "README.md", f"# Start\n\n[broken]({destination})\n")
    _write(tmp_path, "docs/guide.md", "# Current\n")
    with pytest.raises(documentation.DocumentationError) as caught:
        documentation.check_local_links(tmp_path, [source])
    assert f"README.md:3: {destination}: {reason}" in str(caught.value)


def test_symlink_link_cannot_escape_repository(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    source = _write(root, "README.md", "# Start\n\n[external](outside.md)\n")
    external = _write(tmp_path, "private.md", "# Private\n")
    (root / "outside.md").symlink_to(external)
    with pytest.raises(
        documentation.DocumentationError, match="outside the repository"
    ):
        documentation.check_local_links(root, [source])


def test_symlink_document_is_rejected_before_formatter_reads_it(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    external = _write(tmp_path, "private.md", "# Private\n")
    (root / "README.md").symlink_to(external)

    def no_process(*_: object, **__: object) -> None:
        pytest.fail("An external document must not reach the Markdown formatter")

    monkeypatch.setattr(subprocess, "run", no_process)
    with pytest.raises(
        documentation.DocumentationError, match="source resolves outside"
    ):
        documentation.check_documents(root)


def test_markdown_parser_ignores_code_and_does_not_fetch_external_links(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    source = _write(
        tmp_path,
        "README.md",
        "# Start\n\n"
        "`[example](missing.md)`\n\n"
        "```markdown\n[example](missing.md#gone)\n```\n\n"
        "    [indented](missing.md)\n\n"
        "[web](https://invalid.example/missing.md#gone)\n"
        "[relative host](//invalid.example/missing.md)\n"
        "[email](mailto:nobody@example.invalid)\n",
    )

    def no_process(*_: object, **__: object) -> None:
        pytest.fail("Link validation must not start a command or network client")

    monkeypatch.setattr(subprocess, "run", no_process)
    documentation.check_local_links(tmp_path, [source])


def test_deleting_heading_detects_unchanged_incoming_link(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "# Start\n\n[guide](docs/guide.md#original)\n")
    _write(tmp_path, "docs/guide.md", "# Replacement\n")
    with pytest.raises(documentation.DocumentationError, match="README.md:3"):
        documentation.check_local_links(
            tmp_path, documentation.markdown_paths(tmp_path)
        )


def test_document_check_runs_only_markdown_formatter_and_links(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    _write(tmp_path, "README.md", "# Start\n")
    _write(tmp_path, "docs/guide.md", "# Guide\n")
    _write(tmp_path, "src/ignored.md", "# Not a user document\n")
    commands: list[list[str]] = []

    def capture(command: list[str], **kwargs: object) -> None:
        commands.append(command)
        assert kwargs["cwd"] == tmp_path
        assert kwargs["check"] is True

    monkeypatch.setattr(subprocess, "run", capture)
    documentation.check_documents(tmp_path)
    assert commands == [
        [
            sys.executable,
            "-m",
            "pymarkdown",
            "-d",
            "MD013",
            "scan",
            "README.md",
            "docs/guide.md",
        ]
    ]
    assert "2 documents" in capsys.readouterr().out


def test_markdown_failure_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    _write(tmp_path, "README.md", "# Title\n")

    def fail(command: list[str], **_: object) -> None:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        documentation.check_documents(tmp_path)
    assert "passed" not in capsys.readouterr().out


def test_empty_document_tree_does_not_start_an_unscoped_markdown_scan(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    def no_process(*_: object, **__: object) -> None:
        pytest.fail("An empty document list must not scan another directory")

    monkeypatch.setattr(subprocess, "run", no_process)
    documentation.check_documents(tmp_path)
