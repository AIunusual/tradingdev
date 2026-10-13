"""CLI lookup must work from a terminal without the editor's PATH additions."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest
from tooling import codex_binary

if TYPE_CHECKING:
    from pytest import MonkeyPatch


@pytest.mark.parametrize("explicit", [True, False])
def test_configured_binary_and_path_take_precedence(
    monkeypatch: MonkeyPatch,
    explicit: bool,
) -> None:
    with TemporaryDirectory(prefix="tradingdev-cli-test-") as temporary:
        binary = Path(temporary) / "codex"
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        binary.chmod(0o700)
        monkeypatch.setenv("PATH", temporary if not explicit else "")
        if explicit:
            monkeypatch.setenv("TRADINGDEV_CODEX_BIN", str(binary))
        else:
            monkeypatch.delenv("TRADINGDEV_CODEX_BIN", raising=False)
        monkeypatch.setattr(codex_binary, "_bundled_candidates", lambda: [])
        assert Path(codex_binary.resolve_codex_binary()).resolve() == binary.resolve()


def test_terminal_without_codex_path_finds_editor_binary(
    monkeypatch: MonkeyPatch,
) -> None:
    with TemporaryDirectory(prefix="tradingdev-cli-test-") as temporary:
        monkeypatch.delenv("TRADINGDEV_CODEX_BIN", raising=False)
        monkeypatch.setenv("PATH", "")
        monkeypatch.setattr(Path, "home", lambda: Path(temporary))
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(platform, "machine", lambda: "arm64")
        versions = ["26.9.1", "26.10.1"]
        binaries = []
        for version in versions:
            binary = (
                Path(temporary)
                / ".vscode/extensions"
                / (f"openai.chatgpt-{version}-darwin-arm64/bin/macos-aarch64/codex")
            )
            binary.parent.mkdir(parents=True)
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(0o700)
            binaries.append(binary)
        assert codex_binary.resolve_codex_binary() == str(binaries[-1].resolve())


def test_invalid_explicit_binary_is_not_silently_replaced(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADINGDEV_CODEX_BIN", "/missing/tradingdev-test-codex")
    with pytest.raises(RuntimeError, match="TRADINGDEV_CODEX_BIN is not executable"):
        codex_binary.resolve_codex_binary()


def test_missing_cli_reports_actionable_error(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("TRADINGDEV_CODEX_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(codex_binary, "_bundled_candidates", lambda: [])
    with pytest.raises(RuntimeError, match="No model request was made"):
        codex_binary.resolve_codex_binary()
