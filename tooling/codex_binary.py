"""Find the CLI without relying on an editor-injected shell PATH."""

from __future__ import annotations

import os
import platform
import re
import shutil
import sys
from pathlib import Path


def _bundled_candidates() -> list[Path]:
    """Locate native macOS binaries in an installed OpenAI VS Code extension."""
    if sys.platform != "darwin":
        return []
    architecture = {"arm64": "aarch64", "x86_64": "x86_64"}.get(platform.machine())
    if architecture is None:
        return []
    extensions = Path.home() / ".vscode" / "extensions"
    candidates = extensions.glob(f"openai.chatgpt-*/bin/macos-{architecture}/codex")
    return sorted(
        candidates,
        key=lambda path: tuple(
            int(part) for part in re.findall(r"\d+", path.parents[2].name)
        ),
        reverse=True,
    )


def resolve_codex_binary() -> str:
    """Prefer explicit configuration, then PATH, then an installed editor CLI."""
    configured = os.environ.get("TRADINGDEV_CODEX_BIN")
    if configured:
        resolved = shutil.which(str(Path(configured).expanduser()))
        if resolved:
            return str(Path(resolved).resolve())
        msg = f"TRADINGDEV_CODEX_BIN is not executable: {configured}"
        raise RuntimeError(msg)

    if resolved := shutil.which("codex"):
        return str(Path(resolved).absolute())

    for candidate in _bundled_candidates():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())

    msg = (
        "Codex CLI was not found on PATH or in the supported VS Code installation. "
        "Set TRADINGDEV_CODEX_BIN=/absolute/path/to/codex, then check that "
        "binary with 'login status'. No model request was made."
    )
    raise RuntimeError(msg)
