"""Rebuild real offline environments without changing their dependency lock."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest
from tooling.process_guard import run_checked

if TYPE_CHECKING:
    from collections.abc import Generator

_PROJECT = Path(__file__).resolve().parents[2]
_PYPROJECT = """\
[project]
name = "rebuild-fixture"
version = "0.1.0"
requires-python = ">=3.14,<3.15"
dependencies = []

[project.optional-dependencies]
dashboard = []

[dependency-groups]
dev = []
"""


@dataclass
class RebuildProject:
    root: Path
    environment: dict[str, str]

    def run(self, *command: str) -> None:
        run_checked(list(command), cwd=self.root, env=self.environment, timeout=30)


@contextmanager
def _project() -> Generator[RebuildProject]:
    with TemporaryDirectory(prefix="tradingdev-rebuild-test-") as temporary:
        directory = Path(temporary)
        root = directory / "project"
        (root / "scripts").mkdir(parents=True)
        shutil.copy2(
            _PROJECT / "scripts/rebuild-env.sh", root / "scripts/rebuild-env.sh"
        )
        (root / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("UV_")
            and key not in {"VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"}
        }
        environment.update(
            {
                "UV_CACHE_DIR": str(directory / "uv-cache"),
                "UV_OFFLINE": "1",
                "UV_PYTHON_DOWNLOADS": "never",
                "UV_PYTHON": sys.executable,
                "TMPDIR": str(directory),
                "TMP": str(directory),
                "TEMP": str(directory),
            }
        )
        project = RebuildProject(root, environment)
        project.run("uv", "venv", "--python", sys.executable, ".venv")
        (root / ".venv/stale-marker").write_text("old environment", encoding="utf-8")
        yield project
    assert not directory.exists()


def test_rebuild_preserves_lock_workspace_and_shared_cache() -> None:
    with _project() as project:
        root = project.root
        project.run("uv", "lock")
        lock = (root / "uv.lock").read_bytes()
        sentinels = [
            root / "workspace/__pycache__/user-data",
            Path(project.environment["UV_CACHE_DIR"]) / "shared-cache-marker",
            root.parent / "other-environment/untouched",
        ]
        for sentinel in sentinels:
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("preserve", encoding="utf-8")
        project.environment["UV_PROJECT_ENVIRONMENT"] = str(sentinels[-1].parent)

        project.run("bash", "scripts/rebuild-env.sh")

        assert (root / "uv.lock").read_bytes() == lock
        assert not (root / ".venv/stale-marker").exists()
        assert all(path.read_text(encoding="utf-8") == "preserve" for path in sentinels)
        project.run(
            str(root / ".venv/bin/python"),
            "-c",
            "import sys; from pathlib import Path; "
            "assert Path(sys.prefix).resolve() == (Path.cwd() / '.venv').resolve(); "
            f"assert sys.version_info[:2] == {tuple(sys.version_info[:2])!r}",
        )


@pytest.mark.parametrize("lock_state", ["missing", "stale"])
def test_invalid_lock_preserves_existing_environment(lock_state: str) -> None:
    with _project() as project:
        root = project.root
        lock_path = root / "uv.lock"
        if lock_state == "stale":
            project.run("uv", "lock")
            (root / "pyproject.toml").write_text(
                _PYPROJECT.replace('version = "0.1.0"', 'version = "0.2.0"'),
                encoding="utf-8",
            )
        lock = lock_path.read_bytes() if lock_path.exists() else None
        configuration = (root / ".venv/pyvenv.cfg").read_bytes()

        with pytest.raises(subprocess.CalledProcessError):
            project.run("bash", "scripts/rebuild-env.sh")

        assert (root / ".venv/stale-marker").read_text() == "old environment"
        assert (root / ".venv/pyvenv.cfg").read_bytes() == configuration
        assert (lock_path.read_bytes() if lock_path.exists() else None) == lock
