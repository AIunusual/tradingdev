"""Run the repository's explicit formatter or read-only quality checks."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_PATHS = ["src", "tests", "tooling", "research"]


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, "-m", *arguments], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "format"])
    arguments = parser.parse_args()
    if arguments.action == "check":
        try:
            importlib.metadata.distribution("streamlit")
            importlib.metadata.distribution("plotly")
        except importlib.metadata.PackageNotFoundError:
            parser.error(
                "Run uv sync --locked --all-extras before checking dashboard types."
            )
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    markdown = sorted(
        str(path.relative_to(ROOT))
        for path in [*ROOT.glob("*.md"), *ROOT.glob("docs/**/*.md")]
    )
    # Tables and Chinese prose are not usefully constrained by ASCII line width.
    markdown_options = ["pymarkdown", "-d", "MD013"]
    if arguments.action == "format":
        run("ruff", "check", "--fix", *PYTHON_PATHS)
        run("ruff", "format", *PYTHON_PATHS)
        run(*markdown_options, "fix", *markdown)
    else:
        run("ruff", "check", "--no-cache", *PYTHON_PATHS)
        run("ruff", "format", "--no-cache", "--check", *PYTHON_PATHS)
        run(
            "mypy",
            "--config-file=pyproject.toml",
            "--no-incremental",
            f"--cache-dir={os.devnull}",
            *PYTHON_PATHS,
        )
        run(*markdown_options, "scan", *markdown)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
