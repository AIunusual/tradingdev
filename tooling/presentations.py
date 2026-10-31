"""Build and verify registered, self-contained Marp presentations."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from html.parser import HTMLParser
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING

from tooling.process_guard import (
    CLEANUP_UNVERIFIED_EXIT_CODE,
    ProcessCleanupError,
    run_captured,
)

if TYPE_CHECKING:
    from collections.abc import Generator

DECK = "docs/talks/tradingdev-introduction"
PRESENTATION_PATHS = frozenset(
    f"{DECK}/{name}" for name in ("README.md", "slides.md", "theme.css", "index.html")
)
_SPECIAL_PATHS = frozenset({"docs/.nojekyll", ".vscode/tasks.json"})
_TOOL_PATHS = frozenset(
    {
        "tooling/presentations.py",
        "scripts/check-presentations.sh",
        "scripts/presentations.sh",
    }
)
_TITLE = "TradingDev｜讓 AI 助手協助研究交易策略"
_DESCRIPTION = "產品介紹、使用場景、分類比較與後續開發方向"
_SETUP_HELP = "Run scripts/presentations.sh setup to install the locked toolchain."
_NODE_HELP = (
    "Install Node.js 18+ and npm, confirm `node` is on PATH, then run "
    "scripts/presentations.sh setup to install the locked toolchain."
)


class PresentationError(RuntimeError):
    """A presentation or its locked compiler cannot be verified."""


class _CleanupError(ProcessCleanupError):
    """Process teardown could not be verified; keep affected workspaces."""


def presentation_task(action: str) -> dict[str, object]:
    """Return the complete allowed VS Code task; unknown keys cannot be inherited."""
    if action not in {"build", "watch"}:
        raise ValueError("Only build and watch are registered presentation tasks.")
    task: dict[str, object] = {
        "label": f"slides: {action}",
        "detail": (
            "儲存 TradingDev 簡報或主題時，自動更新 index.html。"
            if action == "watch"
            else "輸出供保存或發布的靜態 index.html。"
        ),
        "type": "process",
        "command": "uv",
        "args": ["run", "--no-sync", "python", "-m", "tooling.presentations", action],
        "options": {"cwd": "${workspaceFolder}"},
        "problemMatcher": [],
        "presentation": {"reveal": "silent", "focus": False, "panel": "dedicated"},
    }
    if action == "watch":
        task.update(
            isBackground=True, runOptions={"runOn": "folderOpen", "instanceLimit": 1}
        )
    else:
        task["group"] = "build"
    return task


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_object(content: bytes) -> dict[str, object]:
    value: object = json.loads(
        content.decode("utf-8"), object_pairs_hook=_unique_object
    )
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def _tasks(content: bytes | None) -> tuple[dict[str, object], list[object]]:
    if content is None:
        return {"version": "2.0.0"}, []
    value = _json_object(content)
    tasks = value.pop("tasks", None)
    if value.get("version") != "2.0.0" or not isinstance(tasks, list):
        raise ValueError("Invalid VS Code tasks document.")
    return value, tasks


def _presentation_tasks_change_allowed(
    before: bytes | None, after: bytes | None
) -> bool:
    old_settings, old_tasks = _tasks(before)
    new_settings, new_tasks = _tasks(after)
    if old_settings != new_settings:
        return False
    # Global commands/options can alter otherwise identical process tasks.
    if set(new_settings) != {"version"}:
        return False
    allowed = [presentation_task(action) for action in ("build", "watch")]
    labels = {task["label"] for task in allowed}

    def unrelated(tasks: list[object]) -> list[object] | None:
        remaining: list[object] = []
        seen: set[str] = set()
        for task in tasks:
            if not isinstance(task, dict) or not isinstance(task.get("label"), str):
                return None
            label = task["label"]
            if label in seen:
                return None
            seen.add(label)
            if label in labels:
                if json.dumps(task, sort_keys=True) not in {
                    json.dumps(known, sort_keys=True) for known in allowed
                }:
                    return None
            else:
                remaining.append(task)
        return remaining

    old_other, new_other = unrelated(old_tasks), unrelated(new_tasks)
    return old_other is not None and json.dumps(
        old_other, sort_keys=True
    ) == json.dumps(new_other, sort_keys=True)


def is_presentation_path(path: str) -> bool:
    """Identify deck or compiler changes requiring the presentation checks."""
    return path in PRESENTATION_PATHS | _SPECIAL_PATHS | _TOOL_PATHS or path.startswith(
        "tooling/marp/"
    )


def presentation_change_allowed(
    path: str, before: bytes | None, after: bytes | None
) -> bool:
    """Classify Git blob contents, without executing candidate configuration.

    The Git caller must also require ordinary file modes on both sides.
    Compiler/configuration changes remain code changes even when deck-related.
    """
    if path in PRESENTATION_PATHS:
        return True
    if path == "docs/.nojekyll":
        return before in {None, b""} and after in {None, b""}
    if path != ".vscode/tasks.json":
        return False
    try:
        return _presentation_tasks_change_allowed(before, after)
    except ValueError, UnicodeError:
        return False


def _css_unescape(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        escaped = match[1]
        if re.fullmatch(r"[0-9a-fA-F]{1,6}\s?", escaped):
            number = int(escaped.strip(), 16)
            return chr(number) if 0 < number <= 0x10FFFF else "\ufffd"
        return escaped

    return re.sub(r"\\([0-9a-fA-F]{1,6}\s?|[^\n\r\f])", replace, value)


def _check_css(value: str) -> None:
    css = _css_unescape(re.sub(r"/\*.*?\*/", "", value, flags=re.DOTALL))
    if re.search(r"@import\b", css, flags=re.IGNORECASE):
        raise PresentationError("External CSS imports are not allowed.")
    if re.search(r"\b(?:image-set|image)\s*\(", css, flags=re.IGNORECASE):
        raise PresentationError(
            "CSS image functions with implicit URLs are not allowed."
        )
    for match in re.finditer(r"\burl\s*\(\s*([^)]*)\)", css, flags=re.IGNORECASE):
        target = match[1].strip().strip("\"'").strip()
        if target and not target.casefold().startswith(("data:", "#")):
            raise PresentationError(f"External CSS resource is not allowed: {target}")


class _PresentationHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[dict[str, str | None]] = []
        self.active: str | None = None
        self.content: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in {"iframe", "object", "embed"}:
            raise PresentationError(f"Embedded active content is not allowed: {tag}")
        if tag == "section":
            self.sections.append(attributes)
        if tag in {"style", "script"}:
            self.active, self.content = tag, []
        if tag == "base" or (
            tag == "meta" and (attributes.get("http-equiv") or "").lower() == "refresh"
        ):
            raise PresentationError(f"Unexpected navigation element: {tag}")
        for name, value in attrs:
            if name == "style" and value:
                _check_css(value)
            if name in {"srcdoc", "srcset", "ping"} or name.startswith("on"):
                raise PresentationError(f"Inline executable HTML attribute: {name}")
            resource = name in {"src", "srcset", "poster", "data", "background"} or (
                name in {"href", "xlink:href"} and tag not in {"a"}
            )
            if (
                resource
                and value
                and not value.strip().lower().startswith(("data:", "#"))
            ):
                raise PresentationError(
                    f"External HTML resource is not allowed: {value}"
                )
            if (
                tag == "a"
                and name == "href"
                and value
                and re.match(r"\s*(?:javascript|data):", value, flags=re.IGNORECASE)
            ):
                raise PresentationError("Executable citation URLs are not allowed.")

    def handle_data(self, data: str) -> None:
        if self.active:
            self.content.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != self.active:
            return
        content = "".join(self.content)
        if tag == "style":
            _check_css(content)
        elif re.search(
            r"WebSocket|EventSource|location\s*\.\s*reload|marp-cli-reload|wss?://",
            content,
            flags=re.IGNORECASE,
        ):
            raise PresentationError(
                "Watch/reload code is not allowed in committed HTML."
            )
        self.active, self.content = None, []


def check_html(content: str) -> int:
    """Inspect generated output as data; never open it in a browser."""
    parser = _PresentationHTML()
    parser.feed(content)
    parser.close()
    count = len(parser.sections)
    if not count:
        raise PresentationError("The presentation contains no sections.")
    for number, section in enumerate(parser.sections, 1):
        if (
            section.get("id") != str(number)
            or section.get("data-marpit-pagination") != str(number)
            or section.get("data-marpit-pagination-total") != str(count)
        ):
            raise PresentationError(
                "Section IDs, page numbers and totals are inconsistent."
            )
    return count


@contextmanager
def _workspace() -> Generator[Path]:
    root = Path(mkdtemp(prefix="tradingdev-presentations-"))
    retain = False
    try:
        yield root
    except ProcessCleanupError as error:
        # Propagate the typed cleanup failure across every containing workspace.
        # Other failures have completed teardown and may remove their files.
        retain = True
        raise _CleanupError(f"Cleanup unverified; retained {root}: {error}") from error
    finally:
        if not retain:
            shutil.rmtree(root)


def _environment(temporary: Path) -> dict[str, str]:
    user_config = temporary / "npm-user.rc"
    global_config = temporary / "npm-global.rc"
    user_config.touch()
    global_config.touch()
    environment = {
        key: value
        for key, value in os.environ.items()
        if key != "DEBUG"
        and not key.upper().startswith(("NODE_", "NPM_CONFIG_", "MARP_"))
    }
    environment.update(
        TMPDIR=str(temporary),
        TMP=str(temporary),
        TEMP=str(temporary),
        npm_config_cache=str(temporary / "npm-cache"),
        npm_config_userconfig=str(user_config),
        npm_config_globalconfig=str(global_config),
        PUPPETEER_SKIP_DOWNLOAD="true",
    )
    return environment


def _execute(command: list[str], *, cwd: Path, temporary: Path, timeout: float) -> None:
    result = run_captured(
        command, cwd=cwd, env=_environment(temporary), timeout=timeout
    )
    if result.returncode:
        raise PresentationError(
            f"Presentation compiler failed ({result.returncode}):\n"
            f"{result.stdout}{result.stderr}"
        )


def _check_root_manifest(
    manifest: dict[str, object], lock: dict[str, object], root_package: object
) -> None:
    """Check npm's root identity and dependency declarations before using installs."""
    if not isinstance(root_package, dict):
        raise PresentationError("The npm lockfile has no root package record.")
    for field in ("name", "version"):
        value = manifest.get(field)
        if (
            not isinstance(value, str)
            or not value
            or lock.get(field) != value
            or root_package.get(field) != value
        ):
            raise PresentationError(f"Root manifest {field} differs from the npm lock.")
    has_cli = False
    for field in (
        "dependencies",
        "devDependencies",
        "optionalDependencies",
        "peerDependencies",
    ):
        declared = manifest.get(field, {})
        if not isinstance(declared, dict) or any(
            not isinstance(name, str) or not isinstance(specification, str)
            for name, specification in declared.items()
        ):
            raise PresentationError(f"Invalid root manifest {field}.")
        if declared != root_package.get(field, {}):
            raise PresentationError(f"Root manifest {field} differs from the npm lock.")
        if "@marp-team/marp-cli" in declared:
            has_cli = True
            if declared["@marp-team/marp-cli"] != "4.5.1":
                raise PresentationError("The root manifest must pin Marp CLI to 4.5.1.")
    if not has_cli:
        raise PresentationError("The root manifest must declare Marp CLI 4.5.1.")
    for field in ("peerDependenciesMeta", "engines", "os", "cpu", "libc"):
        if manifest.get(field) != root_package.get(field):
            raise PresentationError(f"Root manifest {field} differs from the npm lock.")
    # These resolution modes cannot be proved from the root package record alone.
    # Supporting them requires an explicit extension of this fixed compiler contract.
    if {
        "overrides",
        "workspaces",
        "bundleDependencies",
        "bundledDependencies",
    } & manifest.keys():
        raise PresentationError("Unsupported root npm dependency resolution settings.")


def _install_help() -> str:
    """Return the install step that can succeed for the missing dependency."""
    if shutil.which("node") is None or shutil.which("npm") is None:
        return _NODE_HELP
    return _SETUP_HELP


def _toolchain(root: Path) -> Path:
    expected = root / "tooling/marp"
    installed = Path(
        os.environ.get("TRADINGDEV_PRESENTATION_TOOLCHAIN", str(expected))
    ).resolve()
    try:
        for name in ("package.json", "package-lock.json"):
            if (expected / name).read_bytes() != (installed / name).read_bytes():
                raise PresentationError(
                    f"Toolchain {name} differs from the checked snapshot."
                )
        lock = _json_object((expected / "package-lock.json").read_bytes())
        packages = lock.get("packages")
        if lock.get("lockfileVersion") != 3 or not isinstance(packages, dict):
            raise PresentationError("A complete npm lockfile version 3 is required.")
        _check_root_manifest(
            _json_object((expected / "package.json").read_bytes()),
            lock,
            packages.get(""),
        )
        cli = packages.get("node_modules/@marp-team/marp-cli")
        if not isinstance(cli, dict) or cli.get("version") != "4.5.1":
            raise PresentationError("The locked Marp CLI version must be 4.5.1.")
        actual = _json_object(
            (installed / "node_modules/.package-lock.json").read_bytes()
        ).get("packages")
        if not isinstance(actual, dict):
            raise PresentationError(
                "The installed toolchain has no dependency manifest."
            )
        for path, package in packages.items():
            if not path:
                continue
            if (
                not isinstance(package, dict)
                or not isinstance(path, str)
                or not path.startswith("node_modules/")
                or ".." in Path(path).parts
            ):
                raise PresentationError("Invalid locked dependency path.")
            present = actual.get(path)
            if not isinstance(present, dict) or any(
                present.get(key) != package.get(key)
                for key in ("version", "resolved", "integrity")
            ):
                if package.get("optional") is True and present is None:
                    continue
                raise PresentationError(
                    f"Installed dependency differs from lock: {path}"
                )
            manifest = installed / path / "package.json"
            if not manifest.is_file() and package.get("optional") is True:
                continue
            if _json_object(manifest.read_bytes()).get("version") != package.get(
                "version"
            ):
                raise PresentationError(f"Installed dependency version differs: {path}")
        if set(actual) - set(packages):
            raise PresentationError(
                "Installed toolchain includes unlocked dependencies."
            )
    except (OSError, ValueError) as error:
        raise PresentationError(
            f"Locked presentation dependencies are unavailable. {_install_help()}"
        ) from error
    return installed


def _command(toolchain: Path, source: Path, theme: Path, output: Path) -> list[str]:
    node = shutil.which("node")
    if node is None:
        raise PresentationError(f"Node.js is unavailable. {_install_help()}")
    return [
        node,
        str(toolchain / "node_modules/@marp-team/marp-cli/marp-cli.js"),
        "--no-config",
        "--no-html",
        "--engine",
        str(toolchain / "node_modules/@marp-team/marp-core/lib/marp.js"),
        str(source),
        "--theme-set",
        str(theme),
        "--theme",
        "tradingdev",
        "--title",
        _TITLE,
        "--description",
        _DESCRIPTION,
        "--output",
        str(output),
    ]


def build(root: Path, *, output: Path | None = None, timeout: float = 120) -> None:
    toolchain = _toolchain(root)
    with _workspace() as temporary:
        generated = temporary / "index.html"
        _execute(
            _command(
                toolchain,
                root / DECK / "slides.md",
                root / DECK / "theme.css",
                generated,
            ),
            cwd=temporary,
            temporary=temporary,
            timeout=timeout,
        )
        (output or root / DECK / "index.html").write_bytes(generated.read_bytes())


def check_presentations(root: Path, *, timeout: float = 120) -> None:
    """Check documents, rebuild with the locked compiler and verify the output."""
    from tooling.documentation import check_documents

    root = root.resolve()
    check_documents(root)
    with _workspace() as temporary:
        output = temporary / "index.html"
        build(root, output=output, timeout=timeout)
        committed = root / DECK / "index.html"
        if output.read_bytes() != committed.read_bytes():
            raise PresentationError(
                "Committed index.html is out of date; "
                "run scripts/presentations.sh build."
            )
        count = check_html(output.read_text(encoding="utf-8"))
    print(
        f"Presentation checks passed ({count} slides; "
        "locked build, links, pages and resources)."
    )


def setup(root: Path, *, timeout: float = 300) -> None:
    npm = shutil.which("npm")
    if npm is None:
        raise PresentationError("Node.js and npm are required to install the compiler.")
    with _workspace() as temporary:
        _execute(
            [npm, "ci", "--ignore-scripts", "--no-audit", "--no-fund"],
            cwd=root / "tooling/marp",
            temporary=temporary,
            timeout=timeout,
        )
    _toolchain(root)
    print("Locked presentation compiler installed.")


def watch(root: Path) -> None:
    # Rebuild static HTML on changes; Marp's native --watch injects a reload
    # WebSocket into the artifact, which must never enter the committed output.
    inputs = [root / DECK / name for name in ("slides.md", "theme.css")]
    previous: list[bytes] | None = None
    try:
        while True:
            current = [path.read_bytes() for path in inputs]
            if current != previous:
                try:
                    build(root)
                except ProcessCleanupError:
                    raise
                except (
                    PresentationError,
                    OSError,
                    subprocess.SubprocessError,
                ) as error:
                    print(f"Presentation rebuild failed: {error}", file=sys.stderr)
                else:
                    print(
                        "Presentation rebuilt; watching slides.md and theme.css.",
                        flush=True,
                    )
                previous = current
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Presentation watch stopped.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "build", "watch", "check"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        {"setup": setup, "build": build, "watch": watch, "check": check_presentations}[
            args.action
        ](root)
    except ProcessCleanupError as error:
        print(f"Presentation cleanup is unverified: {error}", file=sys.stderr)
        return CLEANUP_UNVERIFIED_EXIT_CODE
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"Presentation checks failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
