"""Presentation routing and validation are deterministic and clean up processes."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING

import psutil
import pytest
from tooling import presentations
from tooling.process_guard import CLEANUP_UNVERIFIED_EXIT_CODE, ProcessCleanupError

if TYPE_CHECKING:
    from pytest import MonkeyPatch


def _tasks(*tasks: dict[str, object], **settings: object) -> bytes:
    return json.dumps({"version": "2.0.0", "tasks": tasks, **settings}).encode()


def test_registered_tasks_allow_only_known_changes_and_unchanged_other_tasks() -> None:
    other: dict[str, object] = {"label": "tests", "command": "existing"}
    watch = presentations.presentation_task("watch")
    build = presentations.presentation_task("build")
    assert presentations.presentation_change_allowed(
        ".vscode/tasks.json", _tasks(other, watch), _tasks(other, watch, build)
    )
    assert presentations.presentation_change_allowed(
        ".vscode/tasks.json", None, _tasks(watch, build)
    )
    assert not presentations.presentation_change_allowed(
        ".vscode/tasks.json", _tasks(other, watch), _tasks(watch, build)
    )
    assert not presentations.presentation_change_allowed(
        ".vscode/tasks.json", _tasks(watch), _tasks(watch, {"label": "unknown"})
    )


@pytest.mark.parametrize(
    "update",
    [
        {"command": "sh"},
        {"args": ["-c", "touch marker"]},
        {"type": "shell"},
        {"dependsOn": "danger"},
        {"linux": {"command": "danger"}},
        {"windows": {"command": "danger"}},
        {"osx": {"command": "danger"}},
        {"options": {"cwd": "${workspaceFolder}", "env": {"NODE_OPTIONS": "danger"}}},
        {"command": "${command:evil}"},
        {"unknown": True},
        {"runOptions": {"runOn": "folderOpen", "instanceLimit": 2}},
    ],
)
def test_task_label_does_not_authorize_arbitrary_execution(
    update: dict[str, object],
) -> None:
    valid = presentations.presentation_task("watch")
    assert not presentations.presentation_change_allowed(
        ".vscode/tasks.json", _tasks(valid), _tasks({**valid, **update})
    )


@pytest.mark.parametrize(
    "content",
    [
        b'{"version":"2.0.0","tasks":[],"tasks":[]}',
        b'{"version":"2.0.0","tasks":[],"options":{"env":{"PATH":"bad"}}}',
        b'{"version":"2.0.0","tasks":[],"command":"danger"}',
        b'{"version":"2.0.0","tasks":[],"inputs":[]}',
        b"\xff",
        b"[]",
    ],
)
def test_ambiguous_or_global_task_configuration_stays_code(content: bytes) -> None:
    assert not presentations.presentation_change_allowed(
        ".vscode/tasks.json", None, content
    )


def test_duplicate_task_labels_and_unsafe_old_task_are_rejected() -> None:
    watch = presentations.presentation_task("watch")
    assert not presentations.presentation_change_allowed(
        ".vscode/tasks.json", None, _tasks(watch, watch)
    )
    assert not presentations.presentation_change_allowed(
        ".vscode/tasks.json", _tasks({**watch, "command": "danger"}), _tasks(watch)
    )


def test_registered_paths_do_not_allow_arbitrary_decks_or_compiler_changes() -> None:
    for path in presentations.PRESENTATION_PATHS:
        assert presentations.is_presentation_path(path)
        assert presentations.presentation_change_allowed(path, b"before", b"after")
    assert presentations.presentation_change_allowed("docs/.nojekyll", None, b"")
    assert not presentations.presentation_change_allowed(
        "docs/.nojekyll", None, b"code"
    )
    for path in (
        "tooling/marp/package-lock.json",
        "tooling/presentations.py",
        "scripts/check-presentations.sh",
        "scripts/presentations.sh",
    ):
        assert presentations.is_presentation_path(path)
        assert not presentations.presentation_change_allowed(path, b"old", b"new")
    assert not presentations.presentation_change_allowed(
        "docs/other/index.html", None, b"x"
    )


def _html(extra: str = "") -> str:
    return (
        "<html><head><style>section {color:red}</style></head><body>"
        '<section id="1" data-marpit-pagination="1" data-marpit-pagination-total="2">'
        '<a href="https://external.invalid/citation">Citation</a></section>'
        '<section id="2" data-marpit-pagination="2" data-marpit-pagination-total="2">'
        f"{extra}</section></body></html>"
    )


_ENTRY = "# Editing the deck\n\nUse Tasks: Run Task → slides: watch.\n"


def test_static_html_accepts_embedded_resources_and_external_citations() -> None:
    assert (
        presentations.check_html(
            _html(
                '<img src="data:image/png;base64,AA==">'
                "<style>a {background:url(#a)}</style>"
            )
        )
        == 2
    )


@pytest.mark.parametrize(
    "extra",
    [
        '<img src="https://external.invalid/a.png">',
        '<img src="//external.invalid/a.png">',
        '<img srcset="data:image/png;base64,AA== 1x, '
        'https://external.invalid/a.png 2x">',
        '<img src="relative.png">',
        '<script src="https://external.invalid/a.js"></script>',
        '<link rel="stylesheet" href="https://external.invalid/a.css">',
        '<style>@import "https://external.invalid/a.css";</style>',
        "<style>section {background:url(https://external.invalid/a.png)}</style>",
        "<style>section {background:image-set("
        '"https://external.invalid/a.png" 1x)}</style>',
        r"<style>section {background:u\72l(\68ttps://external.invalid/a.png)}</style>",
        '<svg><image href="https://external.invalid/a.svg"/></svg>',
        '<script>new WebSocket("ws://localhost:1234")</script>',
        "<script>location.reload()</script>",
        '<img src="data:x" onload="fetch(1)">',
        '<iframe srcdoc="code"></iframe>',
        '<meta http-equiv="refresh" content="0;url=https://external.invalid">',
        '<a href="javascript:alert(1)">bad</a>',
    ],
)
def test_loaded_resources_and_watch_injections_are_rejected(extra: str) -> None:
    with pytest.raises(presentations.PresentationError):
        presentations.check_html(_html(extra))


@pytest.mark.parametrize(
    "content",
    [
        _html().replace('id="2"', 'id="3"'),
        _html().replace('pagination="2"', 'pagination="1"'),
        _html().replace('total="2"', 'total="3"'),
        "<html></html>",
    ],
)
def test_html_page_counts_and_section_identity_must_agree(content: str) -> None:
    with pytest.raises(presentations.PresentationError):
        presentations.check_html(content)


def _deck(root: Path) -> Path:
    deck = root / presentations.DECK
    deck.mkdir(parents=True)
    (deck / "slides.md").write_text("# Slides\n")
    (deck / "theme.css").write_text("section {color:red}")
    (deck / "README.md").write_text(_ENTRY)
    (deck / "index.html").write_text(_html())
    return deck


def _fake_toolchain(root: Path) -> Path:
    toolchain = root / "tooling/marp"
    cli = "node_modules/@marp-team/marp-cli"
    package: dict[str, object] = {
        "version": "4.5.1",
        "resolved": "https://registry.npmjs.org/cli.tgz",
        "integrity": "sha512-test",
    }
    (toolchain / cli).mkdir(parents=True)
    identity = {"name": "tradingdev-presentations", "version": "1.0.0"}
    manifest = {**identity, "dependencies": {"@marp-team/marp-cli": "4.5.1"}}
    (toolchain / "package.json").write_text(json.dumps(manifest))
    (toolchain / "package-lock.json").write_text(
        json.dumps(
            {**identity, "lockfileVersion": 3, "packages": {"": manifest, cli: package}}
        )
    )
    (toolchain / "node_modules/.package-lock.json").write_text(
        json.dumps({"packages": {cli: package}})
    )
    (toolchain / cli / "package.json").write_text('{"version":"4.5.1"}')
    return toolchain


@pytest.mark.parametrize(
    "update",
    [
        {"dependencies": {"@marp-team/marp-cli": "4.6.0"}},
        {"dependencies": {}},
        {"devDependencies": {"new-package": "1.0.0"}},
        {"optionalDependencies": {"new-package": "1.0.0"}},
        {"peerDependencies": {"new-package": "1.0.0"}},
        {"peerDependenciesMeta": {"new-package": {"optional": True}}},
        {"name": "different-toolchain"},
        {"version": "2.0.0"},
        {"engines": {"node": ">=30"}},
        {"os": ["linux"]},
        {"cpu": ["x64"]},
        {"libc": ["glibc"]},
        {"dependencies": ["invalid"]},
    ],
)
def test_changed_root_manifest_cannot_reuse_stale_lock_and_installation(
    tmp_path: Path, monkeypatch: MonkeyPatch, update: dict[str, object]
) -> None:
    installed = _fake_toolchain(tmp_path)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    manifest = json.loads((installed / "package.json").read_text())
    (installed / "package.json").write_text(json.dumps({**manifest, **update}))
    with pytest.raises(presentations.PresentationError, match="[Rr]oot manifest"):
        presentations._toolchain(tmp_path)


@pytest.mark.parametrize("field", ["name", "version"])
@pytest.mark.parametrize("record", ["root", "top"])
def test_lock_identity_must_match_root_manifest(
    tmp_path: Path, monkeypatch: MonkeyPatch, field: str, record: str
) -> None:
    installed = _fake_toolchain(tmp_path)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    lock = json.loads((installed / "package-lock.json").read_text())
    target = lock["packages"][""] if record == "root" else lock
    target[field] = "different"
    (installed / "package-lock.json").write_text(json.dumps(lock))
    with pytest.raises(presentations.PresentationError, match=f"manifest {field}"):
        presentations._toolchain(tmp_path)


@pytest.mark.parametrize(
    "field", ["overrides", "workspaces", "bundleDependencies", "bundledDependencies"]
)
def test_unverified_root_resolution_modes_are_rejected(
    tmp_path: Path, monkeypatch: MonkeyPatch, field: str
) -> None:
    installed = _fake_toolchain(tmp_path)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    manifest = json.loads((installed / "package.json").read_text())
    manifest[field] = {}
    (installed / "package.json").write_text(json.dumps(manifest))
    with pytest.raises(presentations.PresentationError, match="Unsupported root npm"):
        presentations._toolchain(tmp_path)


def test_matching_manifest_and_lock_can_change_dependency_category(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    installed = _fake_toolchain(tmp_path)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    manifest = json.loads((installed / "package.json").read_text())
    manifest["devDependencies"] = manifest.pop("dependencies")
    lock = json.loads((installed / "package-lock.json").read_text())
    lock["packages"][""] = manifest
    (installed / "package.json").write_text(json.dumps(manifest))
    (installed / "package-lock.json").write_text(json.dumps(lock))
    assert presentations._toolchain(tmp_path) == installed


@pytest.mark.parametrize("specification", [None, "^4.5.1", "4.6.0"])
def test_root_must_declare_exact_cli_version_even_with_matching_root_lock(
    tmp_path: Path, monkeypatch: MonkeyPatch, specification: str | None
) -> None:
    installed = _fake_toolchain(tmp_path)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    manifest = json.loads((installed / "package.json").read_text())
    manifest["dependencies"] = (
        {} if specification is None else {"@marp-team/marp-cli": specification}
    )
    lock = json.loads((installed / "package-lock.json").read_text())
    lock["packages"][""] = manifest
    (installed / "package.json").write_text(json.dumps(manifest))
    (installed / "package-lock.json").write_text(json.dumps(lock))
    with pytest.raises(presentations.PresentationError, match="root manifest must"):
        presentations._toolchain(tmp_path)


def test_toolchain_requires_lock_matching_snapshot_and_installed_versions(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    root = tmp_path / "candidate"
    installed = _fake_toolchain(root)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    assert presentations._toolchain(root) == installed
    copy = tmp_path / "installed"
    shutil.copytree(installed, copy)
    monkeypatch.setenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", str(copy))
    assert presentations._toolchain(root) == copy
    (copy / "package-lock.json").write_text("{}")
    with pytest.raises(
        presentations.PresentationError, match="differs from the checked snapshot"
    ):
        presentations._toolchain(root)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN")
    (installed / "node_modules/@marp-team/marp-cli/package.json").write_text(
        '{"version":"4.6.0"}'
    )
    with pytest.raises(presentations.PresentationError, match="version differs"):
        presentations._toolchain(root)


def test_missing_dependency_fails_without_implicit_install(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _deck(tmp_path)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)

    def no_launch(*args: object, **kwargs: object) -> None:
        pytest.fail("Missing dependencies must not trigger npm or network installation")

    monkeypatch.setattr(presentations, "run_captured", no_launch)
    with pytest.raises(
        presentations.PresentationError, match="scripts/presentations.sh setup"
    ):
        presentations.check_presentations(tmp_path)


def test_missing_node_directs_to_installing_node_before_setup(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _deck(tmp_path)
    monkeypatch.setattr(presentations, "_toolchain", lambda root: root)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(presentations.PresentationError) as error:
        presentations.build(tmp_path)

    message = str(error.value)
    assert "Node.js 18" in message
    assert "npm" in message
    assert "scripts/presentations.sh setup" in message
    assert message.index("Node.js 18") < message.index("scripts/presentations.sh")


@pytest.mark.parametrize(
    ("present", "node_step_required"),
    [(set(), True), ({"node"}, True), ({"npm"}, True), ({"node", "npm"}, False)],
)
def test_missing_toolchain_reports_the_install_step_that_can_succeed(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    present: set[str],
    node_step_required: bool,
) -> None:
    _deck(tmp_path)
    monkeypatch.delenv("TRADINGDEV_PRESENTATION_TOOLCHAIN", raising=False)
    monkeypatch.setattr(
        shutil, "which", lambda name: f"/usr/bin/{name}" if name in present else None
    )

    with pytest.raises(presentations.PresentationError) as error:
        presentations.build(tmp_path)

    message = str(error.value)
    assert "Locked presentation dependencies are unavailable" in message
    assert ("Node.js 18" in message) is node_step_required


def test_build_command_cannot_load_candidate_config_or_local_engine(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/node")
    command = presentations._command(
        tmp_path / "tools",
        tmp_path / "slides.md",
        tmp_path / "theme.css",
        tmp_path / "index.html",
    )
    assert "--no-config" in command
    assert "--no-html" in command
    assert command[command.index("--engine") + 1] == str(
        tmp_path / "tools/node_modules/@marp-team/marp-core/lib/marp.js"
    )
    assert "--watch" not in command
    monkeypatch.setenv("NODE_OPTIONS", "--require bad.js")
    monkeypatch.setenv("MARP_ENGINE", "bad.js")
    monkeypatch.setenv("NPM_CONFIG_IGNORE_SCRIPTS", "false")
    environment = presentations._environment(tmp_path)
    assert (
        not {"NODE_OPTIONS", "MARP_ENGINE", "NPM_CONFIG_IGNORE_SCRIPTS"}
        & environment.keys()
    )


@pytest.mark.parametrize("mode", ["success", "failure", "timeout", "stale"])
def test_check_cleans_temporary_files_and_child_processes(
    tmp_path: Path, monkeypatch: MonkeyPatch, mode: str
) -> None:
    deck = _deck(tmp_path)
    original = (deck / "index.html").read_bytes()
    spaces: list[Path] = []
    real_mkdtemp = mkdtemp

    def make_space(*, prefix: str) -> str:
        directory = real_mkdtemp(prefix=prefix, dir=tmp_path)
        spaces.append(Path(directory))
        return directory

    monkeypatch.setattr(presentations, "mkdtemp", make_space)
    monkeypatch.setattr(presentations, "_toolchain", lambda root: root)
    marker = tmp_path / "child.pid"
    child_code = "import time; time.sleep(60)"

    def command(toolchain: Path, source: Path, theme: Path, output: Path) -> list[str]:
        code = (
            "import os,subprocess,sys,time\nfrom pathlib import Path\n"
            "child=subprocess.Popen([sys.executable,'-c',sys.argv[4]],start_new_session=True)\n"
            "Path(sys.argv[1]).write_text(str(child.pid))\n"
            "time.sleep(60 if sys.argv[3]=='timeout' else 0.2)\n"
            "text=sys.argv[5] + ('stale' if sys.argv[3]=='stale' else '')\n"
            "Path(sys.argv[2]).write_text(text)\n"
            "raise SystemExit(9 if sys.argv[3]=='failure' else 0)\n"
        )
        return [
            sys.executable,
            "-c",
            code,
            str(marker),
            str(output),
            mode,
            child_code,
            _html(),
        ]

    monkeypatch.setattr(presentations, "_command", command)
    try:
        if mode == "success":
            presentations.check_presentations(tmp_path, timeout=5)
        else:
            expected = (
                subprocess.TimeoutExpired
                if mode == "timeout"
                else presentations.PresentationError
            )
            with pytest.raises(expected):
                presentations.check_presentations(
                    tmp_path, timeout=0.6 if mode == "timeout" else 5
                )
        assert spaces and all(not path.exists() for path in spaces)
        assert (deck / "index.html").read_bytes() == original
        process = psutil.Process(int(marker.read_text()))
        assert process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass
    finally:
        if marker.exists():
            try:
                child = psutil.Process(int(marker.read_text()))
                if child_code in child.cmdline():
                    child.kill()
                    child.wait(timeout=5)
            except psutil.NoSuchProcess, psutil.ZombieProcess:
                pass


def test_unverified_process_cleanup_preserves_workspace(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _deck(tmp_path)
    spaces: list[Path] = []
    real_mkdtemp = mkdtemp

    def make_space(*, prefix: str) -> str:
        directory = real_mkdtemp(prefix=prefix, dir=tmp_path)
        spaces.append(Path(directory))
        return directory

    monkeypatch.setattr(presentations, "mkdtemp", make_space)
    monkeypatch.setattr(presentations, "_toolchain", lambda root: root)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/node")

    def fail(*args: object, **kwargs: object) -> None:
        raise ProcessCleanupError("Could not terminate subprocesses")

    monkeypatch.setattr(presentations, "run_captured", fail)
    with pytest.raises(ProcessCleanupError, match="Cleanup unverified; retained"):
        presentations.check_presentations(tmp_path)
    assert len(spaces) == 2 and all(path.exists() for path in spaces)


@pytest.mark.parametrize("failure_kind", ["cleanup", "runtime", "build", "timeout"])
def test_cli_propagates_only_typed_cleanup_failure_and_retains_its_workspace(
    tmp_path: Path, monkeypatch: MonkeyPatch, failure_kind: str
) -> None:
    errors: dict[str, Exception] = {
        "cleanup": ProcessCleanupError("opaque failure"),
        "runtime": RuntimeError("Could not terminate subprocesses"),
        "build": presentations.PresentationError("compiler failed"),
        "timeout": subprocess.TimeoutExpired(["compiler"], 1),
    }
    spaces: list[Path] = []

    def make_space(*, prefix: str) -> str:
        return mkdtemp(prefix=prefix, dir=tmp_path)

    def fail(root: Path) -> None:
        with presentations._workspace() as temporary:
            spaces.append(temporary)
            (temporary / "evidence").write_text("diagnostics")
            raise errors[failure_kind]

    monkeypatch.setattr(presentations, "mkdtemp", make_space)
    monkeypatch.setattr(presentations, "check_presentations", fail)
    monkeypatch.setattr(sys, "argv", ["presentations", "check"])
    expected_status = CLEANUP_UNVERIFIED_EXIT_CODE if failure_kind == "cleanup" else 1
    assert presentations.main() == expected_status
    assert len(spaces) == 1
    assert spaces[0].exists() is (failure_kind == "cleanup")


def test_check_reuses_markdown_local_link_validation(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    deck = _deck(tmp_path)
    (deck / "slides.md").write_text("# Slides\n\n[broken](missing.md)\n")

    def no_build(*args: object, **kwargs: object) -> None:
        pytest.fail("Broken Markdown links must fail before rebuilding")

    monkeypatch.setattr(presentations, "build", no_build)
    from tooling.documentation import DocumentationError

    with pytest.raises(DocumentationError, match="missing.md"):
        presentations.check_presentations(tmp_path)


def test_watch_rebuilds_static_output_for_source_and_theme_changes(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    deck = _deck(tmp_path)
    built: list[tuple[str, str]] = []

    def record(root: Path) -> None:
        built.append(
            ((deck / "slides.md").read_text(), (deck / "theme.css").read_text())
        )

    monkeypatch.setattr(presentations, "build", record)
    pauses = 0

    def change(seconds: float) -> None:
        nonlocal pauses
        pauses += 1
        if pauses == 1:
            (deck / "theme.css").write_text("updated theme")
        elif pauses == 2:
            (deck / "slides.md").write_text("updated source")
        elif pauses == 4:
            raise KeyboardInterrupt

    monkeypatch.setattr(time, "sleep", change)
    presentations.watch(tmp_path)
    assert len(built) == 3
    assert built[1][1] == "updated theme"
    assert built[2][0] == "updated source"


def test_check_reuses_markdown_formatting_validation(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    deck = _deck(tmp_path)
    (deck / "slides.md").write_text("# Slides\nMissing heading spacing\n")

    def no_build(*args: object, **kwargs: object) -> None:
        pytest.fail("Markdown formatting errors must fail before rebuilding")

    monkeypatch.setattr(presentations, "build", no_build)
    with pytest.raises(subprocess.CalledProcessError):
        presentations.check_presentations(tmp_path)


@pytest.mark.parametrize("timed_out", [False, True])
def test_watch_recovers_from_build_failure_when_input_is_fixed(
    tmp_path: Path, monkeypatch: MonkeyPatch, timed_out: bool
) -> None:
    deck = _deck(tmp_path)
    calls = 0

    def compile_deck(root: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            if timed_out:
                raise subprocess.TimeoutExpired(["compiler"], 120)
            raise presentations.PresentationError("invalid source")

    monkeypatch.setattr(presentations, "build", compile_deck)
    pauses = 0

    def change(seconds: float) -> None:
        nonlocal pauses
        pauses += 1
        if pauses == 1:
            (deck / "index.html").write_text("unrelated output edit")
            (deck / "README.md").write_text("unrelated entry edit")
        elif pauses == 2:
            (deck / "slides.md").write_text("fixed source")
        elif pauses == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(time, "sleep", change)
    presentations.watch(tmp_path)
    assert calls == 2
