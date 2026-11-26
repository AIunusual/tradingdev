"""Exercise the live-test launcher with a blocked, entirely offline pytest case."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from typing import TYPE_CHECKING

import pytest
from tooling.process_guard import run_captured

if TYPE_CHECKING:
    from collections.abc import Iterator

_REPOSITORY = Path(__file__).resolve().parents[2]
_PROGRESS = "offline fixture progress before completion"
_SUITE = f"""\
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.live_llm


@pytest.fixture(autouse=True)
def record_setup():
    Path("setup.ran").touch()


def test_01_workflow(pytestconfig):
    assert pytestconfig.getoption("llm_provider") == "local"
    assert pytestconfig.getoption("llm_model") == "offline-fixture"
    print({_PROGRESS!r}, flush=True)
    assert sys.__stdin__.readline() == "release\\n"
    Path("completed.ran").write_text("done", encoding="utf-8")
    pytest.fail("intentional offline workflow failure")


def test_02_after_failure():
    Path("second.ran").write_text("unexpected", encoding="utf-8")


def test_03_excluded():
    pytest.fail("the -k option was not forwarded")
"""


def _script_environment(tmp_path: Path) -> dict[str, str]:
    """Install model-test paths as isolated, entirely offline fixtures."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests/e2e").mkdir(parents=True)
    shutil.copy2(
        _REPOSITORY / "scripts/check-llm.sh", tmp_path / "scripts/check-llm.sh"
    )
    shutil.copy2(_REPOSITORY / "tests/llm_plugin.py", tmp_path / "conftest.py")
    (tmp_path / "tooling").mkdir()
    (tmp_path / "tooling/__init__.py").touch()
    for name in ("environment.py", "llm_check.py", "process_guard.py"):
        shutil.copy2(_REPOSITORY / "tooling" / name, tmp_path / "tooling" / name)
    with (tmp_path / "conftest.py").open("a") as conftest:
        conftest.write("""
from pathlib import Path
from tooling import environment

def probe(name):
    with Path("environment.ran").open("a") as log:
        log.write(name + "\\n")
    if os.environ.get("TRADINGDEV_TEST_ENV_FAILURE") == name:
        raise environment.EnvironmentCheckError("fixture prerequisite unavailable")
    if os.environ.get("TRADINGDEV_TEST_ENV_FAILURE") == "cleanup":
        from tooling.process_guard import ProcessCleanupError
        raise ProcessCleanupError("fixture probe cleanup unverified")
    return "fixture"

environment.require_strategy_runtime = lambda: probe("runtime")
environment.require_codex = lambda: probe("codex")
environment.require_local_service = lambda _url: probe("local")
""")
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    live_llm: offline launcher fixture\n",
        encoding="utf-8",
    )
    (tmp_path / "tests/e2e/test_llm_workflows.py").write_text(_SUITE, encoding="utf-8")
    (tmp_path / "tests/e2e/test_llm_cleanup.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "pytestmark = pytest.mark.live_llm\n"
        "def test_cleanup_only():\n"
        "    Path('cleanup.ran').write_text('done', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "tests/e2e/test_llm_execution_plans.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "pytestmark = pytest.mark.live_llm\n"
        "def test_execution_plan_only():\n"
        "    Path('execution_plan.ran').write_text('done', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "tests/e2e/test_llm_ml.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "pytestmark = pytest.mark.live_llm\n"
        "def test_ml_pipeline_only():\n"
        "    Path('ml_pipeline.ran').write_text('done', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "tests/e2e/test_llm_evaluation_series.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "pytestmark = pytest.mark.live_llm\n"
        "def test_evaluation_series_only():\n"
        "    Path('evaluation_series.ran').write_text('done', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "tests/e2e/test_llm_discovery.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "pytestmark = pytest.mark.live_llm\n"
        "def test_natural_request_only():\n"
        "    Path('natural_request.ran').write_text('done', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "tests/e2e/test_llm_strategy_execution_guidance.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "pytestmark = pytest.mark.live_llm\n"
        "def test_strategy_execution_guidance_only():\n"
        "    Path('strategy_execution_guidance.ran').write_text(\n"
        "        'done', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "tests/e2e/test_llm_novice_intake.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "pytestmark = pytest.mark.live_llm\n"
        "def test_novice_intake_only():\n"
        "    Path('novice_intake.ran').write_text('done', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "tests/e2e/test_llm_comparison.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "pytestmark = pytest.mark.live_llm\n"
        "def test_fixed_comparison_only():\n"
        "    Path('fixed_comparison.ran').write_text('done', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "tests/e2e/test_llm_research_proposal.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "pytestmark = pytest.mark.live_llm\n"
        "def test_research_proposal_only():\n"
        "    Path('research_proposal.ran').write_text('done', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "tests/e2e/test_llm_research_delivery.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "pytestmark = pytest.mark.live_llm\n"
        "def test_research_delivery_only():\n"
        "    Path('research_delivery.ran').write_text('done', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "tests/e2e/test_llm_http_delivery.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "pytestmark = pytest.mark.live_llm\n"
        "def test_http_delivery_only():\n"
        "    Path('http_delivery.ran').write_text('done', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "tests/e2e/test_llm_automatic_walk_forward.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "pytestmark = pytest.mark.live_llm\n"
        "def test_automatic_walk_forward_only():\n"
        "    Path('automatic_walk_forward.ran').write_text('done', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "tests/e2e/test_llm_plateau.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "pytestmark = pytest.mark.live_llm\n"
        "def test_parameter_plateau_only():\n"
        "    Path('parameter_plateau.ran').write_text('done', encoding='utf-8')\n",
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "TRADINGDEV_CHECK_PYTHON": sys.executable,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
        "PYTEST_PLUGINS": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(tmp_path),
        "TMP": str(tmp_path),
        "TEMP": str(tmp_path),
    }
    # The launcher itself must disable buffering; an inherited setting must not
    # conceal a regression when stdout is piped by an IDE or another process.
    environment.pop("PYTHONUNBUFFERED", None)
    return environment


@pytest.fixture
def script_workspace(tmp_path: Path) -> Iterator[Path]:
    with TemporaryDirectory(dir=tmp_path) as directory:
        workspace = Path(directory)
        yield workspace
    assert not workspace.exists()


@pytest.mark.parametrize("source", ["arguments", "environment"])
@pytest.mark.parametrize("mode", ["parallel", "looponfail"])
def test_live_launcher_rejects_parallelism_before_setup(
    script_workspace: Path, source: str, mode: str
) -> None:
    environment = _script_environment(script_workspace)
    environment["TRADINGDEV_TEST_ENV_FAILURE"] = "runtime"
    options = (
        ["-p", "xdist.plugin", "-n", "2"]
        if mode == "parallel"
        else ["-p", "xdist.looponfail", "--looponfail"]
    )
    if source == "environment":
        environment["PYTEST_ADDOPTS"] = " ".join(options)
        options = []
    result = run_captured(
        [
            "bash",
            "scripts/check-llm.sh",
            "local",
            "--llm-model",
            "offline-fixture",
            "-k",
            "cleanup_only",
            *options,
        ],
        cwd=script_workspace,
        env=environment,
        timeout=20,
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, (
        result.stdout + result.stderr
    )
    assert "requires serial execution" in result.stderr
    assert not (script_workspace / "cleanup.ran").exists()
    assert not (script_workspace / "setup.ran").exists()
    assert not (script_workspace / "environment.ran").exists()


def test_script_streams_progress_through_pipe_and_stops_at_first_failure(
    script_workspace: Path,
) -> None:
    environment = _script_environment(script_workspace)
    output: list[str] = []
    progress = Event()
    with subprocess.Popen(
        [
            "bash",
            "scripts/check-llm.sh",
            "local",
            "--llm-model",
            "offline-fixture",
            "-k",
            "not excluded",
            "-p",
            "no:cacheprovider",
        ],
        cwd=script_workspace,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as process:
        assert process.stdin is not None
        assert process.stdout is not None
        stdout = process.stdout

        def read_output() -> None:
            for line in stdout:
                output.append(line)
                if _PROGRESS in line:
                    progress.set()

        reader = Thread(target=read_output, daemon=True)
        reader.start()
        try:
            assert progress.wait(timeout=20), (
                "No progress reached the stdout pipe before test completion:\n"
                + "".join(output)
            )
            assert process.poll() is None
            assert not (script_workspace / "completed.ran").exists()
            process.stdin.write("release\n")
            process.stdin.flush()
            returncode = process.wait(timeout=20)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            reader.join(timeout=5)
            assert not reader.is_alive(), "Output reader did not stop with pytest"

    transcript = "".join(output)
    assert returncode == pytest.ExitCode.TESTS_FAILED, transcript
    assert "intentional offline workflow failure" in transcript
    assert "stopping after 1 failures" in transcript
    assert "1 failed, 1 deselected" in transcript
    assert (script_workspace / "completed.ran").exists()
    assert not (script_workspace / "second.ran").exists()


@pytest.mark.parametrize(
    "scenario",
    [
        "cleanup",
        "execution_plan",
        "ml_pipeline",
        "evaluation_series",
        "natural_request",
        "strategy_execution_guidance",
        "novice_intake",
        "fixed_comparison",
        "parameter_plateau",
        "research_proposal",
        "research_delivery",
        "http_delivery",
        "automatic_walk_forward",
    ],
)
def test_script_selects_scenario_without_running_authoring_suite(
    script_workspace: Path,
    scenario: str,
) -> None:
    environment = _script_environment(script_workspace)
    result = subprocess.run(
        [
            "bash",
            "scripts/check-llm.sh",
            "local",
            "--llm-model",
            "offline-fixture",
            "-k",
            f"{scenario}_only",
            "-p",
            "no:cacheprovider",
        ],
        cwd=script_workspace,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == pytest.ExitCode.OK, result.stdout
    assert (script_workspace / f"{scenario}.ran").read_text() == "done"
    assert {
        path.name
        for path in script_workspace.glob("*.ran")
        if path.name != "environment.ran"
    } == {f"{scenario}.ran"}


@pytest.mark.parametrize(
    "provider, failing_probe, options",
    [
        ("local", "runtime", []),
        ("local", "local", []),
        ("codex", "codex", []),
        ("local", "runtime", ["--setup-only"]),
        ("local", "runtime", ["-p", "xdist.plugin", "-n", "0"]),
    ],
)
def test_missing_prerequisite_stops_live_suite_before_test_setup(
    script_workspace: Path,
    provider: str,
    failing_probe: str,
    options: list[str],
) -> None:
    environment = _script_environment(script_workspace)
    environment["TRADINGDEV_TEST_ENV_FAILURE"] = failing_probe
    result = subprocess.run(
        [
            "bash",
            "scripts/check-llm.sh",
            provider,
            "--llm-model",
            "offline-fixture",
            "-k",
            "test_01_workflow",
            *options,
            "-p",
            "no:cacheprovider",
        ],
        cwd=script_workspace,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "Live-test environment check failed" in result.stderr
    assert not (script_workspace / "setup.ran").exists()
    assert (script_workspace / "environment.ran").read_text().splitlines() == (
        ["runtime"] if failing_probe == "runtime" else ["runtime", provider]
    )


@pytest.mark.parametrize(
    "selection, expected",
    [
        (["--collect-only"], 0),
        (["--help"], 0),
        (["--fixtures"], 0),
        (["--fixtures-per-test"], 0),
        (["--setup-plan"], 0),
        (["-k", "no_matching_scenario"], 5),
    ],
)
@pytest.mark.parametrize("provider", ["codex", "local"])
def test_information_or_empty_selection_does_not_probe_services(
    script_workspace: Path,
    provider: str,
    selection: list[str],
    expected: int,
) -> None:
    environment = _script_environment(script_workspace)
    environment["TRADINGDEV_TEST_ENV_FAILURE"] = "runtime"
    result = subprocess.run(
        [
            "bash",
            "scripts/check-llm.sh",
            provider,
            "--llm-model",
            "offline-fixture",
            *selection,
            "-p",
            "no:cacheprovider",
        ],
        cwd=script_workspace,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    assert not (script_workspace / "environment.ran").exists()
    assert not (script_workspace / "setup.ran").exists()
    assert not (script_workspace / "cleanup.ran").exists()


def test_live_launcher_preserves_unverified_cleanup_exit_status(
    script_workspace: Path,
) -> None:
    environment = _script_environment(script_workspace)
    environment["TRADINGDEV_TEST_ENV_FAILURE"] = "cleanup"
    result = subprocess.run(
        [
            "bash",
            "scripts/check-llm.sh",
            "codex",
            "-k",
            "cleanup_only",
            "-p",
            "no:cacheprovider",
        ],
        cwd=script_workspace,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 86
    assert '"code": "process_cleanup_unverified"' in result.stderr
    assert not (script_workspace / "cleanup.ran").exists()
