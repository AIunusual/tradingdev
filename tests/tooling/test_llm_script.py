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

if TYPE_CHECKING:
    from collections.abc import Iterator

_REPOSITORY = Path(__file__).resolve().parents[2]
_PROGRESS = "offline fixture progress before completion"
_SUITE = f"""\
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.live_llm


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
    (tmp_path / "tests/e2e/test_llm_automatic_walk_forward.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "pytestmark = pytest.mark.live_llm\n"
        "def test_automatic_walk_forward_only():\n"
        "    Path('automatic_walk_forward.ran').write_text('done', encoding='utf-8')\n",
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
        "natural_request",
        "strategy_execution_guidance",
        "novice_intake",
        "fixed_comparison",
        "research_proposal",
        "research_delivery",
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
    assert "1 passed, 12 deselected" in result.stdout
    assert (script_workspace / f"{scenario}.ran").read_text() == "done"
    assert not (script_workspace / "completed.ran").exists()
