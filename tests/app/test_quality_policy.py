"""Generated strategies use the project policy independently of their workspace."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sysconfig
import venv
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import TYPE_CHECKING

import pytest
from tooling.process_guard import ProcessCleanupError, run_captured

import tradingdev
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app import quality_policy
from tradingdev.app.quality_policy import quality_config_path
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.strategies.templates import strategy_contract_payload

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def _installed_package_workspace() -> Iterator[Path]:
    root = Path(mkdtemp(prefix="tradingdev-installed-quality-")).resolve()
    retained = False
    try:
        yield root
    except ProcessCleanupError as error:
        retained = True
        raise ProcessCleanupError(f"{error}; retained {root}") from error
    finally:
        if not retained:
            shutil.rmtree(root)


@pytest.mark.parametrize(
    "failure",
    [
        None,
        AssertionError("failed"),
        subprocess.TimeoutExpired("probe", 1),
        ProcessCleanupError("cleanup unverified"),
    ],
)
def test_installed_workspace_cleanup_requires_stopped_processes(
    failure: Exception | None,
) -> None:
    root: Path | None = None
    try:
        try:
            with _installed_package_workspace() as root:
                (root / "evidence").write_text("child evidence")
                if failure is not None:
                    raise failure
        except Exception as error:
            assert failure is not None and type(error) is type(failure)
            if isinstance(error, ProcessCleanupError):
                assert f"retained {root}" in str(error)
        assert root is not None
        if isinstance(failure, ProcessCleanupError):
            assert (root / "evidence").read_text() == "child evidence"
        else:
            assert not root.exists()
    finally:
        # The failure above is injected without starting a child process.
        if root is not None and root.exists():
            shutil.rmtree(root)


def test_quality_gates_ignore_working_directory_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TemporaryDirectory(prefix="tradingdev-quality-test-") as directory:
        root = Path(directory)
        monkeypatch.chdir(root)
        monkeypatch.setenv("MYPY_CACHE_DIR", str(root / "mypy-cache"))
        monkeypatch.setenv("RUFF_CACHE_DIR", str(root / "ruff-cache"))
        (root / "ruff.toml").write_text('[lint]\nignore = ["ALL"]\n')
        (root / "mypy.ini").write_text("[mypy]\nignore_errors = True\n")
        service = StrategyService(WorkspacePaths(root / "workspace"))
        source = root / "strategy.py"
        code = strategy_contract_payload(root, "example")["content"][
            "example_strategy_code"
        ]
        source.write_text(code)

        assert service._quality_gate_diagnostics(source) == []

        source.write_text(code.replace("result = df.copy()", "result = missing_frame"))
        diagnostics = service._quality_gate_diagnostics(source)
        assert {item.code for item in diagnostics} == {"ruff_failed", "mypy_failed"}
    assert not root.exists()


@pytest.mark.parametrize("failure", [None, RuntimeError, subprocess.TimeoutExpired])
def test_installed_policy_copy_is_exact_and_always_removed(
    monkeypatch: pytest.MonkeyPatch, failure: type[Exception] | None
) -> None:
    with TemporaryDirectory(prefix="tradingdev-policy-lifetime-") as directory:
        policy = Path(directory) / "_quality" / "pyproject.toml"
        policy.parent.mkdir()
        policy.write_bytes(quality_config_path().read_bytes())
        monkeypatch.setattr(quality_policy, "quality_config_path", lambda: policy)
        copied: Path | None = None
        try:
            with quality_policy.quality_check_config() as copied:
                assert copied != policy
                assert copied.read_bytes() == policy.read_bytes()
                if failure is subprocess.TimeoutExpired:
                    raise subprocess.TimeoutExpired("mypy", 60)
                if failure is not None:
                    raise failure("quality check failed")
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            assert type(error) is failure
        assert copied is not None and not copied.parent.exists()
        assert policy.is_file()


def test_installed_package_keeps_strict_quality_checks() -> None:
    """Exercise the wheel layout with real Ruff/Mypy, outside an editable checkout."""
    with _installed_package_workspace() as root:
        environment = root / "venv"
        venv.EnvBuilder(with_pip=False).create(environment)
        site = Path(
            sysconfig.get_path(
                "purelib", vars={"base": str(environment), "platbase": str(environment)}
            )
        )
        (site / "runtime-dependencies.pth").write_text(
            sysconfig.get_path("purelib") + "\n"
        )
        package = site / "tradingdev"
        shutil.copytree(
            Path(tradingdev.__file__).parent,
            package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (package / "_quality").mkdir(exist_ok=True)
        shutil.copyfile(quality_config_path(), package / "_quality" / "pyproject.toml")
        source = root / "strategy.py"
        code = strategy_contract_payload(root, "example")["content"][
            "example_strategy_code"
        ]
        child = (
            "import json; from pathlib import Path; import tradingdev; "
            "from tradingdev.adapters.storage.filesystem import WorkspacePaths; "
            "from tradingdev.app.strategy_service import StrategyService; "
            f"assert Path(tradingdev.__file__).parent == Path({str(package)!r}); "
            "service = StrategyService(WorkspacePaths(Path('workspace'))); "
            "print(json.dumps([d.model_dump() for d in "
            "service._quality_gate_diagnostics(Path('strategy.py').resolve())]))"
        )
        for text, expected in (
            (code, set()),
            (
                code.replace("result = df.copy()", "result = missing_frame"),
                {"ruff_failed", "mypy_failed"},
            ),
        ):
            source.write_text(text)
            result = run_captured(
                [str(environment / "bin" / "python"), "-I", "-c", child],
                cwd=root,
                env={
                    **os.environ,
                    "MYPY_CACHE_DIR": str(root / "mypy-cache"),
                    "RUFF_CACHE_DIR": str(root / "ruff-cache"),
                },
                timeout=120,
            )
            assert result.returncode == 0, result.stderr
            diagnostics = json.loads(result.stdout)
            assert {item["code"] for item in diagnostics} == expected, diagnostics
    assert not root.exists()


def test_generated_strategy_keeps_strict_types_and_reports_unknown_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TemporaryDirectory(prefix="tradingdev-quality-test-") as directory:
        root = Path(directory)
        monkeypatch.setenv("MYPY_CACHE_DIR", str(root / "mypy-cache"))
        monkeypatch.setenv("RUFF_CACHE_DIR", str(root / "ruff-cache"))
        service = StrategyService(WorkspacePaths(root / "workspace"))
        source = root / "strategy.py"
        source.write_text(
            "import tradingdev_nonexistent_dependency\n\n"
            "def calculate(value):\n"
            "    return tradingdev_nonexistent_dependency.calculate(value)\n"
        )

        diagnostics = service._quality_gate_diagnostics(source)
        mypy = next(item for item in diagnostics if item.code == "mypy_failed")
        assert "[import-not-found]" in mypy.message
        assert "[no-untyped-def]" in mypy.message
    assert not root.exists()


def test_quality_gates_accept_indicator_layer_and_talib_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TemporaryDirectory(prefix="tradingdev-quality-test-") as directory:
        root = Path(directory)
        monkeypatch.setenv("MYPY_CACHE_DIR", str(root / "mypy-cache"))
        monkeypatch.setenv("RUFF_CACHE_DIR", str(root / "ruff-cache"))
        service = StrategyService(WorkspacePaths(root / "workspace"))
        source = root / "strategy.py"
        source.write_text(
            "from __future__ import annotations\n\n"
            "import pandas as pd\n"
            "import talib\n\n"
            "from tradingdev.domain import indicators\n\n"
            "\n"
            "def fair_value(close: pd.Series) -> pd.Series:\n"
            "    fast = indicators.ema(close, 5)\n"
            "    _upper, middle, _lower = talib.BBANDS(\n"
            "        close.to_numpy(dtype=float), timeperiod=20\n"
            "    )\n"
            "    slow = pd.Series(middle, index=close.index)\n"
            "    return fast - slow\n"
        )

        assert service._quality_gate_diagnostics(source) == []
    assert not root.exists()
