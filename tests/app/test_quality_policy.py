"""Generated strategies use the project policy independently of their workspace."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.strategies.templates import strategy_contract_payload

if TYPE_CHECKING:
    import pytest


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
