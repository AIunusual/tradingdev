"""CLI uses the same registered training requirements as MCP preparation."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from tradingdev.adapters.cli import main as cli_main
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.strategies.capabilities import StrategyExecutionModeError
from tradingdev.domain.strategies.catalog import BundledStrategyCatalog


def test_cli_rejects_untrained_mode_before_data_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = BundledStrategyCatalog().get("quantile_volume")
    assert entry is not None

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Rejected execution must not load data or save a result")

    with TemporaryDirectory(prefix="cli-guidance-", dir=tmp_path) as directory:
        root = Path(directory)
        workspace = WorkspacePaths(root / "workspace")
        service = BacktestService(
            strategy_gate=StrategyService(workspace),
            data_service=DataService(workspace),
        )
        monkeypatch.setattr(DataService, "load", fail)
        monkeypatch.setattr(cli_main, "BacktestService", lambda: service)
        monkeypatch.setattr(cli_main, "ArtifactService", fail)
        monkeypatch.setattr(
            "sys.argv", ["tradingdev", "--config", str(entry.config_path)]
        )
        with pytest.raises(StrategyExecutionModeError, match="walk-forward") as caught:
            cli_main.main()
        assert caught.value.code == "strategy_requires_training"
        assert caught.value.guidance.required_configuration == ["validation"]
    assert not root.exists()
