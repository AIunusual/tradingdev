"""Actual execution rejects faults that the synthetic fixtures cannot expose."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from tests.app.test_backtest_service import _raw_config, _service
from tests.mcp.test_optimization_manifest import _prepared_strategy

from tradingdev.app.backtest_service import BacktestService
from tradingdev.domain.backtest.schemas import (
    BacktestConfig,
    ParallelConfig,
)
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.bundled.kd_strategy.strategy import KDStrategy
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.strategies.signals import SignalContractError
from tradingdev.domain.validation.config import (
    AutomaticWalkForwardConfig,
)
from tradingdev.domain.validation.walk_forward import WalkForwardValidator
from tradingdev.mcp.workers.optimization import _run_single_combo

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine


@pytest.mark.parametrize("fault", ["nan", "rows", "timestamp", "mutation"])
def test_real_backtest_fails_before_engine_after_passing_both_program_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    service, data, _loader, strategy = _service(tmp_path, rows=37)
    before = data.dataset.frame.copy(deep=True)
    calls: list[int] = []
    engine = Mock()

    def generate(frame: pd.DataFrame) -> pd.DataFrame:
        calls.append(len(frame))
        result = frame.assign(signal=0)
        if len(frame) == 37:
            if fault == "nan":
                result["signal"] = np.nan
            elif fault == "rows":
                result = result.iloc[:-1].copy()
            elif fault == "timestamp":
                result["timestamp"] += pd.Timedelta(hours=1)
            else:
                frame["close"] *= 2
        return result

    monkeypatch.setattr(strategy, "generate_signals", generate)
    monkeypatch.setattr(service, "create_engine", lambda *args: engine)
    with pytest.raises(SignalContractError):
        service.run_raw_config(_raw_config())
    assert 80 in calls and 240 in calls and calls[-1] == 37
    engine.run.assert_not_called()
    pd.testing.assert_frame_equal(data.dataset.frame, before)


@pytest.mark.parametrize("fault_phase", [1, 2])
def test_walk_forward_checks_training_and_test_outputs_before_their_engine(
    fault_phase: int,
) -> None:
    class FaultyFoldStrategy(BaseStrategy):
        calls = 0

        def get_parameters(self) -> dict[str, Any]:
            return {}

        def generate_signals(self, frame: pd.DataFrame) -> pd.DataFrame:
            self.calls += 1
            return frame.assign(signal=np.nan if self.calls == fault_phase else 0)

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
            "close": np.arange(20) + 100.0,
        }
    )
    engine = Mock()
    engine.run.return_value.metrics = {}
    validator = WalkForwardValidator(
        AutomaticWalkForwardConfig(
            mode="automatic",
        ),
        cast("BaseBacktestEngine", engine),
    )
    with pytest.raises(SignalContractError, match="invalid_signal_values"):
        validator.run_fold(0, FaultyFoldStrategy(), frame.iloc[:12], frame.iloc[12:])
    assert engine.run.call_count == fault_phase - 1


def test_optimization_checks_bundled_strategy_output_before_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy_config = {"id": "kd_crossover", "parameters": {}}
    execution = StrategyLoader().resolve_execution(strategy_config)
    engine = Mock()
    monkeypatch.setattr(BacktestService, "prepare_strategy", _prepared_strategy)
    monkeypatch.setattr(BacktestService, "create_engine", lambda *args: engine)
    monkeypatch.setattr(
        KDStrategy, "generate_signals", lambda self, frame: frame.assign(signal=np.nan)
    )
    with pytest.raises(SignalContractError, match="invalid_signal_values"):
        _run_single_combo(
            strategy_config,
            execution,
            BacktestConfig(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date="2024-01-01",
                end_date="2024-01-08",
                init_cash=10000,
            ),
            pd.DataFrame({"close": [100.0, 101.0]}),
            {},
            "total_return",
            ParallelConfig(),
            42,
        )
    engine.run.assert_not_called()
