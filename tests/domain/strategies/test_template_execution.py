"""The published SMA example executes the long-only position rule it describes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
import yaml

from tradingdev.domain.backtest.engines import create_backtest_engine
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.strategies.templates import strategy_contract_payload

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.domain.strategies.base import BaseStrategy


def _example(
    tmp_path: Path, *, fast_period: int = 2, slow_period: int = 3
) -> tuple[BaseStrategy, dict[str, Any]]:
    example = strategy_contract_payload(tmp_path, "example")["content"]
    source = tmp_path / "generated_strategies" / "example.py"
    source.parent.mkdir()
    source.write_text(example["example_strategy_code"], encoding="utf-8")
    config = yaml.safe_load(example["example_yaml_config"])
    config["strategy"]["source_path"] = str(source)
    config["strategy"]["parameters"] = {
        "fast_period": fast_period,
        "slow_period": slow_period,
    }
    strategy = StrategyLoader(workspace_root=tmp_path).create_from_config(
        config, engine=None
    )
    return strategy, config


def _market(prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(prices), freq="h"),
            "open": [price + 0.1 for price in prices],
            "high": [price + 0.2 for price in prices],
            "low": [price - 0.2 for price in prices],
            "close": prices,
            "volume": [1000.0] * len(prices),
        },
        index=[10 + 3 * index for index in range(len(prices))],
    )


@pytest.mark.parametrize(
    ("prices", "expected", "entry_index", "exit_index"),
    [
        ([1, 2, 3, 3, 3, 3], [0, 0, 1, 1, 0, 0], 3, 5),
        (
            [3, 2, 1, 2, 3, 4, 3, 2, 1, 2],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            5,
            8,
        ),
    ],
    ids=["exit-on-equality", "exit-on-decline"],
)
@pytest.mark.parametrize("signal_as_position", [False, True])
def test_example_holds_long_until_flat_and_executes_at_next_open(
    tmp_path: Path,
    prices: list[float],
    expected: list[int],
    entry_index: int,
    exit_index: int,
    signal_as_position: bool,
) -> None:
    strategy, config = _example(tmp_path)
    frame = _market(prices)
    original = frame.copy(deep=True)

    signals = strategy.generate_signals(frame)

    pd.testing.assert_frame_equal(frame, original)
    assert signals.index.equals(frame.index)
    assert signals["signal"].tolist() == expected
    assert strategy.get_parameters() == {"fast_period": 2, "slow_period": 3}
    config["backtest"].update(
        fees=0.0, slippage=0.0, signal_as_position=signal_as_position
    )
    engine = create_backtest_engine(BacktestConfig.model_validate(config["backtest"]))
    result = engine.run(signals)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade["direction"] == 1
    assert trade["entry_idx"] == entry_index
    assert trade["exit_idx"] == exit_index
    assert trade["entry_price"] == pytest.approx(frame["open"].iloc[entry_index])
    assert trade["exit_price"] == pytest.approx(frame["open"].iloc[exit_index])


def test_example_respects_changed_parameters_and_their_warmup(tmp_path: Path) -> None:
    strategy, _config = _example(tmp_path, fast_period=2, slow_period=4)

    signals = strategy.generate_signals(_market([1, 2, 3, 3, 3, 3]))

    assert strategy.get_parameters() == {"fast_period": 2, "slow_period": 4}
    assert signals["signal"].tolist() == [0, 0, 0, 1, 1, 0]


def test_example_stays_flat_when_averages_are_equal(tmp_path: Path) -> None:
    strategy, config = _example(tmp_path)

    signals = strategy.generate_signals(_market([3.0] * 6))

    assert signals["signal"].tolist() == [0] * 6
    result = create_backtest_engine(
        BacktestConfig.model_validate(config["backtest"])
    ).run(signals)
    assert result.trades == []


def test_example_does_not_hold_when_an_average_is_missing(tmp_path: Path) -> None:
    strategy, _config = _example(tmp_path)

    signals = strategy.generate_signals(_market([1, 2, 3, float("nan"), 5, 6]))

    assert signals["signal"].tolist() == [0, 0, 1, 0, 0, 0]


@pytest.mark.parametrize("invalid", [float("inf"), float("-inf")])
def test_example_rejects_infinite_prices(tmp_path: Path, invalid: float) -> None:
    strategy, _config = _example(tmp_path)

    with pytest.raises(ValueError, match="indicator inputs must not contain infinity"):
        strategy.generate_signals(_market([1, 2, 3, invalid, 5, 6]))
