"""Strategy construction, fitting, and evaluation share the recorded run seed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from tests.app.test_backtest_service import _raw_config, _service, _SignalStrategy

from tradingdev.domain.backtest.schemas import BacktestConfig, ParallelConfig
from tradingdev.domain.randomness import (
    execution_randomness,
    get_numpy_rng,
    get_random,
    get_seed,
)
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.mcp.workers import optimization

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


def _random_strategy(events: list[tuple[str, int | None, float]]) -> _SignalStrategy:
    class RandomStrategy(_SignalStrategy):
        def __init__(self) -> None:
            super().__init__()
            events.append(("constructor", get_seed(), get_random().random()))

        def fit(self, df: pd.DataFrame) -> None:
            events.append(("fit", get_seed(), float(get_numpy_rng().random())))

        def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
            events.append(("signals", get_seed(), float(get_numpy_rng().random())))
            result = df.copy()
            result["signal"] = get_numpy_rng().choice([-1, 0, 1], size=len(df))
            return result

    return RandomStrategy()


@pytest.mark.parametrize("walk_forward", [False, True])
def test_service_replays_seeded_constructor_fit_and_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, walk_forward: bool
) -> None:
    service, _data, loader, _strategy = _service(tmp_path, rows=100)
    events: list[tuple[str, int | None, float]] = []
    monkeypatch.setattr(
        loader, "create_from_execution", lambda *args: _random_strategy(events)
    )
    config = _raw_config(random_seed=42)
    if walk_forward:
        config["validation"] = {"mode": "automatic", "n_splits": 2, "train_ratio": 0.5}
    service.run_raw_config(config, walk_forward=walk_forward)
    first = events.copy()
    assert first[0][0] == "constructor"
    assert all(seed == 42 for _, seed, _ in first)
    assert any(stage == "fit" for stage, _, _ in first) == walk_forward
    events.clear()
    service.run_raw_config(config, walk_forward=walk_forward)
    assert events == first
    events.clear()
    config["random_seed"] = 7
    service.run_raw_config(config, walk_forward=walk_forward)
    assert events != first
    with pytest.raises(RuntimeError, match="active strategy execution"):
        get_seed()


def test_service_failure_restores_enclosing_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _data, loader, _strategy = _service(tmp_path)

    def fail(*_args: Any) -> None:
        assert get_seed() == 42
        raise ValueError("constructor failed")

    monkeypatch.setattr(loader, "create_from_execution", fail)
    with execution_randomness(7):
        generator = get_numpy_rng()
        with pytest.raises(ValueError, match="constructor failed"):
            service.run_raw_config(_raw_config(random_seed=42))
        assert get_seed() == 7 and get_numpy_rng() is generator


def test_trial_streams_do_not_depend_on_prior_trials_or_execution_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, data, loader, _strategy = _service(tmp_path)
    events: list[tuple[str, int | None, float]] = []
    prepared_spec = service.prepare_strategy(_raw_config())
    monkeypatch.setattr(service, "prepare_strategy", lambda *args: prepared_spec)
    monkeypatch.setattr(optimization, "BacktestService", lambda: service)
    monkeypatch.setattr(optimization, "StrategyLoader", lambda: loader)
    monkeypatch.setattr(
        loader,
        "create_from_execution",
        lambda *args, **kwargs: _random_strategy(events),
    )
    config = BacktestConfig(**_raw_config()["backtest"])
    strategy = {"id": "fixture", "parameters": {}}
    execution = StrategyExecution(kind="generated", constructor_kwargs={})

    def evaluate(seed: int) -> None:
        result = optimization._run_single_combo(
            strategy,
            execution,
            config,
            data.dataset.frame,
            {},
            "total_return",
            ParallelConfig(),
            seed,
        )
        assert result.result.metric_metadata["execution_context"]["random_seed"] == seed

    evaluate(42)
    first = events.copy()
    evaluate(7)
    events.clear()
    evaluate(42)
    assert events == first
    events.clear()
    optimization._evaluate_combo(
        strategy,
        execution,
        config.model_dump(),
        data.dataset.frame,
        {},
        "total_return",
        ParallelConfig().model_dump(),
        42,
    )
    assert events == first
