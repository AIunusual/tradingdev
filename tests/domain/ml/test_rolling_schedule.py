"""The declared schedule must be the schedule that runs.

Bounds are hand-computed from the declared numbers (``lookback=10`` gives a
100-bar sliding window, ``retrain_interval=7`` gives a roll every 7 bars), never
read back from the function under test.  The last test compares the enumeration
with the retrains the engine actually performed, so the preview cannot drift
into a second, prettier description of a different run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.rolling_gen import (
    RollingScheme,
    RollType,
    enumerate_rolls,
    get_rolling_scheme,
    rolling_probabilities,
)
from tradingdev.domain.ml.schemas import XGBoostModelConfig

_LOOKBACK = 10
"""lookback=10 declares a 10 x lookback = 100-bar sliding window."""

_INTERVAL = 7
"""A roll every 7 bars, taken from the strategy's classic retrain cadence."""

_HISTORY = 200
_SCORED = 20


def _schedule(rtype: RollType, **overrides: int) -> tuple[tuple[int, ...], ...]:
    """Enumerate and flatten, so a wrong bound shows up as a wrong number."""
    scheme = RollingScheme(
        name="test_scheme",
        description="",
        rtype=rtype,
        **overrides,
    )
    rolls = enumerate_rolls(
        scheme,
        n_scored_bars=_SCORED,
        history_bars=_HISTORY,
        lookback=_LOOKBACK,
        retrain_interval=_INTERVAL,
    )
    return tuple(
        (
            roll.index,
            roll.retrain_bar,
            roll.combined_idx,
            roll.train_start,
            roll.train_end,
            roll.test_start,
            roll.test_end,
        )
        for roll in rolls
    )


def test_sliding_schedule_matches_hand_computed_bounds() -> None:
    """Bars 0, 7 and 14 retrain; the last roll is clipped at 20."""
    assert _schedule(RollType.ROLL_SD) == (
        # index, retrain, combined, train_start, train_end, test_start, test_end
        (0, 0, 200, 100, 200, 0, 7),
        (1, 7, 207, 107, 207, 7, 14),
        (2, 14, 214, 114, 214, 14, 20),
    )


def test_expanding_schedule_keeps_the_first_bar() -> None:
    assert _schedule(RollType.ROLL_EX) == (
        (0, 0, 200, 0, 200, 0, 7),
        (1, 7, 207, 0, 207, 7, 14),
        (2, 14, 214, 0, 214, 14, 20),
    )


@pytest.mark.parametrize("rtype", list(RollType), ids=lambda t: t.value)
def test_no_roll_trains_on_a_bar_it_scores(rtype: RollType) -> None:
    """``train_end`` is the bar being scored, and the bound is half-open."""
    rolls = enumerate_rolls(
        get_rolling_scheme("roll_sd" if rtype is RollType.ROLL_SD else "roll_ex"),
        n_scored_bars=_SCORED,
        history_bars=_HISTORY,
        lookback=_LOOKBACK,
        retrain_interval=_INTERVAL,
    )

    for roll in rolls:
        first_scored = _HISTORY + roll.test_start
        assert roll.train_end == first_scored, (
            f"roll {roll.index}: training stops at {roll.train_end}, "
            f"but its first scored bar is {first_scored}"
        )


def test_rolls_cover_every_bar_exactly_once() -> None:
    rolls = enumerate_rolls(
        get_rolling_scheme("roll_sd"),
        n_scored_bars=_SCORED,
        history_bars=_HISTORY,
        lookback=_LOOKBACK,
        retrain_interval=_INTERVAL,
    )

    covered: list[int] = []
    for roll in rolls:
        covered.extend(range(roll.test_start, roll.test_end))
    assert covered == list(range(_SCORED))


def test_a_step_covering_the_run_gives_one_roll() -> None:
    """A cadence longer than the run is qlib's degenerate single-roll case."""
    rolls = enumerate_rolls(
        get_rolling_scheme("roll_sd"),
        n_scored_bars=_SCORED,
        history_bars=_HISTORY,
        lookback=_LOOKBACK,
        retrain_interval=10_000,
    )

    assert len(rolls) == 1
    assert (rolls[0].test_start, rolls[0].test_end) == (0, _SCORED)


def test_an_empty_scored_frame_has_no_rolls() -> None:
    rolls = enumerate_rolls(
        get_rolling_scheme("roll_sd"),
        n_scored_bars=0,
        history_bars=_HISTORY,
        lookback=_LOOKBACK,
        retrain_interval=_INTERVAL,
    )

    assert rolls == ()


def test_a_full_replay_starts_with_an_empty_window() -> None:
    """Scoring the fitted frame leaves the first roll nothing to train on."""
    rolls = enumerate_rolls(
        get_rolling_scheme("roll_sd"),
        n_scored_bars=_SCORED,
        history_bars=0,
        lookback=_LOOKBACK,
        retrain_interval=_INTERVAL,
    )

    assert (rolls[0].train_start, rolls[0].train_end) == (0, 0)
    assert rolls[0].train_bars == 0
    assert rolls[1].train_bars == 7


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_scored_bars": -1}, "n_scored_bars must not be negative"),
        ({"history_bars": -1}, "history_bars must not be negative"),
    ],
)
def test_negative_counts_are_rejected(kwargs: dict[str, int], message: str) -> None:
    arguments = {
        "n_scored_bars": _SCORED,
        "history_bars": _HISTORY,
        "lookback": _LOOKBACK,
        "retrain_interval": _INTERVAL,
    } | kwargs

    with pytest.raises(ValueError, match=message):
        enumerate_rolls(get_rolling_scheme("roll_sd"), **arguments)


class _RecordingFeatureEngineer(FeatureEngineer):
    """Record the training window the engine actually uses."""

    def __init__(self) -> None:
        super().__init__(lookback=_LOOKBACK, target_horizon=1)
        self.windows: list[pd.DataFrame] = []

    def transform(
        self, df: pd.DataFrame, *, include_target: bool = True
    ) -> pd.DataFrame:
        if include_target:
            self.windows.append(df.copy())
        return df.copy()


class _StubModel(XGBoostDirectionModel):
    """Train and predict without touching xgboost."""

    def __init__(self, config: XGBoostModelConfig | None = None) -> None:
        super().__init__(config or XGBoostModelConfig())

    def train(self, df: pd.DataFrame, eval_df: pd.DataFrame | None = None) -> None:
        return None

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({-1: 0.0, 1: 1.0}, index=df.index)


def _market_frame(rows: int) -> pd.DataFrame:
    positions = np.arange(rows, dtype=float)
    close = 1000.0 + positions + 0.5 * np.sin(positions)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0 + positions,
        }
    )


def test_the_schedule_is_what_the_engine_runs() -> None:
    """Every enumerated roll must match a retrain the engine really performed."""
    frame = _market_frame(_HISTORY + _SCORED)
    history = frame.iloc[:_HISTORY].copy()
    scored = frame.iloc[_HISTORY:].copy()
    recorder = _RecordingFeatureEngineer()
    scheme = get_rolling_scheme("roll_sd")

    rolling_probabilities(
        history,
        scored,
        scheme,
        model=_StubModel(),
        model_config=XGBoostModelConfig(),
        fe=recorder,
        lookback=_LOOKBACK,
        retrain_interval=_INTERVAL,
        label_horizon=1,
        model_factory=_StubModel,
    )

    schedule = enumerate_rolls(
        scheme,
        n_scored_bars=len(scored),
        history_bars=len(history),
        lookback=_LOOKBACK,
        retrain_interval=_INTERVAL,
    )
    assert len(recorder.windows) == len(schedule) == 3
    for roll, window in zip(schedule, recorder.windows, strict=True):
        assert (window.index[0], window.index[-1] + 1) == (
            roll.train_start,
            roll.train_end,
        ), f"roll {roll.index}: the engine trained on a different window"
