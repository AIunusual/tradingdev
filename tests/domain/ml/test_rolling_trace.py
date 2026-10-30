"""Every scored bar must be attributable to the roll that produced it.

The trace is the answer to "which model scored this bar, and what did it train
on".  Expectations here are arithmetic on the declared numbers (``step=7`` over
20 bars, a 100-bar sliding window), and the trace's windows are compared with
the schedule by equality so the two cannot describe different runs.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.rolling_gen import (
    RollingOutcome,
    RollingScheme,
    RollType,
    RollWindow,
    enumerate_rolls,
    get_rolling_scheme,
    rolling_outcome,
    trace_payload,
)
from tradingdev.domain.ml.schemas import XGBoostModelConfig

_LOOKBACK = 10
_INTERVAL = 7
_HISTORY = 200
_SCORED = 20
_WINDOW = 100
"""lookback=10 declares a 10 x lookback = 100-bar sliding window."""


class _StubFeatureEngineer(FeatureEngineer):
    """Record training windows and drop one scoring bar on request."""

    def __init__(self, drop_label: int | None = None) -> None:
        super().__init__(lookback=_LOOKBACK, target_horizon=1)
        self._drop_label = drop_label
        self.windows: list[pd.DataFrame] = []

    def transform(
        self, df: pd.DataFrame, *, include_target: bool = True
    ) -> pd.DataFrame:
        result = df.copy()
        if include_target:
            self.windows.append(result)
            return result
        if self._drop_label is not None:
            return result.drop(index=self._drop_label, errors="ignore")
        return result


class _StubModel(XGBoostDirectionModel):
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


def _outcome(
    scheme: RollingScheme,
    *,
    drop_label: int | None = None,
) -> tuple[RollingOutcome, list[pd.DataFrame]]:
    frame = _market_frame(_HISTORY + _SCORED)
    recorder = _StubFeatureEngineer(drop_label)
    outcome = rolling_outcome(
        frame.iloc[:_HISTORY].copy(),
        frame.iloc[_HISTORY:].copy(),
        scheme,
        model=_StubModel(),
        model_config=XGBoostModelConfig(),
        fe=recorder,
        lookback=_LOOKBACK,
        retrain_interval=_INTERVAL,
        label_horizon=1,
        model_factory=_StubModel,
    )
    return outcome, recorder.windows


def _sliding() -> RollingScheme:
    return RollingScheme(name="roll_sd", description="", rtype=RollType.ROLL_SD)


def test_every_scored_bar_is_attributed_to_the_roll_that_scored_it() -> None:
    """With step=7 over 20 bars the roll boundaries are 0 and 7 and 14."""
    outcome, _ = _outcome(_sliding())

    assert outcome.bar_roll == tuple(bar // _INTERVAL for bar in range(_SCORED))


def test_traces_carry_the_declared_window() -> None:
    """A trace and the schedule must be the same object, not two answers."""
    outcome, _ = _outcome(_sliding())

    declared = enumerate_rolls(
        _sliding(),
        n_scored_bars=_SCORED,
        history_bars=_HISTORY,
        lookback=_LOOKBACK,
        retrain_interval=_INTERVAL,
    )
    assert tuple(trace.window for trace in outcome.rolls) == declared
    assert all(isinstance(trace.window, RollWindow) for trace in outcome.rolls)


def test_an_unscored_bar_belongs_to_no_roll() -> None:
    """Bar 5 cannot be scored, so it is attributed to nothing and counted once."""
    outcome, _ = _outcome(_sliding(), drop_label=_HISTORY + 5)

    assert outcome.bar_roll[5] == -1
    assert [trace.bars_scored for trace in outcome.rolls] == [6, 7, 6]
    # The payload cannot store one entry per bar, so it records the holes.
    assert trace_payload(outcome)["unscored_bars"] == [5]


def test_a_starved_window_is_recorded_as_untrained() -> None:
    """A skipped retrain says so instead of looking like a fitted model."""
    scheme = RollingScheme(
        name="roll_sd",
        description="",
        rtype=RollType.ROLL_SD,
        min_train_bars=10_000,
    )

    outcome, windows = _outcome(scheme)

    assert len(windows) == len(outcome.rolls)
    for trace in outcome.rolls:
        assert trace.trained is False
        # The one-bar target of the last raw row is outside the training window.
        assert trace.n_train_rows == _WINDOW - 1


def test_a_fitted_window_reports_the_rows_it_trained_on() -> None:
    scheme = RollingScheme(
        name="roll_sd",
        description="",
        rtype=RollType.ROLL_SD,
        min_train_bars=1,
    )

    outcome, _ = _outcome(scheme)

    assert all(trace.trained is True for trace in outcome.rolls)
    assert [trace.n_train_rows for trace in outcome.rolls] == [_WINDOW - 1] * 3


def test_the_payload_records_the_effective_geometry() -> None:
    """An unset step must be reported as the cadence that actually ran."""
    outcome, _ = _outcome(_sliding())

    payload = trace_payload(outcome)

    assert payload["scheme"] == "roll_sd"
    assert payload["step"] == _INTERVAL
    assert payload["window_size"] == _WINDOW
    assert payload["bars"] == _SCORED
    assert payload["bars_scored"] == _SCORED
    assert payload["initial_model_available"] is True
    assert payload["unscored_bars"] == []
    rolls = payload["rolls"]
    assert isinstance(rolls, list)
    assert [entry["retrain_bar"] for entry in rolls] == [0, 7, 14]
    assert [entry["train_start"] for entry in rolls] == [100, 107, 114]


def test_the_payload_is_json_ready() -> None:
    """Run metadata is stored as JSON, so the record has to survive a round trip."""
    outcome, _ = _outcome(_sliding())

    restored = json.loads(json.dumps(trace_payload(outcome)))

    assert restored["rolls"][0]["trained"] is True


def test_an_empty_run_has_no_rolls() -> None:
    frame = _market_frame(_HISTORY)
    outcome = rolling_outcome(
        frame,
        frame.iloc[:0].copy(),
        _sliding(),
        model=_StubModel(),
        model_config=XGBoostModelConfig(),
        fe=_StubFeatureEngineer(),
        lookback=_LOOKBACK,
        retrain_interval=_INTERVAL,
        label_horizon=1,
        model_factory=_StubModel,
    )

    assert outcome.rolls == ()
    assert outcome.bar_roll == ()


@pytest.mark.parametrize("rtype", list(RollType), ids=lambda t: t.value)
def test_the_trace_never_claims_a_bar_before_its_own_training(rtype: RollType) -> None:
    """A roll's window must end at the first bar it scores, half-open."""
    name = "roll_sd" if rtype is RollType.ROLL_SD else "roll_ex"
    outcome, _ = _outcome(get_rolling_scheme(name))

    for trace in outcome.rolls:
        assert trace.window.train_end == _HISTORY + trace.window.test_start
