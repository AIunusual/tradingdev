"""The rolling engine may only train on bars that already happened.

Every assertion is stated in bar terms -- which timestamp trained the model that
scored which bar -- and checked against frames the feature engineer recorded.
That way a change to the loop has to break recorded evidence, not just disagree
with a helper the test could have asked for the same answer.

The expected signal sequences below are worked out from the fixture layout
(``test_offset`` plus a per-bar parity rule), never copied from a run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.rolling_gen import (
    RollingScheme,
    RollType,
    rolling_outcome,
    rolling_probabilities,
    run_rolling,
    signals_from_probabilities,
)
from tradingdev.domain.ml.schemas import XGBoostModelConfig

if TYPE_CHECKING:
    from collections.abc import Callable

_MODEL_CONFIG = XGBoostModelConfig()
"""One shared config: every stub model below ignores it."""


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


class _RecordingFeatureEngineer(FeatureEngineer):
    """Record the windows and contexts instead of computing real indicators."""

    def __init__(self) -> None:
        super().__init__(lookback=10, target_horizon=1)
        self.windows: list[pd.DataFrame] = []
        self.contexts: list[pd.DataFrame] = []

    def transform(
        self, df: pd.DataFrame, *, include_target: bool = True
    ) -> pd.DataFrame:
        if include_target:
            self.windows.append(df.copy())
        else:
            self.contexts.append(df.copy())
        return df.copy()


class _DroppingFeatureEngineer(_RecordingFeatureEngineer):
    """Record as usual, but drop one bar from every scoring context."""

    def __init__(self, drop_index: int) -> None:
        super().__init__()
        self._drop_index = drop_index

    def transform(
        self, df: pd.DataFrame, *, include_target: bool = True
    ) -> pd.DataFrame:
        result = super().transform(df, include_target=include_target)
        if include_target:
            return result
        return result.drop(index=self._drop_index, errors="ignore")


class _EmptyFeatureEngineer(_RecordingFeatureEngineer):
    """A feature engineer that never returns a usable row."""

    def transform(
        self, df: pd.DataFrame, *, include_target: bool = True
    ) -> pd.DataFrame:
        return df.iloc[:0]


class _RowLabelModel(XGBoostDirectionModel):
    """Call long on row labels that are multiples of 7, flat on the rest.

    The label is the bar's own position in the concatenated frame, so a run that
    read the wrong row would shift the whole sequence.  The modulus is 7 rather
    than 2 on purpose: a shift by the 20-bar context length is invisible modulo
    2, and a check that cannot fail is not a check.
    """

    def __init__(self, config: XGBoostModelConfig | None = None) -> None:
        super().__init__(config or XGBoostModelConfig())
        self.trained_on: list[pd.DataFrame] = []

    def train(self, df: pd.DataFrame, eval_df: pd.DataFrame | None = None) -> None:
        self.trained_on.append(df.copy())

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        p_long = 0.9 if int(df.index[0]) % 7 == 0 else 0.1
        return pd.DataFrame({-1: 1.0 - p_long, 1: p_long}, index=df.index)


class _AlwaysLong(XGBoostDirectionModel):
    """Call long on every bar, so a suppressed bar stands out."""

    def __init__(self, config: XGBoostModelConfig | None = None) -> None:
        super().__init__(config or XGBoostModelConfig())

    def train(self, df: pd.DataFrame, eval_df: pd.DataFrame | None = None) -> None:
        return None

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({-1: 0.0, 1: 1.0}, index=df.index)


class _CountingFactory:
    """Count how many models the engine actually trained."""

    def __init__(self, model: XGBoostDirectionModel) -> None:
        self._model = model
        self.calls = 0

    def __call__(self, config: XGBoostModelConfig) -> XGBoostDirectionModel:
        self.calls += 1
        return self._model


def _scheme(rtype: RollType, *, window_size: int | None = None) -> RollingScheme:
    """A scheme that retrains every bar, so one window exists per bar."""
    return RollingScheme(
        name="test_scheme",
        description="",
        rtype=rtype,
        step=1,
        min_train_bars=1,
        window_size=window_size,
    )


def _probabilities(
    history: pd.DataFrame,
    scored: pd.DataFrame,
    scheme: RollingScheme,
    fe: FeatureEngineer,
    model: XGBoostDirectionModel,
    *,
    factory: Callable[[XGBoostModelConfig], XGBoostDirectionModel] | None = None,
) -> pd.DataFrame:
    return rolling_probabilities(
        history,
        scored,
        scheme,
        model=model,
        model_config=_MODEL_CONFIG,
        fe=fe,
        lookback=10,
        retrain_interval=1,
        label_horizon=1,
        model_factory=type(model) if factory is None else factory,
    )


def _signals(
    history: pd.DataFrame,
    scored: pd.DataFrame,
    scheme: RollingScheme,
    fe: FeatureEngineer,
    model: XGBoostDirectionModel,
    *,
    threshold: float = 0.5,
    cooldown: int = 0,
) -> np.ndarray:
    return run_rolling(
        history,
        scored,
        scheme,
        model=model,
        model_config=_MODEL_CONFIG,
        fe=fe,
        threshold=threshold,
        cooldown=cooldown,
        lookback=10,
        retrain_interval=1,
        label_horizon=1,
        model_factory=type(model),
    )


@pytest.mark.parametrize("rtype", list(RollType), ids=lambda t: t.value)
def test_every_retrained_window_ends_before_the_bar_it_scores(
    rtype: RollType,
) -> None:
    """The scored bar and everything after it stay out of training."""
    frame = _market_frame(30)
    history = frame.iloc[:20].copy()
    scored = frame.iloc[20:].copy()
    recorder = _RecordingFeatureEngineer()

    _probabilities(history, scored, _scheme(rtype), recorder, _RowLabelModel())

    assert len(recorder.windows) == len(scored)
    for bar, window in enumerate(recorder.windows):
        last_trained = pd.Timestamp(window["timestamp"].iloc[-1])
        assert last_trained < pd.Timestamp(scored["timestamp"].iloc[bar]), (
            f"bar {bar}: training window reaches {last_trained}, "
            f"which is not before {scored['timestamp'].iloc[bar]}"
        )


def test_expanding_window_keeps_the_first_history_bar() -> None:
    frame = _market_frame(30)
    history = frame.iloc[:20].copy()
    scored = frame.iloc[20:].copy()
    recorder = _RecordingFeatureEngineer()

    _probabilities(
        history, scored, _scheme(RollType.ROLL_EX), recorder, _RowLabelModel()
    )

    for bar, window in enumerate(recorder.windows):
        assert pd.Timestamp(window["timestamp"].iloc[0]) == pd.Timestamp(
            history["timestamp"].iloc[0]
        )
        # 20 history bars plus the bars already scored before this one.
        assert len(window) == 20 + bar


def test_sliding_window_keeps_its_declared_length() -> None:
    frame = _market_frame(30)
    history = frame.iloc[:20].copy()
    scored = frame.iloc[20:].copy()
    recorder = _RecordingFeatureEngineer()

    _probabilities(
        history,
        scored,
        _scheme(RollType.ROLL_SD, window_size=15),
        recorder,
        _RowLabelModel(),
    )

    assert [len(window) for window in recorder.windows] == [15] * len(scored)
    # The last bar is index 29; a 15-bar window starts at 29 - 15 = 14.
    assert pd.Timestamp(recorder.windows[-1]["timestamp"].iloc[0]) == pd.Timestamp(
        frame["timestamp"].iloc[14]
    )


def test_predictions_land_on_the_bar_they_describe() -> None:
    """Row labels, not positions: parity of the concatenated index decides."""
    frame = _market_frame(30)
    history = frame.iloc[:20].copy()
    scored = frame.iloc[20:].copy()

    signals = _signals(
        history,
        scored,
        _scheme(RollType.ROLL_EX),
        _RecordingFeatureEngineer(),
        _RowLabelModel(),
    )

    # Combined index 20..29 -> labels divisible by 7 are long, the rest short.
    expected = [1 if (20 + bar) % 7 == 0 else -1 for bar in range(len(scored))]
    assert signals.tolist() == expected


def test_a_bar_without_features_emits_no_signal() -> None:
    """A missing feature is not an invitation to reuse the previous bar."""
    frame = _market_frame(30)
    history = frame.iloc[:20].copy()
    scored = frame.iloc[20:].copy()
    silent_bar = 3

    signals = _signals(
        history,
        scored,
        _scheme(RollType.ROLL_EX),
        _DroppingFeatureEngineer(drop_index=20 + silent_bar),
        _AlwaysLong(),
    )

    assert signals[silent_bar] == 0
    # Every other bar still trades, so the zero belongs to the dropped bar.
    others = np.arange(len(scored)) != silent_bar
    assert signals[others].tolist() == [1] * (len(scored) - 1)


def test_a_window_below_the_minimum_keeps_the_previous_model() -> None:
    """Starving the window must skip training, never stretch it forward."""
    frame = _market_frame(30)
    history = frame.iloc[:20].copy()
    scored = frame.iloc[20:].copy()
    model = _AlwaysLong()
    factory = _CountingFactory(model)

    probabilities = _probabilities(
        history,
        scored,
        RollingScheme(
            name="test_scheme",
            description="",
            rtype=RollType.ROLL_EX,
            step=1,
            min_train_bars=10_000,
        ),
        _RecordingFeatureEngineer(),
        model,
        factory=factory,
    )

    assert factory.calls == 0
    assert probabilities["p_long"].tolist() == [1.0] * len(scored)


def test_replaying_the_training_frame_does_not_replay_it() -> None:
    """``fit(frame)`` then scoring the same frame must not concatenate twice."""
    frame = _market_frame(25)
    recorder = _RecordingFeatureEngineer()

    _probabilities(frame, frame, _scheme(RollType.ROLL_EX), recorder, _RowLabelModel())

    # Bars 0 and 1 have fewer than two earlier rows, so no window is recorded
    # for them; the first recorded window belongs to bar 2. A window recorded
    # for those bars could only have come from the frame's own future.
    assert len(recorder.windows) == len(frame) - 2, (
        f"{len(recorder.windows)} training windows for {len(frame)} bars: the "
        "first two bars have no earlier row, so any window recorded for them "
        "was trained on the bar being scored"
    )
    for bar, window in enumerate(recorder.windows, start=2):
        stamps = pd.DatetimeIndex(window["timestamp"])
        assert stamps.is_unique
        assert stamps.is_monotonic_increasing
        assert pd.Timestamp(stamps[-1]) < pd.Timestamp(frame["timestamp"].iloc[bar])


def test_a_run_that_scores_nothing_is_rejected() -> None:
    """All-zero signals would look like a strategy that never trades."""
    frame = _market_frame(30)

    with pytest.raises(ValueError, match="scored none of 10 bars"):
        _probabilities(
            frame.iloc[:20].copy(),
            frame.iloc[20:].copy(),
            _scheme(RollType.ROLL_EX),
            _EmptyFeatureEngineer(),
            _AlwaysLong(),
        )


def test_an_empty_scored_frame_produces_no_signals() -> None:
    """Nothing to order, so nothing to refuse: same as the classic loop."""
    frame = _market_frame(20)

    signals = _signals(
        frame,
        frame.iloc[:0].copy(),
        _scheme(RollType.ROLL_SD),
        _RecordingFeatureEngineer(),
        _AlwaysLong(),
    )

    assert signals.tolist() == []


@pytest.mark.parametrize(
    ("rows", "cooldown", "expected"),
    [
        (
            [(0.9, 0.1), (0.1, 0.9), (0.5, 0.5), (None, None), (0.7, 0.7)],
            0,
            [1, -1, 0, 0, 1],
        ),
        (
            [(0.9, 0.1), (0.1, 0.9), (0.5, 0.5), (None, None), (0.7, 0.7)],
            2,
            [1, 0, 0, 0, 1],
        ),
        ([(0.9, 0.1), (0.9, 0.1), (0.9, 0.1)], 2, [1, 0, 1]),
    ],
    ids=["no-cooldown", "cooldown-two", "cooldown-two-bars"],
)
def test_signals_follow_the_threshold_and_cooldown(
    rows: list[tuple[float | None, float | None]],
    cooldown: int,
    expected: list[int],
) -> None:
    frame = pd.DataFrame(rows, columns=["p_long", "p_short"])

    signals = signals_from_probabilities(frame, threshold=0.6, cooldown=cooldown)

    assert signals.tolist() == expected


def _training_recorder(
    sink: list[pd.DataFrame],
) -> Callable[[XGBoostModelConfig], XGBoostDirectionModel]:
    """A model that records the frame it was fitted on, and nothing else."""

    class _Recorder(XGBoostDirectionModel):
        def __init__(self, config: XGBoostModelConfig) -> None:
            super().__init__(config)

        def train(self, df: pd.DataFrame, eval_df: pd.DataFrame | None = None) -> None:
            sink.append(df.copy())

        def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({-1: 0.0, 1: 1.0}, index=df.index)

    return _Recorder


def test_the_engine_never_trains_on_an_unrealised_label() -> None:
    """Issue #24: unknown tail labels must not be treated as realised.

    With a real feature engineer, the last ``target_horizon`` rows of a training
    window have no future close to label them. They must be dropped, not filled
    with a flat label: the window ends at bar 299 and the horizon is 5, so the
    last bar that can carry a realised label is bar 294.
    """
    horizon = 5
    history_bars = 300
    frame = _market_frame(history_bars + 20)
    sink: list[pd.DataFrame] = []
    factory = _training_recorder(sink)

    rolling_outcome(
        frame.iloc[:history_bars].copy(),
        frame.iloc[history_bars:].copy(),
        _scheme(RollType.ROLL_EX),
        model=factory(_MODEL_CONFIG),
        model_config=_MODEL_CONFIG,
        fe=FeatureEngineer(lookback=24, target_horizon=horizon),
        lookback=24,
        retrain_interval=1,
        label_horizon=horizon,
        model_factory=factory,
    )

    assert sink, "the run must have trained at least one model"
    # The first roll trains on bars 0..299 of the concatenated frame. Its last
    # label needs close[299 + 1 + horizon - 1], which is outside the window, so
    # the last usable bar is 299 - horizon = 294.
    first = sink[0]
    assert pd.Timestamp(first["timestamp"].max()) == pd.Timestamp(
        frame["timestamp"].iloc[history_bars - 1 - horizon]
    ), "the window trained on a bar whose label was not realised yet"
    # Every roll's labels are realised: none was filled in to keep the row.
    for trained in sink:
        assert trained["target"].notna().all(), "an unrealised label was filled in"


class _FillingFeatureEngineer(FeatureEngineer):
    """Fill unknown tail labels with 0, the way issue #24's defect did.

    A feature engineer that does this hands the engine rows whose label reads
    past the bar being scored. The engine has to refuse them on its own.
    """

    def __init__(self, horizon: int) -> None:
        super().__init__(lookback=1, target_horizon=horizon)
        self._horizon = horizon

    def transform(
        self, df: pd.DataFrame, *, include_target: bool = True
    ) -> pd.DataFrame:
        result = df.copy()
        result["stub_feature"] = np.arange(len(result), dtype=float)
        if include_target:
            forward = result["close"].shift(-self._horizon) / result["close"] - 1
            result["target"] = np.sign(forward).fillna(0).astype(int)
        return result


def test_the_engine_drops_labels_the_engineer_filled_in() -> None:
    """Issue #24: label maturity is the engine's rule, not the engineer's.

    The engineer here fills every tail label, so nothing is missing for
    ``dropna`` to remove. The engine still refuses to fit past the last
    realisable label: with the window ending at bar 99 and a horizon of 5, that
    is bar 94: close[99] is the last price the transformer received.
    """
    horizon = 5
    history_bars = 100
    frame = _market_frame(history_bars + 10)
    sink: list[pd.DataFrame] = []
    factory = _training_recorder(sink)

    rolling_outcome(
        frame.iloc[:history_bars].copy(),
        frame.iloc[history_bars:].copy(),
        _scheme(RollType.ROLL_EX),
        model=factory(_MODEL_CONFIG),
        model_config=_MODEL_CONFIG,
        fe=_FillingFeatureEngineer(horizon),
        lookback=1,
        retrain_interval=1000,
        label_horizon=horizon,
        model_factory=factory,
    )

    assert sink, "the run must have trained at least one model"
    first = sink[0]
    assert pd.Timestamp(first["timestamp"].max()) == pd.Timestamp(
        frame["timestamp"].iloc[history_bars - 1 - horizon]
    ), "the engine trained on a label the engineer had filled in"


def test_replay_waits_for_a_model_that_has_not_seen_the_scored_future() -> None:
    frame = _market_frame(40)

    class FutureModel(_AlwaysLong):
        def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
            raise AssertionError("model fitted on replay's future must not score")

    outcome = rolling_outcome(
        frame,
        frame.iloc[:20],
        RollingScheme(
            name="test",
            description="",
            rtype=RollType.ROLL_EX,
            step=5,
            min_train_bars=1,
            predict_lookback=20,
        ),
        model=FutureModel(),
        model_config=_MODEL_CONFIG,
        fe=_RecordingFeatureEngineer(),
        lookback=10,
        retrain_interval=5,
        label_horizon=1,
        model_factory=_AlwaysLong,
    )
    assert outcome.probabilities.iloc[:5].isna().all().all()
    assert outcome.probabilities.iloc[5:]["p_long"].eq(1).all()


def test_replay_without_a_successful_refit_reports_model_availability() -> None:
    frame = _market_frame(40)
    with pytest.raises(ValueError, match="no causal model"):
        rolling_outcome(
            frame,
            frame.iloc[:20],
            RollingScheme(
                name="test",
                description="",
                rtype=RollType.ROLL_EX,
                step=100,
                min_train_bars=1,
                predict_lookback=20,
            ),
            model=_AlwaysLong(),
            model_config=_MODEL_CONFIG,
            fe=_RecordingFeatureEngineer(),
            lookback=10,
            retrain_interval=100,
            label_horizon=1,
            model_factory=_AlwaysLong,
        )


@pytest.mark.parametrize("horizon", [0, -1])
def test_a_non_positive_label_horizon_is_rejected(horizon: int) -> None:
    frame = _market_frame(40)

    with pytest.raises(ValueError, match="label_horizon must be at least 1"):
        rolling_outcome(
            frame.iloc[:30].copy(),
            frame.iloc[30:].copy(),
            _scheme(RollType.ROLL_EX),
            model=_AlwaysLong(),
            model_config=_MODEL_CONFIG,
            fe=_RecordingFeatureEngineer(),
            lookback=10,
            retrain_interval=1,
            label_horizon=horizon,
        )
