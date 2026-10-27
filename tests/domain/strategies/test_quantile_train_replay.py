"""Regression tests for the train-replay context (issue #24 follow-up 1).

``walk_forward`` calls ``fit(train_df)`` and then ``generate_signals(train_df)``
to report training metrics. ``generate_signals`` prepends ``self._train_df`` to
the frame it scores, so that call concatenated the training window with itself:
the 60-bar prediction context spanned both copies, the timestamp filter kept the
duplicate rows, and the state machine started writing at bar 0 while reading
from bar 940.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.features.quantile_features import QuantileFeatureEngineer
from tradingdev.domain.strategies.bundled.quantile_strategy import strategy as quantile
from tradingdev.domain.strategies.bundled.quantile_strategy.config import (
    QuantileStrategyConfig,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

_LONG_ONLY = (1.0, 0.0, 0.0, 0.0)


class _RecordingModel:
    """Stand-in classifier that records how many rows it was asked to score."""

    def __init__(
        self,
        calls: list[int],
        probabilities: tuple[float, float, float, float] = (0.6, 0.0, 0.1, 0.3),
    ) -> None:
        self.calls = calls
        self.probabilities = probabilities

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        self.calls.append(len(features))
        probabilities = np.zeros((len(features), 4), dtype=np.float64)
        for column, value in enumerate(self.probabilities):
            probabilities[:, column] = value
        return probabilities


class _GapFeatureEngineer(QuantileFeatureEngineer):
    """Feature engineer that drops one fixed row from every context window.

    Subclasses the real engineer so the strategy accepts it without a cast and
    so a change to the ``transform`` signature fails type checking here.
    """

    def __init__(self, drop_position: int) -> None:
        super().__init__(horizon=1, profit_target=0.0)
        self.drop_position = drop_position

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        include_target: bool = True,
        target_type: str = "path",
    ) -> pd.DataFrame:
        out = pd.DataFrame(
            {"close": frame["close"].astype(float), "f1": 1.0},
            index=frame.index,
        )
        out = out.drop(index=self.drop_position, errors="ignore")
        return out


class _WindowRecordingFeatureEngineer(QuantileFeatureEngineer):
    """Records how many rows each prediction window contains."""

    def __init__(self, windows: list[int]) -> None:
        super().__init__(horizon=1, profit_target=0.0)
        self.windows = windows

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        include_target: bool = True,
        target_type: str = "path",
    ) -> pd.DataFrame:
        self.windows.append(len(frame))
        return super().transform(
            frame,
            include_target=include_target,
            target_type=target_type,
        )


def _fitted_strategy(
    frame: pd.DataFrame,
    calls: list[int],
    monkeypatch: pytest.MonkeyPatch,
    *,
    retrain_interval: int,
    horizon: int = 3,
    min_holding_bars: int = 5,
    strategy_sl: float = 0.003,
    dynamic_sizing: bool = True,
    train_window: int = 20160,
    probabilities: tuple[float, float, float, float] = (0.6, 0.0, 0.1, 0.3),
) -> quantile.QuantileStrategy:
    """A strategy that already ran ``fit(frame)``."""
    config = QuantileStrategyConfig(
        horizon=horizon,
        retrain_interval=retrain_interval,
        train_window=train_window,
        min_entry_edge=0.0015,
        min_holding_bars=min_holding_bars,
        strategy_sl=strategy_sl,
        dynamic_sizing=dynamic_sizing,
    )
    feature_eng = QuantileFeatureEngineer(
        horizon=horizon,
        profit_target=config.profit_target,
    )
    feature_eng.transform(frame, include_target=True, target_type="regime")
    feature_names = [
        name
        for name in feature_eng.get_feature_names()
        if name not in ("target_regime", "target_long", "target_short", "target")
    ]

    model = _RecordingModel(calls, probabilities)
    monkeypatch.setattr(
        quantile, "_train_regime_classifier", lambda *args, **kwargs: model
    )

    strategy = quantile.QuantileStrategy(config=config)
    strategy._train_df = frame.copy()  # noqa: SLF001
    strategy._feature_eng = feature_eng  # noqa: SLF001
    strategy._feature_names = feature_names  # noqa: SLF001
    strategy._best_horizon = horizon  # noqa: SLF001
    strategy._regime_model = model  # noqa: SLF001
    return strategy


def _expected_rows(frame: pd.DataFrame, strategy: quantile.QuantileStrategy) -> int:
    """Bars of ``frame`` that have features at all, computed independently."""
    return len(
        QuantileFeatureEngineer(
            horizon=strategy._best_horizon,  # noqa: SLF001
            profit_target=strategy._config.profit_target,  # noqa: SLF001
        ).transform(frame, include_target=False)
    )


def test_replaying_the_training_frame_scores_each_bar_once(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scoring the training frame must not score the duplicated context too."""
    retrain_interval = 60
    frame = large_ohlcv_df.iloc[:320]
    calls: list[int] = []
    strategy = _fitted_strategy(
        frame, calls, monkeypatch, retrain_interval=retrain_interval
    )

    signals = strategy.generate_signals(frame)

    assert sum(calls) == _expected_rows(frame, strategy), (
        "each bar must be scored exactly once; the difference is duplicated context"
    )
    assert max(calls) <= retrain_interval, (
        "a chunk must never be handed more rows than it covers"
    )
    assert len(signals) == len(frame)


def test_scoring_the_training_frame_keeps_the_output_aligned(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first chunk must start at bar 0, not at the end of the frame.

    The duplicated copy made the state machine consume bars 260..319 before
    writing output row 0, so a warm-up bar could inherit a late signal.
    """
    retrain_interval = 60
    frame = large_ohlcv_df.iloc[:320]
    calls: list[int] = []
    strategy = _fitted_strategy(
        frame, calls, monkeypatch, retrain_interval=retrain_interval
    )

    signals = strategy.generate_signals(frame)

    # Bar 0 has no usable history, so the context transform drops it and the
    # chunk only covers the bars that survived. The chunk therefore reports
    # strictly fewer rows than a full interval, never more.
    assert calls[0] < retrain_interval
    assert signals["signal"].iloc[0] == 0.0


def test_a_datetime_index_frame_is_not_duplicated(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timestamps may arrive as the index instead of a ``timestamp`` column."""
    retrain_interval = 60
    frame = large_ohlcv_df.iloc[:320].set_index("timestamp")
    calls: list[int] = []
    strategy = _fitted_strategy(
        frame, calls, monkeypatch, retrain_interval=retrain_interval
    )

    strategy.generate_signals(frame)

    assert sum(calls) == _expected_rows(frame, strategy), (
        "an index-only frame must be trimmed like a timestamp-column frame"
    )
    assert max(calls) <= retrain_interval


def test_a_dropped_row_does_not_shift_the_bars_after_it(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Predictions are written back by row position, not by loop counter.

    The fake feature engineer removes one row from the middle of the first
    context window. The model is long on every bar, holds for the whole frame
    (horizon 1000, no stop, one-bar minimum hold), so every scored bar must
    carry signal 1 and the removed bar must stay flat. Writing predictions by
    loop counter instead moved bar 30's signal to bar 29 and left bar 59 empty.
    """
    retrain_interval = 60
    frame = large_ohlcv_df.iloc[:120]
    calls: list[int] = []
    strategy = _fitted_strategy(
        frame,
        calls,
        monkeypatch,
        retrain_interval=retrain_interval,
        horizon=1000,
        min_holding_bars=1,
        strategy_sl=0.0,
        dynamic_sizing=False,
        train_window=1,
        probabilities=_LONG_ONLY,
    )
    strategy._feature_eng = _GapFeatureEngineer(drop_position=30)  # noqa: SLF001
    strategy._feature_names = ["f1"]  # noqa: SLF001

    signals = strategy.generate_signals(frame)

    assert signals["signal"].iloc[30] == 0.0, "the dropped bar has no signal"
    assert signals["signal"].iloc[59] == 1.0, "the last bar of the chunk is scored"
    assert (signals["signal"].iloc[:30] == 1.0).all()
    assert (signals["signal"].iloc[31:60] == 1.0).all()


@pytest.mark.parametrize("missing_from", ["history", "scoring", "both"])
def test_scoring_requires_time_identities_on_both_frames(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    missing_from: str,
) -> None:
    frame = large_ohlcv_df.iloc[:120]
    history = frame.drop(columns="timestamp") if missing_from != "scoring" else frame
    scoring = frame.drop(columns="timestamp") if missing_from != "history" else frame
    calls: list[int] = []
    strategy = _fitted_strategy(history, calls, monkeypatch, retrain_interval=60)

    with pytest.raises(ValueError, match="requires timestamps"):
        strategy.generate_signals(scoring)
    assert calls == []


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "reversed"])
@pytest.mark.parametrize("in_history", [False, True])
def test_scoring_rejects_invalid_time_order(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
    in_history: bool,
) -> None:
    frame = large_ohlcv_df.iloc[:120].copy()
    broken = frame.copy()
    if invalid == "missing":
        broken.loc[10, "timestamp"] = pd.NaT
    elif invalid == "duplicate":
        broken.loc[10, "timestamp"] = broken.loc[9, "timestamp"]
    else:
        broken = broken.iloc[::-1]
    calls: list[int] = []
    strategy = _fitted_strategy(
        broken if in_history else frame, calls, monkeypatch, retrain_interval=60
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        strategy.generate_signals(frame if in_history else broken)
    assert calls == []


@pytest.mark.parametrize("index_history", [False, True])
def test_partial_overlap_uses_only_earlier_history(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    index_history: bool,
) -> None:
    frame = large_ohlcv_df.iloc[:180]
    history = frame.iloc[:120]
    scoring = frame.iloc[60:]
    if index_history:
        history = history.set_index("timestamp")
    else:
        scoring = scoring.set_index("timestamp")
        scoring.index = pd.DatetimeIndex(scoring.index).tz_convert("Asia/Taipei")
    calls: list[int] = []
    strategy = _fitted_strategy(history, calls, monkeypatch, retrain_interval=60)
    windows: list[int] = []
    strategy._feature_eng = _WindowRecordingFeatureEngineer(windows)
    original = scoring.copy(deep=True)

    signals = strategy.generate_signals(scoring)

    assert windows == [120, 120]
    assert calls == [60, 60]
    pd.testing.assert_frame_equal(scoring, original)
    pd.testing.assert_frame_equal(signals[scoring.columns], scoring)


def test_empty_scoring_frame_preserves_schema(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    strategy = _fitted_strategy(large_ohlcv_df, calls, monkeypatch, retrain_interval=60)
    empty = large_ohlcv_df.iloc[:0]
    result = strategy.generate_signals(empty)
    pd.testing.assert_frame_equal(result, empty.assign(signal=0.0, size_weight=1.0))
    assert calls == []


def test_retraining_waits_for_mature_observations(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    strategy = _fitted_strategy(
        large_ohlcv_df, calls, monkeypatch, retrain_interval=200, horizon=200
    )
    trained: list[pd.DataFrame] = []

    def record(frame: pd.DataFrame, *args: object, **kwargs: object) -> object:
        assert not frame.empty, "immature rolling windows must not reach training"
        trained.append(frame.copy())
        return _RecordingModel(calls)

    monkeypatch.setattr(quantile, "_train_regime_classifier", record)
    signals = strategy.generate_signals(large_ohlcv_df)

    assert len(signals) == len(large_ohlcv_df)
    assert sum(calls) == 970
    assert [len(frame) for frame in trained] == [170, 370, 570]


class _SequenceModel:
    """Return a specified sequence across prediction chunks."""

    def __init__(self, probabilities: NDArray[np.float64]) -> None:
        self.probabilities = probabilities
        self.offset = 0

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        start = self.offset
        self.offset += len(features)
        return self.probabilities[start : self.offset]


@pytest.mark.parametrize("gap_start,gap_length", [(30, 1), (59, 1), (60, 60)])
def test_missing_features_flatten_before_rechecking_entry_conditions(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    gap_start: int,
    gap_length: int,
) -> None:
    """A flat candle, chunk tail, or wholly unready chunk cannot retain a position."""
    history = large_ohlcv_df.iloc[:120]
    scored = large_ohlcv_df.iloc[120:300].copy()
    missing_rows = scored.index[gap_start : gap_start + gap_length]
    for column in ("open", "high", "low"):
        scored.loc[missing_rows, column] = scored.loc[missing_rows, "close"]
    original = scored.copy(deep=True)
    strategy = _fitted_strategy(
        history,
        [],
        monkeypatch,
        retrain_interval=60,
        horizon=1000,
        min_holding_bars=1,
        strategy_sl=0.0,
        dynamic_sizing=False,
        train_window=1,
    )
    probabilities = np.zeros((len(scored) - gap_length, 4), dtype=np.float64)
    probabilities[:gap_start, 0] = 1.0
    probabilities[gap_start:, 3] = 1.0
    model = _SequenceModel(probabilities)
    strategy._regime_model = model

    signals = strategy.generate_signals(scored)

    assert model.offset == len(probabilities)
    assert (signals["signal"].iloc[:gap_start] == 1.0).all()
    assert (signals["signal"].iloc[gap_start:] == 0.0).all()
    assert (signals["size_weight"].iloc[gap_start:] == 1.0).all()
    pd.testing.assert_frame_equal(scored, original)
    pd.testing.assert_frame_equal(signals[scored.columns], scored)


@pytest.mark.parametrize("invalid", ["columns", "rows", "nan", "infinity"])
def test_malformed_model_probabilities_are_rejected(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    history = large_ohlcv_df.iloc[:120]
    scored = large_ohlcv_df.iloc[120:180]
    strategy = _fitted_strategy(
        history, [], monkeypatch, retrain_interval=60, train_window=1
    )
    probabilities = np.zeros((len(scored), 4), dtype=np.float64)
    if invalid == "columns":
        probabilities = probabilities[:, :3]
    elif invalid == "rows":
        probabilities = probabilities[:-1]
    else:
        probabilities[0, 0] = np.nan if invalid == "nan" else np.inf
    strategy._regime_model = _SequenceModel(probabilities)

    with pytest.raises(ValueError, match="Regime classifier probabilities"):
        strategy.generate_signals(scored)
