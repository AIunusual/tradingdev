"""Regression tests for the train-replay context (issue #24 follow-up 1).

``walk_forward`` calls ``fit(train_df)`` and then ``generate_signals(train_df)``
to report training metrics. ``generate_signals`` prepends ``self._train_df`` to
the frame it scores, so that call concatenated the training window with itself:
the 60-bar prediction context spanned both copies, the timestamp filter kept the
duplicate rows, and the state machine started writing at bar 0 while reading
from bar 940.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from tradingdev.domain.ml.features.quantile_features import QuantileFeatureEngineer
from tradingdev.domain.strategies.bundled.quantile_strategy import strategy as quantile
from tradingdev.domain.strategies.bundled.quantile_strategy.config import (
    QuantileStrategyConfig,
)

if TYPE_CHECKING:
    import pytest
    import xgboost as xgb

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

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
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
        preserve_index: bool = False,
    ) -> pd.DataFrame:
        out = pd.DataFrame(
            {"close": frame["close"].astype(float), "f1": 1.0},
            index=frame.index,
        )
        out = out.drop(index=self.drop_position, errors="ignore")
        if not preserve_index:
            out = out.reset_index(drop=True)
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
        preserve_index: bool = False,
    ) -> pd.DataFrame:
        self.windows.append(len(frame))
        return super().transform(
            frame,
            include_target=include_target,
            target_type=target_type,
            preserve_index=preserve_index,
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
    strategy._regime_model = cast("xgb.XGBClassifier", model)  # noqa: SLF001
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


def test_a_frame_without_timestamps_keeps_the_whole_training_frame(
    large_ohlcv_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented fallback when nothing can be matched or ordered.

    `docs/strategy_contract.md` says a frame with neither a `timestamp` column
    nor a `DatetimeIndex` cannot be matched against the training history, so the
    whole training frame becomes context. The first prediction window therefore
    holds one chunk of history plus one chunk of scored bars. With timestamps on
    the same overlapping frame the history trims to nothing and that window
    holds a single chunk, which the other tests in this module cover.
    """
    retrain_interval = 60
    frame = large_ohlcv_df.iloc[:120].drop(columns=["timestamp"])
    calls: list[int] = []
    strategy = _fitted_strategy(
        frame,
        calls,
        monkeypatch,
        retrain_interval=retrain_interval,
        train_window=1,
    )
    windows: list[int] = []
    strategy._feature_eng = _WindowRecordingFeatureEngineer(windows)  # noqa: SLF001

    strategy.generate_signals(frame)

    assert windows[0] == 2 * retrain_interval, (
        "without timestamps the training frame is not trimmed, so the first "
        "prediction window carries a full chunk of history before the chunk"
    )
