"""A prediction context is sized by the features, not by the lookback.

Issue #88: with the bundled default candidate ``lookback=12`` the classic
retrainer sliced ``2 * lookback + 1 = 25`` rows, while the fixed indicators need
34.  Every scored bar therefore came back with an empty feature frame, the model
was never called, and the run reported an all-flat backtest as if the strategy
had decided not to trade.

Expected values here are arithmetic on the feature definitions and on the
fixture's bar offsets, never copied from a run: a bar's features first exist once
it has ``26 + 9 - 1 = 34`` rows behind it (TA-Lib's MACD histogram starts at
``slow + signal - 2``). A preferred seed-decay margin adds a further
``ceil(ln(1e-3) / ln(1 - 2/27)) = 90`` bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.retrainer import RollingRetrainer
from tradingdev.domain.ml.schemas import XGBoostModelConfig
from tradingdev.domain.strategies.bundled.xgboost_strategy.config import (
    XGBoostStrategyConfig,
)
from tradingdev.domain.strategies.bundled.xgboost_strategy.strategy import (
    XGBoostStrategy,
)

_EXISTING_ROWS = 26 + 9 - 1
"""Rows one bar needs before the MACD histogram exists at all."""

_SETTLED_ROWS = _EXISTING_ROWS + 90
"""Preferred rows from the declared seed-decay margin, not an error bound."""

_FIT_BARS = 450
_SCORED_BARS = 50


def _market_frame(rows: int) -> pd.DataFrame:
    """Fixed-seed random walk with the OHLCV columns the features read."""
    rng = np.random.default_rng(42)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, rows)))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.uniform(100.0, 200.0, rows),
        }
    )


class _CountingModel(XGBoostDirectionModel):
    """Record every prediction request and always answer "long"."""

    def __init__(self) -> None:
        super().__init__(XGBoostModelConfig())
        self.frames: list[pd.DataFrame] = []

    def train(
        self,
        df: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
    ) -> None:
        return None

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        self.frames.append(df.copy(deep=True))
        return pd.DataFrame({1: 1.0, -1: 0.0}, index=df.index)


class _RecordingFeatureEngineer(FeatureEngineer):
    """Real features, plus a copy of every scoring context that produced them."""

    def __init__(self, lookback: int) -> None:
        super().__init__(lookback=lookback, target_horizon=1)
        self.contexts: list[pd.DataFrame] = []

    def transform(
        self,
        df: pd.DataFrame,
        *,
        include_target: bool = True,
    ) -> pd.DataFrame:
        if not include_target:
            self.contexts.append(df.copy(deep=True))
        return super().transform(df, include_target=include_target)


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return frame.iloc[:_FIT_BARS].copy(), frame.iloc[_FIT_BARS:].copy()


def test_the_shortest_default_candidate_predicts_on_every_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #88: ``lookback=12`` must not silence the whole scored frame.

    The prediction is counted on the model class, so the model that bar 0's
    retrain produces counts as well; the initial one is not the only predictor.
    """
    frame = _market_frame(_FIT_BARS + _SCORED_BARS)
    history, scored = _split(frame)
    calls: list[list[int]] = []

    def predict_proba(self: XGBoostDirectionModel, df: pd.DataFrame) -> pd.DataFrame:
        calls.append(df.index.tolist())
        return pd.DataFrame({1: 1.0, -1: 0.0}, index=df.index)

    monkeypatch.setattr(XGBoostDirectionModel, "predict_proba", predict_proba)

    signals = RollingRetrainer(
        XGBoostModelConfig(n_estimators=3, n_jobs=1, max_depth=2),
        retrain_interval=10_000,
        threshold=0.55,
        cooldown=1,
        lookback=12,
    ).run(scored, history, _CountingModel(), FeatureEngineer(lookback=12))

    assert len(calls) == _SCORED_BARS
    np.testing.assert_array_equal(signals, np.ones(_SCORED_BARS, dtype=int))


def test_the_scored_row_agrees_with_a_full_history_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #88, criterion 3: the row must match the legitimate history.

    The model is now called, so the next question is *what* it is called with.
    A context cut to the bare warm-up would report a MACD histogram that is off
    by more than a standard deviation on some bars, so this compares the row
    actually handed to the model against the same bar computed from every bar up
    to it, normalised by each column's own spread.
    """
    frame = _market_frame(_FIT_BARS + _SCORED_BARS)
    history, scored = _split(frame)
    requests: list[pd.DataFrame] = []

    def predict_proba(self: XGBoostDirectionModel, df: pd.DataFrame) -> pd.DataFrame:
        requests.append(df.copy(deep=True))
        return pd.DataFrame({1: 1.0, -1: 0.0}, index=df.index)

    monkeypatch.setattr(XGBoostDirectionModel, "predict_proba", predict_proba)
    fe = FeatureEngineer(lookback=12)

    RollingRetrainer(
        XGBoostModelConfig(n_estimators=3, n_jobs=1, max_depth=2),
        retrain_interval=10_000,
        threshold=0.55,
        cooldown=1,
        lookback=12,
    ).run(scored, history, _CountingModel(), fe)

    reference = fe.transform(frame, include_target=False)
    columns = fe.get_feature_names()
    spread = reference[columns].std().to_numpy()
    assert len(requests) == _SCORED_BARS
    for offset, request in enumerate(requests):
        scored_idx = _FIT_BARS + offset
        assert request.index.tolist() == [scored_idx]
        delta = np.abs(
            request[columns].to_numpy(dtype=float)
            - reference.loc[[scored_idx], columns].to_numpy(dtype=float)
        )
        worst = int(np.argmax(delta / spread))
        assert (delta <= 0.01 * spread).all(), (
            f"bar {scored_idx}: {columns[worst]} is "
            f"{delta[worst] / spread[worst]:.2%} of a standard deviation away "
            "from the full-history value"
        )


def test_the_context_covers_the_settling_margin_and_ends_on_the_scored_bar() -> None:
    """``lookback=12`` asks for 25 rows, so the feature requirement decides.

    The slice must be long enough for a settled row and still stop at the bar
    being scored rather than reaching past it.
    """
    frame = _market_frame(_FIT_BARS + _SCORED_BARS)
    history, scored = _split(frame)
    fe = _RecordingFeatureEngineer(lookback=12)

    RollingRetrainer(
        XGBoostModelConfig(),
        retrain_interval=10_000,
        threshold=0.55,
        cooldown=1,
        lookback=12,
    ).run(scored, history, _CountingModel(), fe)

    assert len(fe.contexts) == _SCORED_BARS
    assert all(len(context) == _SETTLED_ROWS for context in fe.contexts)
    for offset, context in enumerate(fe.contexts):
        bar = _FIT_BARS + offset
        assert pd.Timestamp(context["timestamp"].iloc[-1]) == pd.Timestamp(
            frame["timestamp"].iloc[bar]
        ), f"bar {bar}: the context ends on a different bar"
        assert (context["timestamp"] <= frame["timestamp"].iloc[bar]).all()


def test_a_longer_lookback_keeps_the_context_it_already_had() -> None:
    """Raising the floor must not shorten a context that already covered it.

    ``lookback=168`` asked for 337 rows before this change and still does, so
    the indicator seeds, and therefore the signals, of an existing run are
    unchanged.
    """
    frame = _market_frame(_FIT_BARS + _SCORED_BARS)
    history, scored = _split(frame)
    fe = _RecordingFeatureEngineer(lookback=168)

    RollingRetrainer(
        XGBoostModelConfig(),
        retrain_interval=10_000,
        threshold=0.55,
        cooldown=1,
        lookback=168,
    ).run(scored, history, _CountingModel(), fe)

    assert {len(context) for context in fe.contexts} == {2 * 168 + 1}


def test_a_frame_that_never_completes_a_feature_row_fails() -> None:
    """Too little history must raise, not report an all-flat result."""
    frame = _market_frame(30)
    model = _CountingModel()

    with pytest.raises(ValueError, match="scored none of 20 bars") as rejected:
        RollingRetrainer(
            XGBoostModelConfig(),
            retrain_interval=10_000,
            threshold=0.55,
            cooldown=1,
            lookback=12,
        ).run(frame.iloc[10:], frame.iloc[:10], model, FeatureEngineer(lookback=12))

    assert "provide more history" in str(rejected.value)
    assert model.frames == []


@pytest.mark.parametrize("history_rows", [40, 150])
def test_a_complete_history_that_still_yields_no_row_does_not_blame_it(
    history_rows: int,
) -> None:
    """Enough rows is not the same as a complete row.

    Equal closes leave the Bollinger width at zero, which the feature engineer
    replaces with NaN, so no row survives however much history is supplied. The
    message must not send the caller looking for more history it already has,
    even when the available history is shorter than the seed-decay context.
    """
    frame = _market_frame(history_rows + 20)
    for column in ("open", "high", "low", "close"):
        frame[column] = 100.0
    frame["volume"] = 1000.0
    model = _CountingModel()

    with pytest.raises(ValueError, match="incomplete for another reason") as rejected:
        RollingRetrainer(
            XGBoostModelConfig(),
            retrain_interval=10_000,
            threshold=0.55,
            cooldown=1,
            lookback=12,
        ).run(
            frame.iloc[history_rows:],
            frame.iloc[:history_rows],
            model,
            FeatureEngineer(lookback=12),
        )

    assert "provide more history" not in str(rejected.value)
    assert model.frames == []


def test_only_the_bars_that_reach_the_warmup_are_scored() -> None:
    """14 history rows plus 20 scored rows reach the warm-up at the last bar.

    A bar is scorable once the rows from the start of the frame through it reach
    ``_EXISTING_ROWS``: ``14 + i + 1 >= 34`` holds only for ``i == 19``.  The
    settling margin does not change that boundary, only the values.
    """
    history_rows = _EXISTING_ROWS - 20
    frame = _market_frame(history_rows + 20)
    history, scored = frame.iloc[:history_rows].copy(), frame.iloc[history_rows:]
    model = _CountingModel()

    signals = RollingRetrainer(
        XGBoostModelConfig(),
        retrain_interval=10_000,
        threshold=0.55,
        cooldown=1,
        lookback=12,
    ).run(scored, history, model, FeatureEngineer(lookback=12))

    assert len(model.frames) == 1
    assert model.frames[0].index.tolist() == [len(frame) - 1]
    assert signals.tolist() == [0] * 19 + [1]


def test_the_fitted_strategy_calls_the_model_with_the_shortest_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The issue's real-strategy reproduction: ``best_lookback=12`` still scores."""
    frame = _market_frame(_FIT_BARS + _SCORED_BARS)
    history, scored = _split(frame)
    calls: list[list[int]] = []

    def predict_proba(self: XGBoostDirectionModel, df: pd.DataFrame) -> pd.DataFrame:
        calls.append(df.index.tolist())
        return pd.DataFrame({1: 1.0, -1: 0.0}, index=df.index)

    monkeypatch.setattr(XGBoostDirectionModel, "predict_proba", predict_proba)
    strategy = XGBoostStrategy(
        XGBoostStrategyConfig(
            model=XGBoostModelConfig(n_estimators=3, n_jobs=1, max_depth=2),
            lookback_candidates=[12],
            retrain_interval=10_000,
        )
    )

    strategy.fit(history)
    result = strategy.generate_signals(scored)

    assert strategy.get_parameters()["best_lookback"] == 12
    assert len(calls) == _SCORED_BARS
    assert result["signal"].tolist() == [1] * _SCORED_BARS
