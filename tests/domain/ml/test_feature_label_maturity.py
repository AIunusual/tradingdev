"""A label may only be used once its outcome lies inside the supplied frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.features.direction_features import DirectionFeatureEngineer
from tradingdev.domain.ml.features.features import FeatureEngineer

_HORIZON = 5
_LOOKBACK = 5
_BARS = 200


def _frame(bars: int) -> pd.DataFrame:
    """Return steadily rising hourly bars, so every realised label is ``1``."""
    index = np.arange(bars)
    close = 100.0 + index
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=bars, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0 + index,
        }
    )


def _engineers(
    horizon: int = _HORIZON,
) -> list[FeatureEngineer | DirectionFeatureEngineer]:
    return [
        FeatureEngineer(lookback=_LOOKBACK, target_horizon=horizon),
        DirectionFeatureEngineer(
            lookback=_LOOKBACK,
            prediction_horizon=horizon,
        ),
    ]


_IDS = ["FeatureEngineer", "DirectionFeatureEngineer"]


@pytest.mark.parametrize(
    "fe,horizon",
    [(fe, horizon) for horizon in (1, _HORIZON) for fe in _engineers(horizon)],
    ids=[f"{name}-horizon-{horizon}" for horizon in (1, _HORIZON) for name in _IDS],
)
def test_a_bar_without_a_realised_outcome_keeps_no_label(
    fe: FeatureEngineer | DirectionFeatureEngineer,
    horizon: int,
) -> None:
    frame = _frame(_BARS)
    labelled = fe.transform(frame, include_target=True)

    last_realised = frame["timestamp"].iloc[-1 - horizon]
    assert not labelled.empty
    assert labelled["timestamp"].max() == last_realised, (
        "the last label needs bars beyond the frame, so that bar cannot be labelled"
    )


@pytest.mark.parametrize("fe", _engineers(), ids=_IDS)
def test_labels_survive_appending_the_bars_they_were_waiting_for(
    fe: FeatureEngineer | DirectionFeatureEngineer,
) -> None:
    before = fe.transform(_frame(_BARS), include_target=True)
    after = fe.transform(_frame(_BARS + _HORIZON), include_target=True)

    shared = before.merge(after, on="timestamp", suffixes=("_before", "_after"))
    assert len(shared) == len(before), "every labelled bar must still be labelled"
    changed = shared.loc[shared["target_before"] != shared["target_after"]]
    assert changed.empty, (
        f"{len(changed)} labels changed once later bars arrived: "
        f"{changed[['timestamp', 'target_before', 'target_after']].to_dict('records')}"
    )


@pytest.mark.parametrize(
    "fe",
    [fe for horizon in (_BARS, _BARS + 1) for fe in _engineers(horizon)],
    ids=[
        f"{name}-horizon-{horizon}" for horizon in (_BARS, _BARS + 1) for name in _IDS
    ],
)
def test_no_training_labels_mature_before_an_overlong_horizon(
    fe: FeatureEngineer | DirectionFeatureEngineer,
) -> None:
    frame = _frame(_BARS)
    labelled = fe.transform(frame, include_target=True)

    assert labelled.empty
    assert "target" in labelled.columns
    assert pd.api.types.is_integer_dtype(labelled["target"])


@pytest.mark.parametrize(
    "fe",
    [fe for horizon in (1, _HORIZON, _BARS + 1) for fe in _engineers(horizon)],
    ids=[
        f"{name}-horizon-{horizon}"
        for horizon in (1, _HORIZON, _BARS + 1)
        for name in _IDS
    ],
)
def test_inference_keeps_the_unlabelled_tail(
    fe: FeatureEngineer | DirectionFeatureEngineer,
) -> None:
    frame = _frame(_BARS)
    original = frame.copy(deep=True)
    features = fe.transform(frame, include_target=False)

    assert not features.empty
    assert "target" not in features.columns
    assert features["timestamp"].iloc[-1] == frame["timestamp"].iloc[-1]
    pd.testing.assert_series_equal(
        features["timestamp"].tail(_HORIZON).reset_index(drop=True),
        frame["timestamp"].tail(_HORIZON).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("fe", _engineers(), ids=_IDS)
def test_missing_interior_outcome_is_not_invented_as_a_flat_or_down_label(
    fe: FeatureEngineer | DirectionFeatureEngineer,
) -> None:
    frame = _frame(_BARS)
    missing_outcome = 150
    frame.loc[missing_outcome, "close"] = np.nan
    original = frame.copy(deep=True)

    labelled = fe.transform(frame, include_target=True)
    inference = fe.transform(frame, include_target=False)

    waiting_bar = frame["timestamp"].iloc[missing_outcome - _HORIZON]
    earlier_bar = frame["timestamp"].iloc[missing_outcome - _HORIZON - 1]
    assert waiting_bar in set(inference["timestamp"])
    assert waiting_bar not in set(labelled["timestamp"])
    assert earlier_bar in set(labelled["timestamp"])
    assert (labelled["target"] == 1).all()
    assert frame["timestamp"].iloc[missing_outcome] not in set(inference["timestamp"])
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("fe", _engineers(), ids=_IDS)
def test_labels_distinguish_realised_up_down_and_flat_outcomes(
    fe: FeatureEngineer | DirectionFeatureEngineer,
) -> None:
    frame = _frame(_BARS)
    # The path keeps moving even when one selected horizon ends at its start,
    # so a flat outcome is not confused with unusable zero-volatility features.
    close = 100.0 + 0.15 * np.arange(_BARS) + 2.0 * np.sin(np.arange(_BARS))
    anchors = (100, 120, 140)
    for anchor, change in zip(anchors, (-2.0, 0.0, 2.0), strict=True):
        close[anchor + _HORIZON] = close[anchor] + change
    frame["close"] = close
    frame["open"] = close + 0.1
    frame["high"] = close + 1.0
    frame["low"] = close - 1.0

    labelled = fe.transform(frame, include_target=True).set_index("timestamp")

    target_timestamps = frame["timestamp"].iloc[np.asarray(anchors, dtype=np.intp)]
    assert target_timestamps.isin(labelled.index).all()
    actual = labelled.loc[target_timestamps, "target"].to_numpy()
    expected = [-1, 0, 1] if isinstance(fe, FeatureEngineer) else [0, 0, 1]
    np.testing.assert_array_equal(actual, expected)
    assert pd.api.types.is_integer_dtype(labelled["target"])
