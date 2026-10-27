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


def _engineers() -> list[FeatureEngineer | DirectionFeatureEngineer]:
    return [
        FeatureEngineer(lookback=_LOOKBACK, target_horizon=_HORIZON),
        DirectionFeatureEngineer(
            lookback=_LOOKBACK,
            prediction_horizon=_HORIZON,
        ),
    ]


_IDS = ["FeatureEngineer", "DirectionFeatureEngineer"]


@pytest.mark.parametrize("fe", _engineers(), ids=_IDS)
def test_a_bar_without_a_realised_outcome_keeps_no_label(
    fe: FeatureEngineer | DirectionFeatureEngineer,
) -> None:
    frame = _frame(_BARS)
    labelled = fe.transform(frame, include_target=True)

    last_realised = frame["timestamp"].iloc[-1 - _HORIZON]
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
