"""Row layouts and irregular label periods must not bypass temporal validation."""

import pandas as pd
import pytest

from tradingdev.domain.ml.validation.temporal import temporal_folds


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    one = pd.DataFrame(
        {
            "timestamp": dates,
            "label_start": dates,
            "label_end": dates + pd.Timedelta(hours=5),
        }
    )
    return pd.concat(
        [one.assign(symbol=symbol) for symbol in ("BTC", "ETH")], ignore_index=True
    )


@pytest.mark.parametrize("scheme", ["purged_walk_forward", "purged_kfold"])
def test_symbol_major_layout_never_splits_a_date_or_overlaps_labels(
    scheme: str,
) -> None:
    frame = _panel()
    folds = temporal_folds(
        frame, scheme=scheme, horizon=5, n_folds=3, embargo=2, min_train_dates=5
    )
    for fold in folds:
        train, test = (
            frame.iloc[list(fold.train_rows)],
            frame.iloc[list(fold.test_rows)],
        )
        assert set(train.timestamp).isdisjoint(test.timestamp)
        assert set(train.groupby("timestamp").size()) == {2}
        assert set(test.groupby("timestamp").size()) == {2}
        assert (
            (train.label_end < test.label_start.min())
            | (train.label_start > test.label_end.max())
        ).all()
        if scheme == "purged_walk_forward":
            assert train.timestamp.max() < test.timestamp.min()


def test_actual_label_end_overrides_nominal_bar_gap() -> None:
    frame = _panel()
    frame.loc[(frame.symbol == "BTC") & (frame.index == 12), "label_end"] = (
        frame.timestamp.iloc[40]
    )
    folds = temporal_folds(
        frame,
        scheme="purged_walk_forward",
        horizon=5,
        n_folds=3,
        embargo=0,
        min_train_dates=5,
    )
    assert 12 not in folds[0].train_rows
    assert 112 not in folds[0].train_rows
    assert folds[0].purged_rows == 12


def test_rejects_legacy_dataset_without_observation_intervals() -> None:
    with pytest.raises(ValueError, match="rebuild"):
        temporal_folds(
            _panel().drop(columns="label_end"),
            scheme="purged_walk_forward",
            horizon=5,
            n_folds=3,
            embargo=0,
            min_train_dates=5,
        )
