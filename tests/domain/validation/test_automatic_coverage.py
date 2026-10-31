"""Automatic folds cover every row after the initial training window."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from tradingdev.domain.validation.config import AutomaticWalkForwardConfig
from tradingdev.domain.validation.date_windows import WalkForwardSplitError
from tradingdev.domain.validation.splitter import DataSplitter


@pytest.mark.parametrize("expanding", [False, True])
def test_hundred_rows_reach_the_last_observation(expanding: bool) -> None:
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=100, tz="UTC")}
    )
    splitter = DataSplitter(
        AutomaticWalkForwardConfig(
            mode="automatic",
            n_splits=5,
            train_ratio=0.8,
            expanding=expanding,
        )
    )
    windows = splitter.split_windows(frame)
    assert [test for _, test in windows] == [
        (44, 55),
        (55, 66),
        (66, 77),
        (77, 88),
        (88, 100),
    ]
    assert [train for train, _ in windows] == (
        [(0, 44), (0, 55), (0, 66), (0, 77), (0, 88)]
        if expanding
        else [(0, 44), (11, 55), (22, 66), (33, 77), (44, 88)]
    )
    splits = splitter.split(frame)
    evaluated = [index for _, test in splits for index in test.index]
    assert evaluated == list(range(44, 100))
    for (train, test), (train_window, test_window) in zip(splits, windows, strict=True):
        assert train.timestamp.iloc[-1] < test.timestamp.iloc[0]
        pd.testing.assert_frame_equal(train, frame.iloc[slice(*train_window)])
        pd.testing.assert_frame_equal(test, frame.iloc[slice(*test_window)])


@pytest.mark.parametrize(
    ("rows", "folds", "ratio", "initial_train"),
    [
        (13, 3, 0.2, 1),
        (26, 3, 0.2, 2),
        (13, 2, 0.7, 7),
        (26, 3, math.nextafter(0.2, 0.0), 1),
        (26, 3, math.nextafter(0.2, 1.0), 2),
    ],
)
def test_decimal_ratio_integer_boundaries_are_floored_exactly(
    rows: int, folds: int, ratio: float, initial_train: int
) -> None:
    windows = DataSplitter(
        AutomaticWalkForwardConfig(mode="automatic", n_splits=folds, train_ratio=ratio)
    ).split_windows(pd.DataFrame({"row": range(rows)}))
    assert windows[0][0] == (0, initial_train)
    assert windows[-1][1][1] == rows


@pytest.mark.parametrize("expanding", [False, True])
@pytest.mark.parametrize(
    ("rows", "folds", "ratio"),
    [(103, 5, 0.8), (11, 1, 0.8), (6, 2, 0.5), (100, 5, 0.05), (100, 2, 0.95)],
)
def test_rounding_preserves_nonempty_contiguous_test_coverage(
    rows: int,
    folds: int,
    ratio: float,
    expanding: bool,
) -> None:
    frame = pd.DataFrame({"row": range(rows)})
    windows = DataSplitter(
        AutomaticWalkForwardConfig(
            mode="automatic",
            n_splits=folds,
            train_ratio=ratio,
            expanding=expanding,
        )
    ).split_windows(frame)
    assert len(windows) == folds
    assert windows[-1][1][1] == rows
    start = windows[0][1][0]
    assert [row for _, (a, b) in windows for row in range(a, b)] == list(
        range(start, rows)
    )
    widths = [b - a for _, (a, b) in windows]
    assert min(widths) >= 1 and max(widths) - min(widths) <= 1
    for (a, b), (c, d) in windows:
        assert 0 <= a < b == c < d <= rows
        assert a == 0 if expanding else b - a == start


@pytest.mark.parametrize(
    ("rows", "folds", "ratio"),
    [(0, 1, 0.8), (1, 1, 0.8), (3, 3, 0.8), (100, 5, 0.999), (100, 5, 0.0001)],
)
def test_impossible_geometry_fails_instead_of_omitting_folds(
    rows: int, folds: int, ratio: float
) -> None:
    with pytest.raises(WalkForwardSplitError, match="nonempty train and test"):
        DataSplitter(
            AutomaticWalkForwardConfig(
                mode="automatic",
                n_splits=folds,
                train_ratio=ratio,
            )
        ).split_windows(pd.DataFrame({"row": range(rows)}))
