"""Regression tests for explicit walk-forward date validation (issue #24).

The explicit split used to build two independent inclusive-day masks, so a
train window ending on the day the test window started trained on bars the
test set also scored. Two layers now prevent that: ``WalkForwardConfig``
rejects overlapping or reversed periods, and ``DataSplitter`` re-checks the
series that actually reaches ``fit()``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine
from tradingdev.domain.strategies.bundled.kd_strategy.config import KDStrategyConfig
from tradingdev.domain.strategies.bundled.kd_strategy.strategy import KDStrategy
from tradingdev.domain.validation.config import (
    AutomaticWalkForwardConfig,
    ExplicitWalkForwardConfig,
    PurgedWalkForwardConfig,
)
from tradingdev.domain.validation.date_windows import (
    WalkForwardSplitError,
    inclusive_day_window,
)
from tradingdev.domain.validation.splitter import DataSplitter
from tradingdev.domain.validation.walk_forward import WalkForwardValidator


def _dates(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401
    """A valid explicit window, with ``overrides`` applied on top."""
    values: dict[str, Any] = {
        "train_start": "2024-01-01",
        "train_end": "2024-01-10",
        "test_start": "2024-01-11",
        "test_end": "2024-01-20",
    }
    values.update(overrides)
    return values


class _FitRecorder(KDStrategy):
    """KD strategy that remembers the timestamps ``fit()`` was given."""

    def __init__(self) -> None:
        super().__init__(config=KDStrategyConfig())
        self.fitted: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> None:
        self.fitted = df["timestamp"].copy()
        super().fit(df)


class TestConfigRejectsLeakingDates:
    def test_adjacent_days_are_accepted(self) -> None:
        config = ExplicitWalkForwardConfig(mode="explicit", **_dates())
        assert config.train_end is not None
        assert config.test_start is not None
        assert config.train_end.date() < config.test_start.date()

    def test_shared_inclusive_end_day_is_rejected(self) -> None:
        """The issue's probe: train_end == test_start shares a day of bars."""
        with pytest.raises(ValidationError, match="precede the test window"):
            ExplicitWalkForwardConfig(
                mode="explicit",
                **_dates(train_end="2024-01-11", test_start="2024-01-11"),
            )

    def test_train_period_ending_inside_the_test_period_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="precede the test window"):
            ExplicitWalkForwardConfig(
                mode="explicit",
                **_dates(train_end="2024-01-15", test_start="2024-01-11"),
            )

    def test_reversed_train_period_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="precede the test window"):
            ExplicitWalkForwardConfig(
                mode="explicit",
                **_dates(train_start="2024-01-10", train_end="2024-01-01"),
            )

    def test_reversed_test_period_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="precede the test window"):
            ExplicitWalkForwardConfig(
                mode="explicit",
                **_dates(test_start="2024-01-20", test_end="2024-01-11"),
            )

    def test_partial_dates_are_rejected_instead_of_falling_back(self) -> None:
        with pytest.raises(ValidationError, match="Field required"):
            ExplicitWalkForwardConfig.model_validate(
                {
                    "mode": "explicit",
                    "train_start": "2024-01-01",
                    "train_end": "2024-01-10",
                }
            )

    def test_no_dates_still_selects_the_automatic_split(self) -> None:
        config = AutomaticWalkForwardConfig(
            mode="automatic", n_splits=2, train_ratio=0.7
        )
        assert config.mode == "automatic"

    def test_non_utc_timestamps_are_compared_as_utc_days(self) -> None:
        """03:00 at +08:00 is the previous UTC day, so the days still overlap."""
        with pytest.raises(ValidationError, match="precede the test window"):
            ExplicitWalkForwardConfig(
                mode="explicit",
                **_dates(
                    train_end=datetime.fromisoformat("2024-01-11T20:00:00+00:00"),
                    test_start=datetime.fromisoformat("2024-01-12T03:00:00+08:00"),
                ),
            )


@pytest.mark.parametrize(
    "dates",
    [
        {"train_start": "2024-01-01"},
        _dates(train_end="2024-01-15"),
        _dates(train_start="2024-01-10", train_end="2024-01-01"),
        _dates(test_start="2024-01-20", test_end="2024-01-11"),
    ],
    ids=["partial", "overlapping", "reversed_train", "reversed_test"],
)
def test_purged_dates_are_rejected(dates: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        PurgedWalkForwardConfig(
            mode="purged", label_horizon=5, embargo_bars=2, n_splits=2, **dates
        )


class TestExplicitSplitGeometry:
    def test_offset_dates_select_the_full_utc_calendar_days(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        config = ExplicitWalkForwardConfig(
            mode="explicit",
            train_start="2024-01-02T03:00:00+08:00",
            train_end="2024-01-03T03:00:00+08:00",
            test_start="2024-01-04T03:00:00+08:00",
            test_end="2024-01-05T03:00:00+08:00",
        )
        splitter = DataSplitter(config)
        train, test = splitter.split(walk_forward_ohlcv_df)[0]
        assert train.equals(walk_forward_ohlcv_df.iloc[:48])
        assert test.equals(walk_forward_ohlcv_df.iloc[48:96])
        assert splitter.split_windows(walk_forward_ohlcv_df) == [((0, 48), (48, 96))]

    def test_end_day_without_a_representable_upper_bound_is_rejected(self) -> None:
        with pytest.raises(WalkForwardSplitError, match="following UTC day"):
            inclusive_day_window(datetime(9999, 12, 30), datetime(9999, 12, 31))

    def test_train_window_covers_the_whole_end_day(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        config = ExplicitWalkForwardConfig(mode="explicit", **_dates())
        train_df, test_df = DataSplitter(config).split(walk_forward_ohlcv_df)[0]
        last_train = train_df["timestamp"].max()
        assert last_train.date() == datetime(2024, 1, 10).date()
        assert last_train.hour == 23
        assert last_train < test_df["timestamp"].min()

    def test_split_halves_are_disjoint_and_adjacent(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        config = ExplicitWalkForwardConfig(mode="explicit", **_dates())
        train_df, test_df = DataSplitter(config).split(walk_forward_ohlcv_df)[0]
        assert set(train_df["timestamp"]).isdisjoint(test_df["timestamp"])
        gap = test_df["timestamp"].min() - train_df["timestamp"].max()
        assert gap == pd.Timedelta(hours=1)

    def test_split_and_split_windows_describe_the_same_rows(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        splitter = DataSplitter(ExplicitWalkForwardConfig(mode="explicit", **_dates()))
        train_df, test_df = splitter.split(walk_forward_ohlcv_df)[0]
        windows = splitter.split_windows(walk_forward_ohlcv_df)
        assert len(windows) == 1
        train_win, test_win = windows[0]
        assert walk_forward_ohlcv_df.iloc[train_win[0] : train_win[1]].equals(train_df)
        assert walk_forward_ohlcv_df.iloc[test_win[0] : test_win[1]].equals(test_df)

    def test_overlap_is_rejected_even_without_config_validation(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        """``model_construct`` bypasses the validator; the splitter must not."""
        ts = walk_forward_ohlcv_df["timestamp"]
        boundary = ts.iloc[len(ts) // 2]
        config = ExplicitWalkForwardConfig.model_construct(
            mode="explicit",
            train_start=ts.iloc[0].to_pydatetime(),
            train_end=boundary.to_pydatetime(),
            test_start=boundary.to_pydatetime(),
            test_end=ts.iloc[-1].to_pydatetime(),
        )
        with pytest.raises(WalkForwardSplitError, match="overlap"):
            DataSplitter(config).split(walk_forward_ohlcv_df)

    def test_window_matching_no_rows_is_reported(
        self, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        config = ExplicitWalkForwardConfig(
            mode="explicit",
            train_start="2030-01-01",
            train_end="2030-01-10",
            test_start="2030-01-11",
            test_end="2030-01-20",
        )
        with pytest.raises(WalkForwardSplitError, match="select no train rows"):
            DataSplitter(config).split(sample_ohlcv_df)

    def test_unsorted_frame_is_reported(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        with pytest.raises(WalkForwardSplitError, match="sorted ascending"):
            DataSplitter(ExplicitWalkForwardConfig(mode="explicit", **_dates())).split(
                walk_forward_ohlcv_df.iloc[::-1]
            )


class TestValidatorTrainsOnPastOnly:
    def test_fit_receives_only_the_train_window(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        config = ExplicitWalkForwardConfig(mode="explicit", **_dates())
        strategy = _FitRecorder()
        engine = SignalBacktestEngine(init_cash=10_000.0, fees=0.0, slippage=0.0)
        WalkForwardValidator(config=config, engine=engine).validate(
            strategy, walk_forward_ohlcv_df
        )
        train_df, test_df = DataSplitter(config).split(walk_forward_ohlcv_df)[0]
        assert strategy.fitted is not None
        assert strategy.fitted.equals(train_df["timestamp"])
        assert set(strategy.fitted).isdisjoint(test_df["timestamp"])

    def test_fold_metadata_does_not_overlap(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        config = ExplicitWalkForwardConfig(mode="explicit", **_dates())
        engine = SignalBacktestEngine(init_cash=10_000.0, fees=0.0, slippage=0.0)
        results = WalkForwardValidator(config=config, engine=engine).validate(
            _FitRecorder(), walk_forward_ohlcv_df
        )
        assert len(results) == 1
        assert results[0].train_end < results[0].test_start
