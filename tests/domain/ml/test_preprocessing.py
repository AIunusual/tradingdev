"""Preprocessing learns from the training segment and nowhere else."""

from __future__ import annotations

import pandas as pd
import pytest

from tradingdev.domain.ml.preprocessing import (
    ClipOutliers,
    DropnaRows,
    Fillna,
    MinMaxNorm,
    ProcessorNotFittedError,
    ProcessorPipeline,
    RobustZScoreNorm,
    ZScoreNorm,
    fit_transform_split,
)
from tradingdev.domain.ml.validation import (
    Split,
    purged_walk_forward_splits,
)

_TRAIN_VALUES = [0.0, 2.0, 4.0, 6.0, 8.0]
_TEST_VALUES = [100.0, 200.0]


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"x": _TRAIN_VALUES + _TEST_VALUES})


def test_zscore_uses_training_statistics_only() -> None:
    frame = _frame()
    train = frame.iloc[:5]
    pipeline = ProcessorPipeline([ZScoreNorm(fields=("x",))]).fit(train)

    transformed = pipeline.transform(frame)
    expected = (frame["x"] - train["x"].mean()) / train["x"].std(ddof=0)
    assert transformed["x"].tolist() == pytest.approx(expected.tolist())


def test_fitting_on_the_full_frame_would_give_a_different_answer() -> None:
    frame = _frame()
    honest = ProcessorPipeline([ZScoreNorm(fields=("x",))]).fit(frame.iloc[:5])
    leaky = ProcessorPipeline([ZScoreNorm(fields=("x",))]).fit(frame)

    last_bar = len(frame) - 1
    assert honest.transform(frame)["x"].iloc[last_bar] > 60.0
    assert leaky.transform(frame)["x"].iloc[last_bar] == pytest.approx(2.17, abs=0.01)


def test_transform_before_fit_is_refused() -> None:
    pipeline = ProcessorPipeline([ZScoreNorm(fields=("x",))])
    with pytest.raises(ProcessorNotFittedError, match="must precede"):
        pipeline.transform(_frame())


def test_a_lone_stateful_processor_also_refuses() -> None:
    """Defence in depth: the guard is not only on the pipeline."""
    with pytest.raises(ProcessorNotFittedError, match="must precede"):
        ZScoreNorm(fields=("x",)).transform(_frame())


def test_fit_returns_the_pipeline_for_chaining() -> None:
    pipeline = ProcessorPipeline([ZScoreNorm(fields=("x",))])
    assert pipeline.fit(_frame().iloc[:5]) is pipeline
    assert pipeline.is_fitted
    assert pipeline.fit_rows == 5


def test_refitting_on_a_wider_segment_changes_the_output() -> None:
    frame = _frame()
    pipeline = ProcessorPipeline([ZScoreNorm(fields=("x",))])

    pipeline.fit(frame.iloc[:5])
    narrow_signature = pipeline.fit_signature
    pipeline.fit(frame)

    assert pipeline.fit_rows == len(frame)
    assert pipeline.fit_signature != narrow_signature


def test_stateless_processors_learn_nothing() -> None:
    frame = pd.DataFrame({"x": [1.0, float("nan"), 3.0]})
    pipeline = ProcessorPipeline([Fillna(fields=("x",), value=0.0)]).fit(frame)
    states = pipeline.describe()

    assert [state.name for state in states] == ["fillna"]
    assert states[0].requires_fit is False
    assert states[0].learned == {}
    assert pipeline.transform(frame)["x"].tolist() == [1.0, 0.0, 3.0]


def test_fillna_respects_the_requested_value() -> None:
    frame = pd.DataFrame({"x": [float("nan"), 2.0], "y": [float("nan"), 4.0]})
    out = Fillna(value=-1.0).transform(frame)
    assert out["x"].tolist() == [-1.0, 2.0]
    assert out["y"].tolist() == [-1.0, 4.0]


def test_dropna_rows_removes_incomplete_rows() -> None:
    frame = pd.DataFrame({"x": [1.0, float("nan"), 3.0], "y": [1.0, 2.0, 3.0]})
    out = DropnaRows(fields=("x",)).transform(frame)
    assert out["x"].tolist() == [1.0, 3.0]
    assert len(out) == 2


def test_ordered_pipeline_fills_before_it_scales() -> None:
    frame = pd.DataFrame({"x": [1.0, float("nan"), 3.0, 5.0]})
    pipeline = ProcessorPipeline([Fillna(fields=("x",)), ZScoreNorm(fields=("x",))])
    out = pipeline.fit_transform(frame)
    assert out["x"].isna().sum() == 0
    assert out["x"].mean() == pytest.approx(0.0, abs=1e-9)


def test_minmax_maps_the_training_range_onto_the_requested_range() -> None:
    frame = _frame()
    pipeline = ProcessorPipeline([MinMaxNorm(fields=("x",))]).fit(frame.iloc[:5])
    out = pipeline.transform(frame)

    assert out["x"].iloc[0] == pytest.approx(0.0)
    assert out["x"].iloc[4] == pytest.approx(1.0)
    assert out["x"].iloc[5] == pytest.approx(12.5)
    assert out["x"].iloc[6] == pytest.approx(25.0)


def test_minmax_honours_a_custom_range() -> None:
    frame = _frame()
    pipeline = ProcessorPipeline(
        [MinMaxNorm(fields=("x",), feature_range=(-1.0, 1.0))]
    ).fit(frame.iloc[:5])
    out = pipeline.transform(frame)
    assert out["x"].iloc[0] == pytest.approx(-1.0)
    assert out["x"].iloc[4] == pytest.approx(1.0)


def test_robust_zscore_is_not_dominated_by_a_training_outlier() -> None:
    train = pd.DataFrame({"x": [0.0, 2.0, 4.0, 6.0, 8.0, 1000.0]})
    robust = ProcessorPipeline([RobustZScoreNorm(fields=("x",))]).fit(train)
    plain = ProcessorPipeline([ZScoreNorm(fields=("x",))]).fit(train)

    robust_out = robust.transform(train)
    plain_out = plain.transform(train)

    # The outlier survives robust scaling and is crushed by the plain z-score.
    assert robust_out["x"].iloc[5] > 100.0
    assert plain_out["x"].iloc[5] < 5.0
    # Ordinary values stay comparable instead of collapsing toward the mean.
    assert robust_out["x"].iloc[4] > plain_out["x"].iloc[4]


def test_clip_outliers_uses_training_bounds() -> None:
    train = pd.DataFrame({"x": [0.0, 2.0, 4.0, 6.0, 8.0, 1000.0]})
    pipeline = ProcessorPipeline([ClipOutliers(fields=("x",), n_sigmas=1.0)])
    pipeline.fit(train)

    bound = train["x"].mean() + train["x"].std(ddof=0)
    out = pipeline.transform(train)
    assert bound < 1000.0
    assert out["x"].max() == pytest.approx(bound)
    assert out["x"].iloc[0] == pytest.approx(0.0)


def test_a_constant_column_does_not_explode() -> None:
    frame = pd.DataFrame({"x": [5.0] * 6})
    pipeline = ProcessorPipeline([ZScoreNorm(fields=("x",))]).fit(frame.iloc[:4])
    out = pipeline.transform(frame)
    assert out["x"].tolist() == pytest.approx([0.0] * 6)


def test_missing_fields_are_rejected_at_fit_time() -> None:
    with pytest.raises(KeyError, match="nope"):
        ZScoreNorm(fields=("nope",)).fit(_frame())


def test_missing_fields_are_rejected_at_transform_time() -> None:
    pipeline = ProcessorPipeline([ZScoreNorm(fields=("x",))]).fit(_frame().iloc[:5])
    with pytest.raises(KeyError, match="x"):
        pipeline.transform(pd.DataFrame({"y": [1.0]}))


def test_fitting_on_an_empty_frame_is_refused() -> None:
    pipeline = ProcessorPipeline([ZScoreNorm(fields=("x",))])
    with pytest.raises(ValueError, match="empty frame"):
        pipeline.fit(_frame().iloc[0:0])


def test_describe_reports_what_was_learned() -> None:
    pipeline = ProcessorPipeline([Fillna(fields=("x",)), ZScoreNorm(fields=("x",))])
    pipeline.fit(_frame().iloc[:5])
    states = pipeline.describe()

    assert [state.name for state in states] == ["fillna", "zscore_norm"]
    assert [state.requires_fit for state in states] == [False, True]
    assert states[1].fields == ("x",)
    assert states[1].learned["x__mean"] == pytest.approx(4.0)
    assert states[1].learned["x__std"] == pytest.approx(8.0**0.5)


def test_processors_never_mutate_their_input() -> None:
    frame = _frame()
    before = frame.copy()
    ProcessorPipeline([ZScoreNorm(fields=("x",))]).fit_transform(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_fit_transform_matches_fit_then_transform() -> None:
    frame = _frame().iloc[:5]
    chained = ProcessorPipeline([ZScoreNorm(fields=("x",))]).fit_transform(frame)
    stepped = ProcessorPipeline([ZScoreNorm(fields=("x",))])
    stepped.fit(frame)
    pd.testing.assert_frame_equal(chained, stepped.transform(frame))


def test_fit_transform_split_scales_the_test_side_with_training_statistics() -> None:
    frame = pd.DataFrame({"x": [float(index) for index in range(80)]})
    split = Split(name="fold/0", train=((0, 60),), test=(60, 80))
    pipeline = ProcessorPipeline([ZScoreNorm(fields=("x",))])

    train_out, test_out = fit_transform_split(frame, split, pipeline)

    assert len(train_out) == 60
    assert len(test_out) == 20
    assert train_out["x"].mean() == pytest.approx(0.0, abs=1e-9)
    assert train_out["x"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)
    assert test_out["x"].mean() > 1.0


def test_pipeline_is_reusable_across_walk_forward_folds() -> None:
    frame = pd.DataFrame({"x": [float(index) for index in range(200)]})
    pipeline = ProcessorPipeline([ZScoreNorm(fields=("x",))])

    for split in purged_walk_forward_splits(200, horizon=5, n_splits=3):
        train_out, test_out = fit_transform_split(frame, split, pipeline)
        assert pipeline.fit_rows == split.train_bars
        assert len(train_out) == split.train_bars
        assert len(test_out) == split.test_bars
        assert train_out["x"].mean() == pytest.approx(0.0, abs=1e-9)
        assert train_out["x"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_kfold_train_windows_on_both_sides_are_concatenated() -> None:
    frame = pd.DataFrame({"x": [float(index) for index in range(60)]})
    split = Split(name="fold/1", train=((0, 20), (40, 60)), test=(20, 40))
    pipeline = ProcessorPipeline([ZScoreNorm(fields=("x",))])

    train_out, test_out = fit_transform_split(frame, split, pipeline)

    assert list(train_out.index) == [*range(20), *range(40, 60)]
    assert len(test_out) == 20
    assert pipeline.fit_rows == 40
