"""Tests for the FLAML AutoML direction model.

The search is driven through a stand-in ``flaml`` module so these tests
assert the contract the wrapper must honour — which columns reach the
search, how ``eval_df`` is forwarded, and what ``predict_proba`` looks
like — without spending a real time budget. One smoke test at the end
does run FLAML for real to prove the wrapper is wired to the library.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.ml.models.flaml_model import (
    DEFAULT_ESTIMATOR_LIST,
    FLAMLAutoMLDirectionModel,
)

if TYPE_CHECKING:
    import numpy.typing as npt

#: Distinguishes "no model was passed" from "the search produced no model",
#: which is the failure mode the wrapper has to detect.
_UNSET = object()


class _FakeEstimator:
    """A leaf estimator that optionally reports per-feature gain."""

    def __init__(self, importances: list[float] | None = None) -> None:
        if importances is not None:
            self.feature_importances_ = importances


class _FakeModel:
    """Stand-in for a FLAML estimator wrapper exposing ``.estimator``."""

    def __init__(self, estimator: _FakeEstimator | None = None) -> None:
        self.estimator = estimator


class _FakeAutoML:
    """Records the ``fit`` call and answers from fixed probabilities."""

    def __init__(
        self,
        *,
        classes: tuple[int, ...] = (0, 1),
        importances: list[float] | None = None,
        model: _FakeModel | None | object = _UNSET,
    ) -> None:
        self.classes_ = classes
        self.best_estimator = "lgbm"
        self.best_loss = 0.1234
        self.model = (
            _FakeModel(_FakeEstimator(importances)) if model is _UNSET else model
        )
        self.fit_calls: list[dict[str, Any]] = []

    # FLAML's own fit() takes X_train/y_train keyword arguments, so the
    # stand-in has to mirror those names even though N803 dislikes them.
    def fit(
        self,
        X_train: pd.DataFrame,  # noqa: N803
        y_train: pd.Series,  # noqa: N803
        **kwargs: Any,
    ) -> None:
        self.fit_calls.append({"X_train": X_train, "y_train": y_train, **kwargs})

    def predict(self, frame: pd.DataFrame) -> npt.NDArray[np.int64]:
        return np.asarray(
            [self.classes_[i % 2] for i in range(len(frame))], dtype=np.int64
        )

    def predict_proba(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.tile([0.25, 0.75], (len(frame), 1))


def _install_fake_flaml(
    monkeypatch: pytest.MonkeyPatch,
    **kwargs: Any,
) -> list[_FakeAutoML]:
    """Put a fake ``flaml`` module in place and return the AutoMLs it makes."""
    created: list[_FakeAutoML] = []
    module = types.ModuleType("flaml")

    def factory() -> _FakeAutoML:
        instance = _FakeAutoML(**kwargs)
        created.append(instance)
        return instance

    monkeypatch.setattr(module, "AutoML", factory, raising=False)
    monkeypatch.setitem(sys.modules, "flaml", module)
    return created


def _frame() -> pd.DataFrame:
    """A feature frame carrying columns that must not become features."""
    n = 6
    return pd.DataFrame(
        {
            "f1": np.arange(n, dtype=float),
            "f2": np.arange(n, dtype=float) * 0.5,
            "target": [0, 1, 0, 1, 0, 1],
            "close": np.linspace(100.0, 101.0, n),
            "dvol": np.full(n, 55.0),
            "timestamp": pd.RangeIndex(n),
        }
    )


def _train(
    monkeypatch: pytest.MonkeyPatch,
    *,
    subject: FLAMLAutoMLDirectionModel | None = None,
    eval_df: pd.DataFrame | None = None,
    **fake_kwargs: Any,
) -> tuple[FLAMLAutoMLDirectionModel, _FakeAutoML]:
    created = _install_fake_flaml(monkeypatch, **fake_kwargs)
    model = subject or FLAMLAutoMLDirectionModel()
    model.train(_frame(), eval_df=eval_df)
    return model, created[0]


class TestTrainContract:
    def test_only_feature_columns_reach_the_search(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, automl = _train(monkeypatch)
        assert list(automl.fit_calls[0]["X_train"].columns) == ["f1", "f2"]

    def test_target_is_passed_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, automl = _train(monkeypatch)
        assert automl.fit_calls[0]["y_train"].tolist() == [0, 1, 0, 1, 0, 1]

    def test_search_is_configured_for_binary_classification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, automl = _train(monkeypatch)
        call = automl.fit_calls[0]
        assert call["task"] == "classification"
        assert call["metric"] == "accuracy"

    def test_default_estimator_list_is_used_when_none_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, automl = _train(monkeypatch)
        assert automl.fit_calls[0]["estimator_list"] == list(DEFAULT_ESTIMATOR_LIST)

    def test_custom_estimator_list_is_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        subject = FLAMLAutoMLDirectionModel(estimator_list=["lgbm", "rf"])
        _, automl = _train(monkeypatch, subject=subject)
        assert automl.fit_calls[0]["estimator_list"] == ["lgbm", "rf"]

    def test_time_budget_is_forwarded_as_seconds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        subject = FLAMLAutoMLDirectionModel(time_budget=30)
        _, automl = _train(monkeypatch, subject=subject)
        assert automl.fit_calls[0]["time_budget"] == 30.0

    def test_validation_set_is_forwarded_when_supplied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        eval_df = _frame()
        _, automl = _train(monkeypatch, eval_df=eval_df)
        call = automl.fit_calls[0]
        assert list(call["X_val"].columns) == ["f1", "f2"]
        assert call["y_val"].tolist() == [0, 1, 0, 1, 0, 1]

    def test_validation_keys_are_omitted_when_no_eval_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, automl = _train(monkeypatch)
        assert "X_val" not in automl.fit_calls[0]
        assert "y_val" not in automl.fit_calls[0]

    def test_optional_search_settings_are_omitted_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        subject = FLAMLAutoMLDirectionModel(n_jobs=None, random_seed=None)
        _, automl = _train(monkeypatch, subject=subject)
        assert "n_jobs" not in automl.fit_calls[0]
        assert "seed" not in automl.fit_calls[0]

    def test_optional_search_settings_are_forwarded_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        subject = FLAMLAutoMLDirectionModel(n_jobs=2, random_seed=7)
        _, automl = _train(monkeypatch, subject=subject)
        call = automl.fit_calls[0]
        assert call["n_jobs"] == 2
        assert call["seed"] == 7

    def test_learner_spills_are_contained_and_removed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # FLAML's CatBoost learner hard-codes a ``catboost_<ts>`` log tree in
        # the current working directory; the search must not let that reach
        # the caller's tree.
        monkeypatch.chdir(tmp_path)
        seen: dict[str, Path] = {}

        class SpillingAutoML(_FakeAutoML):
            def fit(
                self,
                X_train: pd.DataFrame,  # noqa: N803
                y_train: pd.Series,  # noqa: N803
                **kwargs: Any,
            ) -> None:
                seen["cwd"] = Path.cwd()
                (Path.cwd() / "catboost_1.0" / "learn").mkdir(parents=True)
                super().fit(X_train=X_train, y_train=y_train, **kwargs)

        module = types.ModuleType("flaml")
        monkeypatch.setattr(module, "AutoML", SpillingAutoML, raising=False)
        monkeypatch.setitem(sys.modules, "flaml", module)

        FLAMLAutoMLDirectionModel().train(_frame())

        scratch = seen["cwd"]
        assert scratch != tmp_path
        assert scratch.name.startswith("flaml_direction_")
        assert not scratch.exists()
        assert list(tmp_path.iterdir()) == []

    def test_a_search_that_returns_no_model_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `model is None` is what FLAML reports when nothing could be fitted.
        _install_fake_flaml(monkeypatch, model=None)
        with pytest.raises(RuntimeError, match="without a model"):
            FLAMLAutoMLDirectionModel().train(_frame())


class TestPredictContract:
    def test_probability_columns_follow_the_fitted_classes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        subject, _ = _train(monkeypatch)
        np.testing.assert_array_equal(
            subject.predict_proba(_frame()).columns.to_numpy(), [0, 1]
        )

    def test_probability_columns_follow_non_standard_labels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        subject, _ = _train(monkeypatch, classes=(-1, 1))
        np.testing.assert_array_equal(
            subject.predict_proba(_frame()).columns.to_numpy(), [-1, 1]
        )

    def test_probabilities_are_row_aligned_with_the_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        subject, _ = _train(monkeypatch)
        frame = _frame()
        proba = subject.predict_proba(frame)
        assert len(proba) == len(frame)
        assert list(proba.index) == list(frame.index)
        assert proba.iloc[0].tolist() == [0.25, 0.75]

    def test_predict_returns_original_labels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        subject, _ = _train(monkeypatch, classes=(-1, 1))
        predicted = subject.predict(_frame())
        assert predicted.tolist() == [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]

    def test_predict_preserves_the_input_index(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        subject, _ = _train(monkeypatch)
        frame = _frame()
        assert list(subject.predict(frame).index) == list(frame.index)

    def test_predict_before_train_is_rejected(self) -> None:
        with pytest.raises(RuntimeError, match="has not been trained"):
            FLAMLAutoMLDirectionModel().predict(_frame())

    def test_predict_proba_before_train_is_rejected(self) -> None:
        with pytest.raises(RuntimeError, match="has not been trained"):
            FLAMLAutoMLDirectionModel().predict_proba(_frame())


class TestReportedParameters:
    def test_config_is_reported_before_the_search(self) -> None:
        params = FLAMLAutoMLDirectionModel(time_budget=15).get_parameters()
        assert params["time_budget"] == 15.0
        assert params["metric"] == "accuracy"
        assert params["estimator_list"] == list(DEFAULT_ESTIMATOR_LIST)
        assert "selected_estimator" not in params

    def test_the_winner_is_reported_after_the_search(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        subject, _ = _train(monkeypatch)
        assert subject.get_parameters()["selected_estimator"] == "lgbm"

    def test_optional_settings_appear_only_when_set(self) -> None:
        bare = FLAMLAutoMLDirectionModel(n_jobs=None, random_seed=None)
        assert "n_jobs" not in bare.get_parameters()
        assert "random_seed" not in bare.get_parameters()
        full = FLAMLAutoMLDirectionModel(n_jobs=4, random_seed=1).get_parameters()
        assert full["n_jobs"] == 4
        assert full["random_seed"] == 1


class TestFeatureImportance:
    def test_gain_is_projected_and_sorted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        subject, _ = _train(monkeypatch, importances=[1.0, 9.0])
        assert subject.feature_importance() == (("f2", 9.0), ("f1", 1.0))

    def test_no_gain_means_absent_not_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        subject, _ = _train(monkeypatch, model=_FakeModel(_FakeEstimator()))
        assert subject.feature_importance() == ()

    def test_untrained_model_has_no_importance(self) -> None:
        assert FLAMLAutoMLDirectionModel().feature_importance() == ()


def test_real_search_produces_a_usable_model() -> None:
    """Smoke test: the wrapper really drives FLAML, not just the stand-in."""
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "close": np.linspace(100.0, 110.0, n),
        }
    )
    logit = 1.8 * frame["f1"] - 1.2 * frame["f2"]
    frame["target"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    subject = FLAMLAutoMLDirectionModel(
        time_budget=20.0,
        estimator_list=["lgbm", "xgboost", "catboost", "rf", "extra_tree"],
        n_jobs=1,
    )
    cwd = Path.cwd()
    before = set(cwd.iterdir())
    subject.train(frame)
    # CatBoost in the estimator list is exactly the case that used to leave
    # ``catboost_<ts>`` directories behind in the working directory.
    assert set(cwd.iterdir()) == before, sorted(set(cwd.iterdir()) - before)

    proba = subject.predict_proba(frame)
    np.testing.assert_array_equal(proba.columns.to_numpy(), [0, 1])
    assert len(proba) == n
    assert np.allclose(proba.sum(axis=1).to_numpy(), 1.0)
    assert subject.get_parameters()["selected_estimator"]

    predicted = subject.predict(frame)
    assert len(predicted) == n
    assert set(predicted.unique()) <= {0.0, 1.0}
