"""Tests for the FLAML AutoML direction model.

The search is driven through a stand-in ``flaml`` module so these tests
assert the contract the wrapper must honour — which columns reach the
search, how ``eval_df`` is forwarded, and what ``predict_proba`` looks
like — without spending a real time budget. Real learner tests at the end
exercise holdout training and prediction, including CatBoost's file output
settings.
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path
from queue import SimpleQueue
from threading import Barrier, Event, Thread
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
    *,
    automl_type: type[_FakeAutoML] = _FakeAutoML,
    **kwargs: Any,
) -> list[_FakeAutoML]:
    """Put a fake ``flaml`` module in place and return the AutoMLs it makes."""
    created: list[_FakeAutoML] = []
    module = types.ModuleType("flaml")

    def factory() -> _FakeAutoML:
        instance = automl_type(**kwargs)
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


@pytest.fixture
def isolated_search_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Contain both working-directory output and library temporary files."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


class TestTrainContract:
    @pytest.mark.parametrize("empty_input", ["training", "validation"])
    @pytest.mark.parametrize("already_trained", [False, True])
    def test_empty_training_inputs_preserve_model_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        empty_input: str,
        already_trained: bool,
    ) -> None:
        created = _install_fake_flaml(monkeypatch)
        model = FLAMLAutoMLDirectionModel()
        before_predictions: pd.DataFrame | None = None
        if already_trained:
            model.train(_frame())
            before_predictions = model.predict_proba(_frame())
        before_automl = model._automl
        before_features = model._feature_names.copy()
        before_classes = model._classes
        before_searches = len(created)
        changed = _frame().rename(columns={"f1": "replacement_feature"})
        train = changed.iloc[:0] if empty_input == "training" else changed
        validation = changed.iloc[:0] if empty_input == "validation" else None

        with pytest.raises(
            ValueError, match=f"{empty_input.title()} data.*labelled row"
        ):
            model.train(train, eval_df=validation)

        assert len(created) == before_searches
        assert model._automl is before_automl
        assert model._feature_names == before_features
        assert model._classes == before_classes
        if before_predictions is not None:
            pd.testing.assert_frame_equal(
                model.predict_proba(_frame()), before_predictions
            )
        else:
            with pytest.raises(RuntimeError, match="not been trained"):
                model.predict_proba(_frame())

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

    @pytest.mark.parametrize("outcome", ["success", "exception", "timeout", "no_model"])
    def test_search_outcomes_leave_no_files_or_directory_changes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolated_search_directory: Path,
        outcome: str,
    ) -> None:
        class OutcomeAutoML(_FakeAutoML):
            def fit(
                self,
                X_train: pd.DataFrame,  # noqa: N803
                y_train: pd.Series,  # noqa: N803
                **kwargs: Any,
            ) -> None:
                assert Path.cwd() == isolated_search_directory
                assert list(isolated_search_directory.iterdir()) == []
                assert kwargs["log_file_name"] == ""
                assert (
                    kwargs["custom_hp"]["catboost"]["allow_writing_files"]["domain"]
                    is False
                )
                super().fit(X_train=X_train, y_train=y_train, **kwargs)
                if outcome == "exception":
                    raise ValueError("search failed")
                if outcome == "timeout":
                    raise TimeoutError("search timed out")
                if outcome == "no_model":
                    self.model = None

        _install_fake_flaml(monkeypatch, automl_type=OutcomeAutoML)
        subject = FLAMLAutoMLDirectionModel()
        if outcome == "success":
            subject.train(_frame())
        else:
            error_type, message = {
                "exception": (ValueError, "search failed"),
                "timeout": (TimeoutError, "search timed out"),
                "no_model": (RuntimeError, "without a model"),
            }[outcome]
            with pytest.raises(error_type, match=message) as caught:
                subject.train(_frame())
            # Keep the traceback alive: cleanup must not depend on garbage
            # collection releasing the failed train() frame.
            assert caught.traceback
        assert Path.cwd() == isolated_search_directory
        assert list(isolated_search_directory.iterdir()) == []

    @pytest.mark.parametrize(
        "missing_column", ["training_target", "validation_feature", "validation_target"]
    )
    def test_data_preparation_failure_leaves_no_temporary_files(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolated_search_directory: Path,
        missing_column: str,
    ) -> None:
        created = _install_fake_flaml(monkeypatch)
        training = _frame()
        validation = _frame()
        if missing_column == "training_target":
            training = training.drop(columns=["target"])
        elif missing_column == "validation_feature":
            validation = validation.drop(columns=["f1"])
        else:
            validation = validation.drop(columns=["target"])

        with pytest.raises(KeyError) as caught:
            FLAMLAutoMLDirectionModel().train(training, eval_df=validation)

        assert caught.traceback
        assert all(not automl.fit_calls for automl in created)
        assert Path.cwd() == isolated_search_directory
        assert list(isolated_search_directory.iterdir()) == []

    def test_concurrent_searches_preserve_caller_relative_file_access(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolated_search_directory: Path,
    ) -> None:
        entered_fit = Barrier(3, timeout=5)
        finish_fit = Event()
        failures: SimpleQueue[BaseException] = SimpleQueue()
        caller_file = isolated_search_directory / "caller.txt"
        caller_file.write_text("before training", encoding="utf-8")

        class ConcurrentAutoML(_FakeAutoML):
            def fit(
                self,
                X_train: pd.DataFrame,  # noqa: N803
                y_train: pd.Series,  # noqa: N803
                **kwargs: Any,
            ) -> None:
                entered_fit.wait()
                assert finish_fit.wait(timeout=5), "Caller did not release training"
                assert Path.cwd() == isolated_search_directory
                assert (
                    Path("caller.txt").read_text(encoding="utf-8") == "during training"
                )
                super().fit(X_train=X_train, y_train=y_train, **kwargs)

        _install_fake_flaml(monkeypatch, automl_type=ConcurrentAutoML)
        subjects = [FLAMLAutoMLDirectionModel(), FLAMLAutoMLDirectionModel()]

        def train(subject: FLAMLAutoMLDirectionModel) -> None:
            try:
                subject.train(_frame())
            except BaseException as error:
                failures.put(error)

        threads: list[Thread] = []
        all_searches_entered = False
        try:
            for subject in subjects:
                thread = Thread(target=train, args=(subject,))
                thread.start()
                threads.append(thread)
            entered_fit.wait()
            all_searches_entered = True
            assert Path.cwd() == isolated_search_directory
            assert Path("caller.txt").read_text(encoding="utf-8") == "before training"
            Path("caller.txt").write_text("during training", encoding="utf-8")
        finally:
            finish_fit.set()
            if not all_searches_entered:
                entered_fit.abort()
            for thread in threads:
                thread.join(timeout=10)
            assert all(not thread.is_alive() for thread in threads), (
                "Training did not exit"
            )

        errors = []
        while not failures.empty():
            errors.append(failures.get_nowait())
        assert not errors, errors
        assert Path.cwd() == isolated_search_directory
        assert list(isolated_search_directory.iterdir()) == [caller_file]
        for subject in subjects:
            assert len(subject.predict_proba(_frame())) == len(_frame())


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


def _real_training_frame() -> pd.DataFrame:
    """Give real learners a reproducible binary classification problem."""
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
    return frame


def test_real_search_produces_a_usable_model(isolated_search_directory: Path) -> None:
    """Smoke test: the wrapper really drives FLAML, not just the stand-in."""
    frame = _real_training_frame()

    subject = FLAMLAutoMLDirectionModel(
        time_budget=20.0,
        estimator_list=["lgbm", "xgboost", "catboost", "rf", "extra_tree"],
        n_jobs=1,
    )
    subject.train(frame)
    assert Path.cwd() == isolated_search_directory
    assert list(isolated_search_directory.iterdir()) == []

    proba = subject.predict_proba(frame)
    np.testing.assert_array_equal(proba.columns.to_numpy(), [0, 1])
    assert len(proba) == len(frame)
    assert np.allclose(proba.sum(axis=1).to_numpy(), 1.0)
    assert subject.get_parameters()["selected_estimator"]

    predicted = subject.predict(frame)
    assert len(predicted) == len(frame)
    assert set(predicted.unique()) <= {0.0, 1.0}


@pytest.mark.parametrize("learner", ["lgbm", "xgboost", "rf", "extra_tree"])
def test_each_real_learner_trains_with_holdout_and_predicts(
    isolated_search_directory: Path, learner: str
) -> None:
    """A winning alternative cannot hide an incompatible learner."""
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.utils.validation import check_is_fitted
    from xgboost import XGBClassifier

    estimator_types = {
        "lgbm": LGBMClassifier,
        "xgboost": XGBClassifier,
        "rf": RandomForestClassifier,
        "extra_tree": ExtraTreesClassifier,
    }
    frame = _real_training_frame()
    training = frame.iloc[:120].copy()
    validation = frame.iloc[120:160].copy()
    prediction_input = frame.iloc[160:].drop(columns=["target"])
    original_training = training.copy(deep=True)
    original_validation = validation.copy(deep=True)
    original_prediction_input = prediction_input.copy(deep=True)

    subject = FLAMLAutoMLDirectionModel(
        time_budget=3.0,
        estimator_list=[learner],
        n_jobs=1,
        random_seed=42,
    )
    subject.train(training, eval_df=validation)

    assert subject.get_parameters()["selected_estimator"] == learner
    estimator = subject._require_trained().model.estimator
    assert isinstance(estimator, estimator_types[learner])
    check_is_fitted(estimator)

    proba = subject.predict_proba(prediction_input)
    np.testing.assert_array_equal(proba.columns.to_numpy(), [0, 1])
    pd.testing.assert_index_equal(proba.index, prediction_input.index)
    probabilities = proba.to_numpy()
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)

    predicted = subject.predict(prediction_input)
    pd.testing.assert_series_equal(predicted, proba.idxmax(axis=1).astype(float))
    pd.testing.assert_frame_equal(training, original_training)
    pd.testing.assert_frame_equal(validation, original_validation)
    pd.testing.assert_frame_equal(prediction_input, original_prediction_input)
    assert Path.cwd() == isolated_search_directory
    assert list(isolated_search_directory.iterdir()) == []


def test_real_catboost_search_disables_file_output(
    isolated_search_directory: Path,
) -> None:
    """Force CatBoost to run so another winning learner cannot hide spills."""
    from catboost import CatBoostClassifier

    frame = _real_training_frame()
    subject = FLAMLAutoMLDirectionModel(
        time_budget=5.0, estimator_list=["catboost"], n_jobs=1
    )
    subject.train(frame)

    assert subject.get_parameters()["selected_estimator"] == "catboost"
    estimator = subject._require_trained().model.estimator
    assert isinstance(estimator, CatBoostClassifier)
    assert estimator.is_fitted()
    assert estimator.get_param("allow_writing_files") is False
    proba = subject.predict_proba(frame)
    np.testing.assert_array_equal(proba.columns.to_numpy(), [0, 1])
    assert len(proba) == len(frame)
    assert np.allclose(proba.sum(axis=1).to_numpy(), 1.0)
    predicted = subject.predict(frame)
    assert len(predicted) == len(frame)
    assert set(predicted.unique()) <= {0.0, 1.0}
    assert Path.cwd() == isolated_search_directory
    assert list(isolated_search_directory.iterdir()) == []
