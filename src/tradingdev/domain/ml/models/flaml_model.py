"""FLAML AutoML direction prediction model.

FLAML searches tree estimators and their hyperparameters under a wall-clock
budget, then keeps the winning model. Bundled strategies use this entry point
without requiring a deep-learning stack.

Deliberately *not* in the model zoo (:mod:`tradingdev.domain.ml.models.registry`):
a zoo kind must be reproducible from its recorded parameters, and an AutoML
search bounded by wall-clock time is not — the same config can select a
different estimator on a loaded machine. Strategy code that needs a
reproducible, auditable estimator should pick a named zoo kind instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from tradingdev.domain.ml.base import BaseModel
from tradingdev.domain.ml.registry import feature_importance_from_gain
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from flaml import AutoML

logger = setup_logger(__name__)

_EXCLUDE_COLS = frozenset(
    {
        "target",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dvol",
        "dvol_open",
        "dvol_high",
        "dvol_low",
        "dvol_close",
    }
)

#: Estimators FLAML searches by default. Names are FLAML's own; members that
#: are not importable (no catboost wheel on some platforms) make the search
#: fail loudly rather than silently narrowing the space.
DEFAULT_ESTIMATOR_LIST: tuple[str, ...] = (
    "lgbm",
    "xgboost",
    "catboost",
    "rf",
    "extra_tree",
)


class FLAMLAutoMLDirectionModel(BaseModel):
    """FLAML AutoML wrapper for direction prediction.

    Searches several gradient-boosted-tree learners and their
    hyper-parameters under a wall-clock budget and keeps the best one
    found. ``predict_proba`` is the surface callers actually need: it
    returns one column per class so a caller can threshold ``P(up)``
    instead of trusting a hard argmax.
    """

    def __init__(
        self,
        time_budget: float = 300.0,
        estimator_list: tuple[str, ...] | list[str] | None = None,
        metric: str = "accuracy",
        verbosity: int = 0,
        n_jobs: int | None = None,
        random_seed: int | None = 42,
    ) -> None:
        self._time_budget = float(time_budget)
        self._estimator_list = (
            tuple(DEFAULT_ESTIMATOR_LIST)
            if estimator_list is None
            else tuple(estimator_list)
        )
        self._metric = metric
        self._verbosity = verbosity
        self._n_jobs = n_jobs
        self._random_seed = random_seed
        self._automl: AutoML | None = None
        self._feature_names: list[str] = []
        self._classes: tuple[Any, ...] = ()

    def train(
        self,
        df: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
    ) -> None:
        """Run the AutoML search on a feature DataFrame.

        Args:
            df: Training DataFrame with feature columns and ``target``.
            eval_df: Optional holdout DataFrame. FLAML scores candidates on
                it during the search instead of splitting ``df`` itself,
                which keeps the search honest when the caller has already
                purged the train/validation boundary.

        Raises:
            RuntimeError: If the search produced no model.
        """
        from flaml import AutoML

        feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
        self._feature_names = feature_cols

        x_train = df[feature_cols]
        y_train = df["target"]

        fit_kwargs: dict[str, Any] = {
            "task": "classification",
            "metric": self._metric,
            "estimator_list": list(self._estimator_list),
            "time_budget": self._time_budget,
            "verbose": self._verbosity,
            "log_file_name": "",
            # CatBoost output is a constructor setting, so fix it in the
            # search space rather than forwarding it to estimator.fit().
            # Disabling output avoids changing the process-wide cwd when
            # independent callers train concurrently.
            "custom_hp": {
                "catboost": {
                    "allow_writing_files": {"domain": False, "init_value": False},
                },
            },
        }
        if self._n_jobs is not None:
            fit_kwargs["n_jobs"] = self._n_jobs
        if self._random_seed is not None:
            fit_kwargs["seed"] = self._random_seed
        if eval_df is not None:
            fit_kwargs["X_val"] = eval_df[feature_cols]
            fit_kwargs["y_val"] = eval_df["target"]

        automl = AutoML()
        automl.fit(X_train=x_train, y_train=y_train, **fit_kwargs)

        if automl.model is None:  # pragma: no cover - defensive
            msg = "FLAML search finished without a model"
            raise RuntimeError(msg)

        self._automl = automl
        self._classes = tuple(automl.classes_)

        logger.info(
            "FLAML trained: best=%s (loss=%.4f), %d estimators searched, "
            "%d features, %d samples",
            automl.best_estimator,
            float(automl.best_loss),
            len(self._estimator_list),
            len(feature_cols),
            len(x_train),
        )

    def _require_trained(self) -> AutoML:
        """Return the fitted AutoML, raising if :meth:`train` was not run."""
        if self._automl is None:
            msg = "Model has not been trained. Call train() first."
            raise RuntimeError(msg)
        return self._automl

    def predict(self, df: pd.DataFrame) -> pd.Series[float]:
        """Predict direction class (``0``=down, ``1``=up).

        Returns:
            Series of predicted classes, indexed like ``df``.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        automl = self._require_trained()
        preds = automl.predict(df[self._feature_names])
        return pd.Series(preds, index=df.index, dtype=float)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities.

        The columns are the classes FLAML saw during training, in
        ``classes_`` order, so a binary fit yields ``[0, 1]`` — callers
        that need ``P(up)`` read the last column rather than assuming a
        positional convention.

        Returns:
            DataFrame with one column per class, indexed like ``df``.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        automl = self._require_trained()
        proba = automl.predict_proba(df[self._feature_names])
        return pd.DataFrame(proba, index=df.index, columns=list(self._classes))

    def get_parameters(self) -> dict[str, Any]:
        """Return the search configuration actually used."""
        params: dict[str, Any] = {
            "time_budget": self._time_budget,
            "estimator_list": list(self._estimator_list),
            "metric": self._metric,
        }
        if self._n_jobs is not None:
            params["n_jobs"] = self._n_jobs
        if self._random_seed is not None:
            params["random_seed"] = self._random_seed
        if self._automl is not None:
            # The winner is a property of the search, not of the config, so
            # it is reported but never fed back into model identity.
            params["selected_estimator"] = self._automl.best_estimator
        return params

    def feature_importance(self) -> tuple[tuple[str, float], ...]:
        """Gain-based importance of the selected estimator, highest first.

        Returns an empty tuple when the winner exposes no per-feature gain
        (for example a linear learner). An empty result means *unavailable*,
        never *unimportant*.
        """
        if self._automl is None:
            return ()
        estimator = getattr(self._automl.model, "estimator", None)
        importance = getattr(estimator, "feature_importances_", None)
        if importance is None:
            return ()
        return feature_importance_from_gain(self._feature_names, importance)
