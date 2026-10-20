"""LightGBM classifier for directional prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tradingdev.domain.ml.base import BaseModel
from tradingdev.domain.ml.registry import feature_importance_from_gain
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from tradingdev.domain.ml.schemas import LightGBMModelConfig

logger = setup_logger(__name__)

_EXCLUDE_COLS = frozenset(
    {"target", "timestamp", "open", "high", "low", "close", "volume"}
)


class LightGBMDirectionModel(BaseModel):
    """LightGBM direction classifier with the shared -1/0/1 model interface."""

    def __init__(self, config: LightGBMModelConfig) -> None:
        self._config: LightGBMModelConfig = config
        self._model: lgb.LGBMClassifier | None = None
        self._feature_names: list[str] = []
        self._label_encoder: LabelEncoder = LabelEncoder()

    def train(
        self,
        df: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
    ) -> None:
        """Fit features against target; optionally use eval_df for early stopping."""
        feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
        self._feature_names = feature_cols

        x_train = df[feature_cols]

        self._label_encoder.fit(df["target"])
        y_train = self._label_encoder.transform(df["target"])

        clf = lgb.LGBMClassifier(
            n_estimators=self._config.n_estimators,
            max_depth=self._config.max_depth,
            learning_rate=self._config.learning_rate,
            subsample=self._config.subsample,
            colsample_bytree=self._config.colsample_bytree,
            random_state=self._config.random_state,
            n_jobs=self._config.n_jobs,
            verbose=-1,
        )

        fit_kwargs: dict[str, Any] = {}
        if eval_df is not None:
            x_val = eval_df[feature_cols]
            known = set(self._label_encoder.classes_)
            y_val_safe = eval_df["target"].where(
                eval_df["target"].isin(known),
                self._label_encoder.classes_[0],
            )
            y_val = self._label_encoder.transform(y_val_safe)
            fit_kwargs["eval_set"] = [(x_val, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self._config.early_stopping_rounds)
            ]

        clf.fit(x_train, y_train, **fit_kwargs)
        self._model = clf

        logger.info(
            "LightGBM trained: %d samples, %d features",
            len(x_train),
            len(feature_cols),
        )

    def predict(self, df: pd.DataFrame) -> pd.Series[float]:
        """Return -1/0/1 classes; raise RuntimeError before training."""
        if self._model is None:
            msg = "Model has not been trained. Call train() first."
            raise RuntimeError(msg)

        x = df[self._feature_names]
        encoded_preds = np.asarray(self._model.predict(x), dtype=int)
        preds = self._label_encoder.inverse_transform(encoded_preds)
        return pd.Series(preds, index=df.index, dtype=float)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities; raise RuntimeError before training."""
        if self._model is None:
            msg = "Model has not been trained. Call train() first."
            raise RuntimeError(msg)

        x = df[self._feature_names]
        proba = np.asarray(self._model.predict_proba(x))

        n_classes_encoder = len(self._label_encoder.classes_)
        n_cols_proba = proba.shape[1]

        if n_cols_proba == n_classes_encoder:
            columns = self._label_encoder.classes_
        elif n_classes_encoder == 1 and n_cols_proba == 2:
            only_class = self._label_encoder.classes_[0]
            other_class = 1 - only_class if only_class in (0, 1) else 0
            columns = [only_class, other_class]
        else:
            columns = list(range(n_cols_proba))

        return pd.DataFrame(proba, index=df.index, columns=columns)

    def get_parameters(self) -> dict[str, Any]:
        """Return model configuration."""
        return self._config.model_dump()

    def feature_importance(self) -> tuple[tuple[str, float], ...]:
        """Gain-based importance of the trained booster, highest first."""
        if self._model is None:
            return ()
        return feature_importance_from_gain(
            self._feature_names, self._model.feature_importances_
        )
