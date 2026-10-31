"""XGBoost classifier for directional prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from tradingdev.domain.ml.base import BaseModel
from tradingdev.domain.ml.models.direction_targets import fit_direction_encoder
from tradingdev.domain.ml.registry import feature_importance_from_gain
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from tradingdev.domain.ml.schemas import XGBoostModelConfig

logger = setup_logger(__name__)

_EXCLUDE_COLS = frozenset(
    {"target", "timestamp", "open", "high", "low", "close", "volume"}
)


class XGBoostDirectionModel(BaseModel):
    """XGBoost-based direction prediction model.

    Wraps :class:`xgb.XGBClassifier` to predict price direction
    (``-1``, ``0``, ``1``).
    """

    def __init__(self, config: XGBoostModelConfig) -> None:
        self._config = config
        self._model: xgb.XGBClassifier | None = None
        self._feature_names: list[str] = []
        self._label_encoder: LabelEncoder = LabelEncoder()

    def train(
        self,
        df: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
    ) -> None:
        """Train the XGBoost classifier.

        Args:
            df: Training DataFrame with feature columns and a
                ``target`` column.
            eval_df: Optional validation DataFrame for early stopping.

        Raises:
            ValueError: If training/validation data is empty or validation
                targets include classes absent from training. Rejected inputs
                leave the previously fitted model intact.
        """
        feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]

        x_train = df[feature_cols]

        encoder = fit_direction_encoder(df, eval_df)
        y_train = encoder.transform(df["target"])

        fit_params: dict[str, Any] = {}

        n_classes = len(encoder.classes_)
        clf = xgb.XGBClassifier(
            n_estimators=self._config.n_estimators,
            max_depth=self._config.max_depth,
            learning_rate=self._config.learning_rate,
            subsample=self._config.subsample,
            colsample_bytree=self._config.colsample_bytree,
            random_state=self._config.random_state,
            n_jobs=self._config.n_jobs,
            eval_metric=("mlogloss" if n_classes > 2 else "logloss"),
            early_stopping_rounds=(
                self._config.early_stopping_rounds if eval_df is not None else None
            ),
        )

        if eval_df is not None:
            x_val = eval_df[feature_cols]
            y_val = encoder.transform(eval_df["target"])
            fit_params["eval_set"] = [(x_val, y_val)]
            fit_params["verbose"] = False

        clf.fit(x_train, y_train, **fit_params)
        self._model = clf
        self._feature_names = feature_cols
        self._label_encoder = encoder

        logger.info(
            "XGBoost trained: %d samples, %d features",
            len(x_train),
            len(feature_cols),
        )

    def predict(self, df: pd.DataFrame) -> pd.Series[float]:
        """Predict direction class for each row.

        Args:
            df: Feature DataFrame (must contain the same columns
                used during training, excluding ``target``).

        Returns:
            Series of predicted classes (``-1``, ``0``, ``1``).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._model is None:
            msg = "Model has not been trained. Call train() first."
            raise RuntimeError(msg)

        x = df[self._feature_names]
        encoded_preds = self._model.predict(x)
        preds = self._label_encoder.inverse_transform(encoded_preds.astype(int))
        return pd.Series(preds, index=df.index, dtype=float)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities for each row.

        Args:
            df: Feature DataFrame.

        Returns:
            DataFrame with one column per class.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._model is None:
            msg = "Model has not been trained. Call train() first."
            raise RuntimeError(msg)

        x = df[self._feature_names]
        proba = self._model.predict_proba(x)

        n_classes_encoder = len(self._label_encoder.classes_)
        n_cols_proba = proba.shape[1]

        if n_cols_proba == n_classes_encoder:
            columns = self._label_encoder.classes_
        elif n_classes_encoder == 1 and n_cols_proba == 2:
            # XGBoost binary classifier always outputs 2 columns
            # even when LabelEncoder saw only 1 class during fit.
            # Internal class 0 = LabelEncoder.classes_[0] (the
            # only known class).  Col-0 → only_class, col-1 → other.
            only_class = self._label_encoder.classes_[0]
            other_class = 1 - only_class if only_class in (0, 1) else 0
            columns = [only_class, other_class]
        else:
            columns = list(range(n_cols_proba))

        return pd.DataFrame(
            proba,
            index=df.index,
            columns=columns,
        )

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
