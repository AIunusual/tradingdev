"""CatBoost classifier for directional prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tradingdev.domain.ml.base import BaseModel
from tradingdev.domain.ml.models.direction_targets import fit_direction_encoder
from tradingdev.domain.ml.registry import feature_importance_from_gain
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from catboost import CatBoostClassifier

    from tradingdev.domain.ml.schemas import CatBoostModelConfig

logger = setup_logger(__name__)

_EXCLUDE_COLS = frozenset(
    {"target", "timestamp", "open", "high", "low", "close", "volume"}
)


class CatBoostDirectionModel(BaseModel):
    """CatBoost direction classifier with the shared -1/0/1 model interface.

    ``allow_writing_files=False`` prevents training snapshots and logs.
    """

    def __init__(self, config: CatBoostModelConfig) -> None:
        self._config = config
        self._model: CatBoostClassifier | None = None
        self._feature_names: list[str] = []
        self._label_encoder: LabelEncoder = LabelEncoder()

    def train(
        self,
        df: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
    ) -> None:
        """Fit features against target; optionally use eval_df for early stopping."""
        from catboost import CatBoostClassifier

        feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]

        x_train = df[feature_cols]

        encoder = fit_direction_encoder(df, eval_df)
        y_train = encoder.transform(df["target"])

        clf = CatBoostClassifier(
            n_estimators=self._config.n_estimators,
            max_depth=self._config.max_depth,
            learning_rate=self._config.learning_rate,
            subsample=self._config.subsample,
            bootstrap_type="Bernoulli",
            colsample_bylevel=self._config.colsample_bytree,
            thread_count=self._config.n_jobs if self._config.n_jobs is not None else -1,
            random_state=self._config.random_state,
            early_stopping_rounds=(
                self._config.early_stopping_rounds if eval_df is not None else None
            ),
            verbose=False,
            allow_writing_files=False,
        )

        fit_kwargs: dict[str, Any] = {}
        if eval_df is not None:
            x_val = eval_df[feature_cols]
            y_val = encoder.transform(eval_df["target"])
            fit_kwargs["eval_set"] = [(x_val, y_val)]

        clf.fit(x_train, y_train, **fit_kwargs)
        # CatBoost embeds a random GUID and wall-clock completion time in its
        # native blob. Neither affects inference, but both would turn identical
        # seeded training into a distinct artifact and leaderboard candidate.
        # Normalize before serialization; the registry still hashes and verifies
        # every byte of the persisted model, including all prediction state.
        metadata = clf.get_metadata()
        for key in ("model_guid", "train_finish_time"):
            if key in metadata:
                del metadata[key]
        self._model = clf
        self._feature_names = feature_cols
        self._label_encoder = encoder

        logger.info(
            "CatBoost trained: %d samples, %d features",
            len(x_train),
            len(feature_cols),
        )

    def predict(self, df: pd.DataFrame) -> pd.Series[float]:
        """Return -1/0/1 classes; raise RuntimeError before training."""
        if self._model is None:
            msg = "Model has not been trained. Call train() first."
            raise RuntimeError(msg)

        x = df[self._feature_names]
        encoded_preds = self._model.predict(x)
        preds = self._label_encoder.inverse_transform(encoded_preds.astype(int))
        return pd.Series(preds, index=df.index, dtype=float)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities; raise RuntimeError before training."""
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
