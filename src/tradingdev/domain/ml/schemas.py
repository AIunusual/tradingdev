"""Domain schemas for ML models."""

from pydantic import BaseModel


class XGBoostModelConfig(BaseModel):
    """XGBoost model hyperparameters."""

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 10
    random_state: int = 42
    n_jobs: int | None = None


class LightGBMModelConfig(BaseModel):
    """LightGBM model hyperparameters.

    The field names mirror :class:`XGBoostModelConfig` so a single uniform
    contract can drive any booster in the model zoo; the wrapper maps
    ``colsample_bytree`` to LightGBM's ``colsample_bytree``.
    """

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 10
    random_state: int = 42
    n_jobs: int | None = None


class CatBoostModelConfig(BaseModel):
    """CatBoost model hyperparameters.

    Field names mirror :class:`XGBoostModelConfig`. The wrapper maps
    ``colsample_bytree`` → ``colsample_bylevel`` and ``n_jobs`` →
    ``thread_count`` so the zoo exposes one consistent schema.
    """

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 10
    random_state: int = 42
    n_jobs: int | None = None
