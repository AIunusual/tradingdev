"""Encode direction targets without changing validation answers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.preprocessing import LabelEncoder

if TYPE_CHECKING:
    import pandas as pd


def fit_direction_encoder(
    train: pd.DataFrame, evaluation: pd.DataFrame | None
) -> LabelEncoder:
    """Fit on training labels and reject unrepresentable validation classes."""
    if train.empty:
        raise ValueError("Training data must contain at least one labelled row")
    if evaluation is not None and evaluation.empty:
        raise ValueError("Validation data must contain at least one labelled row")
    encoder = LabelEncoder().fit(train["target"])
    if evaluation is not None:
        targets = evaluation["target"]
        unknown = targets[~targets.isin(encoder.classes_)].unique().tolist()
        if unknown:
            raise ValueError(
                f"Validation targets contain unseen classes: {unknown}; "
                f"training classes: {encoder.classes_.tolist()}. "
                "Use a training window containing the required classes; "
                "validation targets cannot be relabelled."
            )
    return encoder
