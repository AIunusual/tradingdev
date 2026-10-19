"""Fit/transform processors with a training-only split helper.

Learn processors retain fitted statistics; infer processors are stateless.
The pipeline records fitting provenance and rejects transforms before fitting.
"""

from __future__ import annotations

from tradingdev.domain.ml.preprocessing.pipelines import (
    ProcessorPipeline,
    ProcessorState,
    fit_transform_split,
)
from tradingdev.domain.ml.preprocessing.processors import (
    ClipOutliers,
    DropnaRows,
    Fillna,
    MinMaxNorm,
    Processor,
    ProcessorNotFittedError,
    RobustZScoreNorm,
    ZScoreNorm,
)

__all__ = [
    "ClipOutliers",
    "DropnaRows",
    "Fillna",
    "MinMaxNorm",
    "Processor",
    "ProcessorNotFittedError",
    "ProcessorPipeline",
    "ProcessorState",
    "RobustZScoreNorm",
    "ZScoreNorm",
    "fit_transform_split",
]
