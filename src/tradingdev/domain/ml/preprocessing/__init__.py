"""Leak-aware preprocessing: processors that declare where they may learn.

The processors here mirror qlib's ``learn_processors`` / ``infer_processors``
distinction. Learn processors capture their parameters in ``fit`` and are only
ever fitted on training bars; infer processors are pure functions and may run
anywhere. :class:`ProcessorPipeline` enforces the boundary structurally, and
:func:`fit_transform_split` ties it to a purged
:class:`~tradingdev.domain.ml.validation.Split`.
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
