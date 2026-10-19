"""Ordered fit/transform pipelines with row-count and content provenance.

``fit_transform_split`` fits on training rows and transforms both partitions
without refitting. ``describe`` exposes each processor's learned statistics.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from tradingdev.domain.ml.preprocessing.processors import ProcessorNotFittedError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tradingdev.domain.ml.preprocessing.processors import Processor
    from tradingdev.domain.ml.validation import Split


@dataclass(frozen=True)
class ProcessorState:
    """What one processor is configured to do and what it actually learned."""

    name: str
    requires_fit: bool
    fields: tuple[str, ...]
    learned: dict[str, float]


def _frame_signature(frame: pd.DataFrame) -> str:
    """A short, deterministic fingerprint of a frame's values and index."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()[:16]


class ProcessorPipeline:
    """Ordered processors; refitting replaces statistics and fitting provenance.

    Retain row count and fingerprint, not the original training frame.
    """

    def __init__(self, processors: Sequence[Processor]) -> None:
        self._processors = tuple(processors)
        self._fit_signature: str | None = None
        self._fit_rows = 0

    @property
    def processors(self) -> tuple[Processor, ...]:
        """The configured processors, in application order."""
        return self._processors

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has run at least once."""
        return self._fit_signature is not None

    @property
    def fit_rows(self) -> int:
        """How many rows the last :meth:`fit` saw; 0 when never fitted."""
        return self._fit_rows

    @property
    def fit_signature(self) -> str | None:
        """Fingerprint of the rows the last :meth:`fit` saw."""
        return self._fit_signature

    def fit(self, frame: pd.DataFrame) -> ProcessorPipeline:
        """Fit and apply each processor in order on the training frame; return self.

        Later processors learn from earlier transformations, so a scaler after
        Fillna fits filled values rather than the original missing values.
        """
        if len(frame) == 0:
            raise ValueError("cannot fit a pipeline on an empty frame")
        self._fit_and_flow(frame)
        self._fit_rows = len(frame)
        self._fit_signature = _frame_signature(frame)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply every processor in order, without learning anything."""
        if not self.is_fitted:
            raise ProcessorNotFittedError(
                "fit() on the training segment must precede transform()"
            )
        out = frame
        for processor in self._processors:
            out = processor.transform(out)
        return out

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``frame`` and transform it; only for the training segment."""
        if len(frame) == 0:
            raise ValueError("cannot fit a pipeline on an empty frame")
        out = self._fit_and_flow(frame)
        self._fit_rows = len(frame)
        self._fit_signature = _frame_signature(frame)
        return out

    def _fit_and_flow(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit and apply each processor in turn, returning the final frame."""
        running = frame
        for processor in self._processors:
            processor.fit(running)
            running = processor.transform(running)
        return running

    def describe(self) -> tuple[ProcessorState, ...]:
        """Report one entry per processor, including its learned parameters."""
        return tuple(
            ProcessorState(
                name=processor.name,
                requires_fit=processor.requires_fit,
                fields=processor.fields,
                learned=processor.learned(),
            )
            for processor in self._processors
        )


def fit_transform_split(
    frame: pd.DataFrame,
    split: Split,
    pipeline: ProcessorPipeline,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit on one split's training rows and return transformed train/test frames.

    ``frame`` is ordered by bar. Test rows use the training-fitted parameters.
    """
    parts = [frame.iloc[start:end] for start, end in split.train]
    train = pd.concat(parts) if parts else frame.iloc[0:0]
    test = frame.iloc[split.test[0] : split.test[1]]
    train_out = pipeline.fit_transform(train)
    return train_out, pipeline.transform(test)
