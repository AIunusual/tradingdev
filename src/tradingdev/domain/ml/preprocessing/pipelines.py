"""Ordered processor pipelines that learn from a training segment only.

A single processor is easy to misuse: calling ``fit`` on the full frame looks
identical to calling it on the training bars. :class:`ProcessorPipeline` makes
the boundary structural:

* :meth:`ProcessorPipeline.fit` is the only entry point that may learn, and it
  records how many rows it saw plus a fingerprint of those rows.
* :meth:`ProcessorPipeline.transform` refuses to run before a ``fit``, so an
  unfitted pipeline fails loudly instead of silently computing statistics from
  whatever frame it is handed.
* :meth:`ProcessorPipeline.describe` reports what each processor learned,
  which is the surface an MCP tool can show to a user who wants to know
  whether the pipeline touched the test data.

:func:`fit_transform_split` wires this to :class:`~tradingdev.domain.ml.
validation.Split`: fit on the training bars, transform both sides, never refit.
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
    """An ordered sequence of processors fitted once per fold.

    The pipeline is reusable: calling :meth:`fit` again replaces the learned
    parameters, which is what a walk-forward loop needs. It deliberately does
    not remember the frame it was fitted on, only its shape and fingerprint,
    so a fitted pipeline stays cheap to pass around.
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
        """Learn every processor's parameters from ``frame``.

        Fitting runs in pipeline order: each processor fits on the frame as it
        currently stands and is then applied, so a later processor sees the
        data the earlier ones already transformed. That is what makes a
        ``Fillna`` followed by a ``ZScoreNorm`` fit the scaler on the filled
        column rather than on the raw, NaN-bearing one.

        Args:
            frame: The training segment. Nothing outside it is observable.

        Returns:
            The pipeline itself, so ``fit(...)`` chains into ``transform(...)``.
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
    """Fit the pipeline on one split's training bars and transform both sides.

    The test bars are transformed by parameters that were computed before they
    were ever looked at, which is the whole point of purging and embargoing
    in the first place.

    Args:
        frame: Feature frame ordered by bar, one row per bar.
        split: The leak-free partition to honour.
        pipeline: A pipeline, refitted here for this split.

    Returns:
        The transformed training rows and the transformed test rows.
    """
    parts = [frame.iloc[start:end] for start, end in split.train]
    train = pd.concat(parts) if parts else frame.iloc[0:0]
    test = frame.iloc[split.test[0] : split.test[1]]
    train_out = pipeline.fit_transform(train)
    return train_out, pipeline.transform(test)
