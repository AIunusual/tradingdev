"""Immutable JSON signal evaluations preserving their metric definitions."""

from __future__ import annotations

import math
import re
from dataclasses import asdict
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tradingdev.domain.ml.evaluation.catalog import SIGNAL_METRIC_CATALOG

# Runtime import: Pydantic resolves these string annotations through module
# globals, so a TYPE_CHECKING import would leave the field types unresolvable.
from tradingdev.domain.ml.evaluation.metrics import MetricScalar  # noqa: TC001

if TYPE_CHECKING:
    import pandas as pd

_DECAY_KEY = re.compile(r"^(rank_)?ic_h[1-9][0-9]*$")


class SignalArtifactModel(BaseModel):
    """Validate stored structure without coercing booleans or accepting NaN."""

    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )


class SignalMetricDefinitionSnapshot(SignalArtifactModel):
    """The exact signal metric definition recorded by the producer."""

    id: str
    unit: str
    category: str
    provider: str
    description: str
    summary: bool
    optimization_direction: Literal["maximize", "minimize"] | None
    # Stored so a signal metric keeps the same shape as a portfolio metric and
    # can be projected into a report without a second vocabulary.
    modes: tuple[str, ...] = ("signal",)
    # Older snapshots did not record this flag. Do not infer historical
    # semantics from the current catalog or mislabel unknown as False.
    requires_annualization: bool | None = None


class StratumStat(SignalArtifactModel):
    """One prediction quantile and its realised outcome."""

    bin: int = Field(ge=1)
    count: int = Field(ge=0)
    mean_prediction: float | None
    mean_forward_return: float | None
    std_forward_return: float | None


class SignalEvaluationBundle(SignalArtifactModel):
    """Complete frozen signal metrics, strata, decay and their definitions."""

    schema_version: Literal[1] = 1
    evaluation_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    dataset_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    horizon: int = Field(ge=1)
    window: int = Field(ge=2)
    n_bins: int = Field(ge=2)
    method: Literal["spearman", "pearson"]
    values: dict[str, MetricScalar]
    strata: list[StratumStat]
    decay: dict[str, MetricScalar]
    definitions: dict[str, SignalMetricDefinitionSnapshot]

    @model_validator(mode="after")
    def consistent_definitions(self) -> Self:
        """Every stored metric must be defined, and decay keys well formed."""
        for key, definition in self.definitions.items():
            if key != definition.id:
                raise ValueError("Signal metric definition ID differs from its key")
        if missing := self.values.keys() - self.definitions.keys():
            raise ValueError(
                f"Missing signal metric definitions: {', '.join(sorted(missing))}"
            )
        for key in self.decay:
            if not _DECAY_KEY.match(key):
                raise ValueError(f"Unsupported decay metric key: {key}")
        return self


def _optional_float(value: object) -> float | None:
    """Copy a finite float, mapping missing or non-finite inputs to None."""
    if value is None:
        return None
    number = float(value)  # type: ignore[arg-type]
    return number if math.isfinite(number) else None


def strata_records(strata: pd.DataFrame) -> list[StratumStat]:
    """Convert a quantile table into frozen, finite artifact records."""
    frame = strata.reset_index()
    bins = frame["bin"].to_numpy()
    counts = frame["count"].to_numpy()
    predictions = frame["mean_prediction"].to_numpy(dtype=float)
    forwards = frame["mean_forward_return"].to_numpy(dtype=float)
    spreads = frame["std_forward_return"].to_numpy(dtype=float)
    return [
        StratumStat(
            bin=int(bins[index]),
            count=int(counts[index]),
            mean_prediction=_optional_float(predictions[index]),
            mean_forward_return=_optional_float(forwards[index]),
            std_forward_return=_optional_float(spreads[index]),
        )
        for index in range(len(frame))
    ]


def build_signal_evaluation(
    evaluation_id: str,
    *,
    values: dict[str, MetricScalar],
    strata: list[StratumStat],
    decay: dict[str, MetricScalar] | None = None,
    horizon: int,
    window: int,
    n_bins: int,
    method: Literal["spearman", "pearson"],
    dataset_hash: str | None = None,
) -> SignalEvaluationBundle:
    """Freeze the producer's definitions alongside its calculated metrics."""
    definitions = {
        key: SignalMetricDefinitionSnapshot.model_validate(
            asdict(SIGNAL_METRIC_CATALOG[key])
        )
        for key in sorted(values)
        if key in SIGNAL_METRIC_CATALOG
    }
    return SignalEvaluationBundle(
        evaluation_id=evaluation_id,
        dataset_hash=dataset_hash,
        horizon=horizon,
        window=window,
        n_bins=n_bins,
        method=method,
        values=values,
        strata=strata,
        decay=decay or {},
        definitions=definitions,
    )
