"""Public contracts for paired comparison of two saved observation series."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, JsonValue, model_validator

from tradingdev.app.contracts.common import ContractModel

type ComparisonColumn = Literal["realized_return", "pearson_ic", "spearman_ic"]
type ComparisonMode = Literal["descriptive", "inferential"]


class SeriesSelection(ContractModel):
    """One immutable source; only the caller's choice, never a recomputation."""

    kind: Literal["run", "signal", "benchmark"]
    run_id: str | None = None
    scope: str | None = None
    artifact_id: str | None = None
    column: ComparisonColumn | None = None
    benchmark_mode: Literal["product", "zero_cost"] | None = None
    price_artifact_id: str | None = None

    @model_validator(mode="after")
    def selection_is_complete(self) -> Self:
        if self.kind in {"run", "benchmark"} and not self.run_id:
            raise ValueError("run_id is required for run and benchmark sources")
        if self.kind == "benchmark" and self.benchmark_mode is None:
            raise ValueError("benchmark_mode is required for benchmark sources")
        if (
            self.kind == "benchmark"
            and self.benchmark_mode == "zero_cost"
            and not self.price_artifact_id
        ):
            raise ValueError("price_artifact_id is required for a zero-cost benchmark")
        if self.kind == "signal" and (not self.artifact_id or not self.column):
            raise ValueError("signal sources require artifact_id and column")
        return self


class ComparisonUnavailable(ContractModel):
    """A data/eligibility absence, distinct from an invalid user request."""

    reason: str
    next_step: str
    source: str
    detail: str | None = None


class SourceIdentity(ContractModel):
    """The identity and frozen settings of the one snapshot behind one side."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    kind: str
    label: str
    source_hash: str
    run_id: str | None = None
    scope: str | None = None
    manifest_hash: str | None = None
    dataset_id: str | None = None
    dataset_hash: str | None = None
    entry_id: str | None = None
    artifact_id: str | None = None
    column: ComparisonColumn | None = None
    mode: str | None = None
    frequency: str | None = None
    capital_basis: str | None = None
    cost_basis: str | None = None
    producer_source: str | None = None
    signal_fingerprint: str | None = None
    training_dataset_id: str | None = None
    training_status: str | None = None
    holdout_relationship: str | None = None
    fold_range: dict[str, JsonValue] | None = None
    benchmark_definition: str | None = None
    benchmark_costs: str | None = None
    settings: dict[str, JsonValue] = Field(default_factory=dict)


class AlignmentRecord(ContractModel):
    """The shared index, audit trail and calendar policy actually applied."""

    calendar_policy: Literal["exact_utc_intersection_no_fill"]
    cadence: str
    cadence_complete: bool
    original_lengths: dict[str, int]
    overlap: int
    valid_observations: int
    common_period: dict[str, str | None]
    excluded_counts: dict[str, int]
    excluded_sample: list[dict[str, str]]
    missing_timestamps_count: int
    missing_timestamps_sample: list[str]
    valid_mask_count: int


class ComparisonResult(ContractModel):
    """Paired mean difference on one shared index, with resampling evidence."""

    estimate: float
    estimate_a: float
    estimate_b: float
    lower: float
    upper: float
    p_value: float
    p_value_interpretation: Literal["conditional_percentile", "descriptive_only"]
    confidence: float
    reps: int
    valid_reps: int
    seed: int
    block_size: int
    correlation: float | None


class ComparisonArtifact(ContractModel):
    artifact_id: str
    sha256: str
    media_type: Literal["text/markdown"]


class PairedComparisonResponse(ContractModel):
    success: Literal[True]
    schema_version: Literal[1]
    algorithm_version: str
    mode: ComparisonMode
    statistic: Literal["mean"]
    difference_direction: Literal["a_minus_b"]
    claim_scope: Literal["verified_out_of_sample", "in_sample_or_unknown"]
    qualification: dict[str, JsonValue]
    status: Literal["available", "unavailable"]
    result: ComparisonResult | None = None
    unavailable: ComparisonUnavailable | None = None
    alignment: AlignmentRecord
    sources: list[SourceIdentity]
    limitations: list[str]
    artifact: ComparisonArtifact | None = None

    @model_validator(mode="after")
    def outcome_matches_status(self) -> Self:
        if self.status == "available" and self.result is None:
            raise ValueError("an available comparison requires a result")
        if self.status == "unavailable" and self.unavailable is None:
            raise ValueError("an unavailable comparison requires a reason")
        if len(self.sources) != 2:
            raise ValueError("exactly two source identities are required")
        return self


MAX_EXCLUDED_IN_RESPONSE = 20
MAX_MISSING_IN_RESPONSE = 20
MAX_REPS = 10_000
MIN_REPS = 100
PageBound = Annotated[int, Field(ge=1, le=MAX_EXCLUDED_IN_RESPONSE)]
