"""Evidence recorded by the optimization worker for each approved grid position."""

from typing import Literal, Self

from pydantic import Field, model_validator

from tradingdev.domain.performance.artifacts import (
    ArtifactModel,
    MetricDefinitionSnapshot,
    PerformanceScope,
    ScopeObservations,
    metric_definition_snapshots,
)
from tradingdev.domain.performance.catalog import MetricScalar  # noqa: TC001

type TrialState = Literal[
    "not_attempted", "running", "completed", "failed", "timed_out"
]
type TrialStateView = TrialState | Literal["interrupted"]


class TrialEvidence(ArtifactModel):
    """One fully observed execution with its producer's metric definitions."""

    scope: PerformanceScope
    observations: ScopeObservations
    definitions: dict[str, MetricDefinitionSnapshot]

    @classmethod
    def capture(cls, scope: PerformanceScope, observations: ScopeObservations) -> Self:
        return cls(
            scope=scope,
            observations=observations,
            definitions=metric_definition_snapshots(scope.values),
        )

    @model_validator(mode="after")
    def validate_definitions(self) -> Self:
        if self.definitions.keys() != self.scope.values.keys() or any(
            name != definition.id for name, definition in self.definitions.items()
        ):
            raise ValueError("Trial definitions differ from recorded metrics")
        return self


class TrialRecord(ArtifactModel):
    """A ledger entry. The test entry uses position -1."""

    schema_version: Literal[1] = 1
    position: int = Field(ge=-1)
    state: TrialState
    selected_index: int | None = Field(default=None, ge=0)
    target_value: MetricScalar = None
    reason: str | None = Field(default=None, min_length=1, max_length=1024)
    evidence_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if (self.state == "completed") != (self.evidence_hash is not None):
            raise ValueError("Only completed trials must have observations")
        if self.state != "completed" and self.target_value is not None:
            raise ValueError("Unobserved trials cannot have an objective value")
        if (self.state in {"failed", "timed_out"}) != (self.reason is not None):
            raise ValueError("Candidate failures require a reason")
        if (self.position == -1 and self.state != "not_attempted") != (
            self.selected_index is not None
        ):
            raise ValueError("Only attempted tests must have a selected index")
        return self
