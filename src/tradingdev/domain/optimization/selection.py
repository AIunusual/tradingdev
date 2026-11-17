"""References from a ranked scope to its run's saved training search."""

from typing import Annotated, Literal

from pydantic import Field

from tradingdev.domain.performance.artifacts import ArtifactModel, OptimizationSelection


class SelectionEvidenceReference(ArtifactModel):
    run_id: str
    manifest_hash: str
    performance_sha256: str
    source: Literal["trial_ledger", "verified_historical_bundle"]


class AvailableSelectionEvidence(ArtifactModel):
    status: Literal["available"] = "available"
    scope_role: Literal["selected_test", "selected_train", "other_train"]
    reference: SelectionEvidenceReference
    summary: OptimizationSelection


class UnavailableSelectionEvidence(ArtifactModel):
    status: Literal["unavailable"] = "unavailable"
    reason: str


type SelectionEvidence = Annotated[
    AvailableSelectionEvidence | UnavailableSelectionEvidence,
    Field(discriminator="status"),
]
