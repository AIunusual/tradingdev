"""Fixed research declarations and simulator policy approved for one execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tradingdev.domain.comparison import ComparisonManifest
from tradingdev.domain.execution_assumptions import (
    ExecutionAssumptions,
    resolve_execution_assumptions,
)
from tradingdev.domain.research_evidence import (
    BoundResearchEvidence,
    ResearchEvidenceRecord,
    bind_research_evidence,
)
from tradingdev.domain.research_evidence_values import digest

if TYPE_CHECKING:
    from tradingdev.domain.execution import ExecutionManifest
    from tradingdev.domain.execution_target import ExecutionTarget


class ApprovedResearch(BaseModel):
    """A detached disclosure; historical verification never resolves today's policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    research: BoundResearchEvidence
    execution_assumptions: tuple[ExecutionAssumptions, ...] = Field(
        min_length=1, max_length=8
    )
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, record: ResearchEvidenceRecord, target: ExecutionTarget) -> Self:
        research = bind_research_evidence(record, target)
        payload = {
            "schema_version": 1,
            "research": research.model_dump(mode="json"),
            "execution_assumptions": [
                resolve_execution_assumptions(manifest).model_dump(mode="json")
                for manifest in execution_manifests(target)
            ],
        }
        result = cls.model_validate({**payload, "content_hash": digest(payload)})
        result.verify(target)
        return result

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Research schema version must be an integer")
        return value

    @model_validator(mode="after")
    def valid_snapshot(self) -> Self:
        self.verify()
        return self

    def verify(self, target: ExecutionTarget | None = None) -> None:
        """Check declarations, saved policy identities, and complete content hash."""
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("Unsupported approved research schema version")
        self.research.verify(target)
        hashes = tuple(item.manifest_hash for item in self.execution_assumptions)
        expected_hashes = tuple(
            dict.fromkeys(item.manifest_hash for item in self.research.targets)
        )
        if hashes != expected_hashes:
            raise ValueError(
                "Research assumptions differ from the bound execution identities"
            )
        for assumptions in self.execution_assumptions:
            ExecutionAssumptions.model_validate_json(assumptions.model_dump_json())
        if target is not None:
            manifests = execution_manifests(target)
            if any(
                assumptions.mode != manifest.config_copy()["backtest"]["mode"]
                for assumptions, manifest in zip(
                    self.execution_assumptions, manifests, strict=True
                )
            ):
                raise ValueError("Research assumptions differ from the execution mode")
        if (
            digest(self.model_dump(mode="json", exclude={"content_hash"}))
            != self.content_hash
        ):
            raise ValueError("Approved research hash does not match its contents")

    def verify_current_policy(self, target: ExecutionTarget) -> None:
        """Reject a policy change between disclosure and starting execution."""
        self.verify(target)
        current = tuple(
            resolve_execution_assumptions(manifest)
            for manifest in execution_manifests(target)
        )
        if current != self.execution_assumptions:
            raise ValueError("Execution policy changed after research confirmation")

    def assumptions_for(self, manifest_hash: str) -> ExecutionAssumptions:
        for assumptions in self.execution_assumptions:
            if assumptions.manifest_hash == manifest_hash:
                return assumptions
        raise ValueError("No approved assumptions for this execution manifest")


def execution_manifests(target: ExecutionTarget) -> tuple[ExecutionManifest, ...]:
    """Return each distinct child manifest once in the accepted target order."""
    candidates = (
        tuple(candidate.manifest for candidate in target.candidates)
        if isinstance(target, ComparisonManifest)
        else (target,)
    )
    return tuple({manifest.manifest_hash: manifest for manifest in candidates}.values())
