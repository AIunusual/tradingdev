"""Complete candidate outcomes and evidence for one fixed comparison job."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from tradingdev.domain.execution import ManifestError

if TYPE_CHECKING:
    from tradingdev.domain.comparison import ComparisonManifest

type ComparisonJobId = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,115}$")
]
type CandidateId = Annotated[str, StringConstraints(pattern=r"^candidate_[1-8]$")]
type Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
type CandidateStatus = Literal[
    "pending", "running", "succeeded", "failed", "not_run", "cancelled"
]
type FailureStatus = Literal["failed", "cancelled"]
type OutcomeReason = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
]

TERMINAL_STATUSES = frozenset({"succeeded", "failed", "not_run", "cancelled"})


def _integer_version(value: object) -> int:
    if type(value) is not int or value != 1:
        raise ValueError("Comparison schema version must be integer 1")
    return value


type SchemaVersion = Annotated[Literal[1], BeforeValidator(_integer_version)]


def canonical_json_object(content: str) -> str:
    """Require finite, unambiguous JSON evidence and return canonical bytes as text."""

    def invalid_constant(value: str) -> object:
        raise ValueError(f"Non-finite JSON value: {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    value = json.loads(
        content, parse_constant=invalid_constant, object_pairs_hook=unique_object
    )
    if not isinstance(value, dict):
        raise ValueError("Comparison data binding must be a JSON object")
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


class ComparisonCandidateResult(BaseModel):
    """A reserved child identity is not proof that an execution succeeded."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )

    candidate_id: CandidateId
    child_manifest_hash: Sha256
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    status: CandidateStatus = "pending"
    reason: OutcomeReason | None = None

    @model_validator(mode="after")
    def reason_matches_status(self) -> Self:
        unsuccessful = self.status in {"failed", "not_run", "cancelled"}
        if unsuccessful != (self.reason is not None):
            raise ValueError("Only unsuccessful terminal candidates require a reason")
        return self


class ComparisonLedger(BaseModel):
    """Retain every approved candidate, including those with no performance result."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )

    schema_version: SchemaVersion = 1
    job_id: ComparisonJobId
    manifest_hash: Sha256
    candidates: tuple[ComparisonCandidateResult, ...] = Field(
        min_length=1, max_length=8
    )

    @classmethod
    def create(cls, job_id: str, manifest: ComparisonManifest) -> Self:
        manifest.verify()
        return cls(
            job_id=job_id,
            manifest_hash=manifest.manifest_hash,
            candidates=tuple(
                ComparisonCandidateResult(
                    candidate_id=candidate.candidate_id,
                    child_manifest_hash=candidate.manifest.manifest_hash,
                    run_id=f"{job_id}_{candidate.candidate_id}",
                )
                for candidate in manifest.candidates
            ),
        )

    @model_validator(mode="after")
    def exact_order_and_ids(self) -> Self:
        for index, candidate in enumerate(self.candidates, start=1):
            candidate_id = f"candidate_{index}"
            if candidate.candidate_id != candidate_id:
                raise ValueError("Comparison ledger must retain exact candidate order")
            if candidate.run_id != f"{self.job_id}_{candidate_id}":
                raise ValueError("Comparison child run ID must match its reservation")
        return self

    @property
    def is_complete(self) -> bool:
        return all(row.status in TERMINAL_STATUSES for row in self.candidates)

    def verify(self, manifest: ComparisonManifest, *, job_id: str) -> None:
        manifest.verify(expected_hash=self.manifest_hash)
        # Revalidate model instances too: model_copy/model_construct bypass validation.
        type(self).model_validate_json(self.model_dump_json())
        if self.job_id != job_id or len(self.candidates) != len(manifest.candidates):
            raise ManifestError("Comparison ledger identity or membership changed")
        for row, member in zip(self.candidates, manifest.candidates, strict=True):
            if (
                row.candidate_id != member.candidate_id
                or row.child_manifest_hash != member.manifest.manifest_hash
            ):
                raise ManifestError("Comparison ledger child binding changed")

    def transition(
        self,
        candidate_id: str,
        status: CandidateStatus,
        *,
        reason: str | None = None,
    ) -> Self:
        """Allow only forward transitions; identical retries never rewrite outcomes."""
        for index, previous in enumerate(self.candidates):
            if previous.candidate_id != candidate_id:
                continue
            row = ComparisonCandidateResult(
                candidate_id=previous.candidate_id,
                child_manifest_hash=previous.child_manifest_hash,
                run_id=previous.run_id,
                status=status,
                reason=reason,
            )
            if row == previous:
                return self
            allowed = {
                "pending": {"running", "not_run"},
                "running": {"succeeded", "failed", "cancelled"},
            }
            if status not in allowed.get(previous.status, set()):
                raise ManifestError(
                    f"Cannot change comparison candidate {previous.status} to {status}"
                )
            rows = (*self.candidates[:index], row, *self.candidates[index + 1 :])
            return type(self)(
                job_id=self.job_id, manifest_hash=self.manifest_hash, candidates=rows
            )
        raise ManifestError(f"Unknown comparison candidate: {candidate_id}")

    def finish_unfinished(self, *, reason: str, active_status: FailureStatus) -> Self:
        """Finalize interrupted work without altering any completed candidate."""
        if active_status not in {"failed", "cancelled"}:
            raise ManifestError("Unfinished candidates must fail or be cancelled")
        result = self
        for row in self.candidates:
            if row.status in {"pending", "running"}:
                result = result.transition(
                    row.candidate_id,
                    "not_run" if row.status == "pending" else active_status,
                    reason=reason,
                )
        return result


class ComparisonReceipt(BaseModel):
    """Bind actual execution data without modifying the approved specification."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )

    schema_version: SchemaVersion = 1
    job_id: ComparisonJobId
    manifest_hash: Sha256
    snapshot_sha256: Sha256
    data_binding_json: str
    source_dataset_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)
    ]

    @field_validator("data_binding_json")
    @classmethod
    def valid_binding(cls, value: str) -> str:
        return canonical_json_object(value)

    def verify(self, *, job_id: str, expected_hash: str) -> None:
        type(self).model_validate_json(self.model_dump_json())
        if self.job_id != job_id or self.manifest_hash != expected_hash:
            raise ManifestError("Comparison receipt identity does not match the job")
