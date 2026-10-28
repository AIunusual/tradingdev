"""A fixed, ordered parameter comparison over one shared research context."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from tradingdev.domain.execution import ExecutionManifest, ManifestError

MAX_COMPARISON_CANDIDATES = 8
type CandidateLabel = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)
]
type CandidateRationale = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class ComparisonCandidate(BaseModel):
    """One complete execution, with wording captured before seeing performance."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    candidate_id: str = Field(pattern=r"^candidate_[1-8]$")
    label: CandidateLabel
    rationale: CandidateRationale
    manifest: ExecutionManifest


class ComparisonManifest(BaseModel):
    """Bind all candidates without adding a search objective or selecting a winner.

    Only constructor parameters may differ. Market data, execution assumptions,
    strategy identity and seed remain identical across the ordered candidates.
    The manifest fixes the data request; its eventual snapshot is execution
    evidence and cannot be known or acquired during offline preparation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[1] = 1
    kind: Literal["comparison"] = "comparison"
    candidates: tuple[ComparisonCandidate, ...] = Field(
        min_length=1, max_length=MAX_COMPARISON_CANDIDATES
    )
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Comparison schema_version must be an integer")
        return value

    @classmethod
    def create(cls, candidates: list[ComparisonCandidate]) -> Self:
        """Own detached children and hash labels, rationales, order and executions."""
        payload = {
            "schema_version": 1,
            "kind": "comparison",
            "candidates": [
                candidate.model_dump(mode="json") for candidate in candidates
            ],
        }
        return cls.model_validate(
            {
                **payload,
                "manifest_hash": hashlib.sha256(_canonical(payload)).hexdigest(),
            }
        )

    @model_validator(mode="after")
    def consistent_candidates(self) -> Self:
        self.verify()
        return self

    def verify(self, expected_hash: str | None = None) -> None:
        """Reject mutated children, duplicated settings or changed shared context."""
        if not 1 <= len(self.candidates) <= MAX_COMPARISON_CANDIDATES:
            raise ManifestError("A comparison requires one to eight fixed candidates")
        shared: bytes | None = None
        constructors: set[bytes] = set()
        labels: set[str] = set()
        for index, candidate in enumerate(self.candidates, start=1):
            if candidate.candidate_id != f"candidate_{index}":
                raise ManifestError("Comparison candidate IDs must follow their order")
            manifest = candidate.manifest
            manifest.verify()
            if manifest.kind != "backtest":
                raise ManifestError(
                    "Fixed comparisons require ordinary backtest candidates"
                )
            config = _shared_config(manifest)
            context = _canonical(
                {
                    "config": config,
                    "strategy_execution": manifest.strategy_execution.model_dump(
                        mode="json", exclude={"constructor_kwargs"}
                    ),
                }
            )
            if shared is not None and context != shared:
                raise ManifestError(
                    "Comparison candidates must share strategy, research, data, "
                    "execution assumptions and seed; only parameters may differ"
                )
            shared = context
            parameters = _canonical(manifest.strategy_execution.constructor_kwargs)
            if parameters in constructors:
                raise ManifestError(
                    "Comparison candidates have duplicate effective parameters"
                )
            constructors.add(parameters)
            if candidate.label in labels:
                raise ManifestError("Comparison candidate labels must be distinct")
            labels.add(candidate.label)
        if expected_hash is not None and expected_hash != self.manifest_hash:
            raise ManifestError("Comparison manifest does not match the expected hash")
        payload = self.model_dump(mode="json", exclude={"manifest_hash"})
        if hashlib.sha256(_canonical(payload)).hexdigest() != self.manifest_hash:
            raise ManifestError("Comparison manifest hash does not match its contents")

    def candidate(self, candidate_id: str) -> ComparisonCandidate:
        """Return a detached, verified member of this fixed comparison."""
        self.verify()
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return ComparisonCandidate.model_validate_json(
                    candidate.model_dump_json()
                )
        raise ManifestError(f"Unknown comparison candidate: {candidate_id}")

    def common_config(self) -> dict[str, Any]:
        """Return shared settings with candidate parameters removed."""
        self.verify()
        return _shared_config(self.candidates[0].manifest)


def _shared_config(manifest: ExecutionManifest) -> dict[str, Any]:
    config = manifest.config_copy()
    strategy = config.get("strategy")
    if (
        not isinstance(strategy, dict)
        or not isinstance(strategy.get("id"), str)
        or not strategy["id"]
    ):
        raise ManifestError("Comparison candidate requires a strategy identity")
    strategy.pop("parameters", None)
    return config
