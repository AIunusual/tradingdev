"""Client-reported research explanations bound to exact execution parameters.

References record what a host reports retrieving. They neither fetch URLs nor
prove that a host browsed a source, understood it, or accurately described it.
Matching an explicit claim value checks its binding, never its semantic truth.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from tradingdev.domain.comparison import ComparisonManifest
from tradingdev.domain.execution_target import ExecutionTarget  # noqa: TC001
from tradingdev.domain.parameter_values import effective_candidates, parameter_leaves
from tradingdev.domain.presentation.confirmation import (
    MAX_PRESENTATION_PARAMETER_DESCRIPTIONS,
    ConfirmationPresentation,
    PresentationExplanation,
    PresentationLabel,
)
from tradingdev.domain.research_evidence_values import (
    MAX_RESEARCH_EVIDENCE_BYTES as MAX_RESEARCH_EVIDENCE_BYTES,
)
from tradingdev.domain.research_evidence_values import (
    FrozenDict,
    bounded_json,
    canonical,
    digest,
    freeze_json,
)

type EvidenceId = Annotated[
    str, Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")
]
type ContentHash = Annotated[
    str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
]
type ReferenceId = Annotated[
    str, StringConstraints(max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
]
type ShortText = PresentationLabel
type Explanation = PresentationExplanation
type ParameterPath = Annotated[
    str,
    StringConstraints(
        min_length=1, max_length=1024, pattern=r"^(?:/(?:[^~/]|~[01])*)+$"
    ),
]
type TargetId = Annotated[
    str,
    StringConstraints(
        max_length=64,
        pattern=r"^(?:base|candidate_[1-8]|search_[1-9][0-9]*_[1-9][0-9]*)$",
    ),
]
type ProvenanceKind = Literal["user", "project_default", "reference", "model_proposal"]

MAX_PARAMETER_BINDINGS = MAX_PRESENTATION_PARAMETER_DESCRIPTIONS
MAX_PARAMETER_TARGETS = 1024


class _EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    def __hash__(self) -> int:
        return hash(canonical(self.model_dump(mode="json")))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _EvidenceModel):
            return NotImplemented
        return type(self) is type(other) and canonical(
            self.model_dump(mode="json")
        ) == canonical(other.model_dump(mode="json"))

    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Research evidence schema_version must be an integer")
        return value


class ParameterClaim(_EvidenceModel):
    """A host's explicit assertion that a source supports one parameter value."""

    claim_id: ReferenceId
    parameter: Explanation
    value: JsonValue

    @field_validator("value", mode="before")
    @classmethod
    def finite_value(cls, value: object) -> JsonValue:
        return bounded_json(value)

    @field_validator("value")
    @classmethod
    def immutable_value(cls, value: JsonValue) -> JsonValue:
        return freeze_json(value)


class ResearchReference(_EvidenceModel):
    """Retrieval metadata reported by a host or supplied by the user."""

    source_id: ReferenceId
    title: ShortText
    url: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    accessed_at: datetime
    retrieval_kind: Literal["host_web", "user_supplied"]
    support: Explanation
    claims: tuple[ParameterClaim, ...] = Field(default=(), max_length=128)

    @field_validator("url")
    @classmethod
    def safe_reference_url(cls, value: str) -> str:
        if any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in value
        ):
            raise ValueError("Reference URLs cannot contain whitespace or controls")
        try:
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or "\\" in value
            ):
                raise ValueError(
                    "References require an HTTP(S) URL without credentials"
                )
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("Invalid HTTP(S) reference URL") from exc
        return value

    @field_validator("accessed_at")
    @classmethod
    def aware_access(cls, value: datetime) -> datetime:
        return _aware(value)

    @model_validator(mode="after")
    def unique_claims(self) -> Self:
        _unique(tuple(claim.claim_id for claim in self.claims), "claim IDs")
        return self


class RuleVariable(_EvidenceModel):
    symbol: ShortText
    meaning: Explanation
    unit: ShortText


class ResearchRule(_EvidenceModel):
    """Unexecuted client wording and formulas for the reader to inspect."""

    rule_id: ReferenceId
    condition: Explanation
    action: Explanation
    formula: Explanation
    variables: tuple[RuleVariable, ...] = Field(default=(), max_length=64)
    source_ids: tuple[ReferenceId, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def unique_links(self) -> Self:
        _unique(self.source_ids, "rule source IDs")
        _unique(
            tuple(variable.symbol for variable in self.variables), "variable symbols"
        )
        return self


class SourceClaimReference(_EvidenceModel):
    source_id: ReferenceId
    claim_id: ReferenceId


class ParameterProvenanceDeclaration(_EvidenceModel):
    """One reason covers the Cartesian product of targets and exact JSON paths."""

    target_ids: tuple[TargetId, ...] = Field(min_length=1, max_length=1024)
    parameter_paths: tuple[ParameterPath, ...] = Field(min_length=1, max_length=256)
    provenance: ProvenanceKind
    reason: Explanation
    source_claims: tuple[SourceClaimReference, ...] = Field(default=(), max_length=128)

    @model_validator(mode="after")
    def valid_declaration(self) -> Self:
        _unique(self.target_ids, "declaration target IDs")
        _unique(self.parameter_paths, "declaration parameter paths")
        _unique(
            tuple((claim.source_id, claim.claim_id) for claim in self.source_claims),
            "source claim references",
        )
        if len(self.target_ids) * len(self.parameter_paths) > MAX_PARAMETER_BINDINGS:
            raise ValueError("Parameter declaration covers too many bindings")
        if (self.provenance == "reference") != bool(self.source_claims):
            raise ValueError(
                "Only reference provenance requires explicit source claim references"
            )
        return self


class ResearchEvidenceDraft(_EvidenceModel):
    """Bounded client explanations; executable values are supplied by manifests."""

    schema_version: Literal[1] = 1
    presentation: ConfirmationPresentation
    rules: tuple[ResearchRule, ...] = Field(min_length=1, max_length=64)
    limits: tuple[Explanation, ...] = Field(min_length=1, max_length=64)
    sources: tuple[ResearchReference, ...] = Field(default=(), max_length=32)
    parameter_provenance: tuple[ParameterProvenanceDeclaration, ...] = Field(
        default=(), max_length=512
    )

    @field_validator("presentation")
    @classmethod
    def owned_presentation(
        cls, value: ConfirmationPresentation
    ) -> ConfirmationPresentation:
        owned = ConfirmationPresentation.model_validate_json(value.model_dump_json())
        object.__setattr__(
            owned, "parameter_descriptions", FrozenDict(owned.parameter_descriptions)
        )
        return owned

    @model_validator(mode="after")
    def consistent_references(self) -> Self:
        _unique(tuple(source.source_id for source in self.sources), "source IDs")
        _unique(tuple(rule.rule_id for rule in self.rules), "rule IDs")
        source_ids = {source.source_id for source in self.sources}
        claims = {
            (source.source_id, claim.claim_id)
            for source in self.sources
            for claim in source.claims
        }
        for rule in self.rules:
            if not set(rule.source_ids) <= source_ids:
                raise ValueError("Research rule refers to an unknown source")
        total = 0
        for declaration in self.parameter_provenance:
            total += len(declaration.target_ids) * len(declaration.parameter_paths)
            if any(
                (reference.source_id, reference.claim_id) not in claims
                for reference in declaration.source_claims
            ):
                raise ValueError(
                    "Parameter provenance refers to an unknown source claim"
                )
        if total > MAX_PARAMETER_BINDINGS:
            raise ValueError("Research declarations exceed the parameter binding limit")
        _check_size(self)
        return self

    def verify(self) -> None:
        """Revalidate even objects changed through unsafe model-copy operations."""
        type(self).model_validate_json(self.model_dump_json())


class ResearchEvidenceRecord(_EvidenceModel):
    """An append-only identity for an independently detached research draft."""

    schema_version: Literal[1] = 1
    evidence_id: EvidenceId
    created_at: datetime
    draft: ResearchEvidenceDraft
    content_hash: ContentHash

    @field_validator("created_at")
    @classmethod
    def aware_creation(cls, value: datetime) -> datetime:
        return _aware(value)

    @classmethod
    def create(
        cls,
        draft: ResearchEvidenceDraft,
        *,
        evidence_id: str | None = None,
        created_at: datetime | None = None,
    ) -> Self:
        draft.verify()
        payload = {
            "schema_version": 1,
            "evidence_id": evidence_id if evidence_id is not None else uuid4().hex,
            "created_at": _datetime_json(_aware(created_at or datetime.now(UTC))),
            "draft": draft.model_dump(mode="json"),
        }
        return cls.model_validate({**payload, "content_hash": digest(payload)})

    @model_validator(mode="after")
    def valid_record(self) -> Self:
        self.verify()
        return self

    def verify(self, expected_hash: str | None = None) -> None:
        self.draft.verify()
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("Unsupported research evidence schema version")
        _aware(self.created_at)
        if expected_hash is not None and expected_hash != self.content_hash:
            raise ValueError("Research evidence does not match the expected hash")
        payload = self.model_dump(mode="json", exclude={"content_hash"})
        if digest(payload) != self.content_hash:
            raise ValueError("Research evidence hash does not match its contents")
        _check_size(self)


class ParameterValue(_EvidenceModel):
    """A backend value at an exact constructor leaf, without client repetition."""

    parameter_path: ParameterPath
    value: JsonValue

    @field_validator("value", mode="before")
    @classmethod
    def finite_value(cls, value: object) -> JsonValue:
        return bounded_json(value)

    @field_validator("value")
    @classmethod
    def immutable_value(cls, value: JsonValue) -> JsonValue:
        return freeze_json(value)


class ParameterBindingTarget(_EvidenceModel):
    target_id: TargetId
    manifest_hash: ContentHash
    parameters: tuple[ParameterValue, ...] = Field(max_length=MAX_PARAMETER_BINDINGS)


class BoundParameterProvenance(ParameterValue):
    """A declaration expanded onto one authoritative, captured leaf value."""

    target_id: TargetId
    provenance: ProvenanceKind
    reason: Explanation
    source_claims: tuple[SourceClaimReference, ...] = Field(default=(), max_length=128)


class ResearchEvidenceBindingError(ValueError):
    """A repairable mismatch with the complete bounded target/path requirements."""

    def __init__(
        self, message: str, *, targets: tuple[ParameterBindingTarget, ...]
    ) -> None:
        super().__init__(message)
        self.required_parameter_bindings = [
            target.model_dump(mode="json") for target in targets
        ]
        self.required_parameter_paths = sorted(
            {
                parameter.parameter_path
                for target in targets
                for parameter in target.parameters
            }
        )


class BoundResearchEvidence(_EvidenceModel):
    """A complete record and expanded provenance tied to parent/child hashes."""

    schema_version: Literal[1] = 1
    record: ResearchEvidenceRecord
    manifest_hash: ContentHash
    targets: tuple[ParameterBindingTarget, ...] = Field(
        min_length=1, max_length=MAX_PARAMETER_TARGETS
    )
    parameters: tuple[BoundParameterProvenance, ...] = Field(
        max_length=MAX_PARAMETER_BINDINGS
    )
    content_hash: ContentHash

    @model_validator(mode="after")
    def valid_binding(self) -> Self:
        self.verify()
        return self

    def verify(self, target: ExecutionTarget | None = None) -> None:
        """Check snapshot integrity and, when supplied, the exact live target."""
        self.record.verify()
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("Unsupported bound research evidence schema version")
        if len(self.targets) > MAX_PARAMETER_TARGETS:
            raise ValueError("Bound research evidence has too many targets")
        expected_parameters = _expand(self.record, self.targets)
        if self.parameters != expected_parameters:
            raise ValueError("Bound parameter provenance differs from its declarations")
        if target is not None:
            expected_targets = enumerate_parameter_bindings(target)
            if (
                self.manifest_hash != target.manifest_hash
                or self.targets != expected_targets
            ):
                raise ValueError(
                    "Research evidence belongs to another execution target"
                )
        payload = self.model_dump(mode="json", exclude={"content_hash"})
        if digest(payload) != self.content_hash:
            raise ValueError("Bound research evidence hash does not match its contents")
        _check_size(self)


def enumerate_parameter_bindings(
    target: ExecutionTarget,
) -> tuple[ParameterBindingTarget, ...]:
    """Enumerate base, fixed candidates, or each merged search-axis subtree.

    Search IDs identify one axis option, not a Cartesian-product execution.
    Captured defaults within the replaced subtree remain explicit leaf values.
    """
    target.verify()
    targets: list[ParameterBindingTarget] = []

    def append(
        target_id: str, manifest_hash: str, value: object, pointer: str = ""
    ) -> None:
        leaves = parameter_leaves(value, pointer)
        if len(leaves) > MAX_PARAMETER_BINDINGS:
            raise ValueError("Research execution has too many parameter leaves")
        targets.append(
            ParameterBindingTarget.model_validate(
                {
                    "target_id": target_id,
                    "manifest_hash": manifest_hash,
                    "parameters": [
                        {"parameter_path": path, "value": leaf}
                        for path, leaf in sorted(leaves.items())
                    ],
                }
            )
        )

    if isinstance(target, ComparisonManifest):
        for candidate in target.candidates:
            append(
                candidate.candidate_id,
                candidate.manifest.manifest_hash,
                candidate.manifest.strategy_execution.constructor_kwargs,
            )
    else:
        if (
            target.optimization is not None
            and 1
            + sum(len(values) for values in target.optimization.param_ranges.values())
            > MAX_PARAMETER_TARGETS
        ):
            raise ValueError("Research execution has too many search targets")
        append(
            "base", target.manifest_hash, target.strategy_execution.constructor_kwargs
        )
        for group, candidate_index, pointer, value in effective_candidates(target):
            append(
                f"search_{group}_{candidate_index}",
                target.manifest_hash,
                value,
                pointer,
            )
    if sum(len(item.parameters) for item in targets) > MAX_PARAMETER_BINDINGS:
        raise ValueError("Research execution exceeds the parameter binding limit")
    return tuple(targets)


def bind_research_evidence(
    record: ResearchEvidenceRecord, target: ExecutionTarget
) -> BoundResearchEvidence:
    """Bind a detached saved record; never fetch a URL or execute a formula."""
    record.verify()
    targets = enumerate_parameter_bindings(target)
    parameters = _expand(record, targets)
    payload = {
        "schema_version": 1,
        "record": record.model_dump(mode="json"),
        "manifest_hash": target.manifest_hash,
        "targets": [item.model_dump(mode="json") for item in targets],
        "parameters": [item.model_dump(mode="json") for item in parameters],
    }
    result = BoundResearchEvidence.model_validate(
        {**payload, "content_hash": digest(payload)}
    )
    result.verify(target)
    return result


def _expand(
    record: ResearchEvidenceRecord, targets: tuple[ParameterBindingTarget, ...]
) -> tuple[BoundParameterProvenance, ...]:
    def fail(message: str) -> None:
        raise ResearchEvidenceBindingError(message, targets=targets)

    _unique(tuple(target.target_id for target in targets), "binding target IDs")
    required: dict[tuple[str, str], JsonValue] = {}
    for target in targets:
        for parameter in target.parameters:
            key = (target.target_id, parameter.parameter_path)
            if key in required:
                fail("Research binding target contains duplicate parameter paths")
            required[key] = parameter.value
    if len(required) > MAX_PARAMETER_BINDINGS:
        fail("Research execution exceeds the parameter binding limit")
    descriptions = record.draft.presentation.parameter_descriptions
    required_paths = {path for _target_id, path in required}
    if set(descriptions) != required_paths:
        fail("Research parameter descriptions must exactly cover the target paths")
    labels = [description.label for description in descriptions.values()]
    if len(labels) != len(set(labels)):
        fail("Research parameter labels must distinguish every captured path")
    claims = {
        (source.source_id, claim.claim_id): claim.value
        for source in record.draft.sources
        for claim in source.claims
    }
    actual: dict[tuple[str, str], BoundParameterProvenance] = {}
    for declaration in record.draft.parameter_provenance:
        for target_id in declaration.target_ids:
            for path in declaration.parameter_paths:
                key = (target_id, path)
                if key not in required:
                    fail(f"Unknown research parameter binding: {target_id}{path}")
                if key in actual:
                    fail(
                        "Overlapping research parameter declarations: "
                        f"{target_id}{path}"
                    )
                value = required[key]
                if declaration.provenance == "reference" and not all(
                    canonical(claims[(reference.source_id, reference.claim_id)])
                    == canonical(value)
                    for reference in declaration.source_claims
                ):
                    fail(
                        "Source claims do not support the effective value: "
                        f"{target_id}{path}"
                    )
                actual[key] = BoundParameterProvenance(
                    target_id=target_id,
                    parameter_path=path,
                    value=value,
                    provenance=declaration.provenance,
                    reason=declaration.reason,
                    source_claims=declaration.source_claims,
                )
    missing = required.keys() - actual.keys()
    if missing:
        fail(f"Research parameter provenance is incomplete: {sorted(missing)}")
    return tuple(actual[key] for key in required)


def _unique(values: tuple[object, ...], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Research evidence requires unique {label}")


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Research evidence timestamps must be timezone-aware")
    return value


def _datetime_json(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _check_size(model: BaseModel) -> None:
    if len(canonical(model.model_dump(mode="json"))) > MAX_RESEARCH_EVIDENCE_BYTES:
        raise ValueError("Research evidence exceeds the maximum serialized byte size")
