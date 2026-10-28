"""Bounded offline preparation requests and explicit program-check evidence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

# Pydantic resolves these annotations when validating the wire schema.
from tradingdev.domain.comparison import ComparisonManifest
from tradingdev.domain.execution import ExecutionKind, ExecutionManifest  # noqa: TC001
from tradingdev.domain.execution_target import ExecutionTarget  # noqa: TC001
from tradingdev.domain.strategies.capabilities import TrainingRequirement  # noqa: TC001
from tradingdev.domain.strategies.contract import (
    DRY_RUN_FIXTURE_ROWS,
    VALIDATE_FIXTURE_ROWS,
)

type PreflightCheck = Literal[
    "configuration",
    "execution_capabilities",
    "parameter_binding",
    "manifest_serialization",
    "signal_contract",
    "candidate_binding",
]
type UnverifiedCheck = Literal[
    "historical_data", "backtest_engine", "training", "performance"
]
type SignalCheck = Literal[
    "synthetic_contract_passed", "deferred_until_training", "bundled_not_executed"
]


class PreflightModel(BaseModel):
    """Reject ambiguous values in preparation requests and saved evidence."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PreflightRequest(PreflightModel):
    """Capture a concrete execution request without authorizing historical work."""

    kind: ExecutionKind | Literal["comparison"]
    arguments: dict[str, JsonValue]


class PreflightReceipt(PreflightModel):
    """Offline evidence, explicitly excluding market, training and engine results."""

    schema_version: Literal[2] = 2
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    elapsed_seconds: float = Field(ge=0)
    checked_paths: list[PreflightCheck]
    synthetic_fixture_rows: list[int]
    signal_check: SignalCheck
    training_requirement: TrainingRequirement
    parameter_combinations_checked: int | None = Field(default=None, ge=1)
    not_checked: list[UnverifiedCheck]

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Preflight schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def consistent_coverage(self) -> Self:
        checks = set(self.checked_paths)
        required: set[PreflightCheck] = {
            "configuration",
            "execution_capabilities",
            "parameter_binding",
            "manifest_serialization",
        }
        if not required.issubset(checks) or len(checks) != len(self.checked_paths):
            raise ValueError("Preflight requires every distinct program check")
        if self.signal_check == "synthetic_contract_passed":
            if (
                self.synthetic_fixture_rows
                != [
                    VALIDATE_FIXTURE_ROWS,
                    DRY_RUN_FIXTURE_ROWS,
                ]
                or "signal_contract" not in checks
            ):
                raise ValueError("Synthetic evidence requires both contract fixtures")
            if self.training_requirement == "required":
                raise ValueError(
                    "Training-required signals cannot be checked untrained"
                )
        elif self.synthetic_fixture_rows or "signal_contract" in checks:
            raise ValueError("Unexecuted signals cannot claim synthetic coverage")
        if (self.signal_check == "deferred_until_training") != (
            self.training_requirement == "required"
        ):
            raise ValueError("Deferred signals must match the training requirement")
        if (self.parameter_combinations_checked is not None) != (
            "candidate_binding" in checks
        ):
            raise ValueError("Candidate coverage must match parameter binding checks")
        if (
            set(self.not_checked)
            != {
                "historical_data",
                "backtest_engine",
                "training",
                "performance",
            }
            or len(self.not_checked) != 4
        ):
            raise ValueError("Offline evidence must disclose every unverified path")
        return self

    def verify_manifest(
        self, manifest: ExecutionManifest, *, comparison_candidate: bool = False
    ) -> None:
        """Validate actual coverage against the exact execution used by every view."""
        # Lists remain mutable after model construction. Revalidate their types and
        # coverage before a receipt is reused by storage, approval or presentation.
        type(self).model_validate(self.model_dump(mode="python"))
        manifest.verify(expected_hash=self.manifest_hash)
        generated = manifest.strategy_execution.kind == "generated"
        if comparison_candidate and self.signal_check != "synthetic_contract_passed":
            raise ValueError(
                "Every comparison candidate requires both contract fixtures"
            )
        if not comparison_candidate and generated != (
            self.signal_check == "synthetic_contract_passed"
        ):
            raise ValueError("Signal evidence differs from the strategy execution kind")
        if generated and self.training_requirement != "unknown":
            raise ValueError(
                "Generated strategies have no declared training capability"
            )
        if self.training_requirement == "required" and manifest.kind != "walk_forward":
            raise ValueError(
                "Training-required preparation needs walk-forward settings"
            )
        search = manifest.optimization
        combinations = None if search is None else search.total_combinations
        if self.parameter_combinations_checked != combinations:
            raise ValueError("Candidate binding evidence differs from the search")


class ComparisonCandidateReceipt(PreflightModel):
    """Program-check evidence for one exact candidate in an ordered comparison."""

    candidate_id: str = Field(pattern=r"^candidate_[1-8]$")
    receipt: PreflightReceipt


class ComparisonPreflightReceipt(PreflightModel):
    """Keep each candidate's checks separate from optimization grid estimates."""

    schema_version: Literal[1] = 1
    kind: Literal["comparison"] = "comparison"
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    elapsed_seconds: float = Field(ge=0)
    candidates: list[ComparisonCandidateReceipt] = Field(min_length=1, max_length=8)

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Comparison preflight schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def ordered_candidates(self) -> Self:
        if [item.candidate_id for item in self.candidates] != [
            f"candidate_{index}" for index in range(1, len(self.candidates) + 1)
        ]:
            raise ValueError("Comparison evidence must retain exact candidate order")
        return self

    def verify_manifest(self, manifest: ComparisonManifest) -> None:
        type(self).model_validate(self.model_dump(mode="python"))
        manifest.verify(expected_hash=self.manifest_hash)
        if len(self.candidates) != len(manifest.candidates):
            raise ValueError("Comparison evidence must cover every candidate")
        for checked, candidate in zip(
            self.candidates, manifest.candidates, strict=True
        ):
            if checked.candidate_id != candidate.candidate_id:
                raise ValueError(
                    "Comparison evidence must retain exact candidate order"
                )
            checked.receipt.verify_manifest(
                candidate.manifest, comparison_candidate=True
            )


type ProgramReceipt = PreflightReceipt | ComparisonPreflightReceipt


def verify_program_receipt(manifest: ExecutionTarget, receipt: ProgramReceipt) -> None:
    """Verify the matching receipt shape and every captured execution identity."""
    if isinstance(manifest, ComparisonManifest):
        if not isinstance(receipt, ComparisonPreflightReceipt):
            raise ValueError("Comparison requires per-candidate program evidence")
        receipt.verify_manifest(manifest)
    else:
        if not isinstance(receipt, PreflightReceipt):
            raise ValueError("Single execution requires single program evidence")
        receipt.verify_manifest(manifest)


class PreflightPayload(PreflightModel):
    """Verified child-process output independent of job and plan storage."""

    manifest: ExecutionTarget
    original_config_path: str
    receipt: ProgramReceipt

    @model_validator(mode="after")
    def matching_manifest(self) -> Self:
        verify_program_receipt(self.manifest, self.receipt)
        return self
