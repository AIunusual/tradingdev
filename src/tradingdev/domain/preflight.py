"""Bounded offline preparation requests and explicit program-check evidence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

# Pydantic resolves these annotations when validating the wire schema.
from tradingdev.domain.execution import ExecutionKind, ExecutionManifest  # noqa: TC001
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

    kind: ExecutionKind
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

    def verify_manifest(self, manifest: ExecutionManifest) -> None:
        """Validate actual coverage against the exact execution used by every view."""
        # Lists remain mutable after model construction. Revalidate their types and
        # coverage before a receipt is reused by storage, approval or presentation.
        type(self).model_validate(self.model_dump(mode="python"))
        manifest.verify(expected_hash=self.manifest_hash)
        generated = manifest.strategy_execution.kind == "generated"
        if generated != (self.signal_check == "synthetic_contract_passed"):
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


class PreflightPayload(PreflightModel):
    """Verified child-process output independent of job and plan storage."""

    manifest: ExecutionManifest
    original_config_path: str
    receipt: PreflightReceipt

    @model_validator(mode="after")
    def matching_manifest(self) -> Self:
        self.receipt.verify_manifest(self.manifest)
        return self
