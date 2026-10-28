"""Immutable reviewed research plans, independent of client approval transport."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime  # noqa: TC003
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tradingdev.domain.approved_research import ApprovedResearch  # noqa: TC001
from tradingdev.domain.execution_target import ExecutionTarget  # noqa: TC001
from tradingdev.domain.preflight import ProgramReceipt, verify_program_receipt
from tradingdev.domain.presentation.confirmation import (
    ConfirmationDocument,  # noqa: TC001
)

type PlanState = Literal[
    "ready", "awaiting_confirmation", "submitting", "submitted", "cancelled", "failed"
]


class ExecutionPlan(BaseModel):
    """Bind the user-visible document and trial evidence to the executed bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[3] = 3
    plan_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    manifest: ExecutionTarget
    original_config_path: str
    preflight: ProgramReceipt
    research_evidence: ApprovedResearch
    document: ConfirmationDocument
    created_at: datetime
    expires_at: datetime
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Execution plan schema_version must be an integer")
        return value

    @staticmethod
    def digest(payload: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    @model_validator(mode="after")
    def _validate_contents(self) -> Self:
        self.verify()
        return self

    def verify(self) -> None:
        if self.schema_version != 3 or type(self.schema_version) is not int:
            raise ValueError("Unsupported execution plan schema version")
        self.manifest.verify()
        self.research_evidence.verify(self.manifest)
        verify_program_receipt(self.manifest, self.preflight)
        if self.document.identity.manifest_hash != self.manifest.manifest_hash:
            raise ValueError("Confirmation document belongs to another execution")
        if self.document.identity.plan_id != self.plan_id:
            raise ValueError("Confirmation document belongs to another plan")
        if (
            self.created_at.tzinfo is None
            or self.expires_at.tzinfo is None
            or self.expires_at <= self.created_at
        ):
            raise ValueError("Plan requires an ordered, timezone-aware validity period")
        payload = self.model_dump(mode="json", exclude={"plan_hash"})
        if self.plan_hash != self.digest(payload):
            raise ValueError("Execution plan hash does not match its contents")
