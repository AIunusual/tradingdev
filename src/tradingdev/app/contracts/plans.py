"""Public review plans and conversation-mediated execution decisions."""

from typing import Literal

from pydantic import Field

from tradingdev.app.contracts.common import ContractModel, ErrorResponse
from tradingdev.domain.execution_plan import PlanState
from tradingdev.domain.preflight import PreflightReceipt

type ExecutionDecision = Literal["approve", "decline", "revise"]


class PlanResponse(ContractModel):
    success: Literal[True] = True
    plan_id: str
    status: PlanState
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    expires_at: str
    confirmation_summary: str
    html_path: str
    artifact_id: str
    preflight: PreflightReceipt
    job_id: str | None = None
    error: str | None = None


class PlanRejected(ErrorResponse):
    required_parameter_paths: list[str] | None = None


class PlanStarted(ContractModel):
    success: Literal[True] = True
    plan_id: str
    job_id: str
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    message: str
