"""Bounded client projections of the optimization trial ledger."""

from typing import Literal

from pydantic import JsonValue

from tradingdev.app.contracts.common import ContractModel, Page
from tradingdev.domain.optimization.trials import TrialStateView


class OptimizationTrial(ContractModel):
    """Stable approved position and its recorded outcome."""

    trial_index: int | None
    parameters: dict[str, JsonValue]
    state: TrialStateView
    scope: str | None
    target_value: float | None
    reason: str | None


class OptimizationProgress(ContractModel):
    """Grid coverage, finite selection evidence and separate test execution."""

    outcome: Literal["in_progress", "partial", "complete"]
    manifest_hash: str
    dataset_id: str | None
    objective: str
    direction: Literal["maximize", "minimize"]
    total_combinations: int
    attempted: int
    completed: int
    rankable: int
    search_complete: bool
    estimation: Literal["pending", "available", "unavailable"]
    stop_reason: str | None
    leading_trial_index: int | None
    leading_target_value: float | None
    test: OptimizationTrial


class OptimizationTrialsResponse(ContractModel):
    """A page of approved grid positions, including unattempted combinations."""

    success: Literal[True]
    job_id: str
    progress: OptimizationProgress
    trials: Page[OptimizationTrial]
