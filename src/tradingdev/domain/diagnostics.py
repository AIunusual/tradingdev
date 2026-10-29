"""Provider-neutral research observations; suggestions are untested hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from tradingdev.domain.performance.artifacts import (
        PerformanceScope,
        ScopeObservations,
    )


class DiagnosticModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class MetricEvidence(DiagnosticModel):
    metric: str
    value: float
    unit: str


class DiagnosticFinding(DiagnosticModel):
    code: str
    observation: str
    evidence: list[MetricEvidence]
    suggested_check: str


class DiagnosticResult(DiagnosticModel):
    provider: str
    version: str
    source: str
    status: Literal["evaluated", "unavailable"]
    findings: list[DiagnosticFinding]
    evaluated_rules: list[str]
    unavailable_rules: dict[str, str]
    limitations: list[str]


@dataclass(frozen=True)
class DiagnosticContext:
    """Retain full native data for future providers, not just imported metrics."""

    performance: PerformanceScope
    observations: ScopeObservations | None


class DiagnosticProvider(Protocol):
    def diagnose(self, context: DiagnosticContext) -> DiagnosticResult:
        """Describe saved evidence without executing or modifying a strategy."""
        ...
