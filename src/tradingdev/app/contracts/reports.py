"""Public identity of a reproducible, offline report artifact."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from tradingdev.app.contracts.common import ContractModel


class ReportResponse(ContractModel):
    """A report and its source manifest, never an unbounded HTML tool response."""

    success: Literal[True]
    report_id: str
    artifact_id: str
    manifest_artifact_id: str
    path: str
    sha256: str
    run_ids: list[str]
    navigation_instructions: str
    scope_count: int
    comparison_job_id: str | None = None
    ml_entry_ids: list[str] = Field(default_factory=list)


class ReportSection(ContractModel):
    """A backend-rendered section that clients may select without writing HTML."""

    id: str
    title: str
    description: str


class ReportSectionCatalog(ContractModel):
    """Common recipes are suggestions; an explicit empty selection is valid."""

    success: Literal[True]
    sections: list[ReportSection]
    templates: dict[str, list[str]]


class ReportCommentary(ContractModel):
    """Client-authored interpretation, displayed separately from computed values."""

    title: str
    text: str


type AnalysisText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
]


class AnalysisReference(ContractModel):
    run_id: str
    scope: str


class ReportAnalysis(ContractModel):
    """Client-authored interpretation, distinct from saved/computed evidence."""

    summary: AnalysisText
    observations: AnalysisText
    hypotheses_and_limits: AnalysisText
    next_steps: AnalysisText
    references: Annotated[list[AnalysisReference], Field(max_length=32)]
