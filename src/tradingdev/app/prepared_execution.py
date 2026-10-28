"""Captured specifications shared by preparation and submission services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.domain.approved_research import ApprovedResearch
    from tradingdev.domain.comparison import ComparisonManifest
    from tradingdev.domain.execution import ExecutionManifest


@dataclass(frozen=True)
class PreparedExecution:
    """A single captured execution and its configuration/check provenance."""

    manifest: ExecutionManifest
    original_config_path: Path
    synthetic_fixture_rows: tuple[int, ...] = ()
    research_evidence: ApprovedResearch | None = None


@dataclass(frozen=True)
class PreparedComparison:
    """All fixed candidates, with separate synthetic check evidence for each."""

    manifest: ComparisonManifest
    original_config_path: Path
    candidate_fixture_rows: dict[str, tuple[int, ...]]
    research_evidence: ApprovedResearch | None = None


type PreparedSubmission = PreparedExecution | PreparedComparison
