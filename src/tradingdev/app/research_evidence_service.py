"""Save research explanations without executing strategies or fetching sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.adapters.storage.research_evidence import ResearchEvidenceStore
from tradingdev.app.contracts.common import ErrorResponse
from tradingdev.app.contracts.research_evidence import ResearchEvidenceSaved
from tradingdev.domain.research_evidence import ResearchEvidenceRecord

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.domain.research_evidence import ResearchEvidenceDraft


class ResearchEvidenceService:
    """Keep complete immutable declarations behind a compact saved identity."""

    def __init__(self, workspace: WorkspacePaths) -> None:
        self.store = ResearchEvidenceStore(workspace)

    def save(
        self, draft: ResearchEvidenceDraft
    ) -> ResearchEvidenceSaved | ErrorResponse:
        """Create a new record each time; this does not verify external sources."""
        try:
            record = ResearchEvidenceRecord.create(draft)
            self.store.save(record)
            return ResearchEvidenceSaved(
                success=True,
                evidence_id=record.evidence_id,
                content_hash=record.content_hash,
                source_count=len(record.draft.sources),
                parameter_declaration_count=len(record.draft.parameter_provenance),
            )
        except (OSError, ValueError) as exc:
            return ErrorResponse(
                success=False,
                code="research_evidence_save_failed",
                error=str(exc),
            )

    def load(self, evidence_id: str) -> ResearchEvidenceRecord:
        """Return verified internal detail for plan preparation, or raise."""
        return self.store.load(evidence_id)
