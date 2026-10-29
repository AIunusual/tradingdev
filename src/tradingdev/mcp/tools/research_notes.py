"""One editable notebook and one export operation across research stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.mcpserver import Context  # noqa: TC002 -- MCP runtime schema
from mcp_types import ToolAnnotations

from tradingdev.app.contracts.common import (  # noqa: TC001 -- MCP runtime schema
    ErrorResponse,
    PageLimit,
    PageOffset,
)
from tradingdev.app.contracts.research_notes import (
    NoteExport,
    NoteList,
    NoteRead,
    NoteSaved,
)
from tradingdev.domain.research_notes import (  # noqa: TC001 -- MCP runtime schema
    NoteId,
    NoteVersion,
    ResearchNoteDraft,
)
from tradingdev.mcp.archive_downloads import ArchiveDownloads

if TYPE_CHECKING:
    from mcp.server import MCPServer

    from tradingdev.app.research_note_service import ResearchNoteService


def register(mcp: MCPServer, service: ResearchNoteService) -> None:
    downloads = ArchiveDownloads(service.archives)

    def deliver(
        result: NoteExport | ErrorResponse, ctx: Context[Any, Any]
    ) -> NoteExport | ErrorResponse:
        if isinstance(result, NoteExport):
            try:
                request = ctx.request_context.request
            except ValueError:
                request = None
            if request is not None:
                result = downloads.attach(result, str(request.base_url))
        return result

    mcp.custom_route("/research-download/{archive_id}", methods=["GET"])(
        downloads.serve
    )

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=False,
        )
    )
    def save_research_note(
        draft: ResearchNoteDraft,
        note_id: NoteId | None = None,
        expected_version: NoteVersion | None = None,
    ) -> NoteSaved | ErrorResponse:
        """Save needs, stage summaries and next steps (16384 UTF-8 bytes).

        New: omit identity/version. Update: read first, send BOTH and the complete
        replacement draft; conflict requires rereading. Preserve prior relevant
        summaries and label unexecuted requests. Not a transcript or run approval.
        Entries may reference report/confirmation artifact IDs for export.
        """
        return service.save(draft, note_id, expected_version)

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def get_research_notes(
        note_id: NoteId | None = None,
        offset: PageOffset = 0,
        limit: PageLimit = 20,
    ) -> NoteRead | NoteList | ErrorResponse:
        """List note identities with pagination, or read one complete bounded note.

        Paging applies only to the list. Read current version before edits/export.
        """
        return service.get(note_id, offset=offset, limit=limit)

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def export_research_note(
        note_id: NoteId,
        expected_version: NoteVersion,
        ctx: Context[Any, Any],
    ) -> NoteExport | ErrorResponse:
        """Export selected note version as offline HTML/JSON/reports ZIP at any stage.

        Up to 20 distinct registered report/confirmation HTMLs, 64 MiB total.
        Missing/changed reports fail explicitly; never reruns. HTTP returns a
        15-minute bearer link; refresh saved copies via get_research_archive.
        Deliver HTTP URL with extraction instructions. Stdio paths need client
        file access; only then link local ZIP/HTML. Keep the whole extracted ZIP.
        """
        return deliver(service.export(note_id, expected_version), ctx)

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def get_research_archive(
        archive_id: str, ctx: Context[Any, Any]
    ) -> NoteExport | ErrorResponse:
        """Read a saved archive and refresh its HTTP download URL without sources.

        Find prior IDs via list_artifacts(artifact_type='research_archive_zip').
        Works after note edits or original reports disappear; verifies ZIP bytes.
        Deliver download_url for HTTP, explaining extraction/opening index.html.
        Local path/html_path links require client filesystem access; HTML may be absent.
        """
        return deliver(service.get_archive(archive_id), ctx)
