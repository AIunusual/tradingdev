"""Shared notebook operations; no strategy changes, executions or transcript hooks."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from filelock import Timeout
from pydantic import TypeAdapter

from tradingdev.adapters.storage.research_archives import ResearchArchiveStore
from tradingdev.adapters.storage.research_notes import (
    NoteConflictError,
    ResearchNoteStore,
)
from tradingdev.app.contracts.common import ErrorResponse, Page, PageRequest
from tradingdev.app.contracts.research_notes import (
    NoteExport,
    NoteList,
    NoteRead,
    NoteSaved,
    NoteSummary,
)
from tradingdev.domain.research_notes import NoteId, NoteVersion, ResearchNoteDraft

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore


def _error(exc: Exception) -> ErrorResponse:
    return ErrorResponse(
        success=False,
        code="research_note_conflict"
        if isinstance(exc, NoteConflictError)
        else "research_note_failed",
        error=str(exc),
    )


class ResearchNoteService:
    def __init__(self, workspace: WorkspacePaths, store: SQLiteStore) -> None:
        self.notes = ResearchNoteStore(store)
        self.archives = ResearchArchiveStore(workspace, store)

    def save(
        self,
        draft: ResearchNoteDraft,
        note_id: str | None = None,
        expected_version: int | None = None,
    ) -> NoteSaved | ErrorResponse:
        try:
            note_id = TypeAdapter(NoteId | None).validate_python(note_id)
            expected_version = TypeAdapter(NoteVersion | None).validate_python(
                expected_version
            )
            # Also enforce the contract for direct Python/CLI callers.
            record = self.notes.save(
                ResearchNoteDraft.model_validate(draft), note_id, expected_version
            )
            return NoteSaved(note_id=record.note_id, version=record.version)
        except (ValueError, OSError, sqlite3.Error) as exc:
            return _error(exc)

    def get(
        self,
        note_id: str | None = None,
        *,
        offset: int = 0,
        limit: int = 20,
    ) -> NoteRead | NoteList | ErrorResponse:
        try:
            page = PageRequest(offset=offset, limit=limit)
            if note_id is not None:
                if offset != 0 or limit != 20:
                    raise ValueError("Pagination applies to the note list, not a note")
                return NoteRead(note=self.notes.load(note_id))
            rows, total = self.notes.list(page.offset, page.limit)
            end = offset + len(rows)
            return NoteList(
                page=Page[NoteSummary](
                    items=[NoteSummary.model_validate(row) for row in rows],
                    total=total,
                    offset=offset,
                    limit=limit,
                    next_offset=end if end < total else None,
                )
            )
        except (ValueError, OSError, sqlite3.Error) as exc:
            return _error(exc)

    def export(self, note_id: str, expected_version: int) -> NoteExport | ErrorResponse:
        try:
            note_id = TypeAdapter(NoteId).validate_python(note_id)
            expected_version = TypeAdapter(NoteVersion).validate_python(
                expected_version
            )
            record = self.notes.load(note_id)
            if record.version != expected_version:
                raise NoteConflictError(
                    "Note changed; read and select its version before exporting"
                )
            return self.archives.publish(record)
        except (ValueError, OSError, sqlite3.Error, Timeout) as exc:
            return _error(exc)

    def get_archive(self, archive_id: str) -> NoteExport | ErrorResponse:
        try:
            return self.archives.describe(archive_id)
        except (ValueError, OSError, sqlite3.Error) as exc:
            return _error(exc)
