"""Transactional current notes; immutable exports preserve selected versions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from uuid import uuid4

from tradingdev.adapters.storage.filesystem import now_iso
from tradingdev.domain.research_notes import MAX_NOTE_BYTES, ResearchNoteRecord

if TYPE_CHECKING:
    from tradingdev.adapters.storage.sqlite import SQLiteStore
    from tradingdev.domain.research_notes import ResearchNoteDraft


class NoteConflictError(ValueError):
    """The selected revision no longer matches the editable note."""


def note_bytes(record: ResearchNoteRecord) -> bytes:
    content = json.dumps(
        record.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(content) > MAX_NOTE_BYTES:
        raise ValueError("Research note exceeds 16384 UTF-8 bytes; condense summaries")
    return content


class ResearchNoteStore:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store
        with store.connect() as conn:
            conn.execute("""create table if not exists research_notes (
                note_id text primary key, version integer not null,
                title text not null, updated_at text not null, document text not null
            )""")

    def load(self, note_id: str) -> ResearchNoteRecord:
        with self.store.connect() as conn:
            row = conn.execute(
                "select * from research_notes where note_id = ?", (note_id,)
            ).fetchone()
        if row is None:
            raise ValueError("Research note not found")
        record = ResearchNoteRecord.model_validate_json(row["document"])
        if (
            record.note_id != note_id
            or record.version != row["version"]
            or record.draft.title != row["title"]
            or record.updated_at != row["updated_at"]
        ):
            raise ValueError("Research note identity differs from stored index")
        note_bytes(record)
        return record

    def save(
        self,
        draft: ResearchNoteDraft,
        note_id: str | None,
        expected_version: int | None,
    ) -> ResearchNoteRecord:
        if (note_id is None) != (expected_version is None):
            raise ValueError("Updates require both note_id and expected_version")
        previous = self.load(note_id) if note_id is not None else None
        if previous is not None and previous.version != expected_version:
            raise NoteConflictError(
                "Note changed; read the current version before editing"
            )
        now = now_iso()
        record = ResearchNoteRecord(
            note_id=note_id or uuid4().hex,
            version=previous.version + 1 if previous else 1,
            created_at=previous.created_at if previous else now,
            updated_at=now,
            draft=draft,
        )
        content = note_bytes(record).decode("utf-8")
        with self.store.connect() as conn:
            if previous is None:
                conn.execute(
                    "insert into research_notes values (?, ?, ?, ?, ?)",
                    (record.note_id, record.version, draft.title, now, content),
                )
            else:
                changed = conn.execute(
                    """update research_notes set version=?, title=?, updated_at=?,
                    document=? where note_id=? and version=?""",
                    (
                        record.version,
                        draft.title,
                        now,
                        content,
                        note_id,
                        expected_version,
                    ),
                )
                if changed.rowcount != 1:
                    raise NoteConflictError("Note changed; read before editing")
        return record

    def list(self, offset: int, limit: int) -> tuple[list[dict[str, str | int]], int]:
        with self.store.connect() as conn:
            # One read transaction keeps count and page on the same snapshot.
            conn.execute("begin")
            total = int(
                conn.execute("select count(*) from research_notes").fetchone()[0]
            )
            rows = conn.execute(
                """select note_id,version,title,updated_at from research_notes
                order by updated_at desc, note_id limit ? offset ?""",
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows], total
