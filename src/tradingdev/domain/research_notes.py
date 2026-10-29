"""Editable research summaries, separate from immutable execution evidence."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

MAX_NOTE_BYTES = 16384
type NoteId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
type NoteTitle = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)
]
type NoteText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
]
type NoteVersion = Annotated[int, Field(strict=True, ge=1)]


class NoteModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )


class ResearchNoteEntry(NoteModel):
    title: NoteTitle
    summary: NoteText
    artifact_ids: Annotated[list[str], Field(max_length=10)] = []


class ResearchNoteDraft(NoteModel):
    title: NoteTitle
    current_brief: NoteText
    entries: Annotated[list[ResearchNoteEntry], Field(max_length=24)] = []
    next_steps: NoteText


class ResearchNoteRecord(NoteModel):
    schema_version: Literal[1] = 1
    note_id: NoteId
    version: NoteVersion
    created_at: str
    updated_at: str
    draft: ResearchNoteDraft
