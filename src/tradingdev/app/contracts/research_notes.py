"""Bounded notebook retrieval and portable archive identities."""

from typing import Literal

from tradingdev.app.contracts.common import ContractModel, Page
from tradingdev.domain.research_notes import NoteId, NoteVersion, ResearchNoteRecord


class NoteSaved(ContractModel):
    success: Literal[True] = True
    note_id: NoteId
    version: NoteVersion


class NoteRead(ContractModel):
    success: Literal[True] = True
    note: ResearchNoteRecord


class NoteSummary(ContractModel):
    note_id: NoteId
    version: NoteVersion
    title: str
    updated_at: str


class NoteList(ContractModel):
    success: Literal[True] = True
    page: Page[NoteSummary]


class NoteExport(ContractModel):
    success: Literal[True] = True
    archive_id: str
    note_id: NoteId
    version: NoteVersion
    artifact_id: str
    path: str
    html_path: str | None
    sha256: str
    download_url: str | None = None
    download_expires_at: int | None = None
    instructions: str = (
        "下載 ZIP 後完整解壓，開啟 index.html；reports/ 保留連結報告。"
        "無下載網址時，path 是 server 本機路徑，需客戶端檔案傳輸能力。"
    )
