"""Persist bounded, immutable research declarations without accessing sources."""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from tradingdev.domain.research_evidence import (
    MAX_RESEARCH_EVIDENCE_BYTES,
    ResearchEvidenceRecord,
)

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths


class ResearchEvidenceStore:
    """Publish complete records once and verify them on every read.

    Hashes detect changed content, not the truth of a source declaration or a
    hostile writer able to replace both a record and its hash in the workspace.
    Saved records are user research data and have no automatic expiry or cleanup.
    """

    def __init__(self, workspace: WorkspacePaths) -> None:
        self.workspace = workspace

    def path(self, evidence_id: str) -> Path:
        """Accept only an evidence identity and reject symlinked storage paths."""
        if re.fullmatch(r"[0-9a-f]{32}", evidence_id) is None:
            raise ValueError("Invalid research evidence ID")
        root = self.workspace.root
        if root.is_symlink() or root.resolve() != root:
            raise ValueError("Research evidence path leaves workspace")
        path = root
        for part in ("research_evidence", f"{evidence_id}.json"):
            path /= part
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("Research evidence path leaves workspace")
        return path

    def save(self, record: ResearchEvidenceRecord) -> None:
        """Atomically publish verified bytes; an identical retry is harmless."""
        record.verify()
        content = self._content(record)
        path = self.path(record.evidence_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path(record.evidence_id)
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb", prefix=".evidence-", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            self.path(record.evidence_id)
            try:
                os.link(temporary, path)
            except FileExistsError:
                saved = self.load(record.evidence_id)
                if self._content(saved) != content:
                    raise ValueError(
                        "Research evidence ID already contains different content"
                    ) from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load(self, evidence_id: str) -> ResearchEvidenceRecord:
        """Load a bounded regular file, validating schema, identity and hash."""
        path = self.path(evidence_id)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Research evidence must be a regular file")
            if info.st_size > MAX_RESEARCH_EVIDENCE_BYTES:
                raise ValueError("Research evidence exceeds the stored byte limit")
            content = stream.read(MAX_RESEARCH_EVIDENCE_BYTES + 1)
        if len(content) > MAX_RESEARCH_EVIDENCE_BYTES:
            raise ValueError("Research evidence exceeds the stored byte limit")
        record = ResearchEvidenceRecord.model_validate_json(content)
        if record.evidence_id != evidence_id:
            raise ValueError("Research evidence identity mismatch")
        record.verify()
        if self._content(record) != content:
            raise ValueError("Saved research evidence bytes have changed")
        return record

    @staticmethod
    def _content(record: ResearchEvidenceRecord) -> bytes:
        content = (
            json.dumps(
                record.model_dump(mode="json"),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        if len(content) > MAX_RESEARCH_EVIDENCE_BYTES:
            raise ValueError("Research evidence exceeds the stored byte limit")
        return content
