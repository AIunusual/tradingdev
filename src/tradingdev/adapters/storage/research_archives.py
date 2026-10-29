"""Publish portable snapshots containing only explicitly referenced HTML reports."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from filelock import FileLock

from tradingdev.adapters.reporting.notebook import render_notebook
from tradingdev.adapters.storage.filesystem import now_iso
from tradingdev.adapters.storage.research_notes import note_bytes
from tradingdev.app.contracts.research_notes import NoteExport
from tradingdev.domain.research_notes import ResearchNoteRecord

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore

MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_ATTACHMENTS = 20


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2
    ).encode("utf-8")


class ResearchArchiveStore:
    def __init__(self, workspace: WorkspacePaths, store: SQLiteStore) -> None:
        self.workspace, self.store = workspace, store

    def safe_path(self, path: Path) -> Path:
        root = self.workspace.root
        if root.is_symlink() or root.resolve() != root:
            raise ValueError("Archive workspace is not a canonical directory")
        relative = path.relative_to(root)
        current = root
        for part in relative.parts:
            current /= part
            if part in {".", ".."} or current.is_symlink():
                raise ValueError("Archive paths must not traverse symlinks or parents")
        return current

    def read(self, path: Path, limit: int = MAX_ARCHIVE_BYTES) -> bytes:
        descriptor = os.open(
            self.safe_path(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ValueError("Archive source is not a bounded regular file")
            content = stream.read(limit + 1)
        if len(content) > limit:
            raise ValueError("Archive exceeds 64 MiB; select fewer reports")
        return content

    def _attachment(self, artifact_id: str) -> tuple[bytes, dict[str, Any]]:
        row = self.store.get_artifact(artifact_id)
        if row is None:
            raise ValueError(f"Referenced artifact not found: {artifact_id}")
        kind = row["artifact_type"]
        if kind == "research_report_html":
            match = re.fullmatch(r"report:([0-9a-f]{64}):html", artifact_id)
            if match is None:
                raise ValueError("Invalid report artifact identity")
            expected = self.workspace.root / "reports" / match[1] / "report.html"
        elif kind == "execution_confirmation":
            match = re.fullmatch(r"([0-9a-f]{32}):confirmation_html", artifact_id)
            if match is None:
                raise ValueError("Invalid confirmation artifact identity")
            expected = (
                self.workspace.root / "execution_plans" / match[1] / "confirmation.html"
            )
        else:
            raise ValueError(
                "Only saved research reports and confirmation HTML can be attached"
            )
        if str(expected) != row["path"]:
            raise ValueError("Artifact path differs from its expected identity")
        content = self.read(expected)
        if _digest(content) != row["sha256"]:
            raise ValueError(f"Referenced report changed: {artifact_id}")
        return content, {
            "artifact_id": artifact_id,
            "artifact_type": kind,
            "sha256": row["sha256"],
        }

    def publish(self, note: ResearchNoteRecord) -> NoteExport:
        files = {"research.json": note_bytes(note)}
        links: dict[str, str] = {}
        attachments = []
        size = len(files["research.json"])
        for entry in note.draft.entries:
            for artifact_id in entry.artifact_ids:
                if artifact_id in links:
                    continue
                if len(links) >= MAX_ATTACHMENTS:
                    raise ValueError("Archive supports at most 20 distinct reports")
                content, metadata = self._attachment(artifact_id)
                size += len(content)
                if size > MAX_ARCHIVE_BYTES:
                    raise ValueError("Archive exceeds 64 MiB; select fewer reports")
                name = f"reports/{len(links) + 1}.html"
                files[name], links[artifact_id] = content, name
                attachments.append({**metadata, "path": name})
        files["index.html"] = render_notebook(note, links).encode("utf-8")
        manifest = {
            "schema_version": 1,
            "note_id": note.note_id,
            "version": note.version,
            "attachments": attachments,
            "files": {name: _digest(content) for name, content in files.items()},
        }
        files["manifest.json"] = _json(manifest)
        archive_id = _digest(files["manifest.json"])
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in sorted(files.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, content)
        files["research.zip"] = output.getvalue()
        if (
            sum(len(c) for n, c in files.items() if n != "research.zip")
            > MAX_ARCHIVE_BYTES
            or len(files["research.zip"]) > MAX_ARCHIVE_BYTES
        ):
            raise ValueError("Archive exceeds 64 MiB including notebook and manifest")
        root = self.safe_path(self.workspace.root / "research_archives")
        root.mkdir(exist_ok=True)
        directory = self.safe_path(root / archive_id)
        artifact_id = f"research:{archive_id}:zip"
        sha = _digest(files["research.zip"])
        lock = self.safe_path(root / f".{archive_id}.lock")
        with FileLock(lock, timeout=5):
            created = False
            try:
                if directory.exists():
                    for name, content in files.items():
                        if self.read(directory / name) != content:
                            raise ValueError(
                                "Existing archive bytes changed; refusing replacement"
                            )
                else:
                    with TemporaryDirectory(prefix=".archive-", dir=root) as temporary:
                        staged = Path(temporary) / "snapshot"
                        staged.mkdir()
                        for name, content in files.items():
                            path = staged / name
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(content)
                        self.safe_path(directory)
                        staged.rename(directory)
                        created = True
                with self.store.connect() as conn:
                    existing = conn.execute(
                        "select * from artifacts where artifact_id=?", (artifact_id,)
                    ).fetchone()
                    zip_path = str(directory / "research.zip")
                    if existing is not None:
                        if (
                            existing["path"] != zip_path
                            or existing["sha256"] != sha
                            or existing["artifact_type"] != "research_archive_zip"
                        ):
                            raise ValueError(
                                "Archive registration differs; refusing replacement"
                            )
                    else:
                        conn.execute(
                            "insert into artifacts values (?,NULL,?,?,?,?,?)",
                            (
                                artifact_id,
                                "research_archive_zip",
                                zip_path,
                                sha,
                                json.dumps(
                                    {"note_id": note.note_id, "version": note.version}
                                ),
                                now_iso(),
                            ),
                        )
            except BaseException:
                if created:
                    shutil.rmtree(directory)
                raise
        return NoteExport(
            archive_id=archive_id,
            note_id=note.note_id,
            version=note.version,
            artifact_id=artifact_id,
            path=str(directory / "research.zip"),
            html_path=str(directory / "index.html"),
            sha256=sha,
        )

    def download(self, archive_id: str, sha256: str) -> bytes:
        if not re.fullmatch(r"[0-9a-f]{64}", archive_id):
            raise ValueError("Invalid archive ID")
        row = self.store.get_artifact(f"research:{archive_id}:zip")
        path = self.workspace.root / "research_archives" / archive_id / "research.zip"
        if (
            row is None
            or row["artifact_type"] != "research_archive_zip"
            or row["path"] != str(path)
            or row["sha256"] != sha256
        ):
            raise ValueError("Unknown archive or mismatched download identity")
        content = self.read(path)
        if _digest(content) != sha256:
            raise ValueError("Archive bytes changed")
        return content

    def describe(self, archive_id: str) -> NoteExport:
        """Reopen only registered snapshot bytes, without its mutable source note."""
        row = self.store.get_artifact(f"research:{archive_id}:zip")
        if row is None:
            raise ValueError("Archive not found")
        content = self.download(archive_id, row["sha256"])
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            manifest = archive.read("manifest.json")
            if _digest(manifest) != archive_id:
                raise ValueError("Archive manifest identity differs")
            note = ResearchNoteRecord.model_validate_json(archive.read("research.json"))
            if sum(info.file_size for info in archive.infolist()) > MAX_ARCHIVE_BYTES:
                raise ValueError("Saved archive exceeds its uncompressed capacity")
            reading_files = {name: archive.read(name) for name in archive.namelist()}
        directory = self.workspace.root / "research_archives" / archive_id
        html_path = None
        try:
            if all(
                self.read(directory / name) == data
                for name, data in reading_files.items()
            ):
                html_path = str(directory / "index.html")
        except ValueError, OSError:
            pass  # The portable ZIP remains independently downloadable.
        return NoteExport(
            archive_id=archive_id,
            note_id=note.note_id,
            version=note.version,
            artifact_id=row["artifact_id"],
            path=str(directory / "research.zip"),
            html_path=html_path,
            sha256=row["sha256"],
        )
