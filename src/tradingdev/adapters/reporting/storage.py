"""Write-once report publication outside historical run directories."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.storage.filesystem import now_iso
from tradingdev.domain.presentation.reports import REPORT_SCHEMA_VERSION

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore


class ReportPublicationError(ValueError):
    """An explicit report identity, path, publication, or download failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _safe_path(workspace: WorkspacePaths, *parts: str) -> Path:
    path = workspace.root
    for part in ("reports", *parts):
        path /= part
        if path.is_symlink() or not path.resolve().is_relative_to(workspace.root):
            raise ReportPublicationError(
                "report_path_invalid", "Report path leaves workspace"
            )
    return path


def read_report_html(
    workspace: WorkspacePaths,
    store: SQLiteStore,
    report_id: str,
    *,
    expected_sha256: str,
) -> bytes:
    """Read registered HTML once and verify the exact bytes returned to the caller."""
    if not isinstance(report_id, str) or not re.fullmatch(r"[0-9a-f]{64}", report_id):
        raise ReportPublicationError("report_path_invalid", "Invalid report ID")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ReportPublicationError(
            "report_artifact_invalid", "Invalid expected report SHA-256"
        )
    path = _safe_path(workspace, report_id, "report.html")
    record = store.get_artifact(f"report:{report_id}:html")
    if record is None:
        raise ReportPublicationError("report_not_found", "Report is not registered")
    if (
        record.get("artifact_type") != "research_report_html"
        or record.get("path") != str(path)
        or record.get("sha256") != expected_sha256
    ):
        raise ReportPublicationError(
            "report_artifact_invalid", "Report registration differs from the download"
        )
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        raise ReportPublicationError(
            "report_artifact_invalid", "Registered report is missing"
        ) from None
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ReportPublicationError(
            "report_artifact_invalid", "Registered report content differs"
        )
    return content


def publish_report(
    workspace: WorkspacePaths,
    store: SQLiteStore,
    report_id: str,
    html: bytes,
    manifest: bytes,
    *,
    run_ids: list[str],
    scope_count: int,
    comparison_job_id: str | None = None,
    ml_entry_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Publish a matching HTML/manifest pair and transactionally register both.

    Concurrent publication returns busy. A failed publication removes only files
    it created. An existing file or record with different content is never replaced.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", report_id):
        raise ReportPublicationError("report_path_invalid", "Invalid report ID")
    reports = _safe_path(workspace)
    reports.mkdir(parents=True, exist_ok=True)
    marker = _safe_path(workspace, f".{report_id}.publishing")
    directory = _safe_path(workspace, report_id)
    existed = directory.exists()
    try:
        marker.mkdir()
    except FileExistsError:
        raise ReportPublicationError(
            "report_busy", "Report publication is already in progress"
        ) from None
    created: list[Path] = []
    temporary: Path | None = None
    artifact_id = f"report:{report_id}:html"
    manifest_id = f"report:{report_id}:manifest"
    records = []
    metadata = json.dumps(
        {
            "report_id": report_id,
            "run_ids": run_ids,
            "schema_version": REPORT_SCHEMA_VERSION,
            "scope_count": scope_count,
            **({"ml_entry_ids": ml_entry_ids} if ml_entry_ids is not None else {}),
            **(
                {"comparison_job_id": comparison_job_id}
                if comparison_job_id is not None
                else {}
            ),
        },
        sort_keys=True,
    )
    try:
        directory.mkdir(exist_ok=True)
        for name, content, record_id, kind in (
            ("report.html", html, artifact_id, "research_report_html"),
            ("manifest.json", manifest, manifest_id, "research_report_manifest"),
        ):
            path = _safe_path(workspace, report_id, name)
            registered = store.get_artifact(record_id)
            if registered is not None and (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest()
                != registered.get("sha256")
            ):
                raise ReportPublicationError(
                    "report_artifact_invalid", "Registered report is missing or corrupt"
                )
            with NamedTemporaryFile(
                prefix=".report-", dir=directory, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            _safe_path(workspace, report_id, name)
            try:
                os.link(temporary, path)
                created.append(path)
            except FileExistsError:
                if path.read_bytes() != content:
                    raise ReportPublicationError(
                        "report_artifact_invalid", "Existing report content differs"
                    ) from None
            finally:
                temporary.unlink(missing_ok=True)
                temporary = None
            records.append(
                (
                    record_id,
                    run_ids[0] if len(run_ids) == 1 else None,
                    kind,
                    str(path),
                    hashlib.sha256(content).hexdigest(),
                    metadata,
                    now_iso(),
                )
            )
        with store.connect() as connection:
            for record in records:
                existing = connection.execute(
                    "select run_id, artifact_type, path, sha256, metadata "
                    "from artifacts "
                    "where artifact_id = ?",
                    (record[0],),
                ).fetchone()
                if existing is not None and tuple(existing) != record[1:6]:
                    raise ReportPublicationError(
                        "report_artifact_invalid",
                        "Existing report registration differs",
                    )
            connection.executemany(
                "insert into artifacts (artifact_id, run_id, artifact_type, "
                "path, sha256, "
                "metadata, created_at) values (?, ?, ?, ?, ?, ?, ?) "
                "on conflict(artifact_id) do nothing",
                records,
            )
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        marker.rmdir()
        if not existed and directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
    return {
        "success": True,
        "report_id": report_id,
        "artifact_id": artifact_id,
        "manifest_artifact_id": manifest_id,
        "path": str(directory / "report.html"),
        "sha256": hashlib.sha256(html).hexdigest(),
        "run_ids": run_ids,
        "scope_count": scope_count,
        **({"ml_entry_ids": ml_entry_ids} if ml_entry_ids is not None else {}),
        **(
            {"comparison_job_id": comparison_job_id}
            if comparison_job_id is not None
            else {}
        ),
    }
