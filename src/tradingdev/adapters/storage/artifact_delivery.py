"""Read only registered, canonical reading artifacts for portable delivery."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from tradingdev.adapters.storage.workspace_files import WorkspaceFiles

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore

MAX_DELIVERY_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ArtifactBytes:
    artifact_id: str
    sha256: str
    mime_type: Literal["text/html", "application/zip", "application/json"]
    filename: str
    content: bytes


def read_delivery_artifact(
    workspace: WorkspacePaths,
    store: SQLiteStore,
    artifact_id: str,
    expected_sha256: str,
) -> ArtifactBytes:
    """Validate registration, canonical location and the exact bounded bytes read."""
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("A saved artifact SHA-256 is required")
    row = store.get_artifact(artifact_id)
    if row is None or row["sha256"] != expected_sha256:
        raise ValueError("Unknown artifact or mismatched saved SHA-256")
    kind = row["artifact_type"]
    mime_type: Literal["text/html", "application/zip", "application/json"] = "text/html"
    if kind == "execution_confirmation":
        match = re.fullmatch(r"([0-9a-f]{32}):confirmation_html", artifact_id)
        directory, basename, prefix = (
            "execution_plans",
            "confirmation.html",
            "confirmation",
        )
    elif kind == "research_report_html":
        match = re.fullmatch(r"report:([0-9a-f]{64}):html", artifact_id)
        directory, basename, prefix = "reports", "report.html", "report"
    elif kind == "research_archive_zip":
        match = re.fullmatch(r"research:([0-9a-f]{64}):zip", artifact_id)
        directory, basename, prefix = "research_archives", "research.zip", "research"
        mime_type = "application/zip"
    elif kind == "parameter_plateau":
        match = re.fullmatch(r"plateau:([0-9a-f]{64})", artifact_id)
        directory, basename, prefix = "analyses", "", "plateau"
        mime_type = "application/json"
    else:
        raise ValueError(
            "Delivery supports confirmation/report HTML, research ZIP and plateau JSON"
        )
    if match is None:
        raise ValueError("Invalid reading artifact identity")
    if kind == "parameter_plateau":
        if match[1] != expected_sha256:
            raise ValueError("Plateau identity differs from the saved SHA-256")
        path = workspace.root / directory / "plateau" / f"{match[1]}.json"
    else:
        path = workspace.root / directory / match[1] / basename
    if row["path"] != str(path):
        raise ValueError("Artifact path differs from its canonical identity")
    content = WorkspaceFiles(workspace.root).read(path, MAX_DELIVERY_BYTES)
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Saved artifact bytes changed")
    return ArtifactBytes(
        artifact_id,
        expected_sha256,
        mime_type,
        f"{prefix}-{match[1][:12]}{path.suffix}",
        content,
    )
