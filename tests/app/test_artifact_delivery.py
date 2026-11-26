"""Portable delivery refuses substituted identities, paths and bytes."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from tradingdev.adapters.storage import artifact_delivery
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.artifact_service import ArtifactService

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "kind,identity,relative,mime",
    [
        (
            "execution_confirmation",
            "a" * 32 + ":confirmation_html",
            f"execution_plans/{'a' * 32}/confirmation.html",
            "text/html",
        ),
        (
            "research_report_html",
            f"report:{'b' * 64}:html",
            f"reports/{'b' * 64}/report.html",
            "text/html",
        ),
        (
            "research_archive_zip",
            f"research:{'c' * 64}:zip",
            f"research_archives/{'c' * 64}/research.zip",
            "application/zip",
        ),
    ],
)
def test_delivery_returns_exact_registered_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    identity: str,
    relative: str,
    mime: str,
) -> None:
    workspace = WorkspacePaths(tmp_path)
    store = SQLiteStore(workspace)
    path = workspace.root / relative
    path.parent.mkdir(parents=True)
    content = "完整 UTF-8\r\n".encode()
    monkeypatch.setattr(artifact_delivery, "MAX_DELIVERY_BYTES", len(content))
    path.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    store.create_artifact(
        artifact_id=identity, run_id=None, artifact_type=kind, path=path, sha256=sha
    )
    result = ArtifactService(workspace=workspace, store=store).read_download(
        identity, sha
    )
    assert result.content == content
    assert result.mime_type == mime and result.sha256 == sha
    assert result.artifact_id == identity


@pytest.mark.parametrize(
    "damage",
    [
        "missing_record",
        "missing_file",
        "changed",
        "expected_hash",
        "registered_hash",
        "identity",
        "type",
        "outside",
        "symlink",
        "parent_symlink",
        "oversize",
        "directory",
    ],
)
def test_delivery_rejects_untrusted_registration_or_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    path = workspace.root / "execution_plans" / ("a" * 32) / "confirmation.html"
    path.parent.mkdir(parents=True)
    content = b"<html>original</html>"
    sha = hashlib.sha256(content).hexdigest()
    path.write_bytes(content)
    identity = "a" * 32 + ":confirmation_html"
    if damage == "identity":
        identity = "arbitrary:confirmation_html"
    if damage != "missing_record":
        store.create_artifact(
            artifact_id=identity,
            run_id=None,
            artifact_type="secret" if damage == "type" else "execution_confirmation",
            path=tmp_path / "outside.html" if damage == "outside" else path,
            sha256="0" * 64 if damage == "registered_hash" else sha,
        )
    if damage in {"missing_file", "symlink", "directory"}:
        path.unlink()
    if damage in {"symlink", "parent_symlink"}:
        outside = tmp_path / "foreign"
        outside.mkdir()
        (outside / path.name).write_bytes(content)
        if damage == "symlink":
            path.symlink_to(outside / path.name)
        else:
            path.unlink()
            path.parent.rmdir()
            path.parent.symlink_to(outside, target_is_directory=True)
    if damage == "directory":
        path.mkdir()
    if damage == "changed":
        path.write_bytes(content + b"tampered")
    if damage == "oversize":
        monkeypatch.setattr(artifact_delivery, "MAX_DELIVERY_BYTES", len(content) - 1)
    service = ArtifactService(workspace=workspace, store=store)
    with pytest.raises((ValueError, OSError)):
        service.read_download(identity, "0" * 64 if damage == "expected_hash" else sha)


def test_parameter_plateau_json_delivery_uses_canonical_digest_path(
    tmp_path: Path,
) -> None:
    from pathlib import Path

    from tests.app.test_parameter_plateau import _record

    from tradingdev.app.plateau_service import PlateauService

    workspace = WorkspacePaths(tmp_path)
    sql = SQLiteStore(workspace)
    result = PlateauService(workspace=workspace, store=sql).analyze(
        metric="signal.ic_mean",
        axes={"/threshold": [1, 2, 3]},
        entry_ids=_record(workspace, [0.5, None, 0.6]),
    )
    descriptor = result["artifact"]
    service = ArtifactService(workspace=workspace, store=sql)
    delivery = service.read_download(descriptor["artifact_id"], descriptor["sha256"])
    assert delivery.mime_type == "application/json"
    assert delivery.content == Path(descriptor["path"]).read_bytes()
    assert delivery.filename.endswith(".json")
    Path(descriptor["path"]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="bytes changed"):
        service.read_download(descriptor["artifact_id"], descriptor["sha256"])
