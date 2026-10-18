"""Filesystem persistence for model artifacts.

Reuses the project's :class:`WorkspacePaths` and hashing helpers so a stored
model lives alongside runs and datasets, and its on-disk blob is
integrity-checked against the content hash recorded in its metadata. The store
is a plain directory of ``<model_id>.json`` (metadata) and ``<model_id>.bin``
(blob) files, so a test can point it at an isolated ``tmp_path`` with no SQLite
or workspace side effects.
"""

from __future__ import annotations

import json
from pathlib import Path

from tradingdev.adapters.storage.filesystem import (
    WorkspacePaths,
    sha256_file,
    write_json,
)
from tradingdev.domain.ml.registry.model_registry import (
    ModelArtifact,
    content_sha256,
)


class ModelArtifactStore:
    """Store model blobs and their metadata keyed by ``model_id``."""

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            root = WorkspacePaths().models
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, model_id: str) -> Path:
        return self.root / f"{model_id}.json"

    def _blob_path(self, model_id: str) -> Path:
        return self.root / f"{model_id}.bin"

    def exists(self, model_id: str) -> bool:
        """Whether a metadata record exists for the id."""
        return self._meta_path(model_id).exists()

    def path_for(self, model_id: str) -> Path:
        """Where the blob for ``model_id`` is stored."""
        return self._blob_path(model_id)

    def save(
        self,
        *,
        model_bytes: bytes,
        artifact: ModelArtifact,
        model_id: str | None = None,
    ) -> ModelArtifact:
        """Persist the blob and its metadata; verify the content hash.

        Raises ``ValueError`` if the blob's computed hash differs from the one
        recorded on the artifact, so a corrupted or mismatched blob cannot be
        silently stored.
        """
        if model_id is None:
            model_id = artifact.model_id
        if content_sha256(model_bytes) != artifact.content_hash:
            raise ValueError(
                "model blob content_hash does not match the recorded artifact"
            )
        self._blob_path(model_id).write_bytes(model_bytes)
        write_json(self._meta_path(model_id), artifact.to_metadata())
        return artifact

    def load(self, model_id: str) -> ModelArtifact:
        """Load metadata and verify the on-disk blob still matches.

        Raises ``FileNotFoundError`` if the record or blob is missing, and
        ``ValueError`` if the blob's current hash differs from the one recorded
        at save time (tamper detection).
        """
        meta_path = self._meta_path(model_id)
        if not meta_path.exists():
            raise FileNotFoundError(f"no model artifact for id: {model_id}")
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        artifact = ModelArtifact.from_metadata(payload)
        blob_path = self._blob_path(model_id)
        if not blob_path.exists():
            raise FileNotFoundError(f"model blob missing for id: {model_id}")
        actual = sha256_file(blob_path)
        if actual != artifact.content_hash:
            raise ValueError(
                f"model blob tampered: expected {artifact.content_hash}, got {actual}"
            )
        return artifact

    def list(self) -> list[ModelArtifact]:
        """Return all stored artifacts newest-by-id deterministic order."""
        result: list[ModelArtifact] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                result.append(ModelArtifact.from_metadata(payload))
            except (ValueError, json.JSONDecodeError):
                continue
        return result


def save_model_artifact(
    *,
    store: ModelArtifactStore,
    model_bytes: bytes,
    artifact: ModelArtifact,
) -> ModelArtifact:
    """Convenience wrapper that saves an artifact into a provided store."""
    return store.save(model_bytes=model_bytes, artifact=artifact)
