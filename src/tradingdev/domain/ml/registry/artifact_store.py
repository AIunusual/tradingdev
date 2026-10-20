"""Filesystem model metadata (.json) and blobs (.bin), verified by content hash.

Integrity checks do not authenticate model authors or make pickle safe.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from filelock import FileLock

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.domain.ml.registry.model_registry import (
    ModelArtifact,
    compute_model_id,
    content_sha256,
)


class ModelArtifactStore:
    """Publish metadata over immutable blobs, with one writer per model ID."""

    def __init__(self, root: Path | None = None, *, lock_timeout: float = 30) -> None:
        if root is None:
            root = WorkspacePaths().models
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_timeout = lock_timeout

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
    ) -> ModelArtifact:
        """Publish a complete artifact without rewriting an existing model blob.

        Writers for one ID serialize validation and publication. Readers need
        no lock: metadata becomes visible only after its immutable blob exists.
        """
        model_id = artifact.model_id
        self._verify_identity(artifact, model_id)
        if content_sha256(model_bytes) != artifact.content_hash:
            raise ValueError(
                "model blob content_hash does not match the recorded artifact"
            )
        metadata = json.dumps(
            artifact.to_metadata(),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        blob_path, meta_path = self._blob_path(model_id), self._meta_path(model_id)
        # Keep the lock inode: deleting it could split waiting writers across
        # separate locks. Different model IDs never share this lock.
        with FileLock(self.root / f".{model_id}.lock", timeout=self._lock_timeout):
            if meta_path.exists():
                existing, _ = self.load_with_bytes(model_id)
                if (
                    existing.model_name != artifact.model_name
                    or existing.parameters != artifact.parameters
                    or existing.feature_names != artifact.feature_names
                    or existing.content_hash != artifact.content_hash
                ):
                    raise ValueError(f"model identity collision for id: {model_id}")
            elif blob_path.exists():
                # An interrupted first publication may leave a complete blob.
                # Reuse it only if it belongs to this exact content.
                self._verify_blob(artifact, blob_path.read_bytes())
            new_blob = not blob_path.exists()
            try:
                if new_blob:
                    self._publish(blob_path, model_bytes)
                self._publish(meta_path, metadata)
            except BaseException:
                if new_blob and not meta_path.exists():
                    blob_path.unlink(missing_ok=True)
                raise
        return artifact

    def load(self, model_id: str) -> ModelArtifact:
        """Load metadata and verify its identity and blob hash."""
        artifact, _ = self.load_with_bytes(model_id)
        return artifact

    def load_with_bytes(self, model_id: str) -> tuple[ModelArtifact, bytes]:
        """Return metadata and the exact bytes whose identity/hash were checked.

        Missing files raise FileNotFoundError; a mismatched hash raises ValueError.
        """
        try:
            payload = json.loads(self._meta_path(model_id).read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise FileNotFoundError(f"no model artifact for id: {model_id}") from error
        artifact = ModelArtifact.from_metadata(payload)
        self._verify_identity(artifact, model_id)
        try:
            model_bytes = self._blob_path(model_id).read_bytes()
        except FileNotFoundError as error:
            raise FileNotFoundError(f"model blob missing for id: {model_id}") from error
        self._verify_blob(artifact, model_bytes)
        return artifact, model_bytes

    @staticmethod
    def _verify_identity(artifact: ModelArtifact, model_id: str) -> None:
        expected = compute_model_id(
            model_name=artifact.model_name,
            parameters=artifact.parameters,
            feature_names=artifact.feature_names,
            content_hash=artifact.content_hash,
        )
        if model_id != artifact.model_id or model_id != expected:
            raise ValueError(f"model_id does not match recorded identity: {model_id}")

    @staticmethod
    def _verify_blob(artifact: ModelArtifact, model_bytes: bytes) -> None:
        actual = content_sha256(model_bytes)
        if actual != artifact.content_hash:
            raise ValueError(
                f"model blob tampered: expected {artifact.content_hash}, got {actual}"
            )

    def _publish(self, path: Path, data: bytes) -> None:
        staging: Path | None = None
        try:
            with NamedTemporaryFile(
                prefix=f".{path.stem}-",
                suffix=f"{path.suffix}.tmp",
                dir=self.root,
                delete=False,
            ) as stream:
                staging = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staging, path)
        finally:
            if staging is not None:
                staging.unlink(missing_ok=True)

    def list(self) -> list[ModelArtifact]:
        """Return all stored artifacts newest-by-id deterministic order."""
        result: list[ModelArtifact] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                artifact = ModelArtifact.from_metadata(payload)
                self._verify_identity(artifact, path.stem)
                result.append(artifact)
            except ValueError, json.JSONDecodeError:
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
