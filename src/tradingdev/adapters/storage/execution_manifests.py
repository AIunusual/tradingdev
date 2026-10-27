"""Publish immutable execution specifications inside a run's artifact directory."""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from filelock import FileLock

from tradingdev.domain.execution import ExecutionManifest, ManifestError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tradingdev.adapters.storage.filesystem import WorkspacePaths

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


class ExecutionManifestStore:
    """Write once, then read using an independently stored expected hash.

    A temporary file is fully written before an atomic hard-link publication.
    Existing content is never replaced, including when two producers race.
    The manifest describes requested execution; it does not snapshot market data
    or provide a security boundary against writers controlling the whole workspace.
    """

    def __init__(self, workspace: WorkspacePaths) -> None:
        self._workspace = workspace

    def path(self, run_id: str) -> Path:
        """Resolve the fixed artifact path without following workspace symlinks."""
        if not _RUN_ID.fullmatch(run_id):
            raise ManifestError("Invalid execution artifact ID")
        path = self._workspace.root
        for part in ("runs", run_id, "manifest.json"):
            path = path / part
            if path.is_symlink() or not path.resolve().is_relative_to(
                self._workspace.root
            ):
                raise ManifestError("Execution manifest path leaves workspace")
        return path

    def publish(self, run_id: str, manifest: ExecutionManifest) -> Path:
        """Atomically publish one verified manifest; identical publication is safe."""
        manifest.verify()
        path = self.path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            json.dumps(
                manifest.model_dump(mode="json"),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix=".manifest-",
                dir=path.parent,
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            self.path(run_id)
            try:
                os.link(temporary, path)
            except FileExistsError:
                self.load(run_id, expected_hash=manifest.manifest_hash)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    @contextmanager
    def submission_lock(self, run_id: str) -> Iterator[None]:
        """Serialize job publication and launch; retain the lock inode for waiters."""
        path = self.path(run_id).with_name(".submission.lock")
        if path.is_symlink():
            raise ManifestError("Execution submission lock must not be a symlink")
        path.parent.mkdir(parents=True, exist_ok=True)
        # This coordinates submitters; it is not a filesystem/SQLite transaction.
        # One object per path makes nested submission/create_job calls reentrant
        # in the same thread while retaining cross-thread/process exclusion.
        with FileLock(path, timeout=30, is_singleton=True):
            self.path(run_id)
            yield

    def load(
        self,
        run_id: str,
        *,
        expected_hash: str,
        expected_file_sha256: str | None = None,
    ) -> ExecutionManifest:
        """Verify execution settings and, when registered, the same file bytes."""
        path = self.path(run_id)
        try:
            content = path.read_bytes()
            if (
                expected_file_sha256 is not None
                and hashlib.sha256(content).hexdigest() != expected_file_sha256
            ):
                raise ManifestError("Manifest file SHA-256 differs from registration")
            manifest = ExecutionManifest.model_validate_json(content)
        except (OSError, ValueError) as exc:
            raise ManifestError(
                f"Cannot read valid execution manifest: {run_id}"
            ) from exc
        manifest.verify(expected_hash=expected_hash)
        return manifest
