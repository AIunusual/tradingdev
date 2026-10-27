"""Atomic comparison evidence and a locked, complete candidate ledger."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Literal

from filelock import FileLock

from tradingdev.domain.comparison import ComparisonManifest
from tradingdev.domain.comparison_results import (
    CandidateStatus,
    ComparisonLedger,
    ComparisonReceipt,
    FailureStatus,
    canonical_json_object,
)
from tradingdev.domain.execution import ManifestError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pydantic import BaseModel

    from tradingdev.adapters.storage.filesystem import WorkspacePaths

_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,115}")
type ComparisonArtifact = Literal[
    "manifest.json",
    "ledger.json",
    "snapshot.parquet",
    "receipt.json",
    ".comparison.lock",
]


class ComparisonStore:
    """Keep immutable specifications/data separate from forward-only outcomes.

    The caller may publish execution data only after approval. This adapter does
    not acquire market data or infer approval. Locks coordinate application
    writers; the workspace is not a security boundary against its owner.
    """

    def __init__(self, workspace: WorkspacePaths) -> None:
        self._root = workspace.root

    def path(self, job_id: str, artifact: ComparisonArtifact = "manifest.json") -> Path:
        """Reject symlinks, invalid IDs and replaced workspace roots."""
        if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
            raise ManifestError("Invalid comparison job ID")
        if artifact not in {
            "manifest.json",
            "ledger.json",
            "snapshot.parquet",
            "receipt.json",
            ".comparison.lock",
        }:
            raise ManifestError("Invalid comparison artifact")
        path = self._root
        for part in (None, "comparisons", job_id, artifact):
            if part is not None:
                path = path / part
            if path.is_symlink() or not path.resolve().is_relative_to(self._root):
                raise ManifestError("Comparison artifact path leaves workspace")
        return path

    @contextmanager
    def _lock(self, job_id: str) -> Iterator[None]:
        path = self.path(job_id, ".comparison.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path(job_id, ".comparison.lock")
        with FileLock(path, timeout=30):
            self.path(job_id, ".comparison.lock")
            yield

    def initialize(self, job_id: str, manifest: ComparisonManifest) -> ComparisonLedger:
        """Publish every pending row before launch; retries preserve existing work."""
        manifest.verify()
        initial = ComparisonLedger.create(job_id, manifest)
        with self._lock(job_id):
            if self.path(job_id, "manifest.json").exists():
                # Once the manifest exists, a missing ledger is corruption, not a
                # new submission. Never erase completed outcomes by rebuilding it.
                return self.load_ledger(job_id, expected_hash=manifest.manifest_hash)
            # Publish the complete initial ledger first. The manifest marks the
            # completed initialization; neither read nor execution can use a lone
            # ledger. An interrupted initialization accepts only identical bytes.
            self._write_once(job_id, "ledger.json", self._json_bytes(initial))
            self._write_once(job_id, "manifest.json", self._json_bytes(manifest))
            return self.load_ledger(job_id, expected_hash=manifest.manifest_hash)

    def load_manifest(self, job_id: str, *, expected_hash: str) -> ComparisonManifest:
        content = self._read(job_id, "manifest.json")
        try:
            manifest = ComparisonManifest.model_validate_json(
                canonical_json_object(content.decode("utf-8"))
            )
            manifest.verify(expected_hash=expected_hash)
            return manifest
        except (ValueError, UnicodeError) as exc:
            raise ManifestError("Cannot read valid comparison manifest") from exc

    def load_ledger(self, job_id: str, *, expected_hash: str) -> ComparisonLedger:
        manifest = self.load_manifest(job_id, expected_hash=expected_hash)
        content = self._read(job_id, "ledger.json")
        try:
            ledger = ComparisonLedger.model_validate_json(
                canonical_json_object(content.decode("utf-8"))
            )
            ledger.verify(manifest, job_id=job_id)
            return ledger
        except (ValueError, UnicodeError) as exc:
            raise ManifestError("Cannot read valid comparison ledger") from exc

    def transition(
        self,
        job_id: str,
        *,
        expected_hash: str,
        candidate_id: str,
        status: CandidateStatus,
        reason: str | None = None,
    ) -> ComparisonLedger:
        """Persist one legal forward transition, with exact retry semantics."""
        with self._lock(job_id):
            previous = self.load_ledger(job_id, expected_hash=expected_hash)
            updated = previous.transition(candidate_id, status, reason=reason)
            if updated != previous:
                self._replace_ledger(job_id, updated)
            return updated

    def finish_unfinished(
        self,
        job_id: str,
        *,
        expected_hash: str,
        reason: str,
        active_status: FailureStatus = "failed",
    ) -> ComparisonLedger:
        """Record cancellation/death after the worker stops, preserving all outcomes."""
        if active_status not in {"failed", "cancelled"}:
            raise ManifestError("Unfinished candidates must fail or be cancelled")
        with self._lock(job_id):
            previous = self.load_ledger(job_id, expected_hash=expected_hash)
            updated = previous.finish_unfinished(
                reason=reason, active_status=active_status
            )
            if updated != previous:
                self._replace_ledger(job_id, updated)
            return updated

    def publish_snapshot(
        self,
        job_id: str,
        *,
        expected_hash: str,
        snapshot_bytes: bytes,
        data_binding_json: str,
        source_dataset_id: str,
    ) -> ComparisonReceipt:
        """Publish post-approval data and receipt once; never change the bundle."""
        if not isinstance(snapshot_bytes, bytes) or not snapshot_bytes:
            raise ManifestError("Comparison snapshot must contain bytes")
        receipt = ComparisonReceipt(
            job_id=job_id,
            manifest_hash=expected_hash,
            snapshot_sha256=hashlib.sha256(snapshot_bytes).hexdigest(),
            data_binding_json=data_binding_json,
            source_dataset_id=source_dataset_id,
        )
        with self._lock(job_id):
            self.load_ledger(job_id, expected_hash=expected_hash)
            if self.path(job_id, "receipt.json").exists():
                previous, _ = self.load_snapshot(job_id, expected_hash=expected_hash)
                if previous != receipt:
                    raise ManifestError("Comparison execution data cannot be replaced")
                return previous
            created = self._write_once(job_id, "snapshot.parquet", snapshot_bytes)
            try:
                self._write_once(job_id, "receipt.json", self._json_bytes(receipt))
            except BaseException:
                # Never remove another publication or an existing crash-recovery file.
                if created and not self.path(job_id, "receipt.json").exists():
                    self.path(job_id, "snapshot.parquet").unlink()
                raise
            return receipt

    def load_receipt(self, job_id: str, *, expected_hash: str) -> ComparisonReceipt:
        """Read bound receipt metadata; this does not verify current snapshot bytes."""
        self.load_manifest(job_id, expected_hash=expected_hash)
        content = self._read(job_id, "receipt.json")
        try:
            receipt = ComparisonReceipt.model_validate_json(
                canonical_json_object(content.decode("utf-8"))
            )
            receipt.verify(job_id=job_id, expected_hash=expected_hash)
        except (ValueError, UnicodeError) as exc:
            raise ManifestError("Cannot read valid comparison receipt") from exc
        return receipt

    def load_snapshot(
        self, job_id: str, *, expected_hash: str
    ) -> tuple[ComparisonReceipt, bytes]:
        """Verify the receipt against this job and the hash of actual snapshot bytes."""
        receipt = self.load_receipt(job_id, expected_hash=expected_hash)
        snapshot = self._read(job_id, "snapshot.parquet")
        if (
            not snapshot
            or hashlib.sha256(snapshot).hexdigest() != receipt.snapshot_sha256
        ):
            raise ManifestError("Comparison snapshot SHA-256 differs from its receipt")
        return receipt, snapshot

    @staticmethod
    def _json_bytes(model: BaseModel) -> bytes:
        return (canonical_json_object(model.model_dump_json()) + "\n").encode("utf-8")

    def _read(self, job_id: str, artifact: ComparisonArtifact) -> bytes:
        path = self.path(job_id, artifact)
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ManifestError("Comparison artifact must be a regular file")
                content = stream.read()
            self.path(job_id, artifact)
            return content
        except OSError as exc:
            raise ManifestError(f"Cannot read comparison artifact: {artifact}") from exc

    def _write_once(
        self, job_id: str, artifact: ComparisonArtifact, content: bytes
    ) -> bool:
        """An atomic hard link never replaces a prior successful publication."""
        path = self.path(job_id, artifact)
        temporary: Path | None = None
        try:
            temporary = self._temporary(path, content)
            self.path(job_id, artifact)
            try:
                os.link(temporary, path)
            except FileExistsError:
                if self._read(job_id, artifact) != content:
                    raise ManifestError(
                        f"Comparison artifact cannot be replaced: {artifact}"
                    ) from None
                return False
            return True
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _replace_ledger(self, job_id: str, ledger: ComparisonLedger) -> None:
        path = self.path(job_id, "ledger.json")
        temporary: Path | None = None
        try:
            temporary = self._temporary(path, self._json_bytes(ledger))
            self.path(job_id, "ledger.json")
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _temporary(path: Path, content: bytes) -> Path:
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb", prefix=f".{path.name}-", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            return temporary
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
