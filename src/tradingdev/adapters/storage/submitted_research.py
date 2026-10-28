"""Immutable submitted and independently readable run research snapshots."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tradingdev.domain.approved_research import ApprovedResearch
from tradingdev.domain.comparison import ComparisonManifest
from tradingdev.domain.execution import ExecutionManifest, ManifestError
from tradingdev.domain.execution_target import ExecutionTarget
from tradingdev.domain.research_evidence_values import canonical

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore

_MAX_BYTES = 4 * 1024 * 1024
_ID = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"


class ResearchSnapshot(BaseModel):
    """Parent identity retained with each run and its detached research."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    job_id: str = Field(pattern=_ID)
    run_id: str | None = Field(default=None, pattern=_ID)
    target: ExecutionTarget
    research_evidence: ApprovedResearch

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Research schema version must be an integer")
        return value

    @model_validator(mode="after")
    def valid_snapshot(self) -> Self:
        if type(self.schema_version) is not int:
            raise ValueError("Invalid submitted research schema")
        self.research_evidence.verify(self.target)
        if self.run_id is not None:
            self.manifest_for_run()
        return self

    def manifest_for_run(self) -> ExecutionManifest:
        if isinstance(self.target, ComparisonManifest):
            for index, candidate in enumerate(self.target.candidates, start=1):
                if self.run_id == f"{self.job_id}_candidate_{index}":
                    return candidate.manifest
        elif self.run_id == self.job_id:
            return self.target
        raise ValueError("Research run is not a member of the submitted target")


class SubmittedResearchStore:
    """Publish exact canonical bytes once; registration provides the external hash."""

    def __init__(self, workspace: WorkspacePaths) -> None:
        self.workspace = workspace

    @staticmethod
    def validate_publication_size(
        target: ExecutionTarget, evidence: ApprovedResearch
    ) -> None:
        """Preflight the largest submission and run envelope before approval."""
        job_id = "a" * (116 if isinstance(target, ComparisonManifest) else 128)
        run_id = (
            f"{job_id}_candidate_1"
            if isinstance(target, ComparisonManifest)
            else job_id
        )
        SubmittedResearchStore._content(
            ResearchSnapshot(
                job_id=job_id, run_id=run_id, target=target, research_evidence=evidence
            )
        )

    def path(self, identifier: str, *, run: bool = False) -> Path:
        if re.fullmatch(_ID, identifier) is None:
            raise ManifestError("Invalid submitted research ID")
        root = self.workspace.root
        if root.is_symlink() or root.resolve() != root:
            raise ManifestError("Research snapshot path leaves workspace")
        parts = (
            ("runs", identifier, "research_evidence.json")
            if run
            else ("submitted_research", f"{identifier}.json")
        )
        path = root
        for part in parts:
            path /= part
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ManifestError("Research snapshot path leaves workspace")
        return path

    def save(
        self, job_id: str, evidence: ApprovedResearch, target: ExecutionTarget
    ) -> Path:
        snapshot = ResearchSnapshot(
            job_id=job_id, target=target, research_evidence=evidence
        )
        return self._publish(snapshot)

    def publish_run(
        self,
        job_id: str,
        run_id: str,
        evidence: ApprovedResearch,
        target: ExecutionTarget,
    ) -> Path:
        snapshot = ResearchSnapshot(
            job_id=job_id, run_id=run_id, target=target, research_evidence=evidence
        )
        return self._publish(snapshot)

    def _publish(self, snapshot: ResearchSnapshot) -> Path:
        content = self._content(snapshot)
        path = self.path(
            snapshot.run_id or snapshot.job_id, run=snapshot.run_id is not None
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb", prefix=".research-", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            self.path(
                snapshot.run_id or snapshot.job_id, run=snapshot.run_id is not None
            )
            try:
                os.link(temporary, path)
            except FileExistsError:
                saved = self._read(
                    snapshot.run_id or snapshot.job_id, run=snapshot.run_id is not None
                )
                if self._content(saved) != content:
                    raise ManifestError(
                        "Research snapshot already contains different content"
                    ) from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    def load(
        self, job_id: str, *, expected_hash: str, target: ExecutionTarget
    ) -> ApprovedResearch:
        snapshot = self._read(job_id)
        if (
            snapshot.job_id != job_id
            or snapshot.run_id is not None
            or snapshot.target.model_dump(mode="json") != target.model_dump(mode="json")
        ):
            raise ManifestError("Submitted research identity differs from its job")
        self._verify_evidence(snapshot, expected_hash)
        snapshot.research_evidence.verify(target)
        return snapshot.research_evidence

    def load_for_job(
        self, job: dict[str, Any], target: ExecutionTarget
    ) -> ApprovedResearch | None:
        expected_hash = job.get("research_evidence_hash")
        path = self.path(str(job["job_id"]))
        if expected_hash is None and not path.exists():
            return None
        if not isinstance(expected_hash, str):
            raise ManifestError("Submitted research has no declared evidence hash")
        return self.load(str(job["job_id"]), expected_hash=expected_hash, target=target)

    def load_run(
        self,
        run: dict[str, Any],
        store: SQLiteStore,
        manifest: ExecutionManifest | None,
    ) -> ApprovedResearch | None:
        run_id = str(run["run_id"])
        job_id = str(run["job_id"])
        path = self.path(run_id, run=True)
        artifact = store.get_artifact(f"{run_id}:research_evidence")
        job = store.get_job(job_id)
        expected_hash = job.get("research_evidence_hash") if job is not None else None
        if (
            expected_hash is None
            and artifact is None
            and not path.exists()
            and not self.path(job_id).exists()
        ):
            return None
        if manifest is None or artifact is None or not isinstance(expected_hash, str):
            raise ManifestError(
                "Recorded research evidence is missing its identity or registration"
            )
        if (
            artifact.get("artifact_id") != f"{run_id}:research_evidence"
            or artifact.get("run_id") != run_id
            or artifact.get("artifact_type") != "research_evidence"
            or artifact.get("path") != str(path)
            or artifact.get("metadata", {}).get("job_id") != job_id
            or artifact.get("metadata", {}).get("research_evidence_hash")
            != expected_hash
            or not isinstance(artifact.get("sha256"), str)
        ):
            raise ManifestError(
                "Registered research evidence identity differs from the run"
            )
        snapshot = self._read(run_id, run=True, expected_sha=artifact["sha256"])
        if (
            snapshot.job_id != job_id
            or snapshot.run_id != run_id
            or snapshot.manifest_for_run().model_dump(mode="json")
            != manifest.model_dump(mode="json")
        ):
            raise ManifestError("Research snapshot differs from its run membership")
        if job is None or snapshot.target.manifest_hash != job.get("manifest_hash"):
            raise ManifestError("Research snapshot differs from its parent identity")
        self._verify_evidence(snapshot, expected_hash)
        return snapshot.research_evidence

    @staticmethod
    def _verify_evidence(snapshot: ResearchSnapshot, expected_hash: str) -> None:
        if snapshot.research_evidence.content_hash != expected_hash:
            raise ManifestError("Research evidence differs from the declared hash")
        snapshot.research_evidence.verify(snapshot.target)

    def _read(
        self, identifier: str, *, run: bool = False, expected_sha: str | None = None
    ) -> ResearchSnapshot:
        path = self.path(identifier, run=run)
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_BYTES:
                    raise ValueError("Invalid research snapshot size or file type")
                content = stream.read(_MAX_BYTES + 1)
            if len(content) > _MAX_BYTES:
                raise ValueError("Research snapshot exceeds the stored byte limit")
            if (
                expected_sha is not None
                and hashlib.sha256(content).hexdigest() != expected_sha
            ):
                raise ValueError("Research snapshot file SHA differs from registration")
            snapshot = ResearchSnapshot.model_validate_json(content)
            if self._content(snapshot) != content:
                raise ValueError("Research snapshot bytes are not canonical")
            if (snapshot.run_id if run else snapshot.job_id) != identifier:
                raise ValueError("Research snapshot ID differs from its file")
            return snapshot
        except (OSError, ValueError) as exc:
            raise ManifestError(
                f"Cannot read valid research snapshot: {identifier}"
            ) from exc

    @staticmethod
    def _content(snapshot: ResearchSnapshot) -> bytes:
        ResearchSnapshot.model_validate_json(snapshot.model_dump_json())
        content = canonical(snapshot.model_dump(mode="json")) + b"\n"
        if len(content) > _MAX_BYTES:
            raise ManifestError("Research snapshot exceeds the stored byte limit")
        return content
