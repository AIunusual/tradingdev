"""Immutable generated strategy revisions with an atomic current pointer."""

from __future__ import annotations

import json
import os
import re
import shutil
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import yaml
from filelock import FileLock, Timeout
from filelock import FileLock as LifecycleFileLock
from pydantic import ValidationError

from tradingdev.adapters.storage.filesystem import (
    WorkspacePaths,
    now_iso,
    sha256_file,
    sha256_text,
)
from tradingdev.domain.strategies.schemas import StrategyMetadata, StrategyStatus

if TYPE_CHECKING:
    from collections.abc import Generator

_STRATEGY_ID = re.compile(r"[a-z][a-z0-9_]*")
_MUTABLE_FIELDS = {"status", "updated_at", "validation", "dry_run"}


class StrategyRevisionError(ValueError):
    """A generated revision identifier or stored revision is invalid."""


class StrategyRevisionIntegrityError(StrategyRevisionError):
    """Stored content no longer matches its revision's identity or hashes."""


class StrategyNamespaceOccupiedError(StrategyRevisionError):
    """A new strategy cannot reuse an existing or partially stored namespace."""


class StrategyRevisionConflictError(StrategyRevisionError):
    """A revision request no longer refers to the current revision."""


class UnsupportedStrategyRevisionError(StrategyRevisionError):
    """A legacy flat strategy must be explicitly saved as a new revision."""


class StrategyRevisionStore:
    """Store immutable source/config pairs and revision-specific evidence.

    Publication completes the revision directory before replacing current.json.
    A failed pointer replacement may leave a complete, unreferenced revision;
    readers continue to see the preceding current revision. This is not a
    transaction across the directory and pointer, nor a security sandbox.
    """

    def __init__(self, workspace: WorkspacePaths) -> None:
        self._workspace = workspace

    def create(
        self,
        strategy_id: str,
        code: str,
        config: dict[str, Any],
        request_summary: str = "",
        *,
        expected_revision_id: str | None = None,
    ) -> StrategyMetadata:
        """Create a new namespace or revise one explicitly selected current revision."""
        with self.lifecycle_lock(strategy_id):
            self.check_save_target(strategy_id, expected_revision_id)
            return self._create(strategy_id, code, config, request_summary)

    def check_save_target(
        self, strategy_id: str, expected_revision_id: str | None
    ) -> StrategyMetadata | None:
        """Check a save target while the caller holds its lifecycle lock.

        New saves reserve the whole namespace, including legacy and incomplete
        publications. Revisions require an intact, explicitly selected current
        revision; neither path silently chooses a base for the caller.
        """
        self._validate_strategy_id(strategy_id)
        if expected_revision_id is None:
            paths = (
                self._path(strategy_id),
                self._path(f"{strategy_id}.json"),
                self._path(f"{strategy_id}.py"),
                self._workspace_path("configs", f"{strategy_id}.yaml"),
            )
            if any(path.exists() for path in paths):
                raise StrategyNamespaceOccupiedError(
                    "Strategy namespace already exists; choose a new ID or supply "
                    "the current expected_revision_id to revise it"
                )
            return None
        self._revision_path(strategy_id, expected_revision_id)
        current = self.load(strategy_id)
        if current is None or current.revision_id != expected_revision_id:
            raise StrategyRevisionConflictError(
                "The expected revision is not current; reload and explicitly "
                "select the revision to update"
            )
        return current

    def _create(
        self,
        strategy_id: str,
        code: str,
        config: dict[str, Any],
        request_summary: str,
    ) -> StrategyMetadata:
        self._validate_strategy_id(strategy_id)
        revision_id = uuid4().hex
        revision_path = self._revision_path(strategy_id, revision_id)
        if revision_path.exists():
            raise StrategyRevisionIntegrityError("Revision already exists")
        source_path = revision_path / "strategy.py"
        config_path = revision_path / "config.yaml"
        normalized = deepcopy(config)
        section = normalized.get("strategy")
        if not isinstance(section, dict):
            raise StrategyRevisionError("YAML strategy section must be a mapping")
        class_name = section.get("class_name")
        if not isinstance(class_name, str) or not class_name:
            raise StrategyRevisionError("YAML missing strategy.class_name")
        section.update(
            id=strategy_id,
            revision_id=revision_id,
            source_path=str(source_path),
        )
        section.pop("source_hash", None)
        normalized_yaml = yaml.safe_dump(normalized, sort_keys=False)
        timestamp = now_iso()
        metadata = StrategyMetadata(
            strategy_id=strategy_id,
            revision_id=revision_id,
            class_name=class_name,
            status=StrategyStatus.DRAFT,
            created_at=timestamp,
            updated_at=timestamp,
            request_summary=request_summary,
            source_path=str(source_path),
            config_path=str(config_path),
            source_hash=sha256_text(code),
            config_hash=sha256_text(normalized_yaml),
        )
        revision_path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=".pending-", dir=revision_path.parent
        ) as temporary:
            pending = Path(temporary)
            (pending / "strategy.py").write_text(code, encoding="utf-8", newline="")
            (pending / "config.yaml").write_text(
                normalized_yaml, encoding="utf-8", newline=""
            )
            (pending / "metadata.json").write_text(
                self._serialize(metadata.model_dump(mode="json")),
                encoding="utf-8",
                newline="",
            )
            self._revision_path(strategy_id, revision_id)
            pending.rename(revision_path)
        self._atomic_json(
            self._path(strategy_id, "current.json"), {"revision_id": revision_id}
        )
        return metadata

    def load(
        self, strategy_id: str, revision_id: str | None = None
    ) -> StrategyMetadata | None:
        """Read and verify one revision, resolving current only when omitted."""
        self._validate_strategy_id(strategy_id)
        current = revision_id is None
        if current:
            pointer = self._path(strategy_id, "current.json")
            if not pointer.exists():
                if self._path(f"{strategy_id}.json").exists():
                    raise UnsupportedStrategyRevisionError(
                        f"Legacy strategy {strategy_id!r} has no immutable revision; "
                        "save its source and config explicitly as a new strategy "
                        "without reusing its ID"
                    )
                return None
            raw_pointer = self._read_json(pointer)
            if set(raw_pointer) != {"revision_id"} or not isinstance(
                raw_pointer["revision_id"], str
            ):
                raise StrategyRevisionIntegrityError("Invalid current revision pointer")
            revision_id = raw_pointer["revision_id"]
        assert revision_id is not None
        revision_path = self._revision_path(strategy_id, revision_id)
        if not revision_path.exists():
            if current:
                raise StrategyRevisionIntegrityError("Current revision is missing")
            return None
        metadata_path = self._path(
            strategy_id, "revisions", revision_id, "metadata.json"
        )
        try:
            metadata = StrategyMetadata.model_validate(self._read_json(metadata_path))
        except ValidationError as exc:
            raise StrategyRevisionIntegrityError("Invalid revision metadata") from exc
        if metadata.strategy_id != strategy_id or metadata.revision_id != revision_id:
            raise StrategyRevisionIntegrityError("Revision metadata identity mismatch")
        self.verify(metadata)
        return metadata

    def list_current(self) -> list[StrategyMetadata]:
        """List each generated strategy's verified current revision."""
        root = self._path()
        if not root.exists():
            return []
        strategy_ids = {path.parent.name for path in root.glob("*/current.json")}
        items = []
        for strategy_id in sorted(strategy_ids):
            metadata = self.load(strategy_id)
            if metadata is not None:
                items.append(metadata)
        return items

    def list_legacy_ids(self) -> list[str]:
        """Discover flat metadata names without trusting their old contents.

        A published current pointer supersedes the retained legacy files.
        Discovery does not load Python or grant execution status.
        """
        root = self._path()
        return sorted(
            path.stem
            for path in root.glob("*.json")
            if _STRATEGY_ID.fullmatch(path.stem)
            and not (root / path.stem / "current.json").exists()
        )

    def read_legacy_source(self, strategy_id: str) -> tuple[str, str]:
        """Read the old fixed source/config locations solely for explicit resaving."""
        self._validate_strategy_id(strategy_id)
        source_path = self._path(f"{strategy_id}.py")
        config_path = self._workspace_path("configs", f"{strategy_id}.yaml")
        try:
            return (
                source_path.read_text(encoding="utf-8"),
                config_path.read_text(encoding="utf-8"),
            )
        except (OSError, UnicodeError) as exc:
            raise StrategyRevisionIntegrityError(
                f"Cannot read legacy source/config for {strategy_id!r}; "
                "restore its generated_strategies/<id>.py and configs/<id>.yaml "
                "files, or provide replacement source/config to save_strategy"
            ) from exc

    def update(
        self,
        metadata: StrategyMetadata,
        *,
        expected_status: StrategyStatus | None = None,
        expected_metadata: StrategyMetadata | None = None,
    ) -> None:
        """Atomically update lifecycle evidence for exactly this revision."""
        with self.lifecycle_lock(metadata.strategy_id):
            self._update(
                metadata,
                expected_status=expected_status,
                expected_metadata=expected_metadata,
            )

    def _update(
        self,
        metadata: StrategyMetadata,
        *,
        expected_status: StrategyStatus | None,
        expected_metadata: StrategyMetadata | None,
    ) -> None:
        revision_path = self._revision_path(metadata.strategy_id, metadata.revision_id)
        if not revision_path.is_dir():
            raise StrategyRevisionIntegrityError("Cannot update a missing revision")
        lock_path = self._path(
            metadata.strategy_id, "revisions", metadata.revision_id, ".metadata.lock"
        )
        try:
            with FileLock(lock_path, timeout=10):
                previous = self.load(metadata.strategy_id, metadata.revision_id)
                if previous is None:
                    raise StrategyRevisionIntegrityError(
                        "Cannot update a missing revision"
                    )
                if (
                    expected_metadata is not None
                    and previous.model_dump() != expected_metadata.model_dump()
                ):
                    raise StrategyRevisionError(
                        "Revision metadata changed during this operation; "
                        "reload the revision"
                    )
                if expected_status is not None and previous.status != expected_status:
                    raise StrategyRevisionError(
                        "Revision status changed during this operation; "
                        "reload the revision"
                    )
                if previous.model_dump(exclude=_MUTABLE_FIELDS) != metadata.model_dump(
                    exclude=_MUTABLE_FIELDS
                ):
                    raise StrategyRevisionIntegrityError(
                        "Cannot modify immutable revision fields"
                    )
                self.verify(metadata)
                self._atomic_json(
                    self._path(
                        metadata.strategy_id,
                        "revisions",
                        metadata.revision_id,
                        "metadata.json",
                    ),
                    metadata.model_dump(mode="json"),
                )
        except Timeout as exc:
            raise StrategyRevisionError(
                "Revision metadata is busy; retry the operation"
            ) from exc

    @contextmanager
    def lifecycle_lock(
        self, strategy_id: str, *, timeout: float = 10
    ) -> Generator[None]:
        """Coordinate publication, checks, and cleanup across threads/processes.

        The lock lives outside deletable revisions. FileLock's singleton keeps
        nested operations reentrant within one thread, including separate stores.
        Never unlink this lock file, since waiters may still hold its inode.
        """
        self._validate_strategy_id(strategy_id)
        lock_path = self._path(".locks", f"{strategy_id}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = LifecycleFileLock(lock_path, is_singleton=True)
        try:
            with lock.acquire(timeout=timeout):
                yield
        except Timeout as exc:
            raise StrategyRevisionError(
                "Strategy lifecycle is busy; retry the operation"
            ) from exc

    def revision_ids(self, strategy_id: str) -> list[str]:
        """List published revision directory names without following symlinks."""
        self._validate_strategy_id(strategy_id)
        root = self._path(strategy_id, "revisions")
        if not root.exists():
            return []
        return sorted(
            path.name
            for path in root.iterdir()
            if not path.name.startswith(".pending-")
        )

    def delete_draft(self, strategy_id: str, revision_id: str) -> None:
        """Delete an intact non-current draft after caller checks references.

        The caller must hold lifecycle_lock from its reference scan through this
        call. Filesystem deletion is not transactional: an I/O failure is reported
        and may leave a partial directory that subsequent checks must reject.
        """
        with self.lifecycle_lock(strategy_id):
            metadata = self.load(strategy_id, revision_id)
            current = self.load(strategy_id)
            if metadata is None:
                raise StrategyRevisionError("Revision no longer exists")
            if current is not None and current.revision_id == revision_id:
                raise StrategyRevisionError("Cannot delete the current revision")
            if metadata.status != StrategyStatus.DRAFT:
                raise StrategyRevisionError("Only draft revisions can be deleted")
            path = self._revision_path(strategy_id, revision_id)
            self.verify_cleanup_tree(strategy_id, revision_id)
            shutil.rmtree(path)

    def verify_cleanup_tree(self, strategy_id: str, revision_id: str) -> None:
        """Reject unexpected files and symlinks before recursive deletion."""
        root = self._revision_path(strategy_id, revision_id)
        allowed_files = {
            "strategy.py",
            "config.yaml",
            "metadata.json",
            ".metadata.lock",
        }
        for child in root.rglob("*"):
            relative = child.relative_to(root)
            if child.is_symlink():
                raise StrategyRevisionIntegrityError("Cleanup tree contains a symlink")
            if relative.parts == ("__pycache__",) and child.is_dir():
                continue
            if child.is_file() and (
                relative.as_posix() in allowed_files
                or len(relative.parts) == 2
                and relative.parts[0] == "__pycache__"
                and child.name.startswith("strategy.")
                and child.suffix == ".pyc"
            ):
                continue
            raise StrategyRevisionIntegrityError(
                f"Cleanup tree contains an unexpected entry: {relative}"
            )

    def verify(self, metadata: StrategyMetadata) -> None:
        """Reject changed bytes, escaped paths, or evidence for another revision."""
        revision_path = self._revision_path(metadata.strategy_id, metadata.revision_id)
        source_path = self._path(
            metadata.strategy_id, "revisions", metadata.revision_id, "strategy.py"
        )
        config_path = self._path(
            metadata.strategy_id, "revisions", metadata.revision_id, "config.yaml"
        )
        if metadata.source_path != str(source_path) or metadata.config_path != str(
            config_path
        ):
            raise StrategyRevisionIntegrityError("Revision file paths do not match")
        try:
            for path, expected_hash in (
                (source_path, metadata.source_hash),
                (config_path, metadata.config_hash),
            ):
                if not path.is_file() or sha256_file(path) != expected_hash:
                    raise StrategyRevisionIntegrityError(
                        f"Revision {revision_path.name} content mismatch: {path.name}"
                    )
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise StrategyRevisionIntegrityError(
                "Cannot read revision content"
            ) from exc
        section = config.get("strategy") if isinstance(config, dict) else None
        expected = {
            "id": metadata.strategy_id,
            "revision_id": metadata.revision_id,
            "source_path": str(source_path),
            "class_name": metadata.class_name,
        }
        if not isinstance(section, dict) or any(
            section.get(key) != value for key, value in expected.items()
        ):
            raise StrategyRevisionIntegrityError("Revision config identity mismatch")
        for evidence in (metadata.validation, metadata.dry_run):
            if evidence is not None and evidence.revision_id != metadata.revision_id:
                raise StrategyRevisionIntegrityError(
                    "Evidence revision identity mismatch"
                )

    def _revision_path(self, strategy_id: str, revision_id: str) -> Path:
        self._validate_strategy_id(strategy_id)
        try:
            parsed = UUID(revision_id)
        except (ValueError, AttributeError) as exc:
            raise StrategyRevisionError("revision_id must be UUID4 hex") from exc
        if parsed.version != 4 or parsed.hex != revision_id:
            raise StrategyRevisionError("revision_id must be UUID4 hex")
        return self._path(strategy_id, "revisions", revision_id)

    @staticmethod
    def _validate_strategy_id(strategy_id: str) -> None:
        if not _STRATEGY_ID.fullmatch(strategy_id):
            raise StrategyRevisionError("strategy_id must be lowercase snake_case")

    def _path(self, *parts: str) -> Path:
        return self._workspace_path("generated_strategies", *parts)

    def _workspace_path(self, *parts: str) -> Path:
        path = self._workspace.root
        for part in (None, *parts):
            if part is not None:
                path = path / part
            if path.is_symlink() or not path.resolve().is_relative_to(
                self._workspace.root
            ):
                raise StrategyRevisionIntegrityError("Revision path leaves workspace")
        return path

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StrategyRevisionIntegrityError(f"Cannot read {path.name}") from exc
        if not isinstance(raw, dict):
            raise StrategyRevisionIntegrityError(f"Expected JSON object: {path.name}")
        return raw

    @staticmethod
    def _serialize(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

    def _atomic_json(self, path: Path, payload: dict[str, Any]) -> None:
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix=".pending-",
                dir=path.parent,
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(self._serialize(payload))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
