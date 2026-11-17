"""Immutable performance JSON artifacts shared by CLI and background jobs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.storage.filesystem import now_iso, sha256_file
from tradingdev.domain.performance.artifacts import (
    ObservationsBundle,
    PerformanceArtifacts,
    PerformanceBundle,
    validate_projection,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


class PerformanceArtifactError(ValueError):
    """A missing historical artifact or an invalid recorded artifact."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PerformanceStore:
    """Publish a matching JSON pair; load values without importing pickle."""

    def __init__(self, workspace: WorkspacePaths, store: SQLiteStore) -> None:
        self._workspace = workspace
        self._store = store

    def _path(self, run_id: str, name: str) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise PerformanceArtifactError(
                "performance_artifact_invalid", "Invalid run ID"
            )
        path = self._workspace.root
        for part in ("runs", run_id, f"{name}.json"):
            path /= part
            if path.is_symlink() or not path.resolve().is_relative_to(
                self._workspace.root
            ):
                raise PerformanceArtifactError(
                    "performance_artifact_invalid", "Performance path leaves workspace"
                )
        return path

    @staticmethod
    def _encode(bundle: PerformanceBundle | ObservationsBundle) -> bytes:
        return (
            json.dumps(
                bundle.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    def publish(self, artifacts: PerformanceArtifacts) -> None:
        """Publish verified files once and register both artifacts in one transaction.

        Files and SQLite are not a shared transaction. A failed publication removes
        only newly created files; an existing result is never replaced or removed.
        Readers independently validate the recorded digests and execution identity.
        """
        artifacts = PerformanceArtifacts.model_validate_json(
            artifacts.model_dump_json()
        )
        run_id = artifacts.performance.run_id
        run = self._store.get_run(run_id)
        if (
            run is None
            or run.get("manifest_hash") != artifacts.performance.manifest_hash
        ):
            raise PerformanceArtifactError(
                "performance_artifact_invalid",
                "Performance identity differs from its run",
            )
        payloads = [
            ("performance", self._encode(artifacts.performance)),
            ("observations", self._encode(artifacts.observations)),
        ]
        paths = [
            (name, self._path(run_id, name), content) for name, content in payloads
        ]
        created: list[Path] = []
        temporaries: list[Path] = []
        try:
            records = []
            for name, path, content in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(
                    prefix=f".{name}-", dir=path.parent, delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    temporaries.append(temporary)
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                self._path(run_id, name)
                try:
                    os.link(temporary, path)
                    created.append(path)
                except FileExistsError:
                    if path.read_bytes() != content:
                        raise PerformanceArtifactError(
                            "performance_artifact_invalid",
                            f"Existing {name} artifact has different content",
                        ) from None
                records.append(
                    (
                        f"{run_id}:{name}_json",
                        run_id,
                        f"{name}_json",
                        str(path),
                        hashlib.sha256(content).hexdigest(),
                        json.dumps(
                            {
                                "schema_version": 1,
                                "manifest_hash": artifacts.performance.manifest_hash,
                            },
                            sort_keys=True,
                        ),
                        now_iso(),
                    )
                )
            with self._store.connect() as connection:
                for record in records:
                    existing = connection.execute(
                        "select run_id, artifact_type, path, sha256 from artifacts "
                        "where artifact_id = ?",
                        (record[0],),
                    ).fetchone()
                    if existing is not None and tuple(existing) != record[1:5]:
                        raise PerformanceArtifactError(
                            "performance_artifact_invalid",
                            "Existing performance artifact identity differs",
                        )
                connection.executemany(
                    "insert into artifacts (artifact_id, run_id, artifact_type, "
                    "path, sha256, metadata, created_at) "
                    "values (?, ?, ?, ?, ?, ?, ?) on conflict(artifact_id) do nothing",
                    records,
                )
        except BaseException:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise
        finally:
            for temporary in temporaries:
                temporary.unlink(missing_ok=True)

    @contextmanager
    def publication(
        self,
        artifacts: PerformanceArtifacts,
        projection: dict[str, Any],
        output_paths: list[Path],
    ) -> Generator[bool]:
        """Protect a whole new run publication, restoring files and rows on failure.

        Yield true for an identical completed result, which the caller must leave
        unchanged. The marker prevents concurrent writers using this protocol;
        a process killed outside Python cleanup leaves an explicit busy marker.
        """
        artifacts = PerformanceArtifacts.model_validate_json(
            artifacts.model_dump_json()
        )
        validate_projection(artifacts, projection)
        run_id = artifacts.performance.run_id
        directory = self._path(run_id, "performance").parent
        existed = directory.exists()
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / ".result-publication"
        try:
            marker.mkdir()
        except FileExistsError:
            raise PerformanceArtifactError(
                "performance_artifact_busy",
                "A result publication is already in progress",
            ) from None
        try:
            existing = self._store.get_run(run_id)
            if existing is not None:
                if (
                    existing.get("metrics") != projection
                    or self._load(run_id, allow_pending=True) != artifacts.performance
                    or self._load_observations(run_id, allow_pending=True)
                    != artifacts.observations
                ):
                    raise PerformanceArtifactError(
                        "performance_artifact_invalid",
                        "An existing run has different results",
                    )
                self._verify_registered_artifacts(run_id)
                yield True
                return
            if self._store.list_artifacts(run_id):
                raise PerformanceArtifactError(
                    "performance_artifact_invalid",
                    "Unowned result artifact records already exist",
                )
            paths = list(
                dict.fromkeys(
                    [
                        *output_paths,
                        self._path(run_id, "performance"),
                        self._path(run_id, "observations"),
                    ]
                )
            )
            backups = {
                path: path.read_bytes() if path.exists() else None for path in paths
            }
            try:
                yield False
            except BaseException as exc:
                failures: list[BaseException] = []
                try:
                    with self._store.connect() as connection:
                        connection.execute(
                            "delete from artifacts where run_id = ?", (run_id,)
                        )
                        connection.execute(
                            "delete from runs where run_id = ?", (run_id,)
                        )
                except BaseException as cleanup_error:
                    failures.append(cleanup_error)
                for path, original in backups.items():
                    try:
                        if original is None:
                            path.unlink(missing_ok=True)
                        elif not path.exists() or path.read_bytes() != original:
                            path.write_bytes(original)
                    except BaseException as cleanup_error:
                        failures.append(cleanup_error)
                if failures:
                    raise BaseExceptionGroup(
                        "Result publication and cleanup failed", [exc, *failures]
                    ) from exc
                raise
        finally:
            marker.rmdir()
            if not existed and not any(directory.iterdir()):
                directory.rmdir()

    def _verify_registered_artifacts(self, run_id: str) -> None:
        """An identical retry succeeds only while every registered file is intact."""
        for artifact in self._store.list_artifacts(run_id):
            try:
                path = Path(artifact["path"])
                if sha256_file(path) != artifact.get("sha256"):
                    raise ValueError("SHA-256 differs from the registered artifact")
            except (OSError, TypeError, ValueError) as exc:
                raise PerformanceArtifactError(
                    "performance_artifact_invalid",
                    "Existing artifact is missing or corrupt: "
                    f"{artifact['artifact_id']}",
                ) from exc

    def _read(
        self, run_id: str, name: str, *, allow_pending: bool = False
    ) -> tuple[bytes, dict[str, Any], str]:
        if (
            not allow_pending
            and (self._path(run_id, name).parent / ".result-publication").exists()
        ):
            raise PerformanceArtifactError(
                "performance_artifact_busy", "Result publication has not completed"
            )
        artifact = self._store.get_artifact(f"{run_id}:{name}_json")
        if artifact is None:
            raise PerformanceArtifactError(
                "performance_artifact_unavailable",
                f"Run has no {name} JSON artifact: {run_id}",
            )
        try:
            run = self._store.get_run(run_id)
            path = self._path(run_id, name)
            if (
                run is None
                or artifact.get("run_id") != run_id
                or artifact.get("artifact_type") != f"{name}_json"
                or artifact.get("path") != str(path)
            ):
                raise ValueError("Artifact does not belong to the requested run")
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != artifact.get("sha256"):
                raise ValueError("Artifact SHA-256 does not match stored metadata")
            return content, run, digest
        except (OSError, ValueError) as exc:
            raise PerformanceArtifactError(
                "performance_artifact_invalid",
                f"Cannot read valid {name} artifact: {exc}",
            ) from exc

    def load(self, run_id: str) -> PerformanceBundle:
        """Read saved scope values and definitions; never recalculate or unpickle."""
        return self._load(run_id)

    def _load(self, run_id: str, *, allow_pending: bool = False) -> PerformanceBundle:
        return self._load_with_hash(run_id, allow_pending=allow_pending)[0]

    def load_with_hash(self, run_id: str) -> tuple[PerformanceBundle, str]:
        """Read a published bundle and the digest of exactly those bytes."""
        return self._load_with_hash(run_id)

    def _load_with_hash(
        self, run_id: str, *, allow_pending: bool = False
    ) -> tuple[PerformanceBundle, str]:
        content, run, digest = self._read(
            run_id, "performance", allow_pending=allow_pending
        )
        try:
            bundle = PerformanceBundle.model_validate_json(content)
            if bundle.run_id != run_id or bundle.manifest_hash != run.get(
                "manifest_hash"
            ):
                raise ValueError(
                    "Performance JSON identity differs from the recorded run"
                )
            return bundle, digest
        except ValueError as exc:
            raise PerformanceArtifactError(
                "performance_artifact_invalid", str(exc)
            ) from exc

    def load_observations(self, run_id: str) -> ObservationsBundle:
        """Read original observations and verify their complete scope mapping."""
        return self._load_observations(run_id)

    def _load_observations(
        self, run_id: str, *, allow_pending: bool = False
    ) -> ObservationsBundle:
        performance = self._load(run_id, allow_pending=allow_pending)
        content, _, _ = self._read(run_id, "observations", allow_pending=allow_pending)
        try:
            observations = ObservationsBundle.model_validate_json(content)
            PerformanceArtifacts(performance=performance, observations=observations)
            return observations
        except ValueError as exc:
            raise PerformanceArtifactError(
                "performance_artifact_invalid", str(exc)
            ) from exc
