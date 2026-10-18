"""Application service for artifact metadata and content."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import (
    WorkspacePaths,
    sha256_file,
    sha256_text,
)
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.adapters.storage.sqlite import SQLiteStore, get_sqlite_store
from tradingdev.app.run_lineage import (
    extract_random_seed,
    read_strategy_snapshot,
)
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.execution import ManifestError
from tradingdev.domain.performance.artifacts import bundles_from_pipeline
from tradingdev.shared.utils.cache import cache_dir, compute_cache_key


class ArtifactService:
    """Read artifact metadata from SQLite and content from disk."""

    def __init__(
        self,
        *,
        workspace: WorkspacePaths | None = None,
        store: SQLiteStore | None = None,
    ) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._workspace.ensure()
        self._store = store or get_sqlite_store(self._workspace)

    def list_artifacts(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """List artifact metadata."""
        return self._store.list_artifacts(run_id)

    def get_artifact(
        self, artifact_id: str, *, include_content: bool = False
    ) -> dict[str, Any]:
        """Return artifact metadata and optional text content."""
        artifact = self._store.get_artifact(artifact_id)
        if artifact is None:
            return {
                "success": False,
                "error": f"Unknown artifact: {artifact_id}",
                "code": "artifact_not_found",
            }
        result: dict[str, Any] = {"success": True, "artifact": artifact}
        path = Path(str(artifact["path"]))
        if include_content:
            if not path.exists():
                return {
                    "success": False,
                    "error": f"Artifact file missing: {path}",
                    "code": "artifact_file_missing",
                }
            try:
                result["content"] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return {
                    "success": False,
                    "error": f"Artifact is not UTF-8 text: {artifact_id}",
                    "code": "artifact_not_text",
                }
        return result

    def load_pipeline_result(self, run_id: str) -> dict[str, Any]:
        """Load a run's pickled PipelineResult artifact."""
        artifact = next(
            (
                item
                for item in self._store.list_artifacts(run_id)
                if item["artifact_type"] == "pipeline_result"
            ),
            None,
        )
        if artifact is None:
            return {
                "success": False,
                "error": f"No pipeline_result artifact for run: {run_id}",
            }
        path = Path(str(artifact["path"]))
        if not path.exists():
            return {"success": False, "error": f"Artifact file missing: {path}"}
        with path.open("rb") as handle:
            pipeline = pickle.load(handle)  # noqa: S301
        if not isinstance(pipeline, PipelineResult):
            return {
                "success": False,
                "error": f"Artifact is not a PipelineResult: {artifact['artifact_id']}",
            }
        return {
            "success": True,
            "artifact": artifact,
            "pipeline": pipeline,
        }

    def cache_pipeline_result(
        self,
        *,
        pipeline: PipelineResult,
        config_path: Path,
        processed_path: Path,
        metrics: dict[str, Any],
        strategy_id: str,
    ) -> Path:
        """Persist a CLI pipeline result and track it as a SQLite artifact."""
        manifest = pipeline.execution_manifest
        if manifest is None:
            raise ManifestError("CLI result must retain the executed manifest")
        config_payload = manifest.config_copy()
        if config_payload != pipeline.config_snapshot:
            raise ManifestError("Result config differs from the executed manifest")
        expected_kind = (
            "walk_forward" if pipeline.mode == "walk_forward" else "backtest"
        )
        if manifest.kind != expected_kind:
            raise ManifestError("Result mode differs from the executed manifest")
        if config_payload["strategy"].get("id") != strategy_id:
            raise ManifestError("Result strategy differs from the executed manifest")
        source = read_strategy_snapshot(
            config_payload, self._workspace, strategy_id=strategy_id
        )
        key = compute_cache_key(
            manifest_hash=manifest.manifest_hash,
            processed_path=processed_path,
            # The code that produced this result is the strategy revision that
            # ran, and its identity is pinned in the manifest. Deriving it here
            # instead would mean asking git about the local checkout, which is
            # a different question - the checkout is not necessarily the code
            # that ran, and it may not be there to ask.
            code_fingerprint=f"{source.revision_id or '-'}:{source.source_hash or '-'}",
        )
        run_id = f"cli_{key}_{uuid4().hex}"
        performance = bundles_from_pipeline(run_id, pipeline, metrics)
        directory = cache_dir()
        cache_path = directory / f"{run_id}.pkl"
        performance_store = PerformanceStore(self._workspace, self._store)
        with performance_store.publication(
            performance,
            metrics,
            [self._workspace.runs / run_id / "manifest.json", cache_path],
        ) as existing:
            if existing:
                return cache_path
            manifest_path = ExecutionManifestStore(self._workspace).publish(
                run_id, manifest
            )
            directory.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as handle:
                pickle.dump(pipeline, handle)

            config_hash = sha256_text(
                yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True)
            )
            dataset_id = (
                sha256_file(processed_path)
                if processed_path.exists()
                else sha256_text(str(processed_path))
            )
            self._store.create_run(
                run_id=run_id,
                job_id=run_id,
                strategy_id=strategy_id,
                revision_id=source.revision_id,
                manifest_hash=manifest.manifest_hash,
                artifact_dir=manifest_path.parent,
                metrics=metrics,
                config_hash=config_hash,
                source_hash=source.source_hash,
                random_seed=extract_random_seed(config_payload),
                dataset_id=dataset_id,
            )
            self._store.create_artifact(
                artifact_id=f"{run_id}:execution_manifest",
                run_id=run_id,
                artifact_type="execution_manifest",
                path=manifest_path,
                sha256=sha256_file(manifest_path),
                metadata={
                    "manifest_hash": manifest.manifest_hash,
                    "schema_version": manifest.schema_version,
                },
            )
            self._store.create_artifact(
                artifact_id=f"{run_id}:pipeline_result",
                run_id=run_id,
                artifact_type="pipeline_result",
                path=cache_path,
                sha256=sha256_file(cache_path),
                metadata={
                    "source": "cli_cache",
                    "config_path": str(config_path),
                    "processed_path": str(processed_path),
                    "cache_key": key,
                },
            )
            PerformanceStore(self._workspace, self._store).publish(performance)
        return cache_path
