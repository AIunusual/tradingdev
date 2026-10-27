"""Submit already prepared execution specifications to background workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.adapters.execution.process_runner import ProcessRunner
    from tradingdev.app.job_store import JobStore
    from tradingdev.domain.execution import ExecutionManifest


@dataclass(frozen=True)
class PreparedExecution:
    """A captured execution and the original configuration's provenance."""

    manifest: ExecutionManifest
    original_config_path: Path
    synthetic_fixture_rows: tuple[int, ...] = ()


class ExecutionSubmissionService:
    """Persist and launch a fixed manifest without resolving configuration again."""

    def __init__(self, job_store: JobStore, process_runner: ProcessRunner) -> None:
        self._job_store = job_store
        self._process_runner = process_runner

    def submit(
        self, prepared: PreparedExecution, *, job_id: str | None = None
    ) -> dict[str, Any]:
        """Create one job and retain startup failures, including interruptions."""
        prepared.manifest.verify()
        selected_id = job_id if job_id is not None else uuid4().hex[:12]
        with self._job_store.submission_lock(selected_id):
            return self._submit_locked(prepared, selected_id)

    def _submit_locked(
        self, prepared: PreparedExecution, selected_id: str
    ) -> dict[str, Any]:
        manifest = prepared.manifest
        manifest.verify()
        if self._job_store.get_job(selected_id) is not None:
            raise ValueError(f"Job already exists: {selected_id}")
        extra: dict[str, Any] = {
            "original_config_path": str(prepared.original_config_path),
        }
        if manifest.optimization is not None:
            extra["total_combinations"] = manifest.optimization.total_combinations
        self._job_store.create_job(
            job_id=selected_id, manifest=manifest, extra_payload=extra
        )
        module = (
            "tradingdev.mcp.workers.optimization"
            if manifest.kind == "optimization"
            else "tradingdev.mcp.workers.backtest"
        )
        try:
            identity = self._process_runner.spawn_module(module, selected_id)
        except BaseException as exc:
            self._job_store.update_job(
                selected_id,
                status="failed",
                error=f"Worker failed to start: {type(exc).__name__}: {exc}",
            )
            raise
        self._job_store.update_job(selected_id, **identity.job_fields())
        return {
            "job_id": selected_id,
            "revision_id": manifest.config_copy()["strategy"].get("revision_id"),
            "manifest_hash": manifest.manifest_hash,
        }
