"""Subprocess worker for backtest and walk-forward jobs."""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

from tradingdev.adapters.execution.process_runner import WorkerHandle
from tradingdev.app import job_store
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.execution_submission import verify_submission_research
from tradingdev.domain.execution import ExecutionManifest, ManifestError
from tradingdev.shared.utils.logger import setup_logger

logger = setup_logger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _run_backtest(job_id: str) -> None:
    """Run a background job and persist metrics."""
    logger.info("Worker started: job=%s", job_id)
    handle = WorkerHandle.from_environment()
    job_store.update_job(
        job_id,
        status="downloading_data",
        **handle.job_fields(),
    )

    try:
        service = BacktestService()
        manifest = job_store.load_manifest(job_id)
        if not isinstance(manifest, ExecutionManifest):
            raise ManifestError("Comparison requires the comparison worker")
        verify_submission_research(job_store.get_default_job_store(), job_id, manifest)
        job_store.update_job(job_id, status="running_backtest")
        run = service.run_manifest(manifest)
        job_store.update_job(job_id, data_downloaded=True, dataset_id=run.dataset_id)
        result_path = job_store.save_result(job_id, run.metrics, pipeline=run.pipeline)
        job_store.update_job(
            job_id,
            status="done",
            ended_at=_now_iso(),
            result_path=str(result_path),
        )
        logger.info("Job %s done -> %s", job_id, result_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Worker failed")
        job_store.update_job(
            job_id,
            status="failed",
            error=str(exc),
            ended_at=_now_iso(),
        )


def main() -> None:
    """Run the worker."""
    parser = argparse.ArgumentParser(
        description="MCP backtest worker (run as subprocess)"
    )
    parser.add_argument("job_id", help="Job ID assigned by MCP server")
    args = parser.parse_args()
    _run_backtest(args.job_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
