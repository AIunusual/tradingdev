"""Subprocess worker for an approved fixed parameter comparison."""

from __future__ import annotations

import argparse
import logging

from tradingdev.adapters.execution.process_runner import WorkerHandle
from tradingdev.app.comparison_service import ComparisonService
from tradingdev.app.job_store import get_default_job_store
from tradingdev.shared.utils.logger import setup_logger

logger = setup_logger(__name__)


def _run_comparison(job_id: str) -> None:
    jobs = get_default_job_store()
    handle = WorkerHandle.from_environment()
    try:
        ComparisonService(job_store=jobs).execute(job_id, worker_handle=handle)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Comparison worker failed")
        job = jobs.get_job(job_id)
        if (
            job is not None
            and job["status"] == "queued"
            and WorkerHandle.from_job(job) in {None, handle}
        ):
            jobs.update_job(
                job_id, status="failed", error=f"{type(exc).__name__}: {exc}"
            )


def main() -> None:
    """Execute the exact comparison submitted by the application service."""
    parser = argparse.ArgumentParser(description="MCP fixed comparison worker")
    parser.add_argument("job_id")
    args = parser.parse_args()
    _run_comparison(args.job_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
