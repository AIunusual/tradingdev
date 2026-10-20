"""Utilities for parallel grid-search execution."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    import pandas as pd

from tradingdev.shared.utils.logger import setup_logger

logger = setup_logger(__name__)


def _get_performance_core_count() -> int:
    """Return the number of performance (P) cores on Apple Silicon.

    On macOS with Apple Silicon, queries ``sysctl hw.perflevel0.logicalcpu``
    to get the P-core count (excluding efficiency cores).

    On Intel Macs, Linux, and Windows, falls back to :func:`os.cpu_count`.

    Returns:
        Number of performance cores (>= 1).
    """
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(  # noqa: S603, S607
                ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                count = int(result.stdout.strip())
                if count > 0:
                    return count
        except OSError, ValueError, subprocess.TimeoutExpired:
            pass

    return os.cpu_count() or 4


def estimate_n_jobs(
    df: pd.DataFrame,
    safety_factor: float = 0.6,
    overhead_multiplier: float = 3.0,
    reserve_cores: int = 2,
) -> int:
    """Estimate safe number of parallel workers based on available memory.

    Each worker receives a copy of *df* plus intermediate allocations
    (signals DataFrame, equity curve, etc.).  We conservatively assume
    each worker needs ``overhead_multiplier`` × the raw DataFrame size.

    On Apple Silicon Macs, the CPU limit is based on the performance
    (P) core count rather than total cores, since efficiency (E) cores
    are significantly slower for CPU-bound workloads and would become
    bottlenecks in parallel grid search.

    Args:
        df: The DataFrame that will be distributed to workers.
        safety_factor: Fraction of available RAM to use (0–1).
        overhead_multiplier: Estimated memory multiplier per worker.
        reserve_cores: Number of cores to reserve for the OS and other
            processes.  Subtracted from the detected core count.

    Returns:
        Number of workers (>= 1, <= performance_cores - reserve_cores).
    """
    per_worker_bytes = int(df.memory_usage(deep=True).sum() * overhead_multiplier)
    available_bytes = int(psutil.virtual_memory().available * safety_factor)
    max_by_memory = max(1, available_bytes // max(per_worker_bytes, 1))

    perf_cores = _get_performance_core_count()
    max_by_cpu = max(1, perf_cores - reserve_cores)

    n_jobs = min(max_by_memory, max_by_cpu)

    logger.info(
        "Parallel estimate: df=%.1f MB, available=%.1f MB, "
        "n_jobs=%d (mem_limit=%d, perf_cores=%d, reserve=%d)",
        per_worker_bytes / overhead_multiplier / 1e6,
        available_bytes / 1e6,
        n_jobs,
        max_by_memory,
        perf_cores,
        reserve_cores,
    )

    return n_jobs
