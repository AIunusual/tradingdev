"""Unified pipeline result for dashboard consumption.

Wraps both simple backtest and walk-forward validation results
so that ``main.py`` can save a single object and the dashboard
can load it without re-running anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tradingdev.domain.backtest.result import BacktestResult
    from tradingdev.domain.benchmark import BenchmarkEvidence
    from tradingdev.domain.execution import ExecutionManifest
    from tradingdev.domain.validation.walk_forward import (
        WalkForwardResult,
    )


@dataclass
class PipelineResult:
    """Top-level result produced by ``main.py``.

    Attributes:
        mode: ``"simple"`` for a single backtest (e.g. KD strategy)
              or ``"walk_forward"`` when a validation section is present.
        backtest_result: The single :class:`BacktestResult` (simple mode).
        fold_results: Per-fold :class:`WalkForwardResult` list
            (walk-forward mode).  Each entry carries full
            ``train_backtest`` / ``test_backtest`` objects.
        config_snapshot: Copy of the resolved execution config so the
            dashboard can access strategy name, timeframe, etc.
        execution_manifest: Verified specification used for this execution.
        benchmark: Baseline closes for the requested period, or explicit
            evidence that none was defined or usable. Scopes derive their own
            aligned series from it.
    """

    mode: str  # "simple" | "walk_forward"
    backtest_result: BacktestResult | None = None
    fold_results: list[WalkForwardResult] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    execution_manifest: ExecutionManifest | None = None
    benchmark: BenchmarkEvidence | None = None
