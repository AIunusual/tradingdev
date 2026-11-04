"""A saved backtest result as a ranking candidate.

The run row records identity; the execution manifest records what was requested.
Both are needed. A metric without its trading conditions cannot be compared with
another strategy's result, and a parameter set without its strategy version cannot
be reproduced later.

This module carries those values. It performs no I/O: the caller reads the run and
its manifest, then builds the candidate here. Missing conditions are refused rather
than defaulted, because an unknown instrument or cost model makes the number
uninterpretable rather than merely imprecise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from tradingdev.domain.ranking import (
    Candidate,
    run_candidate_id,
    run_metric_id,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

MetricSource = Literal["bundle", "legacy_summary"]
"""Where a candidate's metric values came from.

A full performance bundle and a legacy summary are both readable, but they are not
the same measurement. Recording which one produced the numbers lets a mixed table
say so instead of implying a single basis.
"""

RUN_GROUP = "run"
"""Runs share one comparison group.

Different instruments, frequencies, periods and costs do not make two backtests
incomparable: they make the comparison conditional, and the conditions travel with
each candidate so a reader can see them. A source-qualified metric id guarantees
only that both sides were asked for the same catalogued metric. Each run's saved
definition snapshot is checked by the service against the current catalogue.
Provider versions and measurement settings remain in each scope's metadata;
matching definitions do not establish identical measurement conditions.
"""


@dataclass(frozen=True)
class TradingConditions:
    """What one backtest was run on.

    Two results measured under different conditions answer different questions, so
    these travel with the candidate and are reported beside its figures.
    """

    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    fees: float
    slippage: float


@dataclass(frozen=True)
class RunCandidate:
    """One scope of a saved run, with the identity needed to interpret its numbers."""

    run_id: str
    scope: str
    strategy_id: str
    revision_id: str
    parameters: Mapping[str, Any]
    metadata: Mapping[str, Any]
    conditions: TradingConditions
    metric_source: MetricSource


def build_run_candidate(
    *,
    run_id: str,
    scope: str,
    strategy_id: str,
    revision_id: str,
    parameters: Mapping[str, Any],
    metadata: Mapping[str, Any],
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    fees: float,
    slippage: float,
    metric_source: MetricSource,
) -> RunCandidate:
    """Assemble a candidate, refusing a missing identity or trading condition.

    Every argument is required: a blank scope or strategy cannot be ranked against
    anything, and a default cost model would silently change what the metrics mean.
    """
    missing = [
        name
        for name, value in (
            ("run_id", run_id),
            ("scope", scope),
            ("strategy_id", strategy_id),
            ("symbol", symbol),
            ("timeframe", timeframe),
            ("start_date", start_date),
            ("end_date", end_date),
        )
        if not value.strip()
    ]
    if missing:
        raise ValueError(
            "a run candidate requires " + ", ".join(missing) + " to be non-empty"
        )
    for name, cost in (("fees", fees), ("slippage", slippage)):
        # A NaN or negative cost would silently change what the metrics mean, so
        # it is refused here rather than compared later.
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(
                f"a run candidate requires a finite, non-negative {name}: {cost!r}"
            )
    if metric_source not in ("bundle", "legacy_summary"):
        raise ValueError(f"unknown metric source: {metric_source!r}")
    return RunCandidate(
        run_id=run_id,
        scope=scope,
        strategy_id=strategy_id,
        revision_id=revision_id,
        parameters=dict(parameters),
        metadata=dict(metadata),
        conditions=TradingConditions(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            fees=fees,
            slippage=slippage,
        ),
        metric_source=metric_source,
    )


def to_ranking_candidate(
    candidate: RunCandidate,
    values: Mapping[str, float | None],
) -> Candidate[RunCandidate]:
    """Project a saved run into the ranking core, qualifying its metric ids.

    A metric the run did not record is absent from ``values`` rather than zero, so
    the core reports it as skipped with a reason instead of ranking it.
    """
    return Candidate(
        candidate_id=run_candidate_id(candidate.run_id, candidate.scope),
        values={run_metric_id(name): value for name, value in values.items()},
        group_key=RUN_GROUP,
        payload=candidate,
    )
