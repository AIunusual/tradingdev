"""Candidate-agnostic ranking: order, ties and missing values.

The rules here are the ones that are easy to get subtly wrong — tie stability,
``isclose`` boundaries, and missing values that must never become zero — so they
live in one place and both candidate sources call them.

Nothing here knows what a candidate is. The payload is carried through untouched,
the comparison group is an opaque key, and the direction comes from the caller's
metric definition. Two callers share these rules without sharing a data model:
signal ranking in ``domain.ml.leaderboard``, and saved run scopes through
``app.ranking_service.RankingService.rank_run_scopes``.

Candidate ids are strings with a fixed grammar so the tie-break stays
deterministic: ``signal:<entry_id>`` for a recorded evaluation, and
``run:<run_id>#<scope>`` for one scope of a saved run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

Direction = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class MetricDefinition:
    """How one metric is ordered and what its numbers mean.

    ``direction`` is required: a metric without one cannot order candidates.
    ``unit`` is carried for presentation and comparison; the core never scales
    values with it.
    """

    name: str
    direction: Direction
    unit: str = ""


@dataclass(frozen=True)
class Candidate[T]:
    """One thing that can be ranked, and the value it is ranked by.

    ``group_key`` names the measurement conditions that must match for two
    candidates to be comparable. The core does not interpret it; it only refuses
    to rank across groups unless the caller says those groups may be mixed.
    """

    candidate_id: str
    values: Mapping[str, float | None]
    group_key: str
    payload: T


@dataclass(frozen=True)
class RankedCandidate[T]:
    """A candidate placed in the ranking."""

    rank: int
    candidate: Candidate[T]
    value: float
    tied: int


@dataclass(frozen=True)
class SkippedCandidate[T]:
    """A candidate that could not be ranked, and why."""

    candidate: Candidate[T]
    reason: str


@dataclass(frozen=True)
class RankingResult[T]:
    """A ranking plus everything a reader needs to discount it.

    ``ranked_total`` counts the candidates that had a usable value before
    ``top_k`` limited the returned slice, so a caller can still describe the
    selection pressure behind a short page.
    """

    metric: str
    direction: Direction
    ranked: tuple[RankedCandidate[T], ...]
    skipped: tuple[SkippedCandidate[T], ...]
    caveats: tuple[str, ...]
    ranked_total: int
    groups: tuple[str, ...]


def signal_candidate_id(entry_id: str) -> str:
    """The ranking id of a recorded signal evaluation."""
    return f"signal:{entry_id}"


def run_candidate_id(run_id: str, scope: str) -> str:
    """The ranking id of one scope of a saved run."""
    return f"run:{run_id}#{scope}"


def signal_metric_id(metric: str) -> str:
    """The ranking id of a signal metric.

    Metric ids are source-qualified because two catalogues can name the same thing
    differently. ``annual_return`` and ``max_drawdown`` exist on both sides with
    different definitions — a cost-free simulated book on one, an executed backtest
    on the other — so an unqualified id would let them share a column and be read
    as one measurement.
    """
    return f"signal.{metric}"


def run_metric_id(metric: str) -> str:
    """The ranking id of a saved backtest metric; see :func:`signal_metric_id`."""
    return f"run.{metric}"


def rank_candidates[T](
    candidates: Iterable[Candidate[T]],
    definition: MetricDefinition,
    *,
    top_k: int | None = None,
    allow_mixed: bool = False,
) -> RankingResult[T]:
    """Order candidates by one metric, keeping ties and missing values visible.

    Ties share a rank and report how many candidates hold it. A candidate with
    no recorded value, or with a value that is not finite, is skipped with the
    reason kept apart from the others; it is never coerced to zero and never
    dropped from the result.

    ``top_k`` limits the returned slice only. Reject a non-positive ``top_k``.
    Mixed groups raise ``ValueError`` unless ``allow_mixed`` is set; reporting
    that mixing is the caller's job, because each source describes its own
    conditions.
    """
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1")
    metric = definition.name
    items = list(candidates)
    groups = tuple(sorted({item.group_key for item in items}))
    if len(groups) > 1 and not allow_mixed:
        shown = ", ".join(groups[:3])
        raise ValueError(
            f"cannot rank {len(groups)} incomparable groups together: {shown}"
        )

    measured: list[tuple[float, Candidate[T]]] = []
    skipped: list[SkippedCandidate[T]] = []
    for item in items:
        if metric not in item.values:
            # A missing key means nobody recorded the metric; a key with a value
            # that cannot be used means the measurement failed. Keeping the two
            # apart stops a failed run from looking like an unmeasured one.
            skipped.append(
                SkippedCandidate(candidate=item, reason=f"no '{metric}' recorded")
            )
            continue
        raw = item.values[metric]
        value = None if raw is None else float(raw)
        if value is None or not math.isfinite(value):
            skipped.append(
                SkippedCandidate(
                    candidate=item,
                    reason=f"'{metric}' is unavailable or not finite",
                )
            )
            continue
        measured.append((value, item))

    # Two passes keep ties in a stable, id-ordered sequence whichever way the
    # values are sorted: a single reverse sort would also reverse the tie-break.
    measured.sort(key=lambda pair: pair[1].candidate_id)
    measured.sort(key=lambda pair: pair[0], reverse=definition.direction == "maximize")

    ranked: list[RankedCandidate[T]] = []
    rank = 0
    previous: float | None = None
    for index, (value, item) in enumerate(measured):
        if previous is None or not math.isclose(value, previous):
            rank = index + 1
        previous = value
        ranked.append(RankedCandidate(rank=rank, candidate=item, value=value, tied=0))

    tie_counts: dict[int, int] = {}
    for row in ranked:
        tie_counts[row.rank] = tie_counts.get(row.rank, 0) + 1
    ranked = [
        RankedCandidate(
            rank=row.rank,
            candidate=row.candidate,
            value=row.value,
            tied=tie_counts[row.rank],
        )
        for row in ranked
    ]

    caveats: list[str] = []
    n_candidates = len(measured)
    if skipped:
        caveats.append(
            f"{len(skipped)} of {n_candidates + len(skipped)} entries were not "
            f"ranked for '{metric}'. They stay recorded, so a parameter grid "
            "keeps their coordinates even when the score is unavailable; the "
            "ranked candidates are not necessarily adjacent settings."
        )
    if n_candidates == 1 and ranked:
        caveats.append(
            "only one candidate was compared; a single entry proves nothing "
            "about the alternatives"
        )

    return RankingResult(
        metric=metric,
        direction=definition.direction,
        ranked=tuple(ranked[:top_k] if top_k is not None else ranked),
        skipped=tuple(skipped),
        caveats=tuple(caveats),
        ranked_total=n_candidates,
        groups=groups,
    )
