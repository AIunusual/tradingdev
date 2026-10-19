"""Pure signal ranking by catalog direction and measurement context.

Reject incomparable contexts unless explicitly allowed. Report candidate
counts and a multiple-comparison caveat rather than treating the selected
winner as unqualified evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal

from tradingdev.domain.ml.evaluation.catalog import SIGNAL_METRIC_CATALOG

Direction = Literal["maximize", "minimize"]

_NORMAL = NormalDist()

# Metrics whose significance the companion t-statistic actually speaks to.
# ``ic_t_stat`` tests whether the mean IC differs from zero, so it backs the
# level metrics only: attaching it to ``ic_std`` or a positive-rate would
# assert a significance those numbers do not carry. Everything outside this
# map is compared on face value alone.
# Below this many non-overlapping IC windows the normal approximation used by
# :func:`required_t_stat` is visibly optimistic, so say so.
_COMFORTABLE_PERIODS = 30

_T_STAT_COMPANION = {
    "ic_mean": "ic_t_stat",
    "icir": "ic_t_stat",
    "ic_t_stat": "ic_t_stat",
    "rank_ic_mean": "rank_ic_t_stat",
    "rank_icir": "rank_ic_t_stat",
    "rank_ic_t_stat": "rank_ic_t_stat",
}


@dataclass(frozen=True)
class EvaluationContext:
    """Measurement identity for comparable signal scores.

    Includes instrument, timeframe, dates, horizon, evaluation settings,
    feature set, build-time code and dataset content. Different identities
    must not silently replace or compare each other's evidence.
    """

    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    horizon: int
    window: int = 60
    n_quantiles: int = 5
    feature_set: str = ""
    """Named feature set the signal's features came from; ``""`` when unknown."""
    code_fingerprint: str = ""
    """Fingerprint of the feature code that built them; ``""`` when unknown."""
    data_fingerprint: str = ""
    """Content hash of the measured dataset; ``""`` when no dataset is bound."""

    @property
    def key(self) -> str:
        """Stable identity used to decide whether entries may be compared."""
        return "|".join(
            (
                self.symbol,
                self.timeframe,
                self.start_date,
                self.end_date,
                f"horizon={self.horizon}",
                f"window={self.window}",
                f"quantiles={self.n_quantiles}",
                f"features={self.feature_set}",
                f"code={self.code_fingerprint}",
                f"data={self.data_fingerprint}",
            )
        )

    def describe(self) -> str:
        """One-line human description of the measurement conditions."""
        return (
            f"{self.symbol} {self.timeframe} {self.start_date}..{self.end_date} "
            f"(horizon={self.horizon}, window={self.window}, "
            f"quantiles={self.n_quantiles}, features={self.feature_set or '-'}, "
            f"code={self.code_fingerprint or '-'}, data={self.data_fingerprint or '-'})"
        )

    def matches(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        horizon: int | None = None,
        feature_set: str | None = None,
    ) -> bool:
        """Whether the context passes a set of optional filters."""
        return (
            (symbol is None or self.symbol == symbol)
            and (timeframe is None or self.timeframe == timeframe)
            and (horizon is None or self.horizon == horizon)
            and (feature_set is None or self.feature_set == feature_set)
        )


@dataclass(frozen=True)
class LeaderboardEntry:
    """One evaluated candidate, with what it was evaluated on."""

    entry_id: str
    label: str
    context: EvaluationContext
    metrics: Mapping[str, float]
    n_observations: int
    model_id: str = ""
    dataset_id: str = ""

    def metric(self, name: str) -> float | None:
        """Return a finite metric value, or None when missing/non-finite."""
        value = self.metrics.get(name)
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection for the store."""
        return {
            "entry_id": self.entry_id,
            "label": self.label,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "n_observations": self.n_observations,
            "metrics": {key: float(value) for key, value in self.metrics.items()},
            "context": {
                "symbol": self.context.symbol,
                "timeframe": self.context.timeframe,
                "start_date": self.context.start_date,
                "end_date": self.context.end_date,
                "horizon": self.context.horizon,
                "window": self.context.window,
                "n_quantiles": self.context.n_quantiles,
                "feature_set": self.context.feature_set,
                "code_fingerprint": self.context.code_fingerprint,
                "data_fingerprint": self.context.data_fingerprint,
            },
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> LeaderboardEntry:
        """Rebuild an entry from a stored projection."""
        raw_context = record.get("context")
        if not isinstance(raw_context, Mapping):
            raise ValueError("leaderboard record is missing its context")
        context = EvaluationContext(
            symbol=str(raw_context["symbol"]),
            timeframe=str(raw_context["timeframe"]),
            start_date=str(raw_context["start_date"]),
            end_date=str(raw_context["end_date"]),
            horizon=int(raw_context["horizon"]),
            window=int(raw_context.get("window", 60)),
            n_quantiles=int(raw_context.get("n_quantiles", 5)),
            # Older records predate these fields; an unknown feature identity
            # stays unknown rather than being guessed.
            feature_set=str(raw_context.get("feature_set", "")),
            code_fingerprint=str(raw_context.get("code_fingerprint", "")),
            data_fingerprint=str(raw_context.get("data_fingerprint", "")),
        )
        raw_metrics = record.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise ValueError("leaderboard record is missing its metrics")
        return cls(
            entry_id=str(record["entry_id"]),
            label=str(record["label"]),
            model_id=str(record.get("model_id", "")),
            dataset_id=str(record.get("dataset_id", "")),
            n_observations=int(record.get("n_observations", 0)),
            metrics={str(key): float(value) for key, value in raw_metrics.items()},
            context=context,
        )


def build_entry_id(
    *,
    model_id: str,
    dataset_id: str,
    context: EvaluationContext,
    signal_fingerprint: str = "",
) -> str:
    """Identify one candidate within a dataset and measurement context.

    Repeated evaluation replaces that row without inflating candidate count.
    Use ``signal_fingerprint`` when model_id is absent so distinct ad-hoc
    signals remain distinct. A supplied model ID is a caller-declared label.
    """
    payload = "|".join((model_id, dataset_id, context.key, signal_fingerprint))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def signal_fingerprint(signal: Sequence[float]) -> str:
    """Hash a finite signal series, preserving every value boundary."""
    payload = json.dumps(
        [float(value) for value in signal],
        allow_nan=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RankedEntry:
    """An entry placed in the ranking."""

    rank: int
    entry: LeaderboardEntry
    value: float
    tied: int


@dataclass(frozen=True)
class SkippedEntry:
    """An entry that could not be ranked, and why."""

    entry: LeaderboardEntry
    reason: str


@dataclass(frozen=True)
class LeaderboardResult:
    """A ranking plus everything a reader needs to discount it."""

    metric: str
    direction: Direction
    ranked: tuple[RankedEntry, ...]
    skipped: tuple[SkippedEntry, ...]
    caveats: tuple[str, ...]

    @property
    def leader(self) -> RankedEntry | None:
        """The top entry, or ``None`` when nothing could be ranked."""
        return self.ranked[0] if self.ranked else None


def metric_direction(metric: str) -> Direction:
    """Return catalog direction; reject unknown or directionless metrics."""
    definition = SIGNAL_METRIC_CATALOG.get(metric)
    if definition is None:
        raise ValueError(f"unknown signal metric: {metric}")
    direction = definition.optimization_direction
    if direction is None:
        raise ValueError(
            f"metric '{metric}' has no optimisation direction and cannot rank "
            "candidates"
        )
    return direction


def rankable_metrics() -> tuple[str, ...]:
    """Every signal metric that declares an optimisation direction."""
    return tuple(
        metric_id
        for metric_id, definition in SIGNAL_METRIC_CATALOG.items()
        if definition.optimization_direction is not None
    )


def required_t_stat(n_candidates: int, *, alpha: float = 0.05) -> float:
    """Two-sided Bonferroni threshold for ``n_candidates`` at family-wise alpha.

    Uses a normal approximation, which is anti-conservative for short IC
    series. Require at least one candidate and ``0 < alpha < 1``.
    """
    if n_candidates < 1:
        raise ValueError("n_candidates must be at least 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    return _NORMAL.inv_cdf(1.0 - alpha / (2.0 * n_candidates))


def comparability_error(entries: Sequence[LeaderboardEntry]) -> str | None:
    """Why these entries cannot be ranked together, or ``None`` if they can."""
    contexts = sorted({entry.context.key for entry in entries})
    if len(contexts) <= 1:
        return None
    described = sorted({entry.context.describe() for entry in entries})
    return (
        f"entries were not measured on the same conditions "
        f"({len(contexts)} distinct: {'; '.join(described)}); "
        "rank them separately per context or pass allow_mixed to override"
    )


def _selection_caveat(
    metric: str, n_candidates: int, leader: RankedEntry, *, alpha: float
) -> str:
    threshold = required_t_stat(n_candidates, alpha=alpha)
    suffix = _approximation_note(leader)
    companion = _T_STAT_COMPANION.get(metric)
    observed = leader.entry.metric(companion) if companion else None
    if companion and observed is not None:
        if observed < threshold:
            return (
                f"best of {n_candidates} candidates: the winner's {companion} "
                f"({observed:.3f}) is below the {threshold:.3f} needed after "
                f"{n_candidates} comparisons at alpha={alpha}, so the lead is "
                f"within what selection alone produces{suffix}"
            )
        return (
            f"best of {n_candidates} candidates: the winner's {companion} "
            f"({observed:.3f}) clears the {threshold:.3f} needed after "
            f"{n_candidates} comparisons at alpha={alpha}{suffix}"
        )
    return (
        f"best of {n_candidates} candidates: the winner is drawn from "
        f"{n_candidates} comparisons, so expect it to be optimistic; a "
        f"t-statistic above {threshold:.3f} would be needed to call it signal"
        f"{suffix}"
    )


def _approximation_note(leader: RankedEntry) -> str:
    """Warn when the normal approximation is least trustworthy."""
    effective = leader.entry.metrics.get("n_effective_periods")
    if effective is None or not math.isfinite(float(effective)):
        return ""
    if float(effective) >= _COMFORTABLE_PERIODS:
        return ""
    return (
        f"; only {int(effective)} effective periods back it, where the normal "
        "approximation understates the bar"
    )


def rank_entries(
    entries: Iterable[LeaderboardEntry],
    metric: str,
    *,
    top_k: int | None = None,
    alpha: float = 0.05,
) -> LeaderboardResult:
    """Rank by catalog direction, returning skipped entries and selection caveats.

    Call ``comparability_error`` first to verify shared measurement context.
    ``alpha`` controls the family-wise caveat; ``top_k`` limits output.
    Reject unknown/directionless metrics and non-positive ``top_k``.
    """
    direction = metric_direction(metric)
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1")

    candidates: list[tuple[float, LeaderboardEntry]] = []
    skipped: list[SkippedEntry] = []
    for entry in entries:
        value = entry.metric(metric)
        if value is None:
            raw = entry.metrics.get(metric)
            reason = (
                f"no '{metric}' recorded"
                if raw is None
                else f"'{metric}' is not finite ({raw})"
            )
            skipped.append(SkippedEntry(entry=entry, reason=reason))
            continue
        candidates.append((value, entry))

    # Two passes keep ties in a stable, id-ordered sequence whichever way the
    # values are sorted: a single reverse sort would also reverse the tie-break.
    candidates.sort(key=lambda pair: pair[1].entry_id)
    candidates.sort(key=lambda pair: pair[0], reverse=direction == "maximize")

    ranked: list[RankedEntry] = []
    rank = 0
    previous: float | None = None
    for index, (value, entry) in enumerate(candidates):
        if previous is None or not math.isclose(value, previous):
            rank = index + 1
        previous = value
        ranked.append(RankedEntry(rank=rank, entry=entry, value=value, tied=0))

    tied_counts = Counter(item.rank for item in ranked)
    ranked = [
        RankedEntry(
            rank=item.rank,
            entry=item.entry,
            value=item.value,
            tied=tied_counts[item.rank],
        )
        for item in ranked
    ]

    caveats: list[str] = []
    n_candidates = len(candidates)
    if skipped:
        caveats.append(
            f"{len(skipped)} of {n_candidates + len(skipped)} entries were not "
            f"ranked (missing or non-finite '{metric}')"
        )
    if n_candidates > 1 and ranked:
        caveats.append(_selection_caveat(metric, n_candidates, ranked[0], alpha=alpha))
    if n_candidates == 1 and ranked:
        caveats.append(
            "only one candidate was compared; a single entry proves nothing "
            "about the alternatives"
        )

    return LeaderboardResult(
        metric=metric,
        direction=direction,
        ranked=tuple(ranked[:top_k] if top_k is not None else ranked),
        skipped=tuple(skipped),
        caveats=tuple(caveats),
    )
