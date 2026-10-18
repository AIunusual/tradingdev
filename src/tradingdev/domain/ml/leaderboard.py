"""Ranking of evaluated signals, and the mistakes a leaderboard invites.

Sorting a column of scores looks harmless and is not:

* every extra candidate raises the winning score by luck alone, so the top of
  a long list is the most optimistic number in it;
* a metric is not "better when bigger" by default - ``signal_turnover`` and
  ``ic_std`` improve when they shrink, and a metric like ``n_periods`` has no
  direction at all;
* two numbers measured on different things cannot be compared, however tidy
  they look side by side: a Rank IC earned on BTC in 2024 says nothing about
  one earned on ETH in 2025.

This module makes all three explicit instead of leaving them to the caller.
Direction always comes from the signal metric catalog, entries that were not
measured on the same context are refused unless the caller insists, and the
size of the comparison is reported back with the critical value the winner
would have to clear.

Everything here is pure: no IO, no workspace, no clock.
"""

from __future__ import annotations

import hashlib
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
    """What a signal was measured on.

    Two entries are comparable only when their contexts are equal: the same
    instrument, bar size, date range, label horizon, evaluation settings *and*
    the feature code that produced the numbers. A difference in any of them
    changes what the metrics mean.

    ``feature_set`` and ``code_fingerprint`` are part of the identity for the
    same reason as the rest, and for one extra one: without them a bug fix in
    the feature code would leave the dataset id - and therefore the entry id -
    unchanged, so the new number would silently *overwrite* the old row and the
    leaderboard would present post-fix evidence as if it were the same
    candidate measured before the fix. With them, the two get different ids and
    the mismatch is reported instead of absorbed.
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
            )
        )

    def describe(self) -> str:
        """One-line human description of the measurement conditions."""
        return (
            f"{self.symbol} {self.timeframe} {self.start_date}..{self.end_date} "
            f"(horizon={self.horizon}, window={self.window}, "
            f"quantiles={self.n_quantiles}, features={self.feature_set or '-'}, "
            f"code={self.code_fingerprint or '-'})"
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
        """The metric value, or ``None`` when missing or not finite.

        A NaN would otherwise sort above every real number in one direction and
        below it in the other, so it is never a rankable value.
        """
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
    """Fingerprint a candidate in one context.

    Re-evaluating the same model on the same data under the same settings
    produces the same id, so the store updates that row instead of adding a
    second one. Counting the same candidate twice would inflate the size of
    the comparison and quietly make the winner look more significant.

    Args:
        model_id: Model that produced the signal. Empty for a hand-supplied
            signal, which is why ``signal_fingerprint`` exists: without a
            model to identify it, two different ad-hoc signals would collide
            into one row.
        dataset_id: Dataset the signal was computed on.
        context: What the signal was measured on.
        signal_fingerprint: Hash of the signal values; set it when there is no
            model id, so distinct ad-hoc signals stay distinct candidates.
    """
    payload = "|".join((model_id, dataset_id, context.key, signal_fingerprint))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def signal_fingerprint(signal: Sequence[float]) -> str:
    """Hash a signal series so an unmodelled candidate is still identifiable."""
    digest = hashlib.sha256()
    for value in signal:
        digest.update(repr(float(value)).encode("utf-8"))
    return digest.hexdigest()[:16]


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
    """The catalog's optimisation direction for ``metric``.

    Raises:
        ValueError: If the metric is unknown, or is declared without a
            direction. A directionless metric (``n_periods``) cannot order
            candidates: larger is not better, it is just larger.
    """
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
    """Critical two-sided t-value after a Bonferroni correction.

    Picking the best of ``n_candidates`` is ``n_candidates`` comparisons, so
    the usual 5% bar is far too lenient for the winner. This returns the value
    a winner's t-statistic must exceed before the ranking says anything.

    The threshold is a normal approximation rather than a t quantile, which is
    slightly anti-conservative when the IC series is short; callers that rank
    on few effective periods should read the winner as even weaker than this
    says.

    Args:
        n_candidates: How many candidates were compared.
        alpha: Family-wise error rate to control.

    Raises:
        ValueError: If ``n_candidates`` is below 1 or ``alpha`` is not in
            ``(0, 1)``.
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
    """Rank evaluated candidates on one catalog-declared metric.

    Args:
        entries: Candidates to rank. They must share an
            :class:`EvaluationContext`; use :func:`comparability_error` first.
        metric: A signal metric id with a declared direction.
        top_k: Keep only the first ``top_k`` ranks.
        alpha: Family-wise error rate used for the selection caveat.

    Returns:
        The ranking, the entries that could not be placed, and the caveats a
        reader needs in order to discount the winner.

    Raises:
        ValueError: On an unknown or directionless metric, or ``top_k`` below 1.
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
