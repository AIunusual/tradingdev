"""Pure signal ranking by catalog direction and measurement context.

Reject incomparable contexts unless explicitly allowed. Report candidate
counts and a multiple-comparison caveat rather than treating the selected
winner as unqualified evidence.

Candidates are not all machine-learning models. A row declares the method
family that produced its signal, so a hand-set indicator rule can be ranked
beside a fitted model without being mistaken for one. The family is identity,
not a measurement condition: it stays out of :class:`EvaluationContext`, so
rules and models measured on one dataset remain comparable, and the ranking
reports the mix instead of hiding it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Literal, cast

from tradingdev.domain.ml.evaluation.catalog import SIGNAL_METRIC_CATALOG

Direction = Literal["maximize", "minimize"]
MethodFamily = Literal["ml_model", "rule", "other"]

# Stable presentation order; family declarations do not prove provenance.
METHOD_FAMILIES: tuple[MethodFamily, ...] = ("ml_model", "rule", "other")

_MAX_METHOD_NAME_CHARACTERS = 160

_NORMAL = NormalDist()
_MAX_CONTEXT_EXAMPLES = 3
_MAX_CONTEXT_EXAMPLE_CHARACTERS = 240

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

    @property
    def identity_key(self) -> str:
        """What these entries were measured *on*, excluding the tunable axes.

        Ranking one metric across different ``window``, ``horizon`` or
        ``n_quantiles`` is a deliberate sensitivity study, not a mistake, so
        the surface needs an identity that survives those changes. Instrument,
        dates, feature set and feature code stay, because no re-parameterisation
        makes those comparable.

        Dataset content is deliberately absent: a different ``horizon`` rebuilds
        the labels, so every point of a horizon sweep necessarily lives on its
        own dataset. Keeping one dataset version per horizon is checked
        separately by :func:`comparability_error`.
        """
        return "|".join(
            (
                self.symbol,
                self.timeframe,
                self.start_date,
                self.end_date,
                f"features={self.feature_set}",
                f"code={self.code_fingerprint}",
            )
        )

    @property
    def tunable_parameters(self) -> dict[str, int]:
        """The numeric settings a parameter surface sweeps."""
        return {
            "horizon": self.horizon,
            "window": self.window,
            "n_quantiles": self.n_quantiles,
        }

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


def _stored_metric(value: float | None) -> float | None:
    """Project a metric onto what JSON can hold without losing the row.

    JSON has no NaN or Infinity, and deleting the key would erase the evidence
    that the run happened at all, so both collapse to an explicit ``null``.
    """
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class LeaderboardEntry:
    """One evaluated candidate, with what it was evaluated on."""

    entry_id: str
    label: str
    context: EvaluationContext
    metrics: Mapping[str, float | None]
    """Measured metrics, where ``None`` means unavailable or non-finite.

    Three states are deliberately distinguishable, because a failed run must
    not disappear from the parameter grid it belongs to:

    - key absent: the metric was never recorded for this candidate;
    - key present, value ``None``: unavailable (including NaN/Inf), so the
      candidate exists at its coordinates but has no score;
    - key present, finite value: an ordinary measurement.

    Dropping the middle case would let a reader mistake 10 and 30 for adjacent
    settings after 20 failed, and any slope or flatness computed over that
    surviving subset would be wrong.
    """
    n_observations: int
    model_id: str = ""
    dataset_id: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    """Artifact settings for models; caller-declared settings for other methods."""
    signal_fingerprint: str = ""
    """Hash of the evaluated signal, kept apart from ``entry_id``.

    Recording it separately lets a re-run be checked against the stored
    candidate without changing how candidates are identified; ``""`` marks
    rows written before it was kept.
    """
    method_family: str = ""
    """Which kind of method produced the signal; ``""`` when not declared.

    Read it through :attr:`family`, which infers the family of older rows from
    the model ID they carry instead of guessing from today's catalog.
    """
    method_name: str = ""
    """Readable name of a non-model candidate, such as a rule's settings."""

    @property
    def family(self) -> MethodFamily:
        """The declared family, or the one this row's identity implies.

        Rows written before families existed read as ``ml_model`` when they
        carry a model ID. That ID is the caller's declaration of a fitted
        model, not proof that one produced the signal, so the family records
        what the row claims. Rows without an ID stay ``other``: nothing in them
        says whether a rule built them, and claiming otherwise would overstate
        their provenance.
        """
        if self.method_family:
            return cast("MethodFamily", self.method_family)
        return "ml_model" if self.model_id else "other"

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
            "parameters": dict(self.parameters),
            "signal_fingerprint": self.signal_fingerprint,
            "method_family": self.method_family,
            "method_name": self.method_name,
            "n_observations": self.n_observations,
            "metrics": {
                key: _stored_metric(value) for key, value in self.metrics.items()
            },
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
        raw_parameters = record.get("parameters")
        if raw_parameters is not None and not isinstance(raw_parameters, Mapping):
            raise ValueError("leaderboard record has malformed parameters")
        family = str(record.get("method_family", "") or "")
        if family and family not in METHOD_FAMILIES:
            raise ValueError(
                f"leaderboard record declares an unknown method family: {family!r}"
            )
        return cls(
            entry_id=str(record["entry_id"]),
            label=str(record["label"]),
            model_id=str(record.get("model_id", "")),
            dataset_id=str(record.get("dataset_id", "")),
            parameters=dict(raw_parameters or {}),
            signal_fingerprint=str(record.get("signal_fingerprint", "")),
            # An empty family stays empty rather than being rewritten to the
            # inferred value, so a read-write cycle does not add evidence the
            # row never carried. ``family`` performs the inference on access.
            method_family=family,
            method_name=str(record.get("method_name", "")),
            n_observations=int(record.get("n_observations", 0)),
            # A stored null is a measured-and-failed score, not a missing key,
            # so it must come back as None instead of being converted or dropped.
            metrics={
                str(key): None if value is None else float(value)
                for key, value in raw_metrics.items()
            },
            context=context,
        )


def build_entry_id(
    *,
    model_id: str,
    dataset_id: str,
    context: EvaluationContext,
    signal_fingerprint: str = "",
    method_name: str = "",
    method_family: MethodFamily = "other",
    method_parameters: Mapping[str, float] | None = None,
) -> str:
    """Identify one candidate within a dataset and measurement context.

    Repeated evaluation replaces that row without inflating candidate count.
    Use ``signal_fingerprint`` when model_id is absent so distinct ad-hoc
    signals remain distinct. A supplied model ID is a caller-declared label.

    Named methods use their family, name and full-precision declared parameters,
    independent of the signal they produced. Display-name rounding cannot merge
    different settings, and different families cannot overwrite one another.
    Existing model and unnamed-signal identities retain their encoding.
    """
    if method_name and not model_id:
        payload = json.dumps(
            [
                method_family,
                method_name,
                dict(method_parameters or {}),
                dataset_id,
                context.key,
            ],
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    else:
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
class MethodIdentity:
    """What produced a candidate's signal, as far as the record can say.

    ``parameters`` means the settings that produced the signal, whatever their
    provenance: fitted hyperparameters copied from a model artifact, or the
    caller's declared rule settings. A reader must not treat the two as the
    same kind of evidence, which is why ``family`` travels with them.
    """

    family: MethodFamily
    name: str
    parameters: Mapping[str, float]


def resolve_method(
    *,
    model_id: str | None,
    method_family: str | None,
    method_name: str | None,
    method_parameters: Mapping[str, float] | None,
) -> MethodIdentity:
    """Validate and complete a candidate's declared method identity.

    A model ID declares a model, without proving signal origin. A rule requires
    a name so its declared construction can be reported.
    A model's parameters are copied from its artifact, so caller-declared
    parameters are refused for model candidates rather than silently replacing
    them.
    """
    family: str = method_family or ("ml_model" if model_id else "other")
    if family not in METHOD_FAMILIES:
        raise ValueError(
            f"method_family must be one of {list(METHOD_FAMILIES)}, got {family!r}"
        )
    name = (method_name or "").strip()
    if len(name) > _MAX_METHOD_NAME_CHARACTERS:
        raise ValueError(
            "method_name must be at most "
            f"{_MAX_METHOD_NAME_CHARACTERS} characters, got {len(name)}"
        )
    if family == "ml_model" and not model_id:
        raise ValueError(
            "a ml_model candidate needs model_id; a signal built by hand is "
            "family 'rule' when it has a name, or 'other' when it has none"
        )
    if model_id and family != "ml_model":
        raise ValueError(
            f"model_id {model_id!r} declares a model, so its method family is "
            f"ml_model, not {family!r}"
        )
    if model_id and name:
        raise ValueError(
            "a model candidate is named by its artifact, not by method_name; "
            "pass method_name only for candidates without a model"
        )
    if family == "rule" and not name:
        raise ValueError(
            "a rule candidate needs method_name to describe its construction"
        )
    parameters = dict(method_parameters or {})
    if parameters and model_id:
        raise ValueError(
            "a model's parameters are copied from its artifact; pass "
            "method_parameters only for candidates without a model"
        )
    for key, value in parameters.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("method_parameters keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"method_parameters[{key!r}] must be a number, got {value!r}"
            )
        if not math.isfinite(float(value)):
            raise ValueError(f"method_parameters[{key!r}] must be finite")
    return MethodIdentity(
        family=family,
        name=name,
        parameters={key: float(value) for key, value in parameters.items()},
    )


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


def _examples(groups: Mapping[str, LeaderboardEntry]) -> str:
    """A bounded, abbreviated sample of the distinct groups behind a refusal."""
    examples = []
    for key in sorted(groups)[:_MAX_CONTEXT_EXAMPLES]:
        description = groups[key].context.describe()
        if len(description) > _MAX_CONTEXT_EXAMPLE_CHARACTERS:
            description = (
                description[:_MAX_CONTEXT_EXAMPLE_CHARACTERS] + "... [abbreviated]"
            )
        examples.append(description)
    omitted = len(groups) - len(examples)
    return (
        f"showing {len(examples)} examples, {omitted} omitted): {'; '.join(examples)}"
    )


def _surface_error(entries: Sequence[LeaderboardEntry]) -> str | None:
    """Why a parameter surface cannot be read, or ``None`` when it can.

    A surface may span datasets across horizons; common source-bar provenance
    is not verified by this context. Within one horizon the data must be a
    single dataset version, so a rebuilt or corrected dataset is never averaged
    into the same row of the grid.
    """
    identities = {entry.context.identity_key: entry for entry in entries}
    if len(identities) > 1:
        return (
            "entries were not measured on the same instrument, dates, feature "
            f"set or feature code ({len(identities)} distinct identities; "
            f"{_examples(identities)}. These identity fields must match even "
            "when window, horizon and n_quantiles vary. Filter by "
            "symbol/timeframe/feature_set to rank separately, or rebuild both "
            "sides on one feature code. allow_mixed=True permits ranking "
            "incompatible contexts but does not make them comparable."
        )
    versions: dict[int, set[tuple[str, str]]] = {}
    for entry in entries:
        versions.setdefault(entry.context.horizon, set()).add(
            (entry.dataset_id, entry.context.data_fingerprint)
        )
    mixed = {horizon: values for horizon, values in versions.items() if len(values) > 1}
    if mixed:
        horizon = sorted(mixed)[0]
        return (
            f"horizon {horizon} was measured on {len(mixed[horizon])} different "
            "dataset versions. A parameter surface may span datasets across "
            "horizons, but one horizon must have "
            "one dataset version: re-record against a single dataset_id, or "
            "rank each version separately."
        )
    return None


def comparability_error(
    entries: Sequence[LeaderboardEntry], *, allow_parameter_axes: bool = False
) -> str | None:
    """Why these entries cannot be ranked together, or ``None`` if they can.

    By default every measurement setting must match, which is what picking a
    champion needs. ``allow_parameter_axes`` relaxes exactly one thing: entries
    measured on the same instrument, dates, feature set and code may differ in
    ``window``, ``horizon`` or ``n_quantiles``, because sweeping those is the
    point of a sensitivity study. It still refuses to mix different
    instruments, dates, feature sets, feature code, or two dataset versions
    within one horizon.
    """
    if allow_parameter_axes:
        return _surface_error(entries)
    contexts = {entry.context.key: entry for entry in entries}
    if len(contexts) <= 1:
        return None
    return (
        "entries were not measured on the same conditions "
        f"({len(contexts)} distinct contexts; {_examples(contexts)}. "
        "Filter by dataset_id or symbol/timeframe/horizon/feature_set to rank "
        "separately. allow_mixed=True permits ranking incompatible contexts "
        "but does not make them comparable."
    )


def parameter_axes(entries: Sequence[LeaderboardEntry]) -> dict[str, list[int]]:
    """The observed values of every tunable setting, per axis.

    A ranking across a parameter surface is only readable if each point says
    where it sits, so the caller gets the grid coordinates alongside the ranks.
    """
    axes: dict[str, list[int]] = {}
    for name in ("horizon", "window", "n_quantiles"):
        values = sorted({entry.context.tunable_parameters[name] for entry in entries})
        axes[name] = values
    return axes


def surface_caveats(axes: dict[str, list[int]]) -> tuple[str, ...]:
    """The warnings a ranking needs once it spans more than one setting.

    Lives here rather than in a caller so every reader of the same axes gets
    the same words: a ranking over a grid is not one comparable measurement,
    and a sweep across horizons does not verify that the source bars match.
    """
    caveats: list[str] = []
    spanned = {name: values for name, values in axes.items() if len(values) > 1}
    if spanned:
        described = ", ".join(f"{name}={values}" for name, values in spanned.items())
        caveats.append(
            f"candidates span more than one evaluation setting ({described}); "
            "the ranking describes a parameter surface, not one comparable measurement"
        )
    if len(axes.get("horizon", ())) > 1:
        caveats.append(
            "cross-horizon source and raw-data equivalence is not verified; "
            "check dataset manifests and source bars before treating the "
            "surface as a controlled comparison"
        )
    return tuple(caveats)


def family_counts(entries: Sequence[LeaderboardEntry]) -> dict[str, int]:
    """How many candidates each method family contributed, in declared order.

    A ranking that mixes families is still a ranking of what was measured, but
    the reader has to know the mix before reading an order into it.
    """
    counts = Counter(entry.family for entry in entries)
    ordered: dict[str, int] = {
        family: counts[family] for family in METHOD_FAMILIES if counts[family] > 0
    }
    ordered.update(
        {family: count for family, count in counts.items() if family not in ordered}
    )
    return ordered


def family_caveats(counts: Mapping[str, int]) -> tuple[str, ...]:
    """What a reader must know about the families behind one ranking.

    Lives here rather than in a caller so the tool reply and a saved report
    say the same thing about the same table.
    """
    caveats: list[str] = []
    if len(counts) > 1:
        described = ", ".join(f"{family}={count}" for family, count in counts.items())
        caveats.append(
            f"candidates span {len(counts)} method families ({described}); "
            "family labels do not establish equal evidence about selection "
            "history or out-of-sample performance. Inspect provenance and "
            "measurement conditions; filter by family to rank one kind"
        )
    unnamed = counts.get("other", 0)
    if unnamed:
        caveats.append(
            f"{unnamed} candidate(s) use the other method family; names and "
            "declared settings alone do not establish reproducible construction"
        )
    return tuple(caveats)


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

    Caveats also describe the method families behind the candidates, so a table
    mixing fitted models with hand-set rules cannot be read as one kind of
    evidence.
    """
    direction = metric_direction(metric)
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1")

    candidates: list[tuple[float, LeaderboardEntry]] = []
    skipped: list[SkippedEntry] = []
    measured: list[LeaderboardEntry] = []
    for entry in entries:
        measured.append(entry)
        value = entry.metric(metric)
        if value is None:
            # A stored None is a measurement that failed, not a metric nobody
            # recorded; saying which one it is keeps the two cases apart.
            reason = (
                f"no '{metric}' recorded"
                if metric not in entry.metrics
                else f"'{metric}' is unavailable or not finite"
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
            f"ranked for '{metric}'. They stay recorded, so a parameter grid "
            "keeps their coordinates even when the score is unavailable; the "
            "ranked candidates are not necessarily adjacent settings."
        )
    if n_candidates > 1 and ranked:
        caveats.append(_selection_caveat(metric, n_candidates, ranked[0], alpha=alpha))
    if n_candidates == 1 and ranked:
        caveats.append(
            "only one candidate was compared; a single entry proves nothing "
            "about the alternatives"
        )
    caveats.extend(family_caveats(family_counts(measured)))

    return LeaderboardResult(
        metric=metric,
        direction=direction,
        ranked=tuple(ranked[:top_k] if top_k is not None else ranked),
        skipped=tuple(skipped),
        caveats=tuple(caveats),
    )
