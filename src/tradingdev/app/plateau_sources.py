"""Read complete saved collections, independently of ranking and display pages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.storage.ml_datasets import MLDatasetStore
from tradingdev.app.ml_surface import compare_market_evidence, validate_dataset_context
from tradingdev.domain.ml.evaluation.catalog import SIGNAL_METRIC_CATALOG
from tradingdev.domain.ml.leaderboard import ProducerSource
from tradingdev.domain.ml.registry.leaderboard_store import LeaderboardStore
from tradingdev.domain.ml.rules import ThresholdRule
from tradingdev.domain.parameter_values import parameter_leaves
from tradingdev.domain.research_scope import (
    effective_scope_parameters,
    verify_research_scopes,
)

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.app.trade_history_service import TradeHistoryService
    from tradingdev.domain.ml.leaderboard import LeaderboardEntry

MAX_CANDIDATES = 1024


def canonical(value: Any) -> bytes:
    """Freeze finite JSON inputs for identity and equality checks."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode()


@dataclass(frozen=True)
class PlateauCollection:
    """All rows and the evidence used to interpret this one collection."""

    rows: list[dict[str, Any]]
    definition: dict[str, Any]
    evidence: dict[str, Any]

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(canonical(self.evidence)).hexdigest()


def optimization_collection(
    history: TradeHistoryService, run_id: str, metric: str
) -> PlateauCollection:
    loaded = history.load_run(run_id)
    manifest = loaded.manifest
    if manifest is None or manifest.optimization is None:
        raise ValueError(
            "Complete saved optimization manifest required; no historical axis"
            " inference"
        )
    search = manifest.optimization
    if search.total_combinations > MAX_CANDIDATES:
        raise ValueError(
            f"Collection exceeds {MAX_CANDIDATES} candidates; "
            "select another saved collection"
        )
    verify_research_scopes(manifest, loaded.performance, loaded.observations)
    definition = loaded.performance.definitions.get(metric)
    if definition is None or definition.optimization_direction is None:
        raise ValueError(
            "Metric needs a saved scalar definition and optimization direction"
        )
    rows: list[dict[str, Any]] = []
    for index in range(search.total_combinations):
        scope_id = f"trial/{index}/train"
        scope = loaded.performance.scopes[scope_id]
        unavailable = scope.metadata.get("unavailable", {})
        reason = (
            unavailable.get(metric, "undefined_metric")
            if isinstance(unavailable, dict)
            else "undefined_metric"
        )
        rows.append(
            {
                "identity": {"run_id": run_id, "scope": scope_id},
                "parameters": parameter_leaves(
                    effective_scope_parameters(manifest, scope)
                ),
                "value": scope.values.get(metric),
                "reason": (reason if metric in scope.values else "metric_not_recorded"),
            }
        )
    return PlateauCollection(
        rows,
        definition.model_dump(mode="json"),
        {
            "source": "optimization_train",
            "run_id": run_id,
            "completeness": "verified_saved_grid",
            "manifest": manifest.model_dump(mode="json"),
            "performance": loaded.performance.model_dump(mode="json"),
            "observations_sha256": hashlib.sha256(
                canonical(loaded.observations.model_dump(mode="json"))
            ).hexdigest(),
            "dataset_id": loaded.run.get("dataset_id"),
        },
    )


def signal_collection(
    workspace: WorkspacePaths, entry_ids: list[str], metric: str, axes: set[str]
) -> PlateauCollection:
    if len(entry_ids) > MAX_CANDIDATES or len(set(entry_ids)) != len(entry_ids):
        raise ValueError(f"Choose at most {MAX_CANDIDATES} distinct entry IDs")
    definition = SIGNAL_METRIC_CATALOG.get(metric)
    if definition is None or definition.optimization_direction is None:
        raise ValueError(
            "Unknown signal metric or metric without optimization direction"
        )
    # One atomic JSONL generation, never public ranking pages.
    saved = LeaderboardStore(workspace.ml_leaderboard).entries()
    by_id = {entry.entry_id: entry for entry in saved}
    if len(by_id) != len(saved):
        raise ValueError("Stored collection repeats an entry ID")
    if any(identity not in by_id for identity in entry_ids):
        raise ValueError(
            "Selected entries are unavailable; read saved candidate identities again"
        )
    entries = [by_id[identity] for identity in sorted(entry_ids)]
    datasets = MLDatasetStore(workspace.ml_datasets)
    manifests: dict[str, dict[str, Any]] = {}
    conditions: list[bytes] = []
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if not entry.context.data_fingerprint or not entry.dataset_id:
            raise ValueError("Historical signal entries lack saved dataset identity")
        if entry.dataset_id not in manifests:
            manifests[entry.dataset_id] = datasets.load(entry.dataset_id)[1]
        validate_dataset_context(entry, manifests[entry.dataset_id])
        parameters = {
            **parameter_leaves(dict(entry.parameters)),
            **entry.context.tunable_parameters,
        }
        fixed = {name: value for name, value in parameters.items() if name not in axes}
        conditions.append(
            canonical(
                {
                    "context": entry.context.identity_key,
                    "fixed": fixed,
                    "method": [
                        entry.family,
                        _method_condition(entry),
                        entry.producer_source.value,
                        entry.training_dataset_id,
                    ],
                }
            )
        )
        rows.append(
            {
                "identity": {
                    "entry_id": entry.entry_id,
                    "dataset_id": entry.dataset_id,
                    "series_artifact_id": entry.series_artifact_id,
                },
                "parameters": parameters,
                "value": entry.metric(metric),
                "reason": "undefined_metric"
                if metric in entry.metrics
                else "metric_not_recorded",
            }
        )
    if conditions and any(item != conditions[0] for item in conditions[1:]):
        raise ValueError(
            "Incompatible data/method/evaluation conditions or undeclared "
            "varying parameters"
        )
    cross_horizon = len({entry.context.horizon for entry in entries}) > 1
    comparison = compare_market_evidence(
        entries, datasets, require_verified_baseline=cross_horizon
    )
    if cross_horizon and (
        comparison is None or comparison["evaluation_rows"]["status"] != "verified"
    ):
        raise ValueError(
            "Cross-horizon plateau requires verified evaluation rows as well "
            "as raw market evidence"
        )
    per_horizon: dict[int, set[str]] = {}
    for entry in entries:
        per_horizon.setdefault(entry.context.horizon, set()).add(
            entry.context.data_fingerprint
        )
    if any(len(values) != 1 for values in per_horizon.values()):
        raise ValueError("Dataset content differs within the same horizon")
    return PlateauCollection(
        rows,
        asdict(definition),
        {
            "source": "signal_entries",
            "completeness": "explicit_selected_entries_only",
            "entries": [entry.to_record() for entry in entries],
            "datasets": manifests,
            "market_comparison": comparison,
        },
    )


def _method_condition(entry: LeaderboardEntry) -> object:
    """Separate a verified backend rule's column/side from its tunable level.

    Check the exact ThresholdRule.name format against the saved parameter;
    arbitrary caller-declared method names retain their original semantics.
    """
    if entry.producer_source != ProducerSource.BACKEND_RULE:
        return entry.method_name or entry.label
    threshold = entry.parameters.get("threshold")
    if (
        entry.family != "rule"
        or isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
    ):
        raise ValueError("Saved backend rule lacks its numeric threshold identity")
    for direction, operator in (("above", ">"), ("below", "<")):
        suffix = f" {operator} {threshold:g}"
        if entry.method_name.endswith(suffix):
            rule = ThresholdRule.create(
                column=entry.method_name[: -len(suffix)],
                threshold=threshold,
                direction=direction,
            )
            if rule.name == entry.method_name:
                return {
                    "algorithm": "threshold",
                    "column": rule.column,
                    "direction": rule.direction,
                }
    raise ValueError("Saved backend rule name differs from its threshold identity")
