"""Shared parameter plateau analysis over complete, fixed saved collections."""

from __future__ import annotations

import itertools
import math
import sqlite3
import warnings
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.plateau import PlateauStore
from tradingdev.adapters.storage.sqlite import SQLiteStore, get_sqlite_store
from tradingdev.app.contracts.plateau import (
    MAX_CANDIDATES,
    MAX_DIMENSIONS,
    MAX_GRID_CELLS,
    plateau_definitions,
)
from tradingdev.app.plateau_sources import (
    canonical,
    optimization_collection,
    signal_collection,
)
from tradingdev.app.trade_history_service import HistoryReadError, TradeHistoryService
from tradingdev.domain.statistics.plateau import (
    DroppedTrialsWarning,
    plateau_scores,
    plateau_summary,
    trials_from_metrics,
)

if TYPE_CHECKING:
    from tradingdev.app.plateau_sources import PlateauCollection

LIMITATIONS = [
    (
        "Neighborhoods use Chebyshev distance on sorted grid indices, not "
        "raw numeric distances; radius and grid density change scores."
    ),
    (
        "Plateau scores average available neighbors only, excluding the "
        "point itself; undefined cells and absent records are never zero."
    ),
    (
        "Tied metrics choose the earliest saved trial index, or entry ID "
        "in ascending order; plateau-score ties use relative tolerance "
        "1e-12."
    ),
    (
        "A broad plateau does not prove out-of-sample validity, "
        "profitability or removal of data-mining bias; optimization "
        "training is in-sample."
    ),
    (
        "Signal axes are explicitly declared for the selected saved "
        "entries; absent cells do not prove that no experiment was ever "
        "attempted elsewhere."
    ),
    (
        "No search, training, backtest, metric rewrite or arbitrary "
        "minimum sample threshold is performed."
    ),
]


class PlateauService:
    """Analyze once; all subsequent pages read the published fixed snapshot."""

    def __init__(
        self,
        *,
        workspace: WorkspacePaths | None = None,
        store: SQLiteStore | None = None,
    ) -> None:
        self.workspace = workspace or WorkspacePaths()
        self.store = store or get_sqlite_store(self.workspace)
        self.history = TradeHistoryService(workspace=self.workspace, store=self.store)
        self.artifacts = PlateauStore(self.workspace, self.store)

    def analyze(
        self,
        *,
        metric: str,
        axes: dict[str, list[float]],
        run_id: str | None = None,
        entry_ids: list[str] | None = None,
        radius: int = 1,
        min_neighborhood: int | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Persist a complete derived collection without changing its source."""
        try:
            normalized = _axes(axes)
            _page(0, limit)
            if type(radius) is not int or radius < 1:
                raise ValueError("radius must be a positive integer")
            if min_neighborhood is not None and (
                type(min_neighborhood) is not int or min_neighborhood < 1
            ):
                raise ValueError("min_neighborhood must be a positive integer or null")
            if (run_id is None) == (entry_ids is None):
                raise ValueError(
                    "Select exactly one optimization run_id or explicit signal "
                    "entry_ids"
                )
            source, separator, name = metric.partition(".")
            if not separator or source != ("run" if run_id is not None else "signal"):
                raise ValueError(
                    "Metric prefix must match the source: run.<metric> or "
                    "signal.<metric>"
                )
            collection = (
                optimization_collection(self.history, run_id, name)
                if run_id is not None
                else signal_collection(
                    self.workspace, entry_ids or [], name, set(normalized)
                )
            )
            payload = _analyze(collection, normalized, metric, radius, min_neighborhood)
            identity = self.artifacts.publish(payload)
            return self.read(identity, limit=limit)
        except (ValueError, OSError, sqlite3.Error) as exc:
            return _error(exc)

    def read(
        self, analysis_id: str, *, offset: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        """Read one bounded page, verifying the complete saved snapshot first."""
        try:
            _page(offset, limit)
            payload, artifact = self.artifacts.load(analysis_id)
            cells = payload.pop("cells")
            payload.pop("evidence")
            if offset > len(cells):
                raise ValueError("offset exceeds grid cell count")
            end = min(offset + limit, len(cells))
            return {
                **payload,
                "success": True,
                "analysis_id": analysis_id,
                "artifact": artifact,
                "cells": {
                    "items": cells[offset:end],
                    "offset": offset,
                    "limit": limit,
                    "total": len(cells),
                    "next_offset": end if end < len(cells) else None,
                },
            }
        except (ValueError, OSError, sqlite3.Error) as exc:
            return _error(exc)


def _page(offset: int, limit: int) -> None:
    if (
        type(offset) is not int
        or offset < 0
        or type(limit) is not int
        or not 1 <= limit <= 100
    ):
        raise ValueError("offset must be nonnegative; limit must be between 1 and 100")


def _axes(axes: dict[str, list[float]]) -> dict[str, list[float]]:
    if not isinstance(axes, dict) or not 1 <= len(axes) <= MAX_DIMENSIONS:
        raise ValueError(
            f"Declare 1 to {MAX_DIMENSIONS} numeric axes; use parameter JSON "
            "pointers or horizon/window/n_quantiles"
        )
    result: dict[str, list[float]] = {}
    for name, levels in sorted(axes.items()):
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(levels, list)
            or not levels
        ):
            raise ValueError(
                "Each named axis needs an explicit nonempty list of levels"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            for value in levels
        ):
            raise ValueError(
                "Axes require finite numeric levels; categorical values cannot be "
                "encoded as distances"
            )
        values = [float(value) for value in levels]
        if len(set(values)) != len(values):
            raise ValueError("Axis levels must be unique")
        result[name] = sorted(values)
    if math.prod(map(len, result.values())) > MAX_GRID_CELLS:
        raise ValueError(f"Declared grid exceeds {MAX_GRID_CELLS} cells")
    return result


def _analyze(
    collection: PlateauCollection,
    axes: dict[str, list[float]],
    metric: str,
    radius: int,
    minimum: int | None,
) -> dict[str, Any]:
    if len(collection.rows) > MAX_CANDIDATES:
        raise ValueError(f"Collection exceeds {MAX_CANDIDATES} candidates")
    names = list(axes)
    coordinates: dict[tuple[float, ...], dict[str, Any]] = {}
    fixed: bytes | None = None
    for row in collection.rows:
        parameters = row["parameters"]
        if any(name not in parameters for name in names):
            raise ValueError(
                "Requested axes are missing from saved parameters. Exact saved "
                f"coordinate names (first 20 of {len(parameters)}): "
                f"{sorted(parameters)[:20]}. Parameter JSON pointers start with /; "
                "bare horizon/window/n_quantiles apply only to signal evaluation. "
                "Historical coordinates cannot be invented."
            )
        selected = _axes({name: [parameters[name]] for name in names})
        point = tuple(selected[name][0] for name in names)
        if any(
            value not in axes[name] for name, value in zip(names, point, strict=True)
        ):
            raise ValueError(
                "Saved coordinate lies outside declared axes; the complete "
                "collection cannot be silently filtered"
            )
        if point in coordinates:
            raise ValueError(
                f"Duplicate coordinate {dict(zip(names, point, strict=True))}; "
                "select one candidate explicitly: "
                f"{coordinates[point]['identity']}, {row['identity']}"
            )
        other = canonical(
            {key: value for key, value in parameters.items() if key not in axes}
        )
        if fixed is not None and fixed != other:
            raise ValueError(
                "Non-axis parameters differ; declare every varying numeric "
                "parameter or select a compatible group"
            )
        fixed = other
        coordinates[point] = row
    rows = [
        {"params": dict(zip(names, point, strict=True)), "metric": row["value"]}
        for point, row in coordinates.items()
    ]
    if collection.evidence["source"] == "optimization_train" and any(
        sorted({point[index] for point in coordinates}) != axes[name]
        for index, name in enumerate(names)
    ):
        raise ValueError(
            "Optimization axes must exactly match the complete saved search, "
            "including undefined levels"
        )
    direction = collection.definition["optimization_direction"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DroppedTrialsWarning)
        trials = trials_from_metrics(
            rows, params_key="params", metric_key="metric", direction=direction
        )
    summary = plateau_summary(
        trials.trials, axes=axes, radius=radius, min_neighborhood=minimum
    )
    scores = {
        tuple(item.params[name] for name in names): item
        for item in plateau_scores(trials.trials, axes=axes, radius=radius)
    }
    sign = -1 if direction == "minimize" else 1
    cells = []
    for point in itertools.product(*(axes[name] for name in names)):
        cell_source = coordinates.get(point)
        score = scores.get(point)
        cells.append(
            {
                "parameters": dict(zip(names, point, strict=True)),
                "identity": cell_source["identity"]
                if cell_source is not None
                else None,
                "status": "absent"
                if cell_source is None
                else "undefined"
                if score is None
                else "available",
                "reason": "no_record_in_selected_collection"
                if cell_source is None
                else cell_source["reason"]
                if score is None
                else None,
                "value": cell_source["value"] if cell_source is not None else None,
                "score": score.metric if score is not None else None,
                "plateau_score": score.score if score is not None else None,
                "neighborhood_size": score.neighborhood_size
                if score is not None
                else None,
            }
        )

    def best(params: Any) -> dict[str, Any] | None:
        if params is None:
            return None
        point = tuple(params[name] for name in names)
        row = coordinates[point]
        return {
            "parameters": dict(params),
            "identity": row["identity"],
            "value": row["value"],
        }

    return {
        "schema_version": 1,
        "algorithm_version": "parameter-plateau-v1",
        "source_hash": collection.source_hash,
        "source": {
            key: value
            for key, value in collection.evidence.items()
            if key in {"source", "run_id", "completeness", "market_comparison"}
        },
        "metric": metric,
        "definition": collection.definition,
        "direction": direction,
        "axes": axes,
        "settings": {"radius": radius, "min_neighborhood": minimum},
        "n_candidates": len(rows),
        "n_available": len(trials.trials),
        "n_undefined": trials.dropped_count,
        "n_absent": len(cells) - len(rows),
        "summary": {
            "best": best(summary.best_params),
            "plateau_best": best(summary.plateau_best_params),
            "best_score": summary.best_metric,
            "plateau_best_score": summary.plateau_best_score,
            "plateau_best_value": sign * summary.plateau_best_metric
            if summary.plateau_best_metric is not None
            else None,
            "score_gap": summary.metric_gap,
            "grid_distance": summary.grid_distance,
            "best_neighborhood_mean_score": summary.best_neighborhood_mean,
            "best_neighborhood_std": summary.best_neighborhood_std,
            "best_neighborhood_size": summary.best_neighborhood_size,
            "spikiness": summary.spikiness,
            "dimension_warning": summary.dimension_warning,
            "n_below_min_neighborhood": summary.n_below_min_neighborhood,
        },
        "limitations": LIMITATIONS,
        "derived_definitions": plateau_definitions(),
        "cells": cells,
        "evidence": collection.evidence,
    }


def _error(exc: Exception) -> dict[str, Any]:
    return {
        "success": False,
        "code": exc.code
        if isinstance(exc, HistoryReadError)
        else "plateau_unavailable",
        "error": str(exc),
        "next_step": (
            "Inspect saved source identities and evidence; select one "
            "compatible complete collection. Do not rerun experiments "
            "automatically."
        ),
    }
