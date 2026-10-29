"""Project stored leaderboard rows into the shape a report is built from.

A report is assembled from a JSON payload and rendered from it; HTML is only a
projection of that payload. So a signal evaluation becomes visible in a report
by being projectable into the payload's existing scope shape, not by anyone
writing markup for it. This module is that projection and nothing else: it
reads what the leaderboard already stores and writes no new state.

Two conventions are copied deliberately from the portfolio side:

* ``values`` may hold ``None``. A candidate that was measured and produced no
  finite score stays in the payload at its own coordinate instead of vanishing
  from the grid.
* Definitions live once at block level, exactly as ``PerformanceBundle`` keeps
  them, rather than being repeated per scope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tradingdev.domain.ml.evaluation.catalog import (
    definition_snapshot,
    report_visibility,
)
from tradingdev.domain.ml.leaderboard import (
    comparability_error,
    parameter_axes,
    rank_entries,
    surface_caveats,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tradingdev.domain.ml.leaderboard import LeaderboardEntry

ML_REPORT_SCHEMA_VERSION = 1
ML_SIGNAL_KIND = "ml_signal"
ML_SIGNAL_MODE = "signal"
ML_SAMPLE_PROVENANCE_NOTE = (
    "排行榜紀錄未保存訓練／評估資料的切分證據，"
    "因此無法判定這些評估結果屬於樣本內或樣本外。"
)


def ml_signal_scope(entry: LeaderboardEntry) -> dict[str, Any]:
    """One stored candidate as a report scope, coordinates included.

    The three tunable settings are reported in ``metadata`` rather than merged
    into ``parameters``: a fitted parameter dictionary may legitimately contain
    a key of the same name, and merging would silently overwrite one of them.
    ``parameters`` therefore means exactly "what the model was fitted with".
    """
    context = entry.context
    return {
        "scope_id": entry.entry_id,
        "kind": ML_SIGNAL_KIND,
        "mode": ML_SIGNAL_MODE,
        "split": "full",
        "fold_index": None,
        "trial_index": None,
        "values": {key: entry.metric(key) for key in entry.metrics},
        "parameters": dict(entry.parameters),
        "metadata": {
            "label": entry.label,
            "model_id": entry.model_id,
            "dataset_id": entry.dataset_id,
            "n_observations": entry.n_observations,
            "signal_fingerprint": entry.signal_fingerprint,
            "horizon": context.horizon,
            "window": context.window,
            "n_quantiles": context.n_quantiles,
            "symbol": context.symbol,
            "timeframe": context.timeframe,
            "start_date": context.start_date,
            "end_date": context.end_date,
            "feature_set": context.feature_set,
            "code_fingerprint": context.code_fingerprint,
            "data_fingerprint": context.data_fingerprint,
        },
        # A signal evaluation records no orders and no equity curve, so there
        # is nothing to observe. Saying so is different from omitting the key.
        "observations": None,
    }


def ml_report_block(
    entries: Sequence[LeaderboardEntry],
    metric: str,
    *,
    alpha: float = 0.05,
    allow_parameter_axes: bool = False,
) -> dict[str, Any]:
    """The whole signal-evidence block: every candidate, ranked or failed.

    Compute a complete ranking from these candidates, so an independent stale
    or truncated result cannot be paired with different evidence. Parameter
    surfaces require an explicit opt-in, with the same comparability rules as
    the leaderboard.
    """
    ordered = tuple(entries)
    if len({entry.entry_id for entry in ordered}) != len(ordered):
        raise ValueError("Report candidates must have unique entry IDs")
    if mismatch := comparability_error(
        ordered, allow_parameter_axes=allow_parameter_axes
    ):
        raise ValueError(mismatch)
    result = rank_entries(ordered, metric, alpha=alpha)
    metric_ids = sorted({key for entry in ordered for key in entry.metrics})
    axes = parameter_axes(ordered)
    caveats = list(result.caveats)
    caveats.extend(surface_caveats(axes))
    return {
        "schema_version": ML_REPORT_SCHEMA_VERSION,
        "kind": ML_SIGNAL_KIND,
        "scopes": [ml_signal_scope(entry) for entry in ordered],
        "definitions": {
            metric_id: definition_snapshot(metric_id) for metric_id in metric_ids
        },
        # The visibility policy travels with the data so a consumer other than
        # the HTML renderer applies the same rule instead of inventing one.
        "visible_metrics": [
            metric_id for metric_id in metric_ids if report_visibility(metric_id)
        ],
        "hidden_metrics": [
            metric_id for metric_id in metric_ids if not report_visibility(metric_id)
        ],
        "parameter_axes": axes,
        "ranking": {
            "metric": result.metric,
            "direction": result.direction,
            "ranked_scope_ids": [item.entry.entry_id for item in result.ranked],
            "skipped": [
                {
                    "scope_id": item.entry.entry_id,
                    "label": item.entry.label,
                    "reason": item.reason,
                    "horizon": item.entry.context.horizon,
                    "window": item.entry.context.window,
                    "n_quantiles": item.entry.context.n_quantiles,
                }
                for item in result.skipped
            ],
        },
        "caveats": caveats,
    }
