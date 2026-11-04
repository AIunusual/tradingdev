"""Rank saved backtest scopes by the metrics their run already recorded.

Ranking reads; it never recomputes. The values come from the performance bundle the
user's own development run saved, the direction and unit come from the metric
catalogue, and the trading conditions come from the execution manifest that was
published with that run. A run whose numbers or conditions cannot be read is
reported as skipped with the reason, never re-run and never zero-filled.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.run_service import RunService
from tradingdev.domain.backtest.schemas import BacktestConfig
from tradingdev.domain.execution import ManifestError
from tradingdev.domain.performance.artifacts import MetricDefinitionSnapshot
from tradingdev.domain.performance.catalog import METRIC_CATALOG
from tradingdev.domain.ranking import Candidate, MetricDefinition, rank_candidates
from tradingdev.domain.run_candidates import (
    RunCandidate,
    build_run_candidate,
    to_ranking_candidate,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tradingdev.adapters.storage.sqlite import SQLiteStore
    from tradingdev.app.ml_service import MLService

_RUN_PREFIX = "run."
_SIGNAL_PREFIX = "signal."


def _duplicate_error(message: str) -> dict[str, Any]:
    """Refuse a selection that names the same measurement twice.

    Ranking the same scope twice would return two identical rows, which reads as
    a comparison of two candidates. ``compare_runs`` already refuses this, and the
    two surfaces must agree on what a duplicate is.
    """
    return {"success": False, "code": "duplicate_selections", "error": message}


def resolve_metric(metric: str) -> MetricDefinition:
    """Resolve a source-qualified backtest metric id to its direction and unit.

    Signal metrics are rejected here: they are ranked through the leaderboard, and
    a signal id can never equal a backtest id because both are source-qualified.
    """
    if not metric.startswith(_RUN_PREFIX):
        raise ValueError(
            f"backtest ranking needs a 'run.' metric id, not {metric!r}; "
            "signal metrics are ranked through the leaderboard"
        )
    name = metric[len(_RUN_PREFIX) :]
    definition = METRIC_CATALOG.get(name)
    if definition is None:
        raise ValueError(f"unknown backtest metric: {name}")
    if definition.optimization_direction is None:
        raise ValueError(
            f"metric '{name}' has no optimisation direction and cannot rank candidates"
        )
    return MetricDefinition(
        name=metric,
        direction=definition.optimization_direction,
        unit=definition.unit,
    )


class RankingService:
    """Order saved backtest scopes without recomputing anything."""

    def __init__(
        self,
        *,
        workspace: WorkspacePaths | None = None,
        store: SQLiteStore | None = None,
        ml_service: MLService | None = None,
    ) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._workspace.ensure()
        self._runs = RunService(workspace=self._workspace, store=store)
        self._ml = ml_service

    def rank_candidates(
        self,
        metric: str,
        *,
        runs: Sequence[tuple[str, str | None]] | None = None,
        dataset_id: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        horizon: int | None = None,
        feature_set: str | None = None,
        family: str | None = None,
        offset: int = 0,
        limit: int = 20,
        allow_mixed: bool = False,
        allow_parameter_axes: bool = False,
    ) -> dict[str, Any]:
        """Rank one source, chosen by the metric id's prefix.

        ``signal.`` reads recorded evaluations and selects them with filters;
        ``run.`` reads saved scopes named as ``(run_id, scope)`` pairs. The two
        cannot share a table because they do not share a metric definition, so
        arguments belonging to the other source are refused rather than ignored:
        a half-filled table would look like a ranking of two things.

        Both sources page with ``offset``/``limit``. A backtest selection is named
        explicitly and is usually small, so its page is normally the whole list.
        """
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        filters = (dataset_id, symbol, timeframe, horizon, feature_set, family)
        if metric.startswith(_SIGNAL_PREFIX):
            if runs is not None:
                raise ValueError(
                    "signal ranking selects recorded evaluations with filters, "
                    "not run selections"
                )
            if self._ml is None:
                raise RuntimeError(
                    "signal ranking requires an ML service; construct RankingService "
                    "with ml_service="
                )
            result = self._ml.rank_signal_candidates(
                metric[len(_SIGNAL_PREFIX) :],
                dataset_id=dataset_id,
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                feature_set=feature_set,
                family=family,
                offset=offset,
                limit=limit,
                allow_mixed=allow_mixed,
                allow_parameter_axes=allow_parameter_axes,
            )
            result["metric"] = metric
            return result
        if metric.startswith(_RUN_PREFIX):
            if (
                any(value is not None for value in filters)
                or allow_parameter_axes
                or allow_mixed
            ):
                raise ValueError(
                    "backtest ranking selects saved scopes with runs, "
                    "not signal filters"
                )
            if not runs:
                raise ValueError("backtest ranking requires at least one run selection")
            result = self.rank_run_scopes(metric, runs)
            if result.get("success") is False:
                return result
            ranked = result["ranked"]
            skipped = result["skipped"]
            total = max(len(ranked), len(skipped))
            if offset > total:
                raise ValueError(f"offset exceeds ranked row count ({total})")
            end = min(offset + limit, total)
            return {
                "success": True,
                "metric": result["metric"],
                "direction": result["direction"],
                "source": "run",
                "n_candidates": len(ranked) + len(skipped),
                "n_ranked": len(ranked),
                "n_skipped": len(skipped),
                "offset": offset,
                "limit": limit,
                "next_offset": end if end < total else None,
                "ranked": [{**row, "source": "run"} for row in ranked[offset:end]],
                "skipped": [{**row, "source": "run"} for row in skipped[offset:end]],
                "caveats": result["caveats"],
            }
        raise ValueError(
            f"metric id needs a 'signal.' or 'run.' prefix, not {metric!r}"
        )

    def rank_run_scopes(
        self,
        metric: str,
        selections: Sequence[tuple[str, str | None]],
        *,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Rank ``(run_id, scope)`` pairs by one stored backtest metric.

        Each pair is one candidate, so several parameter sets of a single run can be
        compared side by side. A null scope selects that run's default. A pair whose
        run, bundle, manifest or trading conditions cannot be read is skipped with
        that reason; the rest are ranked.
        """
        definition = resolve_metric(metric)
        requested: set[tuple[str, str | None]] = set()
        for run_id, scope in selections:
            if (run_id, scope) in requested:
                return _duplicate_error(
                    f"duplicate selection: {run_id} scope {scope or 'default'}"
                )
            requested.add((run_id, scope))
        candidates = []
        skipped: list[dict[str, str]] = []
        for run_id, scope in selections:
            try:
                candidate = self._load_candidate(run_id, scope, metric)
            except ValueError as exc:
                skipped.append(
                    {
                        "candidate_id": f"run:{run_id}#{scope or 'default'}",
                        "reason": str(exc),
                    }
                )
                continue
            candidates.append(candidate)

        # An omitted scope is resolved by the run itself, so (alpha, None) and
        # (alpha, "full") can be the same measurement while looking different.
        # The check runs on resolved pairs, not on what the caller typed.
        seen_resolved: set[tuple[str, str]] = set()
        for candidate in candidates:
            pair = (candidate.payload.run_id, candidate.payload.scope)
            if pair in seen_resolved:
                return _duplicate_error(
                    f"duplicate selection: {pair[0]} scope {pair[1]} was selected twice"
                )
            seen_resolved.add(pair)

        result = rank_candidates(candidates, definition, top_k=top_k)
        return {
            "success": True,
            "metric": result.metric,
            "direction": result.direction,
            "unit": definition.unit,
            "ranked_total": result.ranked_total,
            "ranked": [
                {
                    "rank": item.rank,
                    "tied": item.tied,
                    "value": item.value,
                    "run_id": item.candidate.payload.run_id,
                    "scope": item.candidate.payload.scope,
                    "strategy_id": item.candidate.payload.strategy_id,
                    "revision_id": item.candidate.payload.revision_id,
                    "parameters": dict(item.candidate.payload.parameters),
                    "metadata": dict(item.candidate.payload.metadata),
                    "conditions": {
                        "symbol": item.candidate.payload.conditions.symbol,
                        "timeframe": item.candidate.payload.conditions.timeframe,
                        "start_date": item.candidate.payload.conditions.start_date,
                        "end_date": item.candidate.payload.conditions.end_date,
                        "fees": item.candidate.payload.conditions.fees,
                        "slippage": item.candidate.payload.conditions.slippage,
                    },
                    "metric_source": item.candidate.payload.metric_source,
                }
                for item in result.ranked
            ],
            "skipped": [
                {
                    "candidate_id": item.candidate.candidate_id,
                    "run_id": item.candidate.payload.run_id,
                    "scope": item.candidate.payload.scope,
                    "reason": item.reason,
                }
                for item in result.skipped
            ]
            + skipped,
            "caveats": [
                *result.caveats,
                "Ranking is conditional on each scope's saved metadata, including "
                "provider versions, settings and observation periods; it does not "
                "establish a controlled comparison. Conditions describe the run's "
                "submitted period, not each scope's observed period.",
            ],
        }

    def _load_candidate(
        self, run_id: str, scope: str | None, metric: str
    ) -> Candidate[RunCandidate]:
        """Build one candidate from stored values; raise ValueError with the reason."""
        run_payload = self._runs.get_run(run_id)
        if run_payload.get("success") is False:
            raise ValueError(f"unknown run: {run_id}")
        run = run_payload["run"]
        metrics_payload = self._runs.get_run_metrics(run_id, scope=scope)
        if metrics_payload.get("success") is False:
            raise ValueError(
                f"{run_id} {scope or 'default'}: "
                f"{metrics_payload.get('error') or metrics_payload.get('code')}"
            )
        # A null request resolves to the run's default, and the row must name the
        # scope it was actually read from rather than the request that found it.
        resolved_scope = str(metrics_payload["scope"])
        if metrics_payload["kind"] == "fold_summary":
            raise ValueError(
                f"{run_id} {resolved_scope}: fold summary cannot be "
                "ranked; select a fold scope"
            )
        name = metric[len(_RUN_PREFIX) :]
        recorded = metrics_payload["definitions"].get(name)
        if recorded is not None:
            current = MetricDefinitionSnapshot.model_validate(
                asdict(METRIC_CATALOG[name])
            ).model_dump()
            differences = [
                key for key, value in current.items() if recorded.get(key) != value
            ]
            if differences:
                raise ValueError(
                    f"{run_id} {resolved_scope}: recorded metric definition differs "
                    f"from the current catalog: {', '.join(differences)}"
                )
        manifest_hash = run.get("manifest_hash")
        if not manifest_hash:
            raise ValueError(f"{run_id}: no execution manifest recorded")
        try:
            manifest = ExecutionManifestStore(self._workspace).load(
                run_id, expected_hash=manifest_hash
            )
        except ManifestError as exc:
            raise ValueError(f"{run_id}: {exc}") from exc
        backtest_section = manifest.config.get("backtest")
        strategy_section = manifest.config.get("strategy")
        if not isinstance(backtest_section, dict) or not isinstance(
            strategy_section, dict
        ):
            raise ValueError(
                f"{run_id}: execution manifest lacks backtest or strategy settings"
            )
        try:
            backtest = BacktestConfig.model_validate(backtest_section)
        except ValueError as exc:
            raise ValueError(
                f"{run_id}: execution manifest has no usable backtest conditions"
            ) from exc
        candidate = build_run_candidate(
            run_id=run_id,
            scope=resolved_scope,
            strategy_id=str(run["strategy_id"]),
            revision_id=str(run.get("revision_id") or ""),
            parameters=metrics_payload["parameters"],
            metadata=metrics_payload["metadata"],
            symbol=backtest.symbol,
            timeframe=backtest.timeframe,
            start_date=backtest.start_date.isoformat(),
            end_date=backtest.end_date.isoformat(),
            fees=backtest.fees,
            slippage=backtest.slippage,
            metric_source="bundle",
        )
        return to_ranking_candidate(candidate, metrics_payload["metrics"])
