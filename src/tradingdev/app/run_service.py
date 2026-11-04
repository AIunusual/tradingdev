"""Read saved performance scopes and compare their declared interpretations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.performance import (
    PerformanceArtifactError,
    PerformanceStore,
)
from tradingdev.adapters.storage.sqlite import SQLiteStore, get_sqlite_store
from tradingdev.app.contracts.common import DEFAULT_PAGE_LIMIT, Page
from tradingdev.app.contracts.research import RunSummary
from tradingdev.app.pagination import paginate
from tradingdev.app.rolling_trace_summary import query_metadata
from tradingdev.domain.performance.catalog import metric_catalog
from tradingdev.domain.presentation.metrics import display_metrics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tradingdev.domain.performance.artifacts import (
        PerformanceBundle,
        PerformanceScope,
    )

_DISCOVERY_FIELDS = (
    "details_available",
    "provenance",
    "available_metric_ids",
    "available_scopes",
    "default_scope",
    "selected_train_scope",
    "detail_error",
)


class RunService:
    """Query stored results; presentation never changes the stored metric set."""

    def __init__(
        self,
        *,
        workspace: WorkspacePaths | None = None,
        store: SQLiteStore | None = None,
    ) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._workspace.ensure()
        self._store = store or get_sqlite_store(self._workspace)
        self._performance = PerformanceStore(self._workspace, self._store)

    def list_runs(self) -> list[dict[str, Any]]:
        """List compact default-scope values with explicit detail discovery."""
        return [self._summarize_run(run) for run in self._store.list_runs()]

    def list_runs_page(
        self,
        *,
        strategy_id: str | None = None,
        revision_id: str | None = None,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> Page[RunSummary]:
        """Page completed identities, newest first, without loading performance."""
        rows = [
            run
            for run in self._store.list_runs()
            if (strategy_id is None or run["strategy_id"] == strategy_id)
            and (revision_id is None or run.get("revision_id") == revision_id)
        ]
        rows.sort(key=lambda row: str(row["run_id"]))
        rows.sort(key=lambda row: str(row["created_at"]), reverse=True)
        return paginate(
            rows,
            project=lambda row: RunSummary.model_validate(
                {name: row.get(name) for name in RunSummary.model_fields}
            ),
            offset=offset,
            limit=limit,
        )

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Return run identity, summary, and available recorded metric scopes."""
        run = self._store.get_run(run_id)
        if run is None:
            return _error("run_not_found", f"Unknown run: {run_id}")
        return {"success": True, "run": self._summarize_run(run)}

    def get_metric_catalog(self, mode: str | None = None) -> dict[str, Any]:
        """Describe the current metric catalog, optionally restricted by mode."""
        if mode not in (None, "signal", "volume"):
            return _error("invalid_mode", "mode must be 'signal' or 'volume'")
        return {
            "success": True,
            "schema_version": 1,
            "definitions": metric_catalog(mode),
        }

    def get_run_metrics(
        self,
        run_id: str,
        metric_ids: list[str] | None = None,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Read original values, definitions, and settings without recomputation."""
        if self._store.get_run(run_id) is None:
            return _error("run_not_found", f"Unknown run: {run_id}")
        try:
            bundle = self._performance.load(run_id)
        except PerformanceArtifactError as exc:
            return _error(exc.code, str(exc))
        scope_id = bundle.default_scope if scope is None else scope
        if scope_id not in bundle.scopes:
            return _selection_error(
                bundle,
                "unknown_metric_scope",
                f"Unknown metric scope: {scope_id}",
                bundle.scopes[bundle.default_scope],
            )
        selected = bundle.scopes[scope_id]
        names = (
            list(selected.values)
            if metric_ids is None
            else list(dict.fromkeys(metric_ids))
        )
        unknown = sorted(set(names) - selected.values.keys())
        if unknown:
            return _selection_error(
                bundle,
                "unknown_metric_id",
                f"Unknown metrics in scope {scope_id}: {', '.join(unknown)}",
                selected,
            )
        scope_payload = selected.model_dump(mode="json")
        payload = {
            key: value for key, value in scope_payload.items() if key != "values"
        }
        payload.update(
            success=True,
            metadata=query_metadata(
                selected.metadata, run_id=run_id, scope_id=scope_id
            ),
            run_id=run_id,
            manifest_hash=bundle.manifest_hash,
            scope=scope_id,
            metrics={name: scope_payload["values"][name] for name in names},
            display_metrics=display_metrics(
                {name: selected.values[name] for name in names}, bundle.definitions
            ),
            definitions={name: bundle.definitions[name].model_dump() for name in names},
            **_discovery(bundle, selected),
        )
        return payload

    def compare_runs(
        self,
        selections: Sequence[tuple[str, str | None]],
        metric_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare saved scopes by ``(run_id, scope)``; nothing is recomputed.

        A pair is the comparison key, so two scopes of one run can be compared.
        Duplicate selections are refused after resolution, which also catches an
        omitted scope that names the same measurement as an explicit one.
        """
        if len(selections) < 2:
            return _error(
                "insufficient_runs", "compare_runs requires at least two selections"
            )
        rows: list[dict[str, Any]] = []
        labels: list[str] = []
        seen: set[tuple[str, str]] = set()
        default_names: set[str] = set()
        for run_id, requested_scope in selections:
            run = self._store.get_run(run_id)
            if run is None:
                return _error("run_not_found", f"Unknown run: {run_id}")
            detail = self.get_run_metrics(
                run_id, metric_ids=metric_ids, scope=requested_scope
            )
            if detail.get("success") is False:
                if (
                    detail["code"] != "performance_artifact_unavailable"
                    or requested_scope is not None
                ):
                    return detail
                summary = self._summarize_run(run)
                metrics = {
                    key: value
                    for key, value in run["metrics"].items()
                    if _is_metric_value(value)
                }
                if metric_ids is not None:
                    unknown = sorted(set(metric_ids) - metrics.keys())
                    if unknown:
                        return _error(
                            "unknown_metric_id",
                            f"Unknown legacy metrics: {', '.join(unknown)}",
                        ) | {
                            "run_id": run_id,
                            "available_metric_ids": sorted(metrics),
                            "available_scopes": [],
                            "default_scope": None,
                        }
                    metrics = {key: metrics[key] for key in metric_ids}
                resolved = (run_id, "")
                if resolved in seen:
                    return _error(
                        "duplicate_selections",
                        f"Duplicate selection: {run_id} has one legacy summary",
                    )
                seen.add(resolved)
                labels.append(_selection_label(run_id, requested_scope))
                rows.append(
                    {
                        "run_id": run_id,
                        "strategy_id": run["strategy_id"],
                        "scope": None,
                        "kind": None,
                        "mode": None,
                        "split": None,
                        "metric_source": "legacy_summary",
                        "metrics": metrics,
                        "display_metrics": {},
                        "metadata": {},
                        "definitions": {},
                        **{key: summary[key] for key in _DISCOVERY_FIELDS},
                    }
                )
                default_names.update(metrics)
            else:
                resolved = (run_id, detail["scope"])
                if resolved in seen:
                    return _error(
                        "duplicate_selections",
                        f"Duplicate selection: {run_id} scope {detail['scope']}",
                    )
                seen.add(resolved)
                labels.append(_selection_label(run_id, requested_scope))
                definitions = detail["definitions"]
                default_names.update(
                    key
                    for key, definition in definitions.items()
                    if definition["summary"] and detail["mode"] in definition["modes"]
                )
                rows.append(
                    {
                        "run_id": run_id,
                        "strategy_id": run["strategy_id"],
                        "metric_source": "bundle",
                        **{
                            key: detail[key]
                            for key in (
                                "scope",
                                "kind",
                                "mode",
                                "split",
                                "metrics",
                                "display_metrics",
                                "metadata",
                                "definitions",
                                *_DISCOVERY_FIELDS,
                            )
                        },
                    }
                )
        names = (
            sorted(default_names)
            if metric_ids is None
            else list(dict.fromkeys(metric_ids))
        )
        for row in rows:
            row["metrics"] = {name: row["metrics"].get(name) for name in names}
            row["display_metrics"] = {
                name: row["display_metrics"][name]
                for name in names
                if name in row["display_metrics"]
            }
            row["definitions"] = {
                name: row["definitions"][name]
                for name in names
                if name in row["definitions"]
            }
        compatibility = {
            name: _metric_compatibility(name, rows, labels) for name in names
        }
        return {
            "success": True,
            "runs": rows,
            "comparable": bool(names)
            and all(item["comparable"] for item in compatibility.values()),
            "metric_compatibility": compatibility,
            "context_differences": _context_differences(rows, labels),
        }

    def _summarize_run(self, run: dict[str, Any]) -> dict[str, Any]:
        try:
            bundle = self._performance.load(run["run_id"])
        except PerformanceArtifactError as exc:
            legacy = exc.code == "performance_artifact_unavailable"
            return run | {
                "metrics": run["metrics"] if legacy else {},
                "display_metrics": {},
                "details_available": False,
                "provenance": "legacy_metrics" if legacy else "invalid_artifact",
                "available_metric_ids": sorted(run["metrics"]) if legacy else [],
                "available_scopes": [],
                "default_scope": None,
                "selected_train_scope": None,
                "detail_error": _error(exc.code, str(exc)),
            }
        selected = bundle.scopes[bundle.default_scope]
        values = selected.model_dump(mode="json")["values"]
        names = [
            key
            for key in values
            if bundle.definitions[key].summary
            and selected.mode in bundle.definitions[key].modes
        ]
        return run | {
            "metrics": {key: values[key] for key in names},
            "display_metrics": display_metrics(
                {key: selected.values[key] for key in names}, bundle.definitions
            ),
            **_discovery(bundle, selected),
        }


def _error(code: str, message: str) -> dict[str, Any]:
    return {"success": False, "error": message, "code": code}


def _discovery(bundle: PerformanceBundle, scope: PerformanceScope) -> dict[str, Any]:
    return {
        "details_available": True,
        "provenance": "performance_artifact",
        "available_metric_ids": sorted(scope.values),
        "available_scopes": list(bundle.scopes),
        "default_scope": bundle.default_scope,
        "selected_train_scope": bundle.selected_train_scope,
        "detail_error": None,
    }


def _selection_error(
    bundle: PerformanceBundle,
    code: str,
    message: str,
    scope: PerformanceScope,
) -> dict[str, Any]:
    return _error(code, message) | {
        "run_id": bundle.run_id,
        "available_metric_ids": sorted(scope.values),
        "available_scopes": list(bundle.scopes),
        "default_scope": bundle.default_scope,
    }


def _is_metric_value(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, int | float) and not isinstance(value, bool))
        or (
            isinstance(value, dict)
            and {"mean", "std", "min", "max", "valid_count"}.issubset(value)
        )
    )


def _selection_label(run_id: str, requested_scope: str | None) -> str:
    """Name one selection inside a comparison response.

    A default scope needs no suffix, because a run contributes it at most once;
    a named scope must be named, because one run can contribute several.
    """
    return run_id if requested_scope is None else f"{run_id}#{requested_scope}"


def _metric_compatibility(
    name: str, rows: list[dict[str, Any]], labels: list[str]
) -> dict[str, Any]:
    reasons: list[str] = []
    for row, label in zip(rows, labels, strict=True):
        if row["provenance"] != "performance_artifact":
            reasons.append(f"unknown_provenance:{label}")
        if name not in row["available_metric_ids"]:
            reasons.append(f"metric_not_recorded:{label}")
        elif row["metrics"].get(name) is None:
            reasons.append(f"metric_unavailable:{label}")
        elif (
            isinstance(row["metrics"][name], dict)
            and row["metrics"][name].get("valid_count") == 0
        ):
            reasons.append(f"no_valid_folds:{label}")
    baseline = rows[0]
    baseline_definition = baseline["definitions"].get(name)
    for row, label in zip(rows[1:], labels[1:], strict=True):
        definition = row["definitions"].get(name)
        if baseline_definition is None or definition is None:
            continue
        for key in ("unit", "category", "provider", "description"):
            if definition.get(key) != baseline_definition.get(key):
                reasons.append(f"definition.{key}_differs:{label}")
        for key in ("mode", "kind", "split"):
            if row[key] != baseline[key]:
                reasons.append(f"{key}_differs:{label}")
        for section in ("providers", "settings"):
            baseline_settings = baseline["metadata"].get(section)
            other_settings = row["metadata"].get(section)
            if baseline_settings is None or other_settings is None:
                reasons.append(f"{section}_unavailable:{label}")
            elif baseline_settings != other_settings:
                reasons.append(f"{section}_differ:{label}")
        baseline_context = baseline["metadata"].get("execution_context", {})
        context = row["metadata"].get("execution_context", {})
        if definition.get("unit") == "amount":
            for key in ("symbol", "init_cash", "initial_cash", "position_size"):
                if baseline_context.get(key) != context.get(key):
                    reasons.append(f"amount_basis.{key}_differs:{label}")
            if not baseline_context.get("symbol") or not context.get("symbol"):
                reasons.append(f"amount_currency_unavailable:{label}")
    return {"comparable": not reasons, "reasons": list(dict.fromkeys(reasons))}


def _context_differences(
    rows: list[dict[str, Any]], labels: list[str]
) -> dict[str, dict[str, Any]]:
    sections = ("execution_context", "observations")
    differences: dict[str, dict[str, Any]] = {}
    for section in sections:
        names = {key for row in rows for key in row["metadata"].get(section, {})}
        for name in sorted(names):
            # Key by selection, not by run: two scopes of one run would otherwise
            # collapse into a single entry and hide the difference being reported.
            values = {
                label: row["metadata"].get(section, {}).get(name)
                for row, label in zip(rows, labels, strict=True)
            }
            if any(value != next(iter(values.values())) for value in values.values()):
                differences[f"{section}.{name}"] = values
    return differences
