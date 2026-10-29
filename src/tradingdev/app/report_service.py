"""Generate offline research reports exclusively from validated saved results."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from typing import Any

from tradingdev.adapters.reporting.bokeh_tables import bokeh_provenance
from tradingdev.adapters.reporting.html import render_report
from tradingdev.adapters.reporting.quantstats_report import (
    QuantStatsReportError,
    build_quantstats_report,
)
from tradingdev.adapters.reporting.research import research_document_payload
from tradingdev.adapters.reporting.storage import (
    ReportPublicationError,
    publish_report,
    read_report_html,
)
from tradingdev.adapters.reporting.tables import (
    comparison_table,
    ledger_table,
    trade_table,
)
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore, get_sqlite_store
from tradingdev.app.contracts.reports import ReportAnalysis
from tradingdev.app.diagnostic_service import DiagnosticService
from tradingdev.app.report_comparison import load_report_comparison
from tradingdev.app.report_ml import (
    MLReportError,
    load_ml_report,
    validate_ml_selection,
)
from tradingdev.app.run_service import RunService
from tradingdev.app.trade_history_service import (
    HistoryReadError,
    LoadedHistoryRun,
    TradeHistoryService,
)
from tradingdev.domain.diagnostics import DiagnosticContext
from tradingdev.domain.execution import ManifestError
from tradingdev.domain.presentation.ledger import build_ledger
from tradingdev.domain.presentation.reports import (
    REPORT_RECIPES,
    REPORT_SCHEMA_VERSION,
    REPORT_SECTIONS,
    REPORT_TEMPLATE_VERSION,
    SCOPED_REPORT_SECTIONS,
    report_navigation_instructions,
)


class ReportService:
    """Publish deterministic, immutable reports without executing a strategy."""

    def __init__(
        self,
        *,
        workspace: WorkspacePaths | None = None,
        store: SQLiteStore | None = None,
    ) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._store = store or get_sqlite_store(self._workspace)
        self._history = TradeHistoryService(
            workspace=self._workspace, store=self._store
        )
        self._diagnostics = DiagnosticService(self._history)

    def get_report_sections(self) -> dict[str, Any]:
        """Offer recipes without requiring their use or accepting raw HTML."""
        return {
            "success": True,
            "sections": [asdict(section) for section in REPORT_SECTIONS.values()],
            "templates": {key: list(value) for key, value in REPORT_RECIPES.items()},
        }

    def get_report_download(
        self, report_id: str, *, expected_sha256: str
    ) -> dict[str, Any]:
        """Return verified HTML bytes for an in-process download adapter.

        Pass the report ID and SHA-256 returned by ``generate_report``. Success
        contains ``content`` as bytes, ``report_id`` and ``sha256``; failures contain
        ``code`` and ``error`` without content. This is not a JSON/MCP response.
        Every call validates the registered identity and the exact bytes returned;
        it neither repairs files nor regenerates historical results.
        """
        try:
            content = read_report_html(
                self._workspace,
                self._store,
                report_id,
                expected_sha256=expected_sha256,
            )
        except ReportPublicationError as exc:
            return _error(exc.code, str(exc))
        except (OSError, ValueError, sqlite3.Error) as exc:
            return _error("report_read_failed", str(exc))
        return {
            "success": True,
            "report_id": report_id,
            "sha256": expected_sha256,
            "content": content,
        }

    def generate_report(
        self,
        run_ids: list[str] | None = None,
        sections: list[str] | None = None,
        commentary: list[dict[str, str]] | None = None,
        *,
        comparison_job_id: str | None = None,
        analysis: dict[str, Any] | None = None,
        ml_entry_ids: list[str] | None = None,
        ml_ranking_metric: str = "ic_mean",
        ml_allow_parameter_axes: bool = False,
    ) -> dict[str, Any]:
        """Publish runs, a finalized comparison and/or selected saved ML evidence.

        Identical inputs reuse the same report, provided existing report files and
        registrations are intact. Missing/corrupt history is an error, never a request
        to reconstruct it from pickle, current configuration, or a fresh simulation.
        """
        try:
            validate_ml_selection(
                ml_entry_ids, ml_ranking_metric, ml_allow_parameter_axes
            )
        except MLReportError as exc:
            return _error(exc.code, str(exc))
        if (run_ids is not None and comparison_job_id is not None) or (
            run_ids is None and comparison_job_id is None and ml_entry_ids is None
        ):
            return _error(
                "invalid_report_selection",
                "Select run_ids OR comparison_job_id, and/or saved ml_entry_ids",
            )
        if comparison_job_id is not None and (
            not isinstance(comparison_job_id, str) or not comparison_job_id.strip()
        ):
            return _error("invalid_report_selection", "Provide a comparison job ID")
        if run_ids is not None and (
            not isinstance(run_ids, list)
            or not 1 <= len(run_ids) <= 8
            or any(not isinstance(run_id, str) for run_id in run_ids)
            or len(set(run_ids)) != len(run_ids)
        ):
            return _error(
                "invalid_report_runs", "Provide one to eight distinct run IDs"
            )
        if sections is None:
            selected = (
                list(REPORT_RECIPES["standard"])
                if run_ids is not None or comparison_job_id is not None
                else []
            )
            if ml_entry_ids is not None:
                selected.append("ml")
        else:
            selected = sections
        if (
            not isinstance(selected, list)
            or any(not isinstance(key, str) for key in selected)
            or len(set(selected)) != len(selected)
            or set(selected) - REPORT_SECTIONS.keys()
        ):
            return _error(
                "invalid_report_sections", "Select unique built-in section IDs"
            )
        if "ml" in selected and ml_entry_ids is None:
            return _error(
                "invalid_report_ml_selection",
                "The ML section requires explicitly selected saved ml_entry_ids",
            )
        notes = [] if commentary is None else commentary
        if (
            not isinstance(notes, list)
            or len(notes) > 20
            or any(
                not isinstance(note, dict)
                or set(note) != {"title", "text"}
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in note.values()
                )
                for note in notes
            )
            or sum(len(value) for note in notes for value in note.values()) > 20000
        ):
            return _error(
                "invalid_report_commentary",
                "Provide up to 20 plain-text title/text notes, 20000 characters total",
            )
        try:
            interpretation = (
                ReportAnalysis.model_validate(analysis)
                if analysis is not None
                else None
            )
            if comparison_job_id is not None:
                comparison = load_report_comparison(
                    self._workspace, self._store, self._history, comparison_job_id
                )
                run_ids = list(comparison.runs)
                payload = self._load(run_ids, loaded_runs=comparison.runs)
                payload["fixed_comparison"] = comparison.evidence
                snapshot = comparison.research_evidence
                if snapshot is not None:
                    payload["research_documents"][snapshot.content_hash] = (
                        research_document_payload(snapshot)
                    )
                    payload["fixed_comparison"]["research_evidence_hash"] = (
                        snapshot.content_hash
                    )
            else:
                run_ids = [] if run_ids is None else run_ids
                payload = self._load(run_ids)
            if ml_entry_ids is not None:
                payload["ml"] = load_ml_report(
                    self._workspace,
                    ml_entry_ids,
                    metric=ml_ranking_metric,
                    allow_parameter_axes=ml_allow_parameter_axes,
                )
            payload["sections"] = list(selected)
            if interpretation is not None:
                allowed = {
                    (run["identity"]["run_id"], scope["scope_id"])
                    for run in payload["runs"]
                    for scope in run["scopes"]
                }
                references = [
                    (ref.run_id, ref.scope) for ref in interpretation.references
                ]
                if (
                    (allowed and not references)
                    or len(set(references)) != len(references)
                    or set(references) - allowed
                ):
                    return _error(
                        "invalid_analysis_reference",
                        "Analysis must reference distinct existing report run/scopes; "
                        "successful reports need at least one reference",
                    )
            payload["analysis"] = (
                interpretation.model_dump(mode="json") if interpretation else None
            )
            payload["navigation_instructions"] = report_navigation_instructions(
                selected,
                has_results=bool(payload["runs"]),
                comparison=comparison_job_id is not None,
                has_ml=ml_entry_ids is not None,
            )
            payload["commentary"] = [dict(note) for note in notes]
            payload["comparison_table"] = comparison_table(
                payload,
                navigate=bool(
                    payload["runs"] and SCOPED_REPORT_SECTIONS.intersection(selected)
                ),
            )
            if comparison_job_id is not None or set(selected) & {
                "overview",
                "ledger",
                "trades",
            }:
                payload["renderers"] = {"bokeh": bokeh_provenance()}
            if "metrics" in selected:
                for run in payload["runs"]:
                    for scope in run["scopes"]:
                        scope["quantstats"] = build_quantstats_report(
                            {**scope, "report_source": run["provenance"]}
                        )
            encoded = _encode(payload)
            report_id = hashlib.sha256(encoded).hexdigest()
            report_html = render_report(payload, report_id).encode("utf-8")
            scope_count = sum(len(run["scopes"]) for run in payload["runs"])
            if "ml" in payload:
                scope_count += len(payload["ml"]["scopes"])
            manifest = {
                "schema_version": REPORT_SCHEMA_VERSION,
                "template_version": REPORT_TEMPLATE_VERSION,
                "report_id": report_id,
                "run_ids": run_ids,
                "content_sha256": report_id,
                "html_sha256": hashlib.sha256(report_html).hexdigest(),
                "sources": [run["provenance"] for run in payload["runs"]],
                "research_documents": {
                    key: {
                        name: document[name]
                        for name in (
                            "evidence_id",
                            "record_hash",
                            "manifest_hash",
                            "created_at",
                        )
                    }
                    for key, document in payload["research_documents"].items()
                },
                "derived_analytics": {
                    run["identity"]["run_id"]: {
                        scope["scope_id"]: {
                            key: value
                            for key, value in scope["quantstats"].items()
                            if key not in {"html", "tables_html"}
                        }
                        for scope in run["scopes"]
                        if "quantstats" in scope
                    }
                    for run in payload["runs"]
                    if "metrics" in selected
                },
                "navigation_instructions": payload["navigation_instructions"],
                "scope_count": scope_count,
                "renderers": payload.get("renderers", {}),
                "sections": list(selected),
                "omitted_sections": [
                    key
                    for key in REPORT_SECTIONS
                    if key not in selected and (key != "ml" or "ml" in payload)
                ],
                "available_data": {
                    run["identity"]["run_id"]: {
                        scope["scope_id"]: {
                            "kind": scope["kind"],
                            "observation_count": len(
                                scope["observations"]["equity_curve"]
                            )
                            if scope["observations"] is not None
                            else None,
                            "trade_count": len(scope["observations"]["trades"])
                            if scope["observations"] is not None
                            else None,
                            "execution_record_count": _record_count(
                                scope["observations"], "execution_records"
                            ),
                            "account_history_count": _record_count(
                                scope["observations"], "account_history"
                            ),
                        }
                        for scope in run["scopes"]
                    }
                    for run in payload["runs"]
                },
                "commentary": [dict(note) for note in notes],
                "analysis": payload["analysis"],
                "diagnostics": {
                    run["identity"]["run_id"]: {
                        scope["scope_id"]: scope["diagnostics"]
                        for scope in run["scopes"]
                    }
                    for run in payload["runs"]
                },
                "available_scopes": {
                    run["identity"]["run_id"]: [
                        scope["scope_id"] for scope in run["scopes"]
                    ]
                    for run in payload["runs"]
                },
                "benchmarks": {
                    run["identity"]["run_id"]: {
                        scope["scope_id"]: _benchmark_summary(scope.get("benchmark"))
                        for scope in run["scopes"]
                        if scope.get("benchmark") is not None
                    }
                    for run in payload["runs"]
                },
            }
            if comparison_job_id is not None:
                manifest["comparison_job_id"] = comparison_job_id
                manifest["fixed_comparison"] = payload["fixed_comparison"]
            if ml_entry_ids is not None:
                manifest["ml_entry_ids"] = list(ml_entry_ids)
                manifest["ml"] = payload["ml"]
            published = publish_report(
                self._workspace,
                self._store,
                report_id,
                report_html,
                _encode(manifest),
                run_ids=run_ids,
                scope_count=scope_count,
                comparison_job_id=comparison_job_id,
                ml_entry_ids=ml_entry_ids,
            )
            return {
                **published,
                "navigation_instructions": payload["navigation_instructions"],
            }
        except (
            HistoryReadError,
            ReportPublicationError,
            MLReportError,
        ) as exc:
            return _error(exc.code, str(exc))
        except ManifestError as exc:
            return _error("comparison_report_invalid", str(exc))
        except (OSError, ValueError, sqlite3.Error, QuantStatsReportError) as exc:
            return _error("report_generation_failed", str(exc))

    def _load(
        self,
        run_ids: list[str],
        *,
        loaded_runs: dict[str, LoadedHistoryRun] | None = None,
    ) -> dict[str, Any]:
        runs: list[dict[str, Any]] = []
        research_documents: dict[str, Any] = {}
        for run_id in run_ids:
            loaded = (
                loaded_runs[run_id]
                if loaded_runs is not None
                else self._history.load_run(run_id)
            )
            run = loaded.run
            bundle = loaded.performance
            research_hash = None
            if loaded.research_evidence is not None:
                research_hash = loaded.research_evidence.content_hash
                research_documents[research_hash] = research_document_payload(
                    loaded.research_evidence
                )
            scopes: list[dict[str, Any]] = []
            config = loaded.manifest.config_copy() if loaded.manifest else None
            for scope_id, scope in bundle.scopes.items():
                entry = {"scope_id": scope_id, **scope.model_dump(mode="json")}
                benchmark = bundle.benchmarks.get(scope_id)
                entry["benchmark"] = (
                    None if benchmark is None else benchmark.model_dump(mode="json")
                )
                definition_keys = list(scope.values) + [
                    name
                    for name in (() if benchmark is None else benchmark.values)
                    if name not in scope.values
                ]
                entry["diagnostics"] = [
                    result.model_dump(mode="json")
                    for result in self._diagnostics.diagnose(
                        DiagnosticContext(
                            scope, loaded.observations.scopes.get(scope_id)
                        )
                    )
                ]
                entry["definitions"] = {
                    name: bundle.definitions[name].model_dump(mode="json")
                    for name in definition_keys
                }
                if scope.kind == "backtest":
                    history = self._history.select_scope(loaded, scope_id)
                    entry.update(
                        observations=history.observations.model_dump(mode="json"),
                        parameters=history.parameters,
                        parameter_provenance=history.parameter_provenance,
                        parameters_complete=history.parameters_complete,
                        parameter_sources=[
                            item.model_dump(mode="json")
                            for item in history.parameter_sources
                        ],
                        execution_assumptions=(
                            history.execution_assumptions.model_dump(mode="json")
                            if history.execution_assumptions is not None
                            else None
                        ),
                    )
                    entry["tables"] = {
                        "ledger": ledger_table(
                            build_ledger(
                                history.observations.execution_records,
                                history.observations.account_history,
                                mode=scope.mode,
                            )
                        ),
                        "trades": trade_table(entry),
                    }
                    config = history.execution_config
                else:
                    entry["observations"] = None
                    entry["parameter_provenance"] = "aggregate_scope_snapshot"
                    entry["parameters_complete"] = False
                scopes.append(entry)
            sources = []
            for name in ("performance_json", "observations_json"):
                record = self._store.get_artifact(f"{run_id}:{name}")
                if record is not None:
                    sources.append(
                        {k: record.get(k) for k in ("artifact_id", "sha256")}
                    )
            if loaded.manifest_artifact is not None:
                sources.append(
                    {
                        key: loaded.manifest_artifact.get(key)
                        for key in ("artifact_id", "sha256")
                    }
                )
            if loaded.research_artifact is not None:
                sources.append(
                    {
                        key: loaded.research_artifact.get(key)
                        for key in ("artifact_id", "sha256")
                    }
                )
            runs.append(
                {
                    "identity": {
                        key: run.get(key)
                        for key in (
                            "run_id",
                            "job_id",
                            "strategy_id",
                            "revision_id",
                            "manifest_hash",
                            "source_hash",
                            "config_hash",
                            "random_seed",
                            "dataset_id",
                            "created_at",
                        )
                    },
                    "default_scope": bundle.default_scope,
                    "selected_train_scope": bundle.selected_train_scope,
                    "execution_config": config,
                    "research_evidence_hash": research_hash,
                    "scopes": scopes,
                    "provenance": {
                        "run_id": run_id,
                        "manifest_hash": bundle.manifest_hash,
                        "artifacts": sources,
                    },
                }
            )
        comparison = None
        if len(run_ids) > 1:
            comparison = RunService(
                workspace=self._workspace, store=self._store
            ).compare_runs(run_ids)
            if not comparison.get("success"):
                raise ReportPublicationError(
                    str(comparison.get("code", "report_comparison_failed")),
                    str(comparison.get("error", "Cannot compare recorded results")),
                )
        payload = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "template_version": REPORT_TEMPLATE_VERSION,
            "runs": runs,
            "research_documents": research_documents,
            "default_scope_comparison": comparison,
        }
        return payload


def _encode(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _error(code: str, message: str) -> dict[str, Any]:
    return {"success": False, "code": code, "error": message}


def _record_count(observations: dict[str, Any] | None, field: str) -> int | None:
    """Keep an unavailable stream distinct from a recorded stream with zero rows."""
    if observations is None or observations.get(field) is None:
        return None
    return len(observations[field])


def _benchmark_summary(benchmark: dict[str, Any]) -> dict[str, Any]:
    """Keep baseline identity and its absence reason auditable from the manifest."""
    definition = benchmark.get("definition")
    definition = definition if isinstance(definition, dict) else {}
    return {
        "status": benchmark.get("status"),
        "id": definition.get("id"),
        "symbol": definition.get("symbol"),
        "asset_class": benchmark.get("asset_class"),
        "source": benchmark.get("source"),
        "dataset_id": benchmark.get("dataset_id"),
        "reason": benchmark.get("reason"),
        "coverage": benchmark.get("coverage"),
    }
