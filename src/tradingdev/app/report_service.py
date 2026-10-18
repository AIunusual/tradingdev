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
from tradingdev.app.run_service import RunService
from tradingdev.app.trade_history_service import HistoryReadError, TradeHistoryService
from tradingdev.domain.presentation.ledger import build_ledger
from tradingdev.domain.presentation.reports import (
    REPORT_RECIPES,
    REPORT_SCHEMA_VERSION,
    REPORT_SECTIONS,
    REPORT_TEMPLATE_VERSION,
    SCOPED_REPORT_SECTIONS,
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
        run_ids: list[str],
        sections: list[str] | None = None,
        commentary: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Read every recorded scope from one to eight distinct runs and publish HTML.

        Identical inputs reuse the same report, provided existing report files and
        registrations are intact. Missing/corrupt history is an error, never a request
        to reconstruct it from pickle, current configuration, or a fresh simulation.
        """
        if (
            not isinstance(run_ids, list)
            or not 1 <= len(run_ids) <= 8
            or any(not isinstance(run_id, str) for run_id in run_ids)
            or len(set(run_ids)) != len(run_ids)
        ):
            return _error(
                "invalid_report_runs", "Provide one to eight distinct run IDs"
            )
        selected = list(REPORT_RECIPES["standard"]) if sections is None else sections
        if (
            not isinstance(selected, list)
            or any(not isinstance(key, str) for key in selected)
            or len(set(selected)) != len(selected)
            or set(selected) - REPORT_SECTIONS.keys()
        ):
            return _error(
                "invalid_report_sections", "Select unique built-in section IDs"
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
            payload = self._load(run_ids)
            payload["sections"] = list(selected)
            payload["commentary"] = [dict(note) for note in notes]
            payload["comparison_table"] = comparison_table(
                payload, navigate=bool(SCOPED_REPORT_SECTIONS.intersection(selected))
            )
            if set(selected) & {"overview", "ledger", "trades"}:
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
            manifest = {
                "schema_version": REPORT_SCHEMA_VERSION,
                "template_version": REPORT_TEMPLATE_VERSION,
                "report_id": report_id,
                "run_ids": run_ids,
                "content_sha256": report_id,
                "html_sha256": hashlib.sha256(report_html).hexdigest(),
                "sources": [run["provenance"] for run in payload["runs"]],
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
                "scope_count": scope_count,
                "renderers": payload.get("renderers", {}),
                "sections": list(selected),
                "omitted_sections": [
                    key for key in REPORT_SECTIONS if key not in selected
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
                "available_scopes": {
                    run["identity"]["run_id"]: [
                        scope["scope_id"] for scope in run["scopes"]
                    ]
                    for run in payload["runs"]
                },
            }
            return publish_report(
                self._workspace,
                self._store,
                report_id,
                report_html,
                _encode(manifest),
                run_ids=run_ids,
                scope_count=scope_count,
            )
        except (
            HistoryReadError,
            ReportPublicationError,
        ) as exc:
            return _error(exc.code, str(exc))
        except (OSError, ValueError, sqlite3.Error, QuantStatsReportError) as exc:
            return _error("report_generation_failed", str(exc))

    def _load(self, run_ids: list[str]) -> dict[str, Any]:
        runs: list[dict[str, Any]] = []
        for run_id in run_ids:
            loaded = self._history.load_run(run_id)
            run = loaded.run
            bundle = loaded.performance
            scopes: list[dict[str, Any]] = []
            config = loaded.manifest.config_copy() if loaded.manifest else None
            for scope_id, scope in bundle.scopes.items():
                entry = {"scope_id": scope_id, **scope.model_dump(mode="json")}
                entry["definitions"] = {
                    name: bundle.definitions[name].model_dump(mode="json")
                    for name in scope.values
                }
                if scope.kind == "backtest":
                    history = self._history.select_scope(loaded, scope_id)
                    entry.update(
                        observations=history.observations.model_dump(mode="json"),
                        parameters=history.parameters,
                        parameter_provenance=history.parameter_provenance,
                        parameters_complete=history.parameters_complete,
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
