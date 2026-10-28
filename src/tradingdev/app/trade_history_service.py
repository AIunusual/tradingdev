"""Read immutable historical observations without executing strategy code."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.performance import (
    PerformanceArtifactError,
    PerformanceStore,
)
from tradingdev.adapters.storage.sqlite import SQLiteStore, get_sqlite_store
from tradingdev.adapters.storage.submitted_research import SubmittedResearchStore
from tradingdev.domain.execution import ManifestError
from tradingdev.domain.research_scope import (
    effective_scope_parameters,
    scope_parameter_sources,
    verify_research_scopes,
)
from tradingdev.domain.strategies.execution import finite_strategy_value

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tradingdev.domain.approved_research import ApprovedResearch
    from tradingdev.domain.execution import ExecutionManifest
    from tradingdev.domain.execution_assumptions import ExecutionAssumptions
    from tradingdev.domain.performance.artifacts import (
        ObservationsBundle,
        PerformanceBundle,
        PerformanceScope,
        ScopeObservations,
    )
    from tradingdev.domain.research_evidence import BoundParameterProvenance


class HistoryReadError(ValueError):
    """Expected inability to read or select a saved result."""

    def __init__(
        self, code: str, message: str, available_scopes: list[str] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.available_scopes = available_scopes or []

    def response(self) -> dict[str, Any]:
        """Return the shared public failure contract."""
        return {
            "success": False,
            "code": self.code,
            "error": str(self),
            "available_scopes": self.available_scopes,
        }


@dataclass(frozen=True)
class LoadedHistoryScope:
    """Validated historical inputs shared by query and reporting adapters."""

    run: dict[str, Any]
    scope_id: str
    performance: PerformanceScope
    observations: ScopeObservations
    parameters: dict[str, JsonValue]
    parameter_provenance: str
    parameters_complete: bool
    execution_config: dict[str, Any] | None
    research_evidence: ApprovedResearch | None = None
    parameter_sources: tuple[BoundParameterProvenance, ...] = ()
    execution_assumptions: ExecutionAssumptions | None = None


@dataclass(frozen=True)
class LoadedHistoryRun:
    """One verified run snapshot reusable across scope selections without I/O."""

    run: dict[str, Any]
    performance: PerformanceBundle
    observations: ObservationsBundle
    manifest: ExecutionManifest | None
    manifest_artifact: dict[str, Any] | None
    research_evidence: ApprovedResearch | None = None
    research_artifact: dict[str, Any] | None = None


class TradeHistoryService:
    """Bounded presentation over the saved JSON integrity-checked read path."""

    def __init__(
        self,
        *,
        workspace: WorkspacePaths | None = None,
        store: SQLiteStore | None = None,
    ) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._store = store or get_sqlite_store(self._workspace)
        self._performance = PerformanceStore(self._workspace, self._store)
        self._manifests = ExecutionManifestStore(self._workspace)

    def load_run(self, run_id: str) -> LoadedHistoryRun:
        """Load a validated run snapshot shared by every selected scope."""
        run = self._store.get_run(run_id)
        if run is None:
            raise HistoryReadError("run_not_found", f"Unknown run: {run_id}")
        try:
            performance = self._performance.load(run_id)
            observations = self._performance.load_observations(run_id)
        except PerformanceArtifactError as exc:
            raise HistoryReadError(exc.code, str(exc)) from exc
        manifest = None
        manifest_artifact = None
        if run.get("manifest_hash") is not None:
            try:
                manifest_artifact = self._store.get_artifact(
                    f"{run_id}:execution_manifest"
                )
                expected_sha = None
                if manifest_artifact is not None:
                    if (
                        manifest_artifact.get("run_id") != run_id
                        or manifest_artifact.get("artifact_type")
                        != "execution_manifest"
                        or manifest_artifact.get("path")
                        != str(self._manifests.path(run_id))
                        or not isinstance(manifest_artifact.get("sha256"), str)
                    ):
                        raise ManifestError(
                            "Registered manifest identity differs from run"
                        )
                    expected_sha = manifest_artifact["sha256"]
                manifest = self._manifests.load(
                    run_id,
                    expected_hash=run["manifest_hash"],
                    expected_file_sha256=expected_sha,
                )
            except ManifestError as exc:
                raise HistoryReadError("execution_manifest_invalid", str(exc)) from exc
            strategy = manifest.config.get("strategy")
            if (
                not isinstance(strategy, dict)
                or strategy.get("id") != run["strategy_id"]
            ):
                raise HistoryReadError(
                    "execution_manifest_invalid",
                    "Manifest strategy identity differs from the recorded run",
                )
        try:
            evidence = SubmittedResearchStore(self._workspace).load_run(
                run, self._store, manifest
            )
            if evidence is not None:
                assert manifest is not None
                verify_research_scopes(manifest, performance, observations)
        except (ManifestError, ValueError) as exc:
            raise HistoryReadError("research_evidence_invalid", str(exc)) from exc
        return LoadedHistoryRun(
            run,
            performance,
            observations,
            manifest,
            manifest_artifact,
            evidence,
            self._store.get_artifact(f"{run_id}:research_evidence")
            if evidence is not None
            else None,
        )

    def load_scope(self, run_id: str, scope: str | None = None) -> LoadedHistoryScope:
        """Load a whole original scalar scope for reports, never a page or replay."""
        loaded = self.load_run(run_id)
        scope_id = loaded.performance.default_scope if scope is None else scope
        return self.select_scope(loaded, scope_id)

    @staticmethod
    def select_scope(loaded: LoadedHistoryRun, scope_id: str) -> LoadedHistoryScope:
        """Select from a verified run snapshot without reading any artifact again."""
        available = list(loaded.observations.scopes)
        if scope_id not in loaded.performance.scopes:
            raise HistoryReadError(
                "unknown_history_scope", f"Unknown history scope: {scope_id}", available
            )
        if scope_id not in loaded.observations.scopes:
            raise HistoryReadError(
                "scope_has_no_observations",
                f"Scope {scope_id} is a summary; select an original scalar scope",
                available,
            )
        selected = loaded.performance.scopes[scope_id]
        manifest = loaded.manifest
        if manifest is None:
            parameters = deepcopy(selected.parameters)
            provenance = "scope_snapshot"
            complete = False
            config = None
        else:
            try:
                parameters = effective_scope_parameters(manifest, selected)
            except ValueError as exc:
                raise HistoryReadError("parameters_unavailable", str(exc)) from exc
            provenance = (
                "execution_manifest_and_trial"
                if manifest.optimization
                else "execution_manifest"
            )
            complete = True
            config = manifest.config_copy()
        return LoadedHistoryScope(
            run=deepcopy(loaded.run),
            scope_id=scope_id,
            performance=selected,
            observations=loaded.observations.scopes[scope_id],
            parameters=parameters,
            parameter_provenance=provenance,
            parameters_complete=complete,
            execution_config=config,
            research_evidence=loaded.research_evidence,
            parameter_sources=scope_parameter_sources(
                loaded.research_evidence, manifest, loaded.performance, scope_id
            )
            if loaded.research_evidence is not None and manifest is not None
            else (),
            execution_assumptions=loaded.research_evidence.assumptions_for(
                manifest.manifest_hash
            )
            if loaded.research_evidence is not None and manifest is not None
            else None,
        )

    def find_runs(
        self,
        strategy_id: str | None = None,
        parameters: dict[str, JsonValue] | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Find matching scalar scopes; unreadable runs remain explicit issues."""
        try:
            _validate_page(offset, limit)
            checked = finite_strategy_value({} if parameters is None else parameters)
            if not isinstance(checked, dict):
                raise ValueError("parameters must be an object")
        except (HistoryReadError, ValueError) as exc:
            return _query_error(exc)
        matches = []
        issues = []
        total = 0
        runs = sorted(
            self._store.list_runs(),
            key=lambda row: (str(row["created_at"]), str(row["run_id"])),
            reverse=True,
        )
        for run in runs:
            if strategy_id is not None and run["strategy_id"] != strategy_id:
                continue
            try:
                loaded = self.load_run(run["run_id"])
                for scope_id in loaded.observations.scopes:
                    total += 1
                    scope = self.select_scope(loaded, scope_id)
                    context = _context(scope)
                    if symbol is not None and context.get("symbol") != symbol:
                        continue
                    if timeframe is not None and context.get("timeframe") != timeframe:
                        continue
                    if checked and not scope.parameters_complete:
                        raise HistoryReadError(
                            "parameters_unavailable",
                            "Parameter filtering requires a verified manifest",
                        )
                    if not _subset(checked, scope.parameters):
                        continue
                    matches.append(
                        {
                            "run_id": run["run_id"],
                            "strategy_id": run["strategy_id"],
                            "revision_id": run.get("revision_id"),
                            "manifest_hash": run.get("manifest_hash"),
                            "created_at": run["created_at"],
                            "scope": scope_id,
                            "mode": scope.performance.mode,
                            "split": scope.performance.split,
                            "symbol": _optional_string(context.get("symbol")),
                            "timeframe": _optional_string(context.get("timeframe")),
                            "parameters": scope.parameters,
                            "parameter_provenance": scope.parameter_provenance,
                            "parameters_complete": scope.parameters_complete,
                            "selected": scope_id
                            == loaded.performance.selected_train_scope,
                            "bar_count": len(scope.observations.equity_curve),
                            "trade_count": len(scope.observations.trades),
                        }
                    )
            except HistoryReadError as exc:
                issues.append(
                    {"run_id": run["run_id"], "code": exc.code, "error": str(exc)}
                )
        return {
            **_page(total, len(matches), offset, limit),
            "runs": matches[offset : offset + limit],
            "complete": not issues,
            "issues": issues,
        }

    def get_run_trades(
        self,
        run_id: str,
        scope: str | None = None,
        offset: int = 0,
        limit: int = 50,
        status: str | None = None,
        direction: str | None = None,
        entry_start: str | None = None,
        entry_end: str | None = None,
    ) -> dict[str, Any]:
        """Page original trades by entry time; dates include the whole UTC day."""
        try:
            _validate_page(offset, limit)
            if status not in (None, "open", "closed"):
                raise HistoryReadError(
                    "invalid_history_query", "status must be open or closed"
                )
            if direction not in (None, "long", "short"):
                raise HistoryReadError(
                    "invalid_history_query", "direction must be long or short"
                )
            bounds = _bounds(entry_start, entry_end)
            loaded = self.load_scope(run_id, scope)
            observations = loaded.observations
            _require_filter_timestamps(observations, bounds)
            rows = []
            for trade_id, record in enumerate(observations.trades):
                row = _trade_row(trade_id, record, observations)
                if status is not None and row["status"] != status:
                    continue
                if direction is not None and row["direction"] != direction:
                    continue
                if not _in_bounds(row["entry_timestamp"], bounds):
                    continue
                rows.append(row)
            return {
                **_identity(loaded),
                **_page(len(observations.trades), len(rows), offset, limit),
                "trades": rows[offset : offset + limit],
                "date_filter_basis": "entry_timestamp",
            }
        except HistoryReadError as exc:
            return exc.response()

    def get_run_equity(
        self,
        run_id: str,
        scope: str | None = None,
        offset: int = 0,
        limit: int = 100,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """Page unmodified bar observations, without interpolation or replay."""
        try:
            _validate_page(offset, limit)
            bounds = _bounds(start, end)
            loaded = self.load_scope(run_id, scope)
            observations = loaded.observations
            _require_filter_timestamps(observations, bounds)
            rows = []
            for index, value in enumerate(observations.equity_curve):
                timestamp = _timestamp(observations, index)
                if not _in_bounds(timestamp, bounds):
                    continue
                rows.append(
                    {
                        "bar_index": index,
                        "timestamp": timestamp,
                        "equity": value,
                        "bar_return": observations.returns[index]
                        if observations.returns is not None
                        else None,
                    }
                )
            return {
                **_identity(loaded),
                **_page(len(observations.equity_curve), len(rows), offset, limit),
                "points": rows[offset : offset + limit],
                "init_cash": observations.init_cash,
                "equity_basis": "account_equity"
                if loaded.performance.mode == "signal"
                else "cumulative_pnl",
            }
        except HistoryReadError as exc:
            return exc.response()

    def get_run_executions(
        self,
        run_id: str,
        scope: str | None = None,
        offset: int = 0,
        limit: int = 50,
        status: str | None = None,
        side: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """Page original order attempts; unavailable history is never replayed."""
        try:
            _validate_page(offset, limit)
            if status not in (None, "filled", "ignored", "rejected"):
                raise HistoryReadError(
                    "invalid_history_query",
                    "status must be filled, ignored, or rejected",
                )
            if side not in (None, "buy", "sell"):
                raise HistoryReadError(
                    "invalid_history_query", "side must be buy or sell"
                )
            bounds = _bounds(start, end)
            loaded = self.load_scope(run_id, scope)
            observations = loaded.observations
            records = observations.execution_records
            rows = []
            if records is not None:
                _require_filter_timestamps(observations, bounds)
                for record in records:
                    if status is not None and record.status != status:
                        continue
                    if side is not None and record.side != side:
                        continue
                    timestamp = _timestamp(observations, record.bar_index)
                    if not _in_bounds(timestamp, bounds):
                        continue
                    rows.append(
                        {**record.model_dump(mode="json"), "timestamp": timestamp}
                    )
            return {
                **_ledger_identity(loaded, records),
                **_page(len(records or []), len(rows), offset, limit),
                "records": rows[offset : offset + limit],
                "valuation_basis": "requested_price_before_slippage",
                "timestamp_semantics": "bar_timestamp_not_exact_intrabar_fill",
            }
        except HistoryReadError as exc:
            return exc.response()

    def get_run_account_history(
        self,
        run_id: str,
        scope: str | None = None,
        offset: int = 0,
        limit: int = 100,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """Page saved close-marked balances, including bars without orders."""
        try:
            _validate_page(offset, limit)
            bounds = _bounds(start, end)
            loaded = self.load_scope(run_id, scope)
            observations = loaded.observations
            states = observations.account_history
            rows = []
            if states is not None:
                _require_filter_timestamps(observations, bounds)
                for state in states:
                    timestamp = _timestamp(observations, state.bar_index)
                    if not _in_bounds(timestamp, bounds):
                        continue
                    rows.append(
                        {**state.model_dump(mode="json"), "timestamp": timestamp}
                    )
            return {
                **_ledger_identity(loaded, states),
                **_page(len(states or []), len(rows), offset, limit),
                "states": rows[offset : offset + limit],
                "valuation_basis": "bar_close",
                "timestamp_semantics": "bar_timestamp_end_of_bar_state",
            }
        except HistoryReadError as exc:
            return exc.response()


def _identity(scope: LoadedHistoryScope) -> dict[str, Any]:
    return {
        "run_id": scope.run["run_id"],
        "scope": scope.scope_id,
        "manifest_hash": scope.run.get("manifest_hash"),
        "mode": scope.performance.mode,
        "parameters": deepcopy(scope.parameters),
        "parameter_provenance": scope.parameter_provenance,
        "parameters_complete": scope.parameters_complete,
        "calendar_timezone": "UTC",
    }


def _ledger_identity(scope: LoadedHistoryScope, records: object) -> dict[str, Any]:
    availability = (
        "available"
        if records is not None
        else "unsupported_volume_accounting"
        if scope.performance.mode == "volume"
        else "not_recorded"
    )
    return {
        **_identity(scope),
        "availability": availability,
        "accounting": "vectorbt_generic"
        if scope.performance.mode == "signal"
        else None,
        "init_cash": scope.observations.init_cash,
        "date_filter_basis": "timestamp",
    }


def _context(scope: LoadedHistoryScope) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if scope.execution_config is not None:
        context.update(scope.execution_config.get("backtest", {}))
    recorded = scope.performance.metadata.get("execution_context")
    if isinstance(recorded, dict):
        context.update(recorded)
    return context


def _subset(requested: dict[str, JsonValue], actual: dict[str, JsonValue]) -> bool:
    for name, value in requested.items():
        if name not in actual:
            return False
        candidate = actual[name]
        if isinstance(value, dict):
            if not isinstance(candidate, dict) or not _subset(value, candidate):
                return False
        elif not _same_value(value, candidate):
            return False
    return True


def _same_value(expected: JsonValue, actual: JsonValue) -> bool:
    if isinstance(expected, bool) != isinstance(actual, bool):
        return False
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(_same_value(a, b) for a, b in zip(expected, actual, strict=True))
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and expected.keys() == actual.keys()
            and all(_same_value(value, actual[key]) for key, value in expected.items())
        )
    return expected == actual


def _validate_page(offset: int, limit: int) -> None:
    if (
        type(offset) is not int
        or offset < 0
        or type(limit) is not int
        or not 1 <= limit <= 500
    ):
        raise HistoryReadError(
            "invalid_history_query",
            "offset must be >= 0 and limit must be between 1 and 500",
        )


def _page(total: int, matched: int, offset: int, limit: int) -> dict[str, Any]:
    return {
        "success": True,
        "offset": offset,
        "limit": limit,
        "total": total,
        "matched": matched,
        "next_offset": offset + limit if offset + limit < matched else None,
    }


def _parse_time(value: str, *, end: bool = False) -> pd.Timestamp:
    try:
        if not isinstance(value, str) or not re.match(
            r"^\d{4}-\d{2}-\d{2}(?:$|T| )", value
        ):
            raise ValueError("Expected an ISO calendar date or timestamp")
        moment = pd.Timestamp(value)
        if pd.isna(moment):
            raise ValueError("Missing date/time")
        if moment.tzinfo is None:
            moment = moment.tz_localize("UTC")
        if end and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            moment = moment + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return moment.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError) as exc:
        raise HistoryReadError(
            "invalid_history_query", f"Invalid ISO date/time: {value}"
        ) from exc


def _bounds(
    start: str | None, end: str | None
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    first = _parse_time(start) if start is not None else None
    last = _parse_time(end, end=True) if end is not None else None
    if first is not None and last is not None and first > last:
        raise HistoryReadError("invalid_history_query", "start must not be after end")
    return first, last


def _in_bounds(
    timestamp: str | None, bounds: tuple[pd.Timestamp | None, pd.Timestamp | None]
) -> bool:
    first, last = bounds
    if first is None and last is None:
        return True
    if timestamp is None:
        return False
    moment = _parse_time(timestamp)
    return (first is None or moment >= first) and (last is None or moment <= last)


def _require_filter_timestamps(
    observations: ScopeObservations,
    bounds: tuple[pd.Timestamp | None, pd.Timestamp | None],
) -> None:
    if any(bound is not None for bound in bounds) and observations.timestamps is None:
        raise HistoryReadError(
            "timestamps_unavailable", "Date filtering requires recorded timestamps"
        )


def _timestamp(observations: ScopeObservations, index: JsonValue) -> str | None:
    if observations.timestamps is None or type(index) is not int:
        return None
    if not 0 <= index < len(observations.timestamps):
        return None
    return _parse_time(observations.timestamps[index]).isoformat()


def _number(value: JsonValue) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _trade_row(
    trade_id: int, record: dict[str, JsonValue], observations: ScopeObservations
) -> dict[str, Any]:
    status = record.get("status")
    known_status = status if status in ("open", "closed") else None
    side = _number(record.get("direction"))
    direction = "long" if side == 1 else "short" if side == -1 else None
    marked = known_status == "open"
    closed = known_status == "closed"
    return {
        "trade_id": trade_id,
        "direction": direction,
        "status": known_status,
        "entry_timestamp": _timestamp(observations, record.get("entry_idx")),
        "exit_timestamp": _timestamp(observations, record.get("exit_idx"))
        if closed
        else None,
        "mark_timestamp": _timestamp(observations, record.get("exit_idx"))
        if marked
        else None,
        "entry_price": _number(record.get("entry_price")),
        "exit_price": _number(record.get("exit_price")) if closed else None,
        "mark_price": _number(record.get("exit_price")) if marked else None,
        **{
            key: _number(record.get(key))
            for key in (
                "size",
                "size_quote",
                "entry_fees",
                "exit_fees",
                "fee",
                "gross_pnl",
                "net_pnl",
            )
        },
        "record": deepcopy(record),
    }


def _query_error(error: HistoryReadError | ValueError) -> dict[str, Any]:
    if isinstance(error, HistoryReadError):
        return error.response()
    return HistoryReadError("invalid_history_query", str(error)).response()
