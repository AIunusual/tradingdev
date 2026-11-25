"""Statistical analysis of one validated saved run/scope.

This service reads immutable performance artifacts through
:class:`TradeHistoryService`; it never executes a strategy, downloads market
data, changes stored metrics, or rebuilds statistics from paged display output.
All figures are derived from the same verified snapshot, with the run's frozen
capital, mode, calendar and inference settings.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

import pandas as pd

from tradingdev.adapters.storage.filesystem import (
    WorkspacePaths,
    sha256_text,
)
from tradingdev.adapters.storage.sqlite import SQLiteStore, get_sqlite_store
from tradingdev.app.trade_history_service import HistoryReadError, TradeHistoryService
from tradingdev.domain.performance.sampling import bar_duration_ns
from tradingdev.domain.statistics.breakeven import summarize_breakeven
from tradingdev.domain.statistics.performance import (
    daily_excess_returns,
    summarize_performance,
)
from tradingdev.domain.statistics.sharpe import (
    MIN_BOOTSTRAP_OBSERVATIONS,
    IntervalUnavailable,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_interval_outcome,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tradingdev.app.trade_history_service import (
        LoadedHistoryRun,
        LoadedHistoryScope,
    )

ALGORITHM_VERSION = "statistics-analysis-v1"
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20261125
BOOTSTRAP_CONFIDENCE = 0.95
ARTIFACT_TYPE = "statistics_analysis"

_LIMITATIONS = (
    "Square-root annualisation does not remove serial dependence or compounding "
    "effects from Sharpe ratios.",
    "PSR uses the iid asymptotic variance; serial correlation can invalidate it.",
    "DSR additionally assumes independent trials and a verifiable search "
    "population; correlated trials may under- or over-correct the hurdle.",
    "Bootstrap intervals are descriptive resamples, not nominal-coverage "
    "guarantees; block-size sensitivity is not measured here.",
    "Missing calendar periods are never filled; inference is refused when "
    "timestamps do not follow the declared fixed bar cadence.",
    "Historical open positions keep their saved mark; no exit fill or cost is "
    "invented for them.",
)

_NEXT = {
    "fold_summary_unsupported": (
        "Select an original scalar run/scope from find_runs or get_run_metrics."
    ),
    "settings_unavailable": (
        "Use a run with a complete frozen settings record; do not apply current "
        "policy defaults to historical results."
    ),
    "missing_annualization": (
        "Use a saved run with periods_per_year recorded by its execution config."
    ),
    "missing_risk_free_rate": (
        "Use a saved run with its risk-free rate recorded by the metric producer."
    ),
    "missing_frequency": (
        "Use a saved run with its declared bar frequency in frozen settings."
    ),
    "timestamps_unavailable": (
        "Use a saved scope with recorded UTC timestamps for calendar analysis."
    ),
    "observation_gap_unsupported": (
        "Use a complete fixed-cadence saved run, or define a gap-aware inference "
        "method before interpreting this series."
    ),
    "bar_cadence_unknown": (
        "Use a run with a fixed, parseable bar frequency such as 1h or 1d."
    ),
    "not_applicable": (
        "Use a signal-mode account-equity scope for return-based inference; "
        "volume-mode cumulative P&L has no capital-relative return."
    ),
    "insufficient_observations": (
        "Save a longer run; the bootstrap requires at least "
        f"{MIN_BOOTSTRAP_OBSERVATIONS} observed periods."
    ),
    "undefined_statistic": (
        "Inspect the saved equity series; a zero-dispersion return sample has "
        "no Sharpe ratio."
    ),
    "no_valid_resample": (
        "Inspect the saved returns and try an explicit block size; no bootstrap "
        "draw produced a defined statistic."
    ),
    "not_selected_training_scope": (
        "Analyze the selected training scope; a test scope never inherits the "
        "training search hurdle."
    ),
    "search_evidence_incomplete": (
        "Use a published optimization with complete same-basis trial Sharpes and "
        "a verified raw attempt count."
    ),
    "trial_basis_differs": (
        "Recompute the analysis with one consistent set of mode, capital, "
        "frequency and annualization settings."
    ),
    "unsupported_return_std_ddof": (
        "Use the recorded ddof=1 return basis or define a matching inference method."
    ),
    "inference_error": (
        "Inspect the saved observations and settings; the statistical function "
        "rejected the derived inputs."
    ),
    "insufficient_time_range": (
        "Use a scope with at least two distinct observation timestamps."
    ),
    "invalid_equity": ("Use a saved scope with finite, aligned equity observations."),
}


@dataclass(frozen=True)
class _Settings:
    init_cash: float | None
    mode: str
    frequency: str
    periods_per_year: float | None
    risk_free_rate: float | None
    period_pnl_ddof: int
    return_std_ddof: int | None
    complete: bool
    missing_fields: tuple[str, ...]

    @property
    def capital_basis(self) -> str:
        return "account_equity" if self.mode == "signal" else "cumulative_pnl"

    def contract(self) -> dict[str, Any]:
        return {
            "capital_basis": self.capital_basis,
            "init_cash": self.init_cash,
            "mode": self.mode,
            "frequency": self.frequency,
            "periods_per_year": self.periods_per_year,
            "risk_free_rate": self.risk_free_rate,
            "period_pnl_ddof": self.period_pnl_ddof,
            "return_std_ddof": self.return_std_ddof,
            "missing_periods": "not_filled",
            "calendar_timezone": "UTC",
            "gap_policy": "exact_declared_bar_cadence",
            "terminal_close_policy": "saved_as_executed",
            "settings_complete": self.complete,
        }


class StatisticsService:
    """One bounded analysis entry point shared by MCP, CLI and artifacts."""

    def __init__(
        self,
        *,
        workspace: WorkspacePaths | None = None,
        store: SQLiteStore | None = None,
        history: TradeHistoryService | None = None,
    ) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._store = store or get_sqlite_store(self._workspace)
        self._history = history or TradeHistoryService(
            workspace=self._workspace, store=self._store
        )

    def analyze(
        self, run_id: str, scope: str | None = None, *, save_artifact: bool = False
    ) -> dict[str, Any]:
        """Analyze one saved run/scope from a single integrity-checked snapshot."""
        try:
            loaded = self._history.load_run(run_id)
            selected = self._history.select_scope(
                loaded,
                loaded.performance.default_scope if scope is None else scope,
            )
        except HistoryReadError as exc:
            if exc.code == "scope_has_no_observations":
                exc = HistoryReadError(
                    "fold_summary_unsupported",
                    "Fold summaries are descriptive aggregates without observations",
                    list(loaded.performance.scopes),
                )
            return _query_error(exc, run_id, scope)
        if selected.performance.kind != "backtest":
            return _query_error(
                HistoryReadError(
                    "fold_summary_unsupported",
                    "Fold summaries are descriptive aggregates without observations",
                    list(loaded.observations.scopes),
                ),
                run_id,
                scope,
            )
        source_hash = _source_hash(loaded, selected)
        source_label = f"{run_id}:{selected.scope_id}@{source_hash[:12]}"
        settings = _read_settings(selected)
        source = {
            "run_id": run_id,
            "scope": selected.scope_id,
            "kind": selected.performance.kind,
            "split": selected.performance.split,
            "mode": selected.performance.mode,
            "manifest_hash": loaded.performance.manifest_hash,
            "dataset_id": selected.run.get("dataset_id"),
            "source_hash": source_hash,
            "selected_train_scope": loaded.performance.selected_train_scope,
            "definition_ids": sorted(loaded.performance.definitions),
        }

        summary = self._summary(selected, settings, source_label)
        breakeven = self._breakeven(selected, settings, source_label)
        inference = self._inference(
            loaded, selected, settings, source_hash, source_label
        )
        payload: dict[str, Any] = {
            "success": True,
            "schema_version": 1,
            "algorithm_version": ALGORITHM_VERSION,
            "source": source,
            "settings": settings.contract(),
            "summary": summary,
            "breakeven": breakeven,
            "inference": inference,
            "limitations": list(_LIMITATIONS),
            "artifact": None,
        }
        if save_artifact:
            try:
                payload["artifact"] = self._publish_artifact(payload)
            except (OSError, ValueError) as exc:
                return _query_error(
                    HistoryReadError(
                        "statistics_artifact_failed",
                        f"Could not publish the derived artifact: {exc}",
                    ),
                    run_id,
                    scope,
                )
        return payload

    @staticmethod
    def _summary(
        scope: LoadedHistoryScope, settings: _Settings, source: str
    ) -> dict[str, Any]:
        problem = _summary_problem(scope, settings, source)
        if problem is not None:
            return _outcome("unavailable", unavailable=problem)
        observations = scope.observations
        assert settings.risk_free_rate is not None
        assert settings.frequency is not None
        assert observations.timestamps is not None
        try:
            result = summarize_performance(
                _equity_values(scope),
                observations.timestamps,
                observations.trades,
                init_cash=settings.init_cash,
                frequency=settings.frequency,
                periods_per_year=settings.periods_per_year,
                risk_free_rate=settings.risk_free_rate,
            )
        except ValueError as exc:
            return _outcome(
                "unavailable",
                unavailable=_unavailable("invalid_equity", source, str(exc)),
            )
        record: dict[str, Any] = {
            "start": result.start.isoformat(),
            "end": result.end.isoformat(),
            "n_bars": result.n_bars,
            "n_days": result.n_days,
            "n_months": result.n_months,
            "init_cash": result.init_cash,
            "frequency": result.frequency,
            "periods_per_year": result.periods_per_year,
            "risk_free_rate": result.risk_free_rate,
            "total_pnl": result.total_pnl,
            "observed_equity_change": result.observed_equity_change,
            "total_return": result.total_return,
            "max_drawdown": result.max_drawdown,
            "max_drawdown_amount": result.max_drawdown_amount,
            "daily_pnl_mean": result.daily_pnl_mean,
            "daily_pnl_std": result.daily_pnl_std,
            "monthly_pnl_mean": result.monthly_pnl_mean,
            "monthly_pnl_std": result.monthly_pnl_std,
            "sharpe_ratio": result.sharpe_ratio,
            "return_information_ratio": result.return_information_ratio,
            "closed_trades": _trade_record(result.closed_trades),
            "marked_trades": _trade_record(result.marked_trades),
            "unavailable": dict(result.unavailable),
        }
        if settings.return_std_ddof != 1:
            record["unavailable"]["sharpe_ratio"] = "unsupported_return_std_ddof"
            record["sharpe_ratio"] = None
        return _outcome("available", result=record)

    @staticmethod
    def _breakeven(
        scope: LoadedHistoryScope, settings: _Settings, source: str
    ) -> dict[str, Any]:
        observations = scope.observations
        if observations.timestamps is None or len(observations.timestamps) < 2:
            return _outcome(
                "unavailable",
                unavailable=_unavailable("insufficient_time_range", source),
            )
        try:
            result = summarize_breakeven(
                observations.trades,
                start=observations.timestamps[0],
                end=observations.timestamps[-1],
                capital=settings.init_cash,
            )
        except ValueError as exc:
            return _outcome(
                "unavailable",
                unavailable=_unavailable("invalid_equity", source, str(exc)),
            )
        return _outcome(
            "available",
            result={
                "total_trades": result.total_trades,
                "winning_trades": result.winning_trades,
                "losing_trades": result.losing_trades,
                "avg_win": result.avg_win,
                "avg_loss": result.avg_loss,
                "payoff_ratio": result.payoff_ratio,
                "breakeven_win_rate": result.breakeven_win_rate,
                "win_rate": result.win_rate,
                "win_rate_edge": result.win_rate_edge,
                "years": result.years,
                "traded_notional": result.traded_notional,
                "annual_traded_notional": result.annual_traded_notional,
                "annual_turnover": result.annual_turnover,
                "total_cost": result.total_cost,
                "annual_cost": result.annual_cost,
                "cost_per_turnover": result.cost_per_turnover,
                "annual_cost_drag": result.annual_cost_drag,
            },
        )

    def _inference(
        self,
        loaded: LoadedHistoryRun,
        scope: LoadedHistoryScope,
        settings: _Settings,
        source_hash: str,
        source: str,
    ) -> dict[str, Any]:
        returns, returns_problem = _daily_returns(scope, settings, source)
        config = {
            "algorithm_version": ALGORITHM_VERSION,
            "seed": BOOTSTRAP_SEED,
            "reps": BOOTSTRAP_REPS,
            "block_size": "auto",
            "effective_observations": len(returns),
            "frequency": "daily",
            "periods_per_year": settings.periods_per_year,
            "risk_free_rate": settings.risk_free_rate,
            "source_hash": source_hash,
            "scope": scope.scope_id,
            "source": source,
        }
        interval: dict[str, Any]
        psr: dict[str, Any]
        dsr: dict[str, Any]
        if returns_problem is not None:
            interval = _outcome(
                "unavailable",
                unavailable=returns_problem,
                method="stationary_bootstrap_percentile",
                config=config,
            )
            psr = _outcome(
                "unavailable",
                unavailable=returns_problem,
                method="normal_iid_psr",
                config=config,
            )
            dsr = _outcome(
                "unavailable",
                unavailable=returns_problem,
                method="deflated_sharpe",
                config=config,
            )
            return {
                "interval": interval,
                "psr": psr,
                "dsr": dsr,
                "limitations": list(_LIMITATIONS),
            }

        assert settings.periods_per_year is not None
        try:
            outcome = sharpe_interval_outcome(
                returns,
                periods_per_year=settings.periods_per_year,
                confidence=BOOTSTRAP_CONFIDENCE,
                reps=BOOTSTRAP_REPS,
                seed=BOOTSTRAP_SEED,
            )
        except ValueError as exc:
            interval = _outcome(
                "unavailable",
                unavailable=_unavailable("inference_error", source, str(exc)),
                method="stationary_bootstrap_percentile",
                config=config,
            )
        else:
            if isinstance(outcome, IntervalUnavailable):
                reason = str(outcome.reason)
                interval = _outcome(
                    "unavailable",
                    unavailable=_unavailable(reason, source, outcome.message),
                    method="stationary_bootstrap_percentile",
                    config=config,
                )
            else:
                interval = _outcome(
                    "available",
                    result={
                        "sharpe": outcome.sharpe,
                        "lower": outcome.lower,
                        "upper": outcome.upper,
                        "confidence": outcome.confidence,
                        "block_size": outcome.block_size,
                        "reps": outcome.reps,
                        "valid_reps": outcome.valid_reps,
                    },
                    method="stationary_bootstrap_percentile",
                    config=config,
                )

        psr_problem: dict[str, Any] | None
        try:
            probability = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
        except ValueError as exc:
            psr_problem = _unavailable("inference_error", source, str(exc))
        else:
            psr_problem = (
                None
                if probability is not None
                else _unavailable("undefined_statistic", source)
            )
        psr = (
            _outcome(
                "available",
                result={"psr": probability, "benchmark_sr": 0.0},
                method="normal_iid_psr",
                config=config,
            )
            if psr_problem is None
            else _outcome(
                "unavailable",
                unavailable=psr_problem,
                method="normal_iid_psr",
                config=config,
            )
        )

        dsr_problem, dsr_result = self._dsr(loaded, scope, settings, returns, source)
        dsr = _outcome(
            "unavailable" if dsr_problem is not None else "available",
            result=dsr_result,
            unavailable=dsr_problem,
            method="deflated_sharpe",
            config=config,
        )
        return {
            "interval": interval,
            "psr": psr,
            "dsr": dsr,
            "limitations": list(_LIMITATIONS),
        }

    def _dsr(
        self,
        loaded: LoadedHistoryRun,
        scope: LoadedHistoryScope,
        settings: _Settings,
        returns: pd.Series[float],
        source: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        bundle = loaded.performance
        selected = bundle.selected_train_scope
        if selected is None or scope.scope_id != selected:
            return (
                _unavailable("not_selected_training_scope", source),
                None,
            )
        selection = bundle.optimization_selection
        if selection is None:
            return _unavailable(
                "search_evidence_incomplete", source, "no selection record"
            ), None
        trial_ids = sorted(
            key
            for key, value in bundle.scopes.items()
            if key.startswith("trial/")
            and key.endswith("/train")
            and value.kind == "backtest"
        )
        if len(trial_ids) != selection.raw_attempted:
            return (
                _unavailable(
                    "search_evidence_incomplete",
                    source,
                    f"{len(trial_ids)} trial scopes for "
                    f"{selection.raw_attempted} attempts",
                ),
                None,
            )
        assert settings.periods_per_year is not None
        assert settings.risk_free_rate is not None
        trial_sharpes: list[float] = []
        for trial_id in trial_ids:
            trial = self._history.select_scope(loaded, trial_id)
            if not _same_trial_basis(trial, settings):
                return _unavailable("trial_basis_differs", source, trial_id), None
            observations = trial.observations
            assert observations.timestamps is not None
            try:
                summary = summarize_performance(
                    _equity_values(trial),
                    observations.timestamps,
                    observations.trades,
                    init_cash=settings.init_cash,
                    frequency=settings.frequency,
                    periods_per_year=settings.periods_per_year,
                    risk_free_rate=settings.risk_free_rate,
                )
            except ValueError as exc:
                return _unavailable(
                    "trial_basis_differs", source, f"{trial_id}: {exc}"
                ), None
            if summary.sharpe_ratio is None or not math.isfinite(summary.sharpe_ratio):
                return (
                    _unavailable(
                        "search_evidence_incomplete",
                        source,
                        f"{trial_id} has no Sharpe",
                    ),
                    None,
                )
            trial_sharpes.append(summary.sharpe_ratio)
        try:
            result = deflated_sharpe_ratio(
                returns,
                trial_sharpes,
                periods_per_year=settings.periods_per_year,
                n_trials=selection.raw_attempted,
            )
        except ValueError as exc:
            return _unavailable("inference_error", source, str(exc)), None
        if result is None:
            return _unavailable("undefined_statistic", source), None
        return None, {
            "dsr": result.dsr,
            "psr_vs_zero": result.psr_vs_zero,
            "hurdle_annualised": result.hurdle_annualised,
            "n_trials": result.n_trials,
            "trial_sharpe_std_annualised": result.trial_sharpe_std_annualised,
            "trial_scope_count": len(trial_ids),
            "search_population": "selected_training_scope",
        }

    def _publish_artifact(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Publish a content-addressed, immutable Markdown analysis artifact."""
        document = _render_artifact(payload)
        digest = sha256_text(document)
        run_id = str(payload["source"]["run_id"])
        artifact_id = f"{run_id}:{ARTIFACT_TYPE}:{digest[:16]}"
        path = self._workspace.runs / run_id / ARTIFACT_TYPE / f"{digest[:16]}.md"
        if path.is_symlink() or not path.resolve().is_relative_to(self._workspace.root):
            raise ValueError("Analysis artifact path leaves the workspace")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(document)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_text(encoding="utf-8") != document:
                    raise ValueError("Existing analysis artifact differs") from None
            finally:
                temporary.unlink(missing_ok=True)
        existing = self._store.get_artifact(artifact_id)
        if existing is not None and (
            existing["path"] != str(path) or existing["sha256"] != digest
        ):
            raise ValueError("Registered analysis artifact identity differs")
        self._store.create_artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            artifact_type=ARTIFACT_TYPE,
            path=path,
            sha256=digest,
            metadata={
                "algorithm_version": ALGORITHM_VERSION,
                "source_hash": payload["source"]["source_hash"],
                "scope": payload["source"]["scope"],
            },
        )
        return {
            "artifact_id": artifact_id,
            "artifact_type": ARTIFACT_TYPE,
            "sha256": digest,
            "path": str(path),
        }


def _read_settings(scope: LoadedHistoryScope) -> _Settings:
    """Read the producer's frozen settings; never substitute policy defaults."""
    metadata = scope.performance.metadata
    raw = metadata.get("settings")
    settings: dict[str, JsonValue] = raw if isinstance(raw, dict) else {}
    missing = [key for key in ("frequency", "risk_free_rate") if key not in settings]
    periods_raw = settings.get("periods_per_year")
    periods = _finite_or_none(periods_raw)
    if "periods_per_year" not in settings or (
        periods_raw is not None and (periods is None or periods <= 0)
    ):
        periods = None
        missing.append("periods_per_year")
    risk_free = _finite_or_none(settings.get("risk_free_rate"))
    frequency = settings.get("frequency")
    if not isinstance(frequency, str) or not frequency:
        frequency = ""
    ddof = settings.get("return_std_ddof")
    return _Settings(
        init_cash=scope.observations.init_cash,
        mode=scope.performance.mode,
        frequency=frequency,
        periods_per_year=periods,
        risk_free_rate=risk_free,
        period_pnl_ddof=0,
        return_std_ddof=ddof if type(ddof) is int else None,
        complete=not missing,
        missing_fields=tuple(missing),
    )


def _summary_problem(
    scope: LoadedHistoryScope, settings: _Settings, source: str
) -> dict[str, Any] | None:
    observations = scope.observations
    if {"risk_free_rate", "frequency"} & set(settings.missing_fields):
        fields = ", ".join(settings.missing_fields)
        reason = (
            "missing_risk_free_rate"
            if "risk_free_rate" in settings.missing_fields
            else "missing_frequency"
        )
        return _unavailable(reason, source, f"missing frozen settings: {fields}")
    if observations.timestamps is None or not observations.timestamps:
        return _unavailable("timestamps_unavailable", source)
    if settings.mode == "signal" and (
        settings.init_cash is None
        or settings.init_cash <= 0
        or not math.isfinite(settings.init_cash)
    ):
        return _unavailable(
            "settings_unavailable", source, "signal mode needs init_cash"
        )
    return None


def _daily_returns(
    scope: LoadedHistoryScope, settings: _Settings, source: str
) -> tuple[pd.Series[float], dict[str, Any] | None]:
    problem = _summary_problem(scope, settings, source)
    if problem is not None:
        return pd.Series(dtype="float64"), problem
    assert settings.frequency is not None
    assert settings.risk_free_rate is not None
    if settings.mode != "signal" or settings.init_cash is None:
        return pd.Series(dtype="float64"), _unavailable("not_applicable", source)
    if settings.periods_per_year is None:
        return pd.Series(dtype="float64"), _unavailable("missing_annualization", source)
    if settings.return_std_ddof != 1:
        return pd.Series(dtype="float64"), _unavailable(
            "unsupported_return_std_ddof", source
        )
    observations = scope.observations
    assert observations.timestamps is not None
    gap = _gap_reason(observations.timestamps, settings.frequency)
    if gap is not None:
        return pd.Series(dtype="float64"), _unavailable(gap, source)
    try:
        returns = daily_excess_returns(
            _equity_values(scope),
            observations.timestamps,
            init_cash=settings.init_cash,
            periods_per_year=settings.periods_per_year,
            risk_free_rate=settings.risk_free_rate,
        )
    except ValueError as exc:
        return pd.Series(dtype="float64"), _unavailable(
            "invalid_equity", source, str(exc)
        )
    if returns.empty:
        return returns, _unavailable("insufficient_observations", source)
    return returns, None


def _gap_reason(timestamps: list[str], frequency: str) -> str | None:
    """Refuse to turn missing bars into adjacent bootstrap observations."""
    duration = bar_duration_ns(frequency)
    if duration is None:
        return "bar_cadence_unknown"
    try:
        parsed = pd.to_datetime(timestamps, utc=True, format="ISO8601")
    except TypeError, ValueError:
        return "timestamps_unavailable"
    if len(parsed) < 2:
        return None
    # pandas 3 preserves the input resolution; compare on a fixed ns basis.
    parsed = parsed.as_unit("ns")
    deltas = parsed[1:].astype("int64") - parsed[:-1].astype("int64")
    return None if bool((deltas == duration).all()) else "observation_gap_unsupported"


def _same_trial_basis(trial: LoadedHistoryScope, settings: _Settings) -> bool:
    if (
        trial.performance.mode != settings.mode
        or trial.observations.init_cash != settings.init_cash
    ):
        return False
    raw = trial.performance.metadata.get("settings")
    trial_settings = raw if isinstance(raw, dict) else {}
    return (
        trial_settings.get("frequency") == settings.frequency
        and trial_settings.get("periods_per_year") == settings.periods_per_year
        and trial_settings.get("risk_free_rate") == settings.risk_free_rate
    )


def _source_hash(loaded: LoadedHistoryRun, scope: LoadedHistoryScope) -> str:
    """Hash the whole validated snapshot, including DSR trial evidence."""
    payload = {
        "run": {
            key: loaded.run.get(key)
            for key in (
                "run_id",
                "job_id",
                "strategy_id",
                "revision_id",
                "manifest_hash",
                "dataset_id",
                "source_hash",
                "config_hash",
                "created_at",
            )
        },
        "bundle": {
            "default_scope": loaded.performance.default_scope,
            "selected_train_scope": loaded.performance.selected_train_scope,
            "optimization_selection": (
                loaded.performance.optimization_selection.model_dump(mode="json")
                if loaded.performance.optimization_selection is not None
                else None
            ),
            "definitions": {
                key: value.model_dump(mode="json")
                for key, value in loaded.performance.definitions.items()
            },
        },
        "selected_scope": scope.scope_id,
        "scopes": {
            key: value.model_dump(mode="json")
            for key, value in loaded.performance.scopes.items()
        },
        "observations": {
            key: value.model_dump(mode="json")
            for key, value in loaded.observations.scopes.items()
        },
        "parameters": scope.parameters,
        "execution_config": scope.execution_config,
    }
    digest = hashlib.sha256()
    digest.update(_canonical(payload))
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _render_artifact(payload: dict[str, Any]) -> str:
    """Human-readable derived artifact with settings, limits, results and source."""
    body = dict(payload)
    body["artifact"] = None
    encoded = json.dumps(body, ensure_ascii=False, allow_nan=False, indent=2)
    return (
        "# TradingDev statistical analysis\n\n"
        f"Algorithm: `{payload['algorithm_version']}`. "
        "This is a derived, immutable analysis of one validated saved run/scope. "
        "It does not rerun a strategy, download data, or change the source "
        "performance bundle or any older report.\n\n"
        f"Source: `{payload['source']['source_hash']}` "
        f"({payload['source']['run_id']}#{payload['source']['scope']}).\n\n"
        "## Fixed settings and limits\n\n"
        + "\n".join(f"- {line}" for line in payload["limitations"])
        + "\n\n## Result\n\n```json\n"
        + encoded
        + "\n```\n"
    )


def _trade_record(summary: Any) -> dict[str, Any]:
    return {
        "total_trades": summary.total_trades,
        "winning_trades": summary.winning_trades,
        "losing_trades": summary.losing_trades,
        "net_pnl": summary.net_pnl,
        "win_rate": summary.win_rate,
        "profit_factor": summary.profit_factor,
        "unavailable": dict(summary.unavailable),
    }


def _equity_values(scope: LoadedHistoryScope) -> list[float]:
    values: list[float] = []
    for position, value in enumerate(scope.observations.equity_curve):
        if value is None or not math.isfinite(value):
            raise ValueError(f"equity observation {position} is not finite")
        values.append(value)
    return values


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _unavailable(reason: str, source: str, detail: str | None = None) -> dict[str, Any]:
    return {
        "reason": reason,
        "next_step": _NEXT.get(reason, "Inspect the saved run/scope and settings."),
        "source": source,
        "detail": detail,
    }


def _outcome(
    status: str,
    *,
    result: dict[str, Any] | None = None,
    unavailable: dict[str, Any] | None = None,
    method: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": status, "result": result}
    if method is not None:
        payload["method"] = method
    if config is not None:
        payload["config"] = config
    if unavailable is not None:
        payload["unavailable"] = unavailable
    return payload


def _query_error(
    exc: HistoryReadError, run_id: str, scope: str | None
) -> dict[str, Any]:
    problem = _unavailable(exc.code, f"{run_id}:{scope or 'default'}")
    return {
        "success": False,
        "code": exc.code,
        "error": str(exc),
        "run_id": run_id,
        "scope": scope,
        "available_scopes": list(exc.available_scopes),
        "unavailable": problem,
    }
