"""Paired comparison of two already saved observation series.

One call freezes both source snapshots, verifies their identities, aligns them
on their exact shared UTC timestamps, and hands the paired values to the pure
stationary-bootstrap function in ``domain.statistics.comparison``. The service
never runs a strategy, trains a model, downloads data, reads a local pickle,
or rebuilds an old series from a later model. Derived Markdown is immutable
and content-addressed; source artifacts are not modified.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_text
from tradingdev.adapters.storage.sqlite import SQLiteStore, get_sqlite_store
from tradingdev.app.contracts.paired_comparison import (
    MAX_EXCLUDED_IN_RESPONSE,
    MAX_MISSING_IN_RESPONSE,
    MAX_REPS,
    MIN_REPS,
    ComparisonMode,
    SeriesSelection,
)
from tradingdev.app.trade_history_service import HistoryReadError, TradeHistoryService
from tradingdev.domain.statistics.alignment import (
    MaskedPairAlignment,
    align_masked_pair,
    is_finite_number,
)
from tradingdev.domain.statistics.benchmark import buy_and_hold_returns
from tradingdev.domain.statistics.comparison import difference_interval_outcome
from tradingdev.domain.statistics.sharpe import (
    MIN_BOOTSTRAP_OBSERVATIONS,
    IntervalUnavailable,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tradingdev.app.trade_history_service import (
        LoadedHistoryRun,
        LoadedHistoryScope,
    )

ALGORITHM_VERSION = "paired-comparison-v1"
SCHEMA_VERSION = 1
DEFAULT_REPS = 1000
DEFAULT_SEED = 20261127
DEFAULT_CONFIDENCE = 0.95
ARTIFACT_TYPE = "paired_comparison"
HTML_ARTIFACT_TYPE = "paired_comparison_html"
RAW_PRICE_ARTIFACT_TYPE = "raw_price_series"

_LIMITATIONS = (
    "Only one pair is analyzed; this is not a multiple-comparison correction "
    "and does not repair a previously searched family.",
    "Bootstrap intervals are paired descriptive resamples, not nominal "
    "coverage guarantees; signficance is conditional on the selected pair.",
    "Missing timestamps are never filled, and gaps are never compressed into "
    "a contiguous calendar before resampling.",
    "A descriptive comparison reports p as a percentile label only; it does "
    "not support a significance claim.",
    "A different dataset or model ID alone does not establish holdout status "
    "or the absence of tuning.",
    "Signal-simulation returns carry no execution costs and are never "
    "treated as cost-bearing trade performance.",
    "A zero-cost price benchmark is unavailable unless a verified raw price "
    "artifact is present; a saved net benchmark cannot be reversed into closes.",
    "Historical open positions keep their saved mark; no exit fill is invented.",
)

_NEXT = {
    "no_common_timestamps": (
        "Choose two sources with at least one shared UTC observation date."
    ),
    "insufficient_observations": (
        f"Use longer saved series; the paired interval needs at least "
        f"{MIN_BOOTSTRAP_OBSERVATIONS} jointly valid observations."
    ),
    "no_valid_observations": (
        "Choose series whose shared dates have jointly valid observations; "
        "warm-up values are kept at their original coordinates but are not paired."
    ),
    "frequency_mismatch": (
        "Choose two sources with the same declared frequency and return definition."
    ),
    "incompatible_definitions": (
        "Do not mix IC with returns, cost-free signal simulation with trade "
        "performance, or different frozen settings in one comparison."
    ),
    "cadence_gap_unsupported": (
        "Use a complete exact UTC daily or hourly pair for inference; a "
        "gap-aware resampler is not part of this first version."
    ),
    "source_qualification_unknown": (
        "Use saved sources with an explicit producer and evaluation identity."
    ),
    "zero_cost_prices_unavailable": (
        "This workspace has no integrity-verified raw price artifact; use the "
        "saved product benchmark instead. No price is downloaded or invented."
    ),
    "undefined_statistic": (
        "Inspect the paired values; a zero-dispersion sample has no defined mean "
        "difference interval."
    ),
    "no_valid_resample": (
        "Inspect the series or set an explicit block size; no bootstrap draw "
        "produced a defined statistic."
    ),
    "comparison_error": ("Inspect the saved inputs and comparison settings."),
}


@dataclass(frozen=True)
class _LoadedSource:
    series: pd.Series
    label: str
    identity: dict[str, Any]
    settings: dict[str, JsonValue]
    cadence: str
    definition: str
    cost_basis: str
    claim_scope: str


@dataclass(frozen=True)
class _LoadFailure:
    reason: str
    detail: str


class PairedComparisonError(ValueError):
    """An invalid caller request, distinct from a data-driven absence."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PairedComparisonService:
    """One shared entry point for MCP and any future CLI consumer."""

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

    def compare(
        self,
        a: SeriesSelection | dict[str, Any],
        b: SeriesSelection | dict[str, Any],
        *,
        mode: ComparisonMode = "descriptive",
        reps: int = DEFAULT_REPS,
        seed: int = DEFAULT_SEED,
        block_size: int | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        save_artifact: bool = False,
    ) -> dict[str, Any]:
        """Compare two saved series without changing either source."""
        left = (
            a if isinstance(a, SeriesSelection) else SeriesSelection.model_validate(a)
        )
        right = (
            b if isinstance(b, SeriesSelection) else SeriesSelection.model_validate(b)
        )
        if mode not in {"descriptive", "inferential"}:
            raise PairedComparisonError(
                "invalid_comparison_request", f"unknown comparison mode: {mode}"
            )
        if type(reps) is not int or not MIN_REPS <= reps <= MAX_REPS:
            raise PairedComparisonError(
                "invalid_comparison_request",
                f"reps must be an integer in [{MIN_REPS}, {MAX_REPS}]",
            )
        if type(seed) is not int or seed < 0:
            raise PairedComparisonError(
                "invalid_comparison_request", "seed must be a non-negative integer"
            )
        if block_size is not None and (type(block_size) is not int or block_size < 1):
            raise PairedComparisonError(
                "invalid_comparison_request", "block_size must be a positive integer"
            )
        if not isinstance(confidence, float) or not 0.0 < confidence < 1.0:
            raise PairedComparisonError(
                "invalid_comparison_request", "confidence must be between 0 and 1"
            )

        # Both selections are resolved from immutable snapshots held within
        # this call. Two scopes of one run share one load, so no query can
        # observe a different generation halfway through the analysis.
        snapshots: dict[str, LoadedHistoryRun] = {}
        left_loaded = self._load(left, snapshots)
        right_loaded = self._load(right, snapshots)
        if isinstance(left_loaded, _LoadFailure):
            return self._unavailable_payload(
                left_loaded,
                mode,
                [_unavailable_identity(left), _unavailable_identity(right)],
                [],
            )
        if isinstance(right_loaded, _LoadFailure):
            return self._unavailable_payload(
                right_loaded,
                mode,
                [_unavailable_identity(left), _unavailable_identity(right)],
                [],
            )

        problem = self._incompatible(left_loaded, right_loaded)
        if problem is not None:
            return self._unavailable_payload(
                _LoadFailure(*problem),
                mode,
                [left_loaded.identity, right_loaded.identity],
                [],
            )

        cadence = (
            left_loaded.cadence
            if left_loaded.cadence == right_loaded.cadence
            else "unsupported"
        )
        try:
            aligned = align_masked_pair(
                left_loaded.series, right_loaded.series, cadence=cadence
            )
        except ValueError as exc:
            return self._unavailable_payload(
                _LoadFailure("comparison_error", str(exc)),
                mode,
                [left_loaded.identity, right_loaded.identity],
                [],
            )
        if aligned.overlap == 0:
            return self._unavailable_payload(
                _LoadFailure(
                    "no_common_timestamps", "the two series share no UTC date"
                ),
                mode,
                [left_loaded.identity, right_loaded.identity],
                [],
                aligned,
            )
        if aligned.valid_count == 0:
            return self._unavailable_payload(
                _LoadFailure(
                    "no_valid_observations", "no paired value is jointly valid"
                ),
                mode,
                [left_loaded.identity, right_loaded.identity],
                [],
                aligned,
            )
        if aligned.valid_count < MIN_BOOTSTRAP_OBSERVATIONS:
            return self._unavailable_payload(
                _LoadFailure(
                    "insufficient_observations",
                    f"{aligned.valid_count} jointly valid observations",
                ),
                mode,
                [left_loaded.identity, right_loaded.identity],
                [],
                aligned,
            )
        interior_gaps = sum(aligned.gap_excluded.values())
        if mode == "inferential" and (
            not aligned.cadence_complete or interior_gaps > 0
        ):
            return self._unavailable_payload(
                _LoadFailure(
                    "cadence_gap_unsupported",
                    f"{len(aligned.missing_timestamps)} missing calendar periods, "
                    f"{interior_gaps} interior invalid observations",
                ),
                mode,
                [left_loaded.identity, right_loaded.identity],
                [],
                aligned,
            )
        if mode == "inferential" and (
            left_loaded.identity.get("producer_source") in {None, "", "unknown"}
            or right_loaded.identity.get("producer_source") in {None, "", "unknown"}
        ):
            return self._unavailable_payload(
                _LoadFailure(
                    "source_qualification_unknown",
                    "one side has no recorded producer",
                ),
                mode,
                [left_loaded.identity, right_loaded.identity],
                [],
                aligned,
            )

        pair_a = aligned.paired["a"]
        pair_b = aligned.paired["b"]
        try:
            outcome = difference_interval_outcome(
                pair_a,
                pair_b,
                np.mean,
                reps=reps,
                confidence=confidence,
                seed=seed,
                block_size=block_size,
            )
        except ValueError as exc:
            return self._unavailable_payload(
                _LoadFailure("comparison_error", str(exc)),
                mode,
                [left_loaded.identity, right_loaded.identity],
                [],
                aligned,
            )
        if isinstance(outcome, IntervalUnavailable):
            reason = str(outcome.reason)
            return self._unavailable_payload(
                _LoadFailure(reason, f"{outcome.observations} observations"),
                mode,
                [left_loaded.identity, right_loaded.identity],
                [],
                aligned,
            )

        claim_scope = (
            "verified_out_of_sample"
            if left_loaded.claim_scope == "verified_out_of_sample"
            and right_loaded.claim_scope == "verified_out_of_sample"
            else "in_sample_or_unknown"
        )
        payload: dict[str, Any] = {
            "success": True,
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "mode": mode,
            "statistic": "mean",
            "difference_direction": "a_minus_b",
            "claim_scope": claim_scope,
            "qualification": {
                "both_snapshots_verified": True,
                "shared_index": True,
                "calendar_complete": aligned.cadence_complete,
                "definitions_compatible": True,
                "p_value_interpretation": (
                    "conditional_percentile"
                    if mode == "inferential"
                    else "descriptive_only"
                ),
                "multiple_comparison_correction": False,
            },
            "status": "available",
            "result": {
                "estimate": outcome.difference,
                "estimate_a": outcome.estimate_a,
                "estimate_b": outcome.estimate_b,
                "lower": outcome.lower,
                "upper": outcome.upper,
                "p_value": outcome.p_value,
                "p_value_interpretation": (
                    "conditional_percentile"
                    if mode == "inferential"
                    else "descriptive_only"
                ),
                "confidence": outcome.confidence,
                "reps": outcome.reps,
                "valid_reps": outcome.valid_reps,
                "seed": seed,
                "block_size": outcome.block_size,
                "correlation": outcome.correlation,
            },
            "unavailable": None,
            "alignment": self._alignment(aligned),
            "sources": [left_loaded.identity, right_loaded.identity],
            "limitations": list(_LIMITATIONS),
            "artifact": None,
            "html_artifact": None,
        }
        if save_artifact:
            try:
                audit = {
                    "valid_mask": list(aligned.valid_mask),
                    "excluded": [
                        {
                            "timestamp": item.timestamp,
                            "absent_from": item.absent_from,
                            "reason": item.reason,
                        }
                        for item in aligned.excluded
                    ],
                    "missing_timestamps": list(aligned.missing_timestamps),
                }
                markdown, html = self._publish_artifacts(payload, audit)
                payload["artifact"] = markdown
                payload["html_artifact"] = html
            except Exception as exc:  # noqa: BLE001 - publication is an operation boundary
                raise PairedComparisonError(
                    "artifact_publication_failed", f"artifact failed: {exc}"
                ) from exc
        return payload

    # -- source resolution -------------------------------------------------

    def _run_snapshot(
        self, run_id: str, snapshots: dict[str, LoadedHistoryRun]
    ) -> LoadedHistoryRun:
        if run_id not in snapshots:
            snapshots[run_id] = self._history.load_run(run_id)
        return snapshots[run_id]

    def _load(
        self,
        selection: SeriesSelection,
        snapshots: dict[str, LoadedHistoryRun],
    ) -> _LoadedSource | _LoadFailure:
        try:
            if selection.kind == "run":
                return self._run_source(selection, snapshots)
            if selection.kind == "benchmark":
                return self._benchmark_source(selection, snapshots)
            return self._signal_source(selection)
        except HistoryReadError as exc:
            return _LoadFailure(exc.code, str(exc))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return _LoadFailure("comparison_error", str(exc))

    def _run_source(
        self, selection: SeriesSelection, snapshots: dict[str, LoadedHistoryRun]
    ) -> _LoadedSource | _LoadFailure:
        assert selection.run_id is not None
        loaded = self._run_snapshot(selection.run_id, snapshots)
        scope = self._history.select_scope(
            loaded, selection.scope or loaded.performance.default_scope
        )
        if scope.performance.kind != "backtest":
            raise HistoryReadError(
                "scope_has_no_observations",
                "fold summaries have no observation series",
                list(loaded.observations.scopes),
            )
        if scope.performance.mode != "signal":
            return _LoadFailure(
                "not_applicable",
                "volume-mode cumulative P&L has no capital-relative return",
            )
        values = _run_returns(scope)
        if values is None:
            return _LoadFailure(
                "not_applicable", "saved observations cannot reconstruct returns"
            )
        timestamps = scope.observations.timestamps
        if timestamps is None or len(timestamps) != len(values):
            return _LoadFailure(
                "timestamps_unavailable", "saved observations have no aligned UTC times"
            )
        series = pd.Series(
            values,
            index=pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)),
            name="run_return",
        )
        settings = _run_settings(scope)
        identity = _run_identity(loaded, scope, settings)
        return _LoadedSource(
            series=series,
            label=identity["label"],
            identity=identity,
            settings=settings,
            cadence=str(settings.get("frequency", "unsupported")),
            definition="cost_bearing_trade_return",
            cost_basis="saved_execution_with_costs",
            claim_scope=_run_claim_scope(scope),
        )

    def _benchmark_source(
        self, selection: SeriesSelection, snapshots: dict[str, LoadedHistoryRun]
    ) -> _LoadedSource | _LoadFailure:
        assert selection.run_id is not None
        loaded = self._run_snapshot(selection.run_id, snapshots)
        scope_id = selection.scope or loaded.performance.default_scope
        if selection.benchmark_mode == "zero_cost":
            return self._zero_cost_source(selection)
        scope = self._history.select_scope(loaded, scope_id)
        benchmark = loaded.performance.benchmarks.get(scope_id)
        if benchmark is None or benchmark.status != "available":
            reason = (
                benchmark.reason if benchmark is not None else "benchmark_unavailable"
            )
            return _LoadFailure(str(reason), "saved product benchmark is unavailable")
        if benchmark.timestamps is None or benchmark.equity_curve is None:
            return _LoadFailure(
                "benchmark_unavailable", "saved product benchmark has no series"
            )
        if benchmark.returns is not None:
            values = [float(v) if v is not None else np.nan for v in benchmark.returns]
        else:
            if not benchmark.init_cash or benchmark.init_cash <= 0:
                return _LoadFailure(
                    "benchmark_unavailable",
                    "benchmark has no capital basis for returns",
                )
            values = _returns_from_equity(benchmark.equity_curve, benchmark.init_cash)
        if len(values) != len(benchmark.timestamps):
            return _LoadFailure(
                "benchmark_unavailable", "benchmark series lengths differ"
            )
        series = pd.Series(
            values,
            index=pd.DatetimeIndex(pd.to_datetime(benchmark.timestamps, utc=True)),
            name="benchmark_return",
        )
        settings = _run_settings(scope)
        identity = _run_identity(loaded, scope, settings)
        identity.update(
            {
                "kind": "benchmark",
                "label": f"{selection.run_id}#{scope_id}:product_benchmark",
                "benchmark_definition": (
                    benchmark.definition.id if benchmark.definition else None
                ),
                "benchmark_costs": json.dumps(
                    {"fees": benchmark.fees, "slippage": benchmark.slippage},
                    sort_keys=True,
                ),
            }
        )
        return _LoadedSource(
            series=series,
            label=identity["label"],
            identity=identity,
            settings=settings,
            cadence=str(settings.get("frequency", "unsupported")),
            definition="cost_bearing_benchmark_return",
            cost_basis="saved_product_benchmark_with_costs",
            claim_scope="in_sample_or_unknown",
        )

    def _zero_cost_source(
        self, selection: SeriesSelection
    ) -> _LoadedSource | _LoadFailure:
        artifact_id = selection.price_artifact_id
        if not artifact_id:
            return _LoadFailure(
                "zero_cost_prices_unavailable", "no price artifact was selected"
            )
        record = self._store.get_artifact(artifact_id)
        if record is None or record.get("artifact_type") != RAW_PRICE_ARTIFACT_TYPE:
            return _LoadFailure(
                "zero_cost_prices_unavailable",
                "no registered verified raw price artifact",
            )
        path = Path(str(record.get("path", "")))
        if not path.is_file() or not path.resolve().is_relative_to(
            self._workspace.root
        ):
            return _LoadFailure(
                "zero_cost_prices_unavailable", "price artifact file is unavailable"
            )
        raw_bytes = path.read_bytes()
        sha = hashlib.sha256(raw_bytes).hexdigest()
        if sha != record.get("sha256"):
            return _LoadFailure(
                "zero_cost_prices_unavailable", "price artifact hash differs"
            )
        try:
            raw = json.loads(raw_bytes.decode("utf-8"))
            prices = raw["closes"]
            timestamps = raw["timestamps"]
            source = raw["source"]
            if not isinstance(source, str) or not source:
                raise ValueError("source is absent")
            series = buy_and_hold_returns(
                prices, timestamps=timestamps, frequency="daily"
            )
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return _LoadFailure(
                "zero_cost_prices_unavailable", f"unverifiable price series: {exc}"
            )
        cadence = "1d"
        identity = {
            "kind": "benchmark",
            "label": f"zero_cost:{artifact_id}",
            "artifact_id": artifact_id,
            "source_hash": sha,
            "benchmark_definition": "zero_cost_buy_and_hold",
            "benchmark_costs": "fees=0,slippage=0",
            "cost_basis": "zero_cost_price",
            "producer_source": "verified_raw_price_artifact",
            "settings": {"start": series.index[0].isoformat(), "costs": 0},
        }
        return _LoadedSource(
            series=series,
            label=str(identity["label"]),
            identity=identity,
            settings={"start": series.index[0].isoformat(), "costs": 0},
            cadence=cadence,
            definition="zero_cost_price_return",
            cost_basis="zero_cost_price",
            claim_scope="in_sample_or_unknown",
        )

    def _signal_source(
        self, selection: SeriesSelection
    ) -> _LoadedSource | _LoadFailure:
        from tradingdev.adapters.storage.evaluation_series import EvaluationSeriesStore

        assert selection.artifact_id is not None and selection.column is not None
        store = EvaluationSeriesStore(self._workspace.ml_evaluation_series)
        try:
            bundle = store.load(selection.artifact_id)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return _LoadFailure("series_unavailable", str(exc))
        manifest = bundle.manifest
        if selection.column not in manifest.columns:
            return _LoadFailure(
                "series_unavailable", f"column {selection.column} is not saved"
            )
        column = selection.column
        raw = bundle.frame[column]
        if not pd.api.types.is_numeric_dtype(raw.dtype):
            return _LoadFailure("series_unavailable", f"column {column} is not numeric")
        timestamps = bundle.frame["timestamp"]
        series = pd.Series(
            pd.to_numeric(raw, errors="coerce").astype("float64").to_numpy(),
            index=pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)),
            name=column,
        )
        # Comparability settings are the evaluation contract only. The
        # fingerprint and entry identity are what distinguish the two sides.
        settings = dict(manifest.settings)
        fold_range = _fold_range(manifest, bundle.frame)
        identity = {
            "kind": "signal",
            "label": f"{selection.artifact_id}:{column}",
            "artifact_id": selection.artifact_id,
            "column": column,
            "dataset_id": manifest.dataset_id,
            "dataset_hash": manifest.dataset_hash,
            "entry_id": manifest.entry_id,
            "producer_source": manifest.producer_source,
            "signal_fingerprint": manifest.signal_fingerprint,
            "training_dataset_id": manifest.training_dataset_id,
            "training_status": manifest.training_status,
            "holdout_relationship": manifest.holdout_relationship,
            "fold_range": fold_range,
            "source_hash": manifest.content_hash,
            "settings": settings,
        }
        if column == "realized_return":
            definition = "zero_cost_signal_simulation"
            cost_basis = "zero_cost_signal_simulation"
        else:
            definition = f"signal_{column}"
            cost_basis = "not_applicable_information_coefficient"
        # A saved pointwise series records a relationship, not an OOS warrant:
        # different dataset IDs and training flags never prove a true holdout.
        claim_scope = "in_sample_or_unknown"
        return _LoadedSource(
            series=series,
            label=str(identity["label"]),
            identity=identity,
            settings=settings,
            cadence=_infer_cadence(pd.DatetimeIndex(series.index)),
            definition=definition,
            cost_basis=cost_basis,
            claim_scope=claim_scope,
        )

    # -- compatibility, payload and artifact ------------------------------

    def _incompatible(
        self, a: _LoadedSource, b: _LoadedSource
    ) -> tuple[str, str] | None:
        allowed = {
            ("cost_bearing_trade_return", "cost_bearing_trade_return"),
            ("cost_bearing_trade_return", "cost_bearing_benchmark_return"),
            ("cost_bearing_benchmark_return", "cost_bearing_trade_return"),
            ("cost_bearing_benchmark_return", "cost_bearing_benchmark_return"),
            ("zero_cost_price_return", "cost_bearing_trade_return"),
            ("cost_bearing_trade_return", "zero_cost_price_return"),
            ("zero_cost_price_return", "zero_cost_price_return"),
            ("signal_pearson_ic", "signal_pearson_ic"),
            ("signal_spearman_ic", "signal_spearman_ic"),
            ("zero_cost_signal_simulation", "zero_cost_signal_simulation"),
        }
        pair = (a.definition, b.definition)
        if pair not in allowed:
            return (
                "incompatible_definitions",
                f"{a.definition} cannot be compared with {b.definition}",
            )
        if a.cadence != b.cadence:
            return (
                "frequency_mismatch",
                f"{a.cadence} versus {b.cadence}",
            )
        if pair[0].startswith("signal_") and a.settings != b.settings:
            return (
                "incompatible_definitions",
                "saved signal evaluation settings differ",
            )
        # A verified zero-cost price benchmark has no capital basis of its own;
        # it inherits the evaluation period from its timestamps, not a run's
        # initial cash or annualisation setting.
        capital_paired = (
            not pair[0].startswith("signal_") and "zero_cost_price_return" not in pair
        )
        if capital_paired:
            for key in ("init_cash", "periods_per_year", "risk_free_rate"):
                if a.settings.get(key) != b.settings.get(key):
                    return (
                        "incompatible_definitions",
                        f"frozen {key} differs between saved sources",
                    )
        return None

    @staticmethod
    def _alignment(aligned: MaskedPairAlignment) -> dict[str, Any]:
        common = aligned.common.index
        excluded_sample = [
            {
                "timestamp": item.timestamp,
                "absent_from": item.absent_from,
                "reason": item.reason,
            }
            for item in aligned.excluded[:MAX_EXCLUDED_IN_RESPONSE]
        ]
        counts: dict[str, int] = {}
        for item in aligned.excluded:
            key = f"{item.reason}:{item.absent_from}"
            counts[key] = counts.get(key, 0) + 1
        return {
            "calendar_policy": aligned.calendar_policy,
            "cadence": aligned.cadence,
            "cadence_complete": aligned.cadence_complete,
            "original_lengths": dict(aligned.original_lengths),
            "overlap": aligned.overlap,
            "valid_observations": aligned.valid_count,
            "common_period": {
                "start": common[0].isoformat() if len(common) else None,
                "end": common[-1].isoformat() if len(common) else None,
            },
            "excluded_counts": counts,
            "excluded_sample": excluded_sample,
            "missing_timestamps_count": len(aligned.missing_timestamps),
            "missing_timestamps_sample": list(
                aligned.missing_timestamps[:MAX_MISSING_IN_RESPONSE]
            ),
            "valid_mask_count": sum(aligned.valid_mask),
        }

    def _unavailable_payload(
        self,
        failure: _LoadFailure,
        mode: ComparisonMode,
        sources: list[dict[str, Any]],
        limitations: list[str],
        aligned: MaskedPairAlignment | None = None,
    ) -> dict[str, Any]:
        if aligned is None:
            empty_alignment: dict[str, Any] = {
                "calendar_policy": "exact_utc_intersection_no_fill",
                "cadence": "unsupported",
                "cadence_complete": False,
                "original_lengths": {"a": 0, "b": 0},
                "overlap": 0,
                "valid_observations": 0,
                "common_period": {"start": None, "end": None},
                "excluded_counts": {},
                "excluded_sample": [],
                "missing_timestamps_count": 0,
                "missing_timestamps_sample": [],
                "valid_mask_count": 0,
            }
        else:
            empty_alignment = self._alignment(aligned)
        return {
            "success": True,
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "mode": mode,
            "statistic": "mean",
            "difference_direction": "a_minus_b",
            "claim_scope": "in_sample_or_unknown",
            "qualification": {"both_snapshots_verified": False},
            "status": "unavailable",
            "result": None,
            "unavailable": {
                "reason": failure.reason,
                "next_step": _NEXT.get(
                    failure.reason, "Inspect the saved sources and settings."
                ),
                "source": "saved_sources",
                "detail": failure.detail,
            },
            "alignment": empty_alignment,
            "sources": sources,
            "limitations": list(_LIMITATIONS),
            "artifact": None,
            "html_artifact": None,
        }

    def _publish_artifacts(
        self, payload: dict[str, Any], audit: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Publish immutable Markdown and self-contained HTML from one payload."""
        body = dict(payload)
        body["artifact"] = None
        body["html_artifact"] = None
        body["audit"] = audit
        # Preflight both targets before writing either one, so a known file or
        # registry conflict cannot leave a half-published pair.
        for document, prefix, suffix in (
            (_render_markdown(body), "paired-comparison", ".md"),
            (_render_html(body), "paired-comparison-html", ".html"),
        ):
            self._check_publication_target(document, prefix, suffix)
        markdown = self._publish_text(
            body,
            document=_render_markdown(body),
            prefix="paired-comparison",
            suffix=".md",
            artifact_type=ARTIFACT_TYPE,
            media_type="text/markdown",
            payload=payload,
        )
        html = self._publish_text(
            body,
            document=_render_html(body),
            prefix="paired-comparison-html",
            suffix=".html",
            artifact_type=HTML_ARTIFACT_TYPE,
            media_type="text/html",
            payload=payload,
        )
        return markdown, html

    def _check_publication_target(
        self, document: str, prefix: str, suffix: str
    ) -> None:
        digest = sha256_text(document)
        artifact_id = f"{prefix}:{digest[:16]}"
        path = self._workspace.root / "paired_comparisons" / f"{digest[:16]}{suffix}"
        if path.is_symlink() or not path.resolve().is_relative_to(self._workspace.root):
            raise ValueError("comparison artifact path leaves the workspace")
        if path.exists() and (
            not path.is_file() or path.read_text(encoding="utf-8") != document
        ):
            raise ValueError("existing comparison artifact differs")
        existing = self._store.get_artifact(artifact_id)
        if existing is not None and (
            existing.get("path") != str(path) or existing.get("sha256") != digest
        ):
            raise ValueError("registered comparison artifact identity differs")

    def _publish_text(
        self,
        body: dict[str, Any],
        *,
        document: str,
        prefix: str,
        suffix: str,
        artifact_type: str,
        media_type: Literal["text/markdown", "text/html"],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        digest = sha256_text(document)
        artifact_id = f"{prefix}:{digest[:16]}"
        path = self._workspace.root / "paired_comparisons" / f"{digest[:16]}{suffix}"
        if path.is_symlink() or not path.resolve().is_relative_to(self._workspace.root):
            raise ValueError("comparison artifact path leaves the workspace")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and (
            not path.is_file() or path.read_text(encoding="utf-8") != document
        ):
            raise ValueError("existing comparison artifact differs")
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
                    raise ValueError("existing comparison artifact differs") from None
            finally:
                temporary.unlink(missing_ok=True)
        existing = self._store.get_artifact(artifact_id)
        if existing is not None and (
            existing.get("path") != str(path) or existing.get("sha256") != digest
        ):
            raise ValueError("registered comparison artifact identity differs")
        self._store.create_artifact(
            artifact_id=artifact_id,
            run_id=None,
            artifact_type=artifact_type,
            path=path,
            sha256=digest,
            metadata={
                "algorithm_version": ALGORITHM_VERSION,
                "mode": payload["mode"],
                "claim_scope": payload["claim_scope"],
                "status": payload["status"],
                "media_type": media_type,
            },
        )
        return {
            "artifact_id": artifact_id,
            "sha256": digest,
            "media_type": media_type,
        }


def _infer_cadence(index: pd.DatetimeIndex) -> str:
    """Derive a supported cadence from saved timestamps, never assume daily."""
    if len(index) < 2:
        return "unsupported"
    deltas = index[1:] - index[:-1]
    if deltas.nunique() != 1:
        return "unsupported"
    step = deltas[0]
    if step == pd.Timedelta(days=1):
        return "1d"
    if step == pd.Timedelta(hours=1):
        return "1h"
    return "unsupported"


def _unavailable_identity(selection: SeriesSelection) -> dict[str, Any]:
    """Keep the two-side contract intact even when a source cannot load."""
    return {
        "kind": selection.kind,
        "label": f"{selection.kind}:unavailable",
        "source_hash": "unavailable",
        "run_id": selection.run_id,
        "scope": selection.scope,
        "artifact_id": selection.artifact_id,
        "column": selection.column,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    )
    return (
        "# TradingDev paired comparison\n\n"
        f"Algorithm: `{payload['algorithm_version']}`. "
        "This is a derived, immutable analysis of two already saved series. "
        "It does not rerun a strategy, train a model, download data, or alter "
        "either source artifact or any older report.\n\n"
        f"Mode: `{payload['mode']}`. "
        f"Claim scope: `{payload['claim_scope']}`. "
        f"Statistic: mean(a) - mean(b) on the shared valid index.\n\n"
        "## Limits\n\n"
        + "\n".join(f"- {line}" for line in payload["limitations"])
        + "\n\n## Result\n\n```json\n"
        + encoded
        + "\n```\n"
    )


def _render_html(payload: dict[str, Any]) -> str:
    """Render a self-contained offline HTML report from the same analysis."""

    def text(value: object) -> str:
        return html.escape("" if value is None else str(value), quote=True)

    encoded_source = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    )
    encoded = html.escape(encoded_source, quote=False)
    result = payload.get("result") or {}
    alignment = payload["alignment"]
    rows = [
        ("演算法版本", payload["algorithm_version"]),
        ("模式", payload["mode"]),
        ("資格範圍", payload["claim_scope"]),
        (
            "統計量 / 方向",
            f"{payload['statistic']} / {payload['difference_direction']}",
        ),
    ]
    if result:
        rows.extend(
            [
                ("估計差額 (a - b)", result["estimate"]),
                (
                    "{:.0%} 區間".format(result["confidence"]),
                    f"{result['lower']} … {result['upper']}",
                ),
                ("p 值", f"{result['p_value']} ({result['p_value_interpretation']})"),
                (
                    "confidence / reps / valid_reps",
                    "{confidence} / {reps} / {valid_reps}".format(**result),
                ),
                ("seed / block size", f"{result['seed']} / {result['block_size']}"),
                ("相關性", result["correlation"]),
            ]
        )
    summary = "\n".join(
        f"<tr><th>{text(label)}</th><td>{text(value)}</td></tr>"
        for label, value in rows
    )
    alignment_rows = [
        ("原始長度", alignment["original_lengths"]),
        (
            "共同日期 / 有效觀察",
            f"{alignment['overlap']} / {alignment['valid_observations']}",
        ),
        ("共同期間", alignment["common_period"]),
        ("排除原因", alignment["excluded_counts"]),
        ("日曆 / 缺口", f"{alignment['cadence']} / {alignment['cadence_complete']}"),
    ]
    audit = "\n".join(
        f"<tr><th>{text(label)}</th><td>{text(value)}</td></tr>"
        for label, value in alignment_rows
    )
    source_cards = "\n".join(
        "<article><h3>{}</h3><dl>{}</dl></article>".format(
            text(source.get("label")),
            "\n".join(
                f"<dt>{text(field)}</dt><dd>{text(source.get(field))}</dd>"
                for field in (
                    "kind",
                    "source_hash",
                    "run_id",
                    "scope",
                    "dataset_id",
                    "entry_id",
                    "signal_fingerprint",
                    "training_status",
                    "holdout_relationship",
                    "fold_range",
                    "benchmark_definition",
                    "benchmark_costs",
                )
            ),
        )
        for source in payload["sources"]
    )
    limits = "\n".join(f"<li>{text(line)}</li>" for line in payload["limitations"])
    evidence = payload.get("audit")
    evidence_section = ""
    if evidence:
        evidence_json = html.escape(
            json.dumps(evidence, ensure_ascii=False, allow_nan=False, indent=2),
            quote=False,
        )
        evidence_section = f"<h2>完整對齊證據</h2><pre>{evidence_json}</pre>"
    return (
        '<!doctype html>\n<html lang="zh-Hant"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>TradingDev paired comparison</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:60rem;margin:2rem auto;"
        "padding:0 1rem;line-height:1.6;color:#222;background:#fff}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;"
        "padding:.4rem .6rem;text-align:left;vertical-align:top}th{width:14rem;"
        "background:#f6f6f6}article{border:1px solid #ddd;padding:.6rem 1rem;"
        "margin:.6rem 0}dt{font-weight:bold}dd{margin:0 0 .3rem}pre{white-space:"
        "pre-wrap;word-break:break-word;background:#f6f6f6;padding:1rem}"
        "</style></head><body>"
        "<h1>TradingDev 配對比較</h1>"
        "<p>這是從兩側已保存序列產生的衍生分析，不會重新回測、訓練、下載或"
        "改寫來源；HTML 與 Markdown 內容相同，可離線閱讀。</p>"
        f"<h2>結果</h2><table>{summary}</table>"
        f"<h2>對齊與排除</h2><table>{audit}</table>"
        f"<h2>來源身分</h2>{source_cards}"
        f"<h2>限制</h2><ul>{limits}</ul>"
        f"<h2>完整分析 JSON</h2><pre>{encoded}</pre>{evidence_section}"
        "</body></html>\n"
    )


def _run_settings(scope: LoadedHistoryScope) -> dict[str, JsonValue]:
    metadata = scope.performance.metadata
    raw = metadata.get("settings")
    settings = dict(raw) if isinstance(raw, dict) else {}
    result: dict[str, JsonValue] = {}
    for key in (
        "frequency",
        "periods_per_year",
        "risk_free_rate",
        "return_std_ddof",
        "terminal_close_policy",
        "trade_scope",
        "calendar_timezone",
    ):
        if key in settings:
            result[key] = settings[key]
    result["init_cash"] = scope.observations.init_cash
    return result


def _run_returns(scope: LoadedHistoryScope) -> list[float] | None:
    observations = scope.observations
    if observations.returns is not None:
        if len(observations.returns) != len(observations.equity_curve):
            return None
        return [
            float(value) if value is not None and np.isfinite(value) else np.nan
            for value in observations.returns
        ]
    capital = observations.init_cash
    if not is_finite_number(capital) or not capital or capital <= 0:
        return None
    return _returns_from_equity(observations.equity_curve, float(capital))


def _returns_from_equity(equity: list[float | None], capital: float) -> list[float]:
    result: list[float] = []
    previous: float | None = capital
    for value in equity:
        if value is None or not np.isfinite(value):
            result.append(np.nan)
            # Do not bridge a missing observation: the next interval has no
            # trustworthy previous close and must stay unavailable too.
            previous = None
            continue
        current = float(value)
        if previous is None or previous == 0:
            result.append(np.nan)
        else:
            result.append(current / previous - 1.0)
        previous = current
    return result


def _run_identity(
    loaded: LoadedHistoryRun,
    scope: LoadedHistoryScope,
    settings: dict[str, JsonValue],
) -> dict[str, Any]:
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
                "created_at",
            )
        },
        "bundle": {
            "default_scope": loaded.performance.default_scope,
            "selected_train_scope": loaded.performance.selected_train_scope,
        },
        "scope": scope.scope_id,
        "performance": scope.performance.model_dump(mode="json"),
        "observations": scope.observations.model_dump(mode="json"),
        "parameters": scope.parameters,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()
    return {
        "kind": "run",
        "label": f"{loaded.run['run_id']}#{scope.scope_id}@{digest[:12]}",
        "source_hash": digest,
        "run_id": str(loaded.run["run_id"]),
        "scope": scope.scope_id,
        "manifest_hash": loaded.performance.manifest_hash,
        "dataset_id": loaded.run.get("dataset_id"),
        "mode": scope.performance.mode,
        "frequency": settings.get("frequency"),
        "capital_basis": "account_equity",
        "producer_source": "saved_backtest_observations",
        "settings": settings,
    }


def _run_claim_scope(scope: LoadedHistoryScope) -> str:
    if scope.performance.split == "test" and scope.research_evidence is not None:
        return "verified_out_of_sample"
    return "in_sample_or_unknown"


def _fold_range(manifest: Any, frame: pd.DataFrame) -> dict[str, JsonValue] | None:
    if manifest.kind != "fold_validation" or frame.empty:
        return None
    columns = ("train_start", "train_end", "eval_start", "eval_end")
    if not set(columns).issubset(frame.columns):
        return None
    return {
        key: str(frame.iloc[0][key]) if pd.notna(frame.iloc[0][key]) else None
        for key in columns
    }
