"""Application service backing the ML MCP tools.

Thin orchestration over the already-tested domain ML programs: it assembles the
enumerable contract menu and runs a prediction through the signal-evaluation
suite (IC / Rank IC, quantile strata, simulated long/short books). It performs
no training and writes nothing, so both tools stay read-only and idempotent.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pandas as pd

from tradingdev.adapters.storage.filesystem import (
    WorkspacePaths,
    sha256_file,
    write_json,
)
from tradingdev.app.data_service import DataService
from tradingdev.domain.ml.calendar import (
    CalendarMode,
    MissingPolicy,
    align_frames,
)
from tradingdev.domain.ml.evaluation import (
    forward_returns,
    quantile_strata,
    signal_metrics,
    simulate_strategy_returns,
    strata_statistics,
)
from tradingdev.domain.ml.evaluation.artifacts import strata_records
from tradingdev.domain.ml.evaluation.catalog import summarize_signal_metrics
from tradingdev.domain.ml.feature_cache import (
    FeatureCacheIdentity,
    data_fingerprint,
    feature_code_fingerprint,
)
from tradingdev.domain.ml.features.registry import (
    FeatureSet,
    feature_set_names,
    get_feature_set,
)
from tradingdev.domain.ml.leaderboard import (
    EvaluationContext,
    LeaderboardEntry,
    build_entry_id,
    comparability_error,
    rank_entries,
    rankable_metrics,
    signal_fingerprint,
)
from tradingdev.domain.ml.models.registry import (
    ModelKind,
    create_direction_model,
    get_model_kind,
    list_model_kinds,
)
from tradingdev.domain.ml.preprocessing import (
    ClipOutliers,
    DropnaRows,
    Fillna,
    MinMaxNorm,
    RobustZScoreNorm,
    ZScoreNorm,
)
from tradingdev.domain.ml.registry import (
    FeatureCacheStore,
    LeaderboardStore,
    ModelArtifactStore,
    build_model_artifact,
    content_sha256,
    load_model_bytes,
)
from tradingdev.domain.ml.validation import (
    audit_validation_splits,
    build_validation_splits,
    get_validation_scheme,
    validation_scheme_names,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from tradingdev.domain.ml.calendar import PanelAlignment
    from tradingdev.domain.ml.validation.splits import Split

_EnumT = TypeVar("_EnumT", bound=Enum)


@dataclass(frozen=True)
class LabelledFrame:
    """One symbol's labelled rows, and whether its features were recomputed."""

    frame: pd.DataFrame
    dropped_rows: int
    features_from_cache: bool


@dataclass(frozen=True)
class _ValidationRequest:
    """A validation request that has been resolved and checked."""

    scheme: str
    label_horizon: int
    """The dataset's own horizon; it alone sets the purge width."""
    n_folds: int
    embargo: int
    min_train_bars: int


_PROCESSOR_TYPES = [
    cls.__name__
    for cls in (
        ZScoreNorm,
        RobustZScoreNorm,
        MinMaxNorm,
        ClipOutliers,
        Fillna,
        DropnaRows,
    )
]


_DATASET_ID = re.compile(r"^ds_[0-9a-f]{16}$")
# model_id is a 16-hex sha256 prefix produced by the artifact registry. Both
# ids reach the filesystem, so they are locked to a literal shape rather than
# trusted: an LLM-supplied value must never be able to walk out of the store.
_MODEL_ID = re.compile(r"^[0-9a-f]{16}$")
_OHLCV_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
# A symbol that never traded on the calendar can report one missing date per
# bar; the count is always exact, the listing is capped for the tool response.
_MAX_REPORTED_DATES = 50
# Rolling IC window for a fold score. It is capped at the fold's own length,
# because a window longer than the test window measures nothing.
_FOLD_IC_WINDOW = 60


class MLService:
    """The enumerable ML vocabulary, signal evaluation, datasets and training.

    Discovery and evaluation are read-only. Dataset building and training
    persist artifacts under the workspace, but both are deterministic given
    their inputs and never modify existing records.
    """

    def __init__(
        self,
        workspace: WorkspacePaths | None = None,
        *,
        data_service: DataService | None = None,
        feature_cache: FeatureCacheStore | None = None,
        feature_code_fingerprint: str | None = None,
    ) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._data_service = data_service or DataService(self._workspace)
        self._feature_cache = feature_cache or FeatureCacheStore(
            self._workspace.ml_feature_cache
        )
        # Deriving the code fingerprint from source (rather than from git, or
        # from the clock) keeps it reproducible, and injection keeps it
        # overridable: a caller that tracks feature versions some other way can
        # say so instead of being handed a guess.
        self._injected_code_fingerprint = feature_code_fingerprint

    def _code_fingerprint(self) -> str:
        """The fingerprint of the code that would build these features."""
        if self._injected_code_fingerprint is None:
            self._injected_code_fingerprint = feature_code_fingerprint()
        return self._injected_code_fingerprint

    def get_contract(self) -> dict[str, Any]:
        """Assemble the full enumerable ML menu for LLM-driven selection."""
        feature_sets = [
            self._feature_set_summary(get_feature_set(name))
            for name in sorted(feature_set_names())
        ]
        model_kinds = [
            self._model_kind_summary(get_model_kind(name))
            for name in list_model_kinds()
        ]
        # Read the same registry train_model resolves names against: a scheme
        # that is advertised but cannot be built is a promise the tool breaks.
        validation_schemes = [
            {"name": scheme.name, "description": scheme.description}
            for scheme in (
                get_validation_scheme(name) for name in validation_scheme_names()
            )
        ]
        return {
            "success": True,
            "schema_version": 1,
            "feature_sets": feature_sets,
            "model_kinds": model_kinds,
            "validation_schemes": validation_schemes,
            "processor_types": _PROCESSOR_TYPES,
            "ic_methods": ["pearson", "spearman"],
            "n_quantiles_range": [2, 10],
            "calendar_modes": [mode.value for mode in CalendarMode],
            "missing_policies": [policy.value for policy in MissingPolicy],
            "leaderboard_metrics": list(rankable_metrics()),
        }

    def evaluate_signal(
        self,
        signal: list[float],
        forward_returns: list[float],
        *,
        horizon: int | None = None,
        window: int = 60,
        n_quantiles: int = 5,
        record: bool = False,
        model_id: str | None = None,
        dataset_id: str | None = None,
    ) -> dict[str, Any]:
        """Judge a prediction against realised forward returns.

        Args:
            signal: Prediction values, one per bar.
            forward_returns: Realised return over the label horizon, one per bar.
            horizon: Label horizon the returns were computed at. Informational
                unless ``record`` is set, in which case it must agree with the
                dataset's own horizon: a record labelled with the wrong horizon
                would later be ranked against numbers that mean something else.
                Left as ``None`` when recording, the dataset's horizon is used.
            window: Rolling IC window in bars.
            n_quantiles: Quantile count for strata and simulated books.
            record: Store the result so it can be ranked by ``leaderboard``.
                Requires ``dataset_id``: without knowing what the signal was
                measured on, the record could later be compared against
                numbers that mean something else.
            model_id: Model that produced the signal, when there is one.
            dataset_id: Dataset the signal was computed on.

        Returns:
            A success dict suitable for ``SignalEvaluationResponse``.

        Raises:
            ValueError: On length mismatch, non-finite inputs, too few finite
                observations for the requested window/quantiles, a record
                request without a dataset, or a recording whose signal length
                or horizon contradicts the dataset it names.
        """
        if len(signal) != len(forward_returns):
            raise ValueError(
                "signal and forward_returns must have equal length "
                f"(got {len(signal)} vs {len(forward_returns)})"
            )
        aligned = self._align_length(signal, forward_returns)
        if len(aligned) < max(window, n_quantiles):
            raise ValueError(
                f"not enough finite observations after alignment "
                f"({len(aligned)}); need at least {max(window, n_quantiles)}"
            )
        if window < 2:
            raise ValueError("window must be at least 2")
        if n_quantiles < 2:
            raise ValueError("n_quantiles must be at least 2")

        pred = pd.Series(signal, dtype=float)
        fwd = pd.Series(forward_returns, dtype=float)

        raw_metrics = signal_metrics(pred, fwd, window=window)
        metrics = {key: self._finite(value) for key, value in raw_metrics.items()}

        strata_df = quantile_strata(pred, fwd, n_bins=n_quantiles)
        strata = [record.model_dump() for record in strata_records(strata_df)]
        strata_stats = strata_statistics(strata_df)

        simulated = simulate_strategy_returns(pred, fwd, n_quantiles=n_quantiles)
        simulated_summary = {
            "long_short_mean": self._finite(simulated.long_short_mean),
            "long_only_mean": self._finite(simulated.long_only_mean),
            "long_short_cumulative": self._finite(simulated.long_short_cumulative),
            "long_only_cumulative": self._finite(simulated.long_only_cumulative),
            "quantile_means": [
                self._finite(value) for value in list(simulated.quantile_means)
            ],
        }

        if record:
            horizon = self._recording_horizon(
                dataset_id, horizon, n_observations=int(len(aligned))
            )
        elif horizon is None:
            horizon = 1

        evaluation_id = self._evaluation_id(
            signal, forward_returns, horizon, window, n_quantiles
        )
        payload: dict[str, Any] = {
            "success": True,
            "evaluation_id": evaluation_id,
            "n_observations": int(len(aligned)),
            "horizon": horizon,
            "window": window,
            "n_bins": n_quantiles,
            "ic_methods": ["pearson", "spearman"],
            "metrics": metrics,
            "strata": strata,
            "strata_monotonicity": self._finite(
                strata_stats.get("strata_monotonicity")
            ),
            "long_short_spread": self._finite(strata_stats.get("long_short_spread")),
            "simulated": simulated_summary,
        }
        if record:
            payload["recorded"] = self._record_evaluation(
                {
                    **metrics,
                    "strata_monotonicity": payload["strata_monotonicity"],
                    "long_short_spread": payload["long_short_spread"],
                },
                signal=signal,
                n_observations=int(len(aligned)),
                horizon=horizon,
                window=window,
                n_quantiles=n_quantiles,
                model_id=model_id,
                dataset_id=dataset_id,
            )
        return payload

    def _recording_horizon(
        self,
        dataset_id: str | None,
        horizon: int | None,
        *,
        n_observations: int,
    ) -> int:
        """The horizon a recorded evaluation may claim.

        The dataset's own horizon is authoritative: it is the horizon the
        forward-return labels were built at. A caller may omit the horizon (it
        is adopted from the dataset) or state one that agrees, but never
        silently overrule it - a record labelled with the wrong horizon would
        be ranked against numbers measured over a different span.
        """
        if not dataset_id:
            raise ValueError(
                "recording an evaluation requires dataset_id, so the record "
                "knows what it was measured on"
            )
        _, manifest = self._load_dataset(dataset_id)
        dataset_horizon = int(manifest["horizon"])
        rows = int(manifest["rows"])
        if n_observations != rows:
            raise ValueError(
                f"signal covers {n_observations} observations but "
                f"{dataset_id} has {rows} rows; the record would not describe "
                "this dataset"
            )
        if horizon is None:
            return dataset_horizon
        if horizon != dataset_horizon:
            raise ValueError(
                f"horizon {horizon} does not match the dataset's horizon "
                f"{dataset_horizon}; omit it to adopt the dataset's, or record "
                "against the dataset the signal was computed on"
            )
        return horizon

    def _record_evaluation(
        self,
        metrics: dict[str, Any],
        *,
        signal: list[float],
        n_observations: int,
        horizon: int,
        window: int,
        n_quantiles: int,
        model_id: str | None,
        dataset_id: str | None,
    ) -> dict[str, Any]:
        """Store one evaluation so it can be ranked against its peers."""
        if not dataset_id:
            raise ValueError(
                "recording an evaluation requires dataset_id, so the record "
                "knows what it was measured on"
            )
        if model_id is not None and not _MODEL_ID.match(model_id):
            raise ValueError(f"malformed model_id: {model_id}")

        _, manifest = self._load_dataset(dataset_id)
        context = EvaluationContext(
            symbol=self._manifest_symbol(manifest),
            timeframe=str(manifest["timeframe"]),
            start_date=str(manifest["start_date"]),
            end_date=str(manifest["end_date"]),
            horizon=horizon,
            window=window,
            n_quantiles=n_quantiles,
            feature_set=str(manifest["feature_set"]),
            code_fingerprint=self._code_fingerprint(),
        )
        # Non-finite metrics are dropped rather than stored: a NaN has no
        # business in a ranking, and storing it would only create a row that
        # can never be placed.
        finite_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        # Without a model to identify the candidate, the signal values are the
        # only thing that tells two hand-supplied entries apart.
        fingerprint = "" if model_id else signal_fingerprint(signal)
        entry = LeaderboardEntry(
            entry_id=build_entry_id(
                model_id=model_id or "",
                dataset_id=dataset_id,
                context=context,
                signal_fingerprint=fingerprint,
            ),
            label=self._entry_label(model_id, fingerprint),
            context=context,
            metrics=finite_metrics,
            n_observations=n_observations,
            model_id=model_id or "",
            dataset_id=dataset_id,
        )
        store = LeaderboardStore(self._workspace.ml_leaderboard)
        replaced = store.record(entry)
        return {
            "entry_id": entry.entry_id,
            "replaced": replaced,
            "context": context.describe(),
            "n_candidates": len(store.entries()),
        }

    @staticmethod
    def _manifest_symbol(manifest: dict[str, Any]) -> str:
        """The instrument label of a dataset, including panels."""
        symbol = manifest.get("symbol")
        if symbol:
            return str(symbol)
        symbols = manifest.get("symbols") or []
        return ",".join(str(item) for item in symbols)

    def _entry_label(self, model_id: str | None, fingerprint: str = "") -> str:
        """A readable name for a recorded candidate."""
        if not model_id:
            return (
                f"ad-hoc signal ({fingerprint[:8]})" if fingerprint else "ad-hoc signal"
            )
        try:
            artifact = ModelArtifactStore(self._workspace.models).load(model_id)
        except (FileNotFoundError, ValueError):
            return model_id
        return f"{artifact.model_name} ({artifact.framework})"

    def leaderboard(
        self,
        metric: str,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        horizon: int | None = None,
        feature_set: str | None = None,
        top_k: int | None = None,
        allow_mixed: bool = False,
    ) -> dict[str, Any]:
        """Rank recorded evaluations on one metric.

        Args:
            metric: A signal metric with a declared optimisation direction.
            symbol: Keep only candidates measured on this instrument.
            timeframe: Keep only candidates measured on this bar size.
            horizon: Keep only candidates measured at this label horizon.
            feature_set: Keep only candidates whose features came from this
                named set.
            top_k: Return only the first ``top_k`` ranks.
            allow_mixed: Rank across different measurement conditions anyway.
                Off by default: the numbers would not be comparable.

        Returns:
            A success dict suitable for ``LeaderboardResponse``.

        Raises:
            ValueError: If nothing is recorded, the metric cannot rank, or the
                remaining entries were measured under different conditions.
        """
        store = LeaderboardStore(self._workspace.ml_leaderboard)
        entries = store.entries(
            symbol=symbol, timeframe=timeframe, horizon=horizon, feature_set=feature_set
        )
        if not entries:
            raise ValueError(
                "no recorded evaluations; call evaluate_signal with record=True first"
            )
        if not allow_mixed:
            mismatch = comparability_error(entries)
            if mismatch:
                raise ValueError(mismatch)

        result = rank_entries(entries, metric, top_k=top_k)
        ranked = [
            {
                "rank": item.rank,
                "entry_id": item.entry.entry_id,
                "label": item.entry.label,
                "model_id": item.entry.model_id,
                "dataset_id": item.entry.dataset_id,
                "value": item.value,
                "tied": item.tied,
                "n_observations": item.entry.n_observations,
                "summary": {
                    key: float(value)
                    for key, value in summarize_signal_metrics(
                        dict(item.entry.metrics)
                    ).items()
                    if isinstance(value, (int, float))
                },
            }
            for item in result.ranked
        ]
        contexts = sorted({entry.context.describe() for entry in entries})
        return {
            "success": True,
            "metric": result.metric,
            "direction": result.direction,
            "n_candidates": len(entries),
            "n_ranked": len(result.ranked),
            "context": contexts[0] if len(contexts) == 1 else "mixed",
            "ranked": ranked,
            "skipped": [
                {
                    "entry_id": item.entry.entry_id,
                    "label": item.entry.label,
                    "reason": item.reason,
                }
                for item in result.skipped
            ],
            "caveats": list(result.caveats),
        }

    def build_dataset(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        feature_set: str,
        horizon: int = 1,
        source: str = "binance_vision",
    ) -> dict[str, Any]:
        """Build a labelled feature dataset from cached OHLCV bars.

        The named feature set supplies the columns, and the label is the sign
        of the realised forward return over ``horizon`` bars. Incomplete rows
        (indicator warm-up and the unlabelled tail) are dropped here rather
        than inside the builders, which stay row-preserving.

        Raises:
            ValueError: On an unknown feature set, a non-positive horizon, or
                missing cached market data.
        """
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        entry = self._feature_set(feature_set)

        labelled = self._symbol_frame(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            entry=entry,
            horizon=horizon,
            source=source,
        )
        dataset = labelled.frame
        dropped_rows = labelled.dropped_rows

        dataset_id = self._dataset_id(
            symbol, timeframe, start_date, end_date, feature_set, horizon, source
        )
        directory = self._workspace.ml_datasets / dataset_id
        directory.mkdir(parents=True, exist_ok=True)
        parquet_path = directory / "dataset.parquet"
        dataset.to_parquet(parquet_path, index=False)
        content_hash = sha256_file(parquet_path)

        feature_columns = list(entry.columns)
        write_json(
            directory / "manifest.json",
            {
                "dataset_id": dataset_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "source": source,
                "feature_set": feature_set,
                "feature_columns": feature_columns,
                "label_column": "target",
                "horizon": horizon,
                "rows": int(len(dataset)),
                "dropped_rows": int(dropped_rows),
                "content_hash": content_hash,
            },
        )
        return {
            "success": True,
            "dataset": {
                "dataset_id": dataset_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "feature_set": feature_set,
                "horizon": horizon,
                "rows": int(len(dataset)),
                "dropped_rows": int(dropped_rows),
                "n_features": len(feature_columns),
                "feature_columns": feature_columns,
                "label_column": "target",
                "content_hash": content_hash,
                "path": str(parquet_path),
                # Deliberately not part of the manifest: the dataset's content
                # does not depend on whether the features were recomputed, and
                # a manifest that changed between two builds of the same id
                # would be a worse audit trail than one that does not.
                "features_from_cache": labelled.features_from_cache,
            },
        }

    def build_panel(
        self,
        *,
        symbols: Sequence[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        feature_set: str,
        horizon: int = 1,
        source: str = "binance_vision",
        calendar_mode: str = "intersection",
        missing_policy: str = "drop",
        max_staleness: int | None = None,
    ) -> dict[str, Any]:
        """Build a labelled multi-symbol panel on one shared calendar.

        Each symbol is featurised and labelled on its own bars first --- the
        label is that symbol's own forward return, so alignment can never
        borrow another instrument's outcome --- and only then are the frames
        put on a common calendar. A date enters the panel only when every
        symbol can supply it, so the panel has no half-populated row.

        Args:
            symbols: Two or more symbols; order does not affect the result.
            calendar_mode: ``intersection`` keeps the dates every symbol
                traded; ``union`` keeps every date any symbol traded and then
                applies ``missing_policy`` to the holes.
            missing_policy: ``drop`` removes dates a symbol cannot supply,
                ``forward_fill`` carries its last bar (bounded by
                ``max_staleness``), ``error`` refuses the panel.
            max_staleness: Largest number of dates one bar may be carried
                across under ``forward_fill``; ``None`` is unbounded.

        Returns:
            A success dict suitable for ``BuildPanelResponse``.

        Raises:
            ValueError: On fewer than two symbols, duplicates, an unknown
                feature set or calendar option, or missing cached data.
        """
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        ordered = self._panel_symbols(symbols)
        mode = self._enum(CalendarMode, calendar_mode, "calendar mode")
        policy = self._enum(MissingPolicy, missing_policy, "missing policy")
        entry = self._feature_set(feature_set)

        frames: dict[str, pd.DataFrame] = {}
        dropped_rows = 0
        features_from_cache = True
        for symbol in ordered:
            labelled = self._symbol_frame(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                entry=entry,
                horizon=horizon,
                source=source,
            )
            dropped_rows += labelled.dropped_rows
            # Reported as one flag: a panel that mixed a recomputed symbol with
            # a cached one is not "cached", and saying so per symbol would
            # invite reading the flag as a statement about the whole panel.
            features_from_cache = features_from_cache and labelled.features_from_cache
            frames[symbol] = labelled.frame.set_index("timestamp")

        alignment = align_frames(
            frames, mode=mode, policy=policy, max_staleness=max_staleness
        )
        panel = (
            pd.concat(
                [alignment.frames[symbol] for symbol in ordered],
                keys=list(ordered),
                names=["symbol", "timestamp"],
            )
            .reset_index()
            .reset_index(drop=True)
        )

        # Under intersection every calendar date exists in every symbol, so the
        # missing-value options cannot change a single row. They are left out of
        # the fingerprint there, otherwise two requests that produce the same
        # panel would be stored twice under different ids.
        inert = mode is CalendarMode.INTERSECTION
        dataset_id = self._panel_id(
            ordered,
            timeframe,
            start_date,
            end_date,
            feature_set,
            horizon,
            source,
            mode.value,
            None if inert else policy.value,
            None if inert else max_staleness,
        )
        directory = self._workspace.ml_datasets / dataset_id
        directory.mkdir(parents=True, exist_ok=True)
        parquet_path = directory / "dataset.parquet"
        panel.to_parquet(parquet_path, index=False)
        content_hash = sha256_file(parquet_path)

        calendar = self._calendar_summary(alignment, mode, policy, max_staleness)
        feature_columns = list(entry.columns)
        write_json(
            directory / "manifest.json",
            {
                "dataset_id": dataset_id,
                "kind": "panel",
                "symbols": list(ordered),
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "source": source,
                "feature_set": feature_set,
                "feature_columns": feature_columns,
                "label_column": "target",
                "horizon": horizon,
                "rows": int(len(panel)),
                "n_dates": int(alignment.calendar.n_dates),
                "dropped_rows": int(dropped_rows),
                "calendar": calendar,
                "content_hash": content_hash,
            },
        )
        return {
            "success": True,
            "panel": {
                "dataset_id": dataset_id,
                "symbols": list(ordered),
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "feature_set": feature_set,
                "horizon": horizon,
                "rows": int(len(panel)),
                "n_symbols": len(ordered),
                "n_dates": int(alignment.calendar.n_dates),
                "dropped_rows": int(dropped_rows),
                "n_features": len(feature_columns),
                "feature_columns": feature_columns,
                "label_column": "target",
                "calendar": calendar,
                "content_hash": content_hash,
                "path": str(parquet_path),
                "features_from_cache": features_from_cache,
            },
        }

    def train_model(
        self,
        *,
        dataset_id: str,
        model_kind: str,
        parameters: dict[str, Any] | None = None,
        validation: str | None = None,
        n_folds: int = 5,
        embargo: int = 0,
        min_train_bars: int | None = None,
    ) -> dict[str, Any]:
        """Train a registered direction model on a built dataset.

        The trained estimator is pickled and stored through the model-artifact
        registry, so the returned ``model_id`` is a stable fingerprint of the
        hyper-parameters, the feature contract and the blob itself.

        Args:
            dataset_id: Dataset to train on.
            model_kind: Registered model kind.
            parameters: Hyper-parameter overrides.
            validation: Named scheme from the advertised
                ``validation_schemes``; ``None`` skips it. When given, the
                model is also fitted once per fold on purged training bars and
                scored on that fold's test window, so the response carries an
                out-of-sample estimate alongside the artifact.
            n_folds: How many test windows the scheme should produce.
            embargo: Extra bars removed around every test window, on top of
                the purge. Guards against serial correlation, which a purge
                sized exactly to the label horizon does not.
            min_train_bars: Smallest training window accepted per fold;
                defaults to half a block's worth. A whole block cannot be
                asked for under walk-forward, because the first fold's
                training window is one block before the purge is carved out
                of it.

        Raises:
            ValueError: On a malformed or unknown ``dataset_id``, an unknown
                ``model_kind``, parameters the model does not accept, an
                unknown ``validation`` scheme, or fold geometry the dataset is
                too small for.
        """
        frame, manifest = self._load_dataset(dataset_id)
        try:
            kind = get_model_kind(model_kind)
        except KeyError as exc:
            raise ValueError(f"unknown model kind: {model_kind}") from exc

        params = self._validated_parameters(kind, parameters or {})
        # Resolve the whole validation request before fitting anything: an
        # unusable request should cost nothing, and a model fitted only to be
        # refused afterwards would be pure waste.
        request = self._validation_request(
            scheme=validation,
            n_folds=n_folds,
            embargo=embargo,
            min_train_bars=min_train_bars,
            n_bars=len(frame),
            label_horizon=int(manifest["horizon"]),
        )

        model = create_direction_model(model_kind, **params)
        # Hand the model only the columns the feature set declared, so extra
        # bookkeeping columns (forward_return, OHLCV) can never become features.
        feature_columns = list(manifest["feature_columns"])
        model.train(frame[feature_columns + ["target"]])

        validation_summary = None
        if request is not None:
            validation_summary = self._validate_in_folds(
                request,
                model_kind=model_kind,
                params=params,
                frame=frame,
                feature_columns=feature_columns,
            )

        # Record the effective settings (declared defaults plus overrides)
        # rather than asking the estimator, so the fingerprint cannot depend
        # on a subclass choosing to report its config.
        effective_parameters = kind.config_cls(**params).model_dump()
        blob = pickle.dumps(model)
        artifact = build_model_artifact(
            model_name=model_kind,
            framework=kind.framework,
            kind="direction",
            parameters=effective_parameters,
            feature_names=tuple(feature_columns),
            content_hash=content_sha256(blob),
            n_train_samples=int(len(frame)),
            feature_importance=model.feature_importance(),
        )
        store = ModelArtifactStore(self._workspace.models)
        store.save(model_bytes=blob, artifact=artifact)

        predictions = model.predict(frame).to_numpy()
        in_sample_accuracy = float((predictions == frame["target"].to_numpy()).mean())
        return {
            "success": True,
            "model": {
                "model_id": artifact.model_id,
                "model_name": model_kind,
                "framework": kind.framework,
                "kind": "direction",
                "dataset_id": dataset_id,
                "n_train_samples": int(len(frame)),
                "n_features": len(manifest["feature_columns"]),
                "parameters": artifact.parameters,
                "feature_importance": [
                    {"feature": name, "importance": value}
                    for name, value in artifact.feature_importance
                ],
                "in_sample_accuracy": in_sample_accuracy,
                "validation": validation_summary,
                "path": str(store.path_for(artifact.model_id)),
            },
        }

    @staticmethod
    def _validation_request(
        *,
        scheme: str | None,
        n_folds: int,
        embargo: int,
        min_train_bars: int | None,
        n_bars: int,
        label_horizon: int,
    ) -> _ValidationRequest | None:
        """Resolve a validation request, or ``None`` when none was asked for.

        Every check that can refuse a request happens here, before any
        estimator is fitted, and the name is resolved against the registry the
        contract advertises so the menu and the parser cannot drift apart.
        """
        if scheme is None:
            return None
        get_validation_scheme(scheme)
        if n_folds < 1:
            raise ValueError(f"n_folds must be at least 1, got {n_folds}")
        if embargo < 0:
            raise ValueError(f"embargo must not be negative, got {embargo}")
        if label_horizon < 1:
            raise ValueError(
                "this dataset has no label horizon, so the purge width cannot "
                "be determined; rebuild it with an explicit horizon"
            )
        floor = min_train_bars
        if floor is None:
            # Half a fold's worth of bars. A whole block is unreachable: the
            # first fold's training window *is* one block, and the purge and
            # embargo are carved out of it, so asking for a full block would
            # reject every walk-forward request that purges anything at all.
            floor = max(1, n_bars // max(1, 2 * (n_folds + 1)))
        return _ValidationRequest(
            scheme=scheme,
            label_horizon=label_horizon,
            n_folds=n_folds,
            embargo=embargo,
            min_train_bars=floor,
        )

    def _validate_in_folds(
        self,
        request: _ValidationRequest,
        *,
        model_kind: str,
        params: dict[str, Any],
        frame: pd.DataFrame,
        feature_columns: list[str],
    ) -> dict[str, Any]:
        """Fit the model once per purged fold and score it on that fold's test bars.

        The purge width is the dataset's own label horizon, read from its
        manifest rather than asked for: the labels already exist, so how far
        their information reaches is a fact about the dataset, not a setting
        the caller should be free to get wrong.

        The artifact handed back alongside these numbers is still fit on every
        bar - that is the model worth deploying - so the folds estimate the
        procedure, and the summary says so.
        """
        splits: list[Split] = build_validation_splits(
            request.scheme,
            n_bars=len(frame),
            horizon=request.label_horizon,
            n_folds=request.n_folds,
            embargo=request.embargo,
            min_train_bars=request.min_train_bars,
        )
        violations = audit_validation_splits(splits, horizon=request.label_horizon)
        if violations:
            # The builders purge by construction; getting here means the
            # geometry or the horizon changed underneath them.
            raise ValueError(
                "the requested validation would leak: " + "; ".join(violations)
            )

        folds: list[dict[str, Any]] = []
        for index, split in enumerate(splits):
            train_rows = self._window_rows(split.train)
            test_rows = self._window_rows([split.test])
            train_frame = frame.iloc[train_rows]
            test_frame = frame.iloc[test_rows]
            fold_model = create_direction_model(model_kind, **params)
            fold_model.train(train_frame[feature_columns + ["target"]])
            predictions = fold_model.predict(test_frame).to_numpy()
            folds.append(
                {
                    "fold": index,
                    "train_bars": int(len(train_frame)),
                    "test_bars": int(len(test_frame)),
                    "purged_bars": int(split.purged_bars),
                    "embargoed_bars": int(split.embargoed_bars),
                    **self._fold_scores(
                        predictions=predictions,
                        forward_returns=test_frame["forward_return"].to_numpy(),
                        targets=test_frame["target"].to_numpy(),
                    ),
                }
            )

        return {
            "scheme": request.scheme,
            "label_horizon": int(request.label_horizon),
            "embargo": int(request.embargo),
            "n_folds": len(folds),
            "purged_bars": int(sum(split.purged_bars for split in splits)),
            "embargoed_bars": int(sum(split.embargoed_bars for split in splits)),
            "leakage_audit_clean": True,
            "artifact_trained_on": "all_bars",
            "folds": folds,
            "aggregate": self._aggregated_scores(folds),
        }

    @staticmethod
    def _window_rows(windows: Sequence[tuple[int, int]]) -> list[int]:
        """Flatten half-open index windows into the row positions they cover."""
        rows: list[int] = []
        for start, end in windows:
            rows.extend(range(start, end))
        return rows

    def _fold_scores(
        self,
        *,
        predictions: npt.NDArray[Any],
        forward_returns: npt.NDArray[Any],
        targets: npt.NDArray[Any],
    ) -> dict[str, Any]:
        """Score one fold's predictions against its own test bars.

        Metrics that need a rolling window are computed on a window no longer
        than the fold itself, and anything that comes back non-finite is
        reported as ``None`` rather than as a number that looks like a result.
        """
        accuracy = float((predictions == targets).mean())
        scores: dict[str, Any] = {"accuracy": self._finite(accuracy)}
        if len(predictions) < 2:
            return {**scores, "ic_mean": None, "rank_ic_mean": None, "hit_rate": None}
        window = max(2, min(_FOLD_IC_WINDOW, len(predictions)))
        raw = signal_metrics(
            pd.Series(predictions, dtype=float),
            pd.Series(forward_returns, dtype=float),
            window=window,
        )
        return {
            **scores,
            "ic_mean": self._finite(raw.get("ic_mean")),
            "rank_ic_mean": self._finite(raw.get("rank_ic_mean")),
            "hit_rate": self._finite(raw.get("hit_rate")),
        }

    @staticmethod
    def _aggregated_scores(folds: list[dict[str, Any]]) -> dict[str, Any]:
        """Mean of each fold score over the folds that produced one.

        A metric every fold failed to produce stays ``None`` rather than
        becoming zero: "not measured" and "measured as nothing" are different
        answers, and only one of them is honest.
        """
        aggregate: dict[str, Any] = {}
        for key in ("accuracy", "ic_mean", "rank_ic_mean", "hit_rate"):
            present = [
                float(fold[key])
                for fold in folds
                if fold.get(key) is not None and math.isfinite(float(fold[key]))
            ]
            aggregate[key] = sum(present) / len(present) if present else None
        return aggregate

    def predict(self, *, model_id: str, dataset_id: str) -> dict[str, Any]:
        """Score a stored model against a stored dataset.

        Loads the artifact (re-verifying its blob hash), unpickles the
        estimator and returns the predicted signal together with the realised
        forward returns, so the pair can be passed straight to
        :meth:`evaluate_signal`.

        Raises:
            ValueError: On a malformed or unknown id, a tampered blob, or a
                dataset whose feature contract differs from the one the model
                was trained on.
        """
        if not _MODEL_ID.match(model_id):
            raise ValueError(f"malformed model_id: {model_id}")
        frame, manifest = self._load_dataset(dataset_id)

        store = ModelArtifactStore(self._workspace.models)
        if not store.exists(model_id):
            raise ValueError(f"unknown model_id: {model_id}")
        try:
            artifact = store.load(model_id)
        except FileNotFoundError as exc:
            raise ValueError(f"model blob missing for id: {model_id}") from exc

        expected = tuple(manifest["feature_columns"])
        if tuple(artifact.feature_names) != expected:
            raise ValueError(
                "feature contract mismatch: model was trained on "
                f"{list(artifact.feature_names)} but the dataset supplies "
                f"{list(expected)}"
            )

        model = pickle.loads(load_model_bytes(store.path_for(model_id)))
        signal = model.predict(frame).to_numpy(dtype=float).tolist()
        realized = frame["forward_return"].to_numpy(dtype=float).tolist()
        return {
            "success": True,
            "prediction": {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "model_name": artifact.model_name,
                "framework": artifact.framework,
                "n_rows": len(signal),
                "horizon": int(manifest["horizon"]),
                "signal": signal,
                "forward_returns": realized,
            },
        }

    def _load_dataset(self, dataset_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Read a built dataset, refusing ids that could escape the store."""
        if not _DATASET_ID.match(dataset_id):
            raise ValueError(f"malformed dataset_id: {dataset_id}")
        directory = self._workspace.ml_datasets / dataset_id
        parquet_path = directory / "dataset.parquet"
        if not parquet_path.exists():
            raise ValueError(f"unknown dataset_id: {dataset_id}")
        manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
        return pd.read_parquet(parquet_path), manifest

    @staticmethod
    def _feature_set(name: str) -> FeatureSet:
        """Resolve a feature set by name, as a ``ValueError`` for the tools."""
        try:
            return get_feature_set(name)
        except KeyError as exc:
            raise ValueError(f"unknown feature set: {name}") from exc

    def _cached_bars(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        source: str,
    ) -> pd.DataFrame:
        """Load one symbol's cached OHLCV bars, or say what is missing."""
        if not self._data_service.data_available(
            symbol, timeframe, start_date, end_date
        ):
            raise ValueError(
                f"no cached market data for {symbol} {timeframe} "
                f"{start_date}..{end_date}; run ensure_data first"
            )
        loaded = self._data_service.ensure(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            source=source,
        )
        frame = loaded.frame.reset_index(drop=True)
        missing = [column for column in _OHLCV_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"cached data is missing columns: {', '.join(missing)}")
        return frame

    def _feature_frame(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        entry: FeatureSet,
        source: str,
    ) -> tuple[pd.DataFrame, bool]:
        """Bars plus the feature set's columns, reusing a verified matrix.

        Only the features are cached. The label is built afterwards, because it
        depends on the horizon: a matrix is reused across every horizon and
        every dataset built from the same bars, which is the whole reason the
        work is worth saving.

        Returns:
            The featurised frame, and whether it came from the cache.
        """
        bars = self._cached_bars(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            source=source,
        )
        identity = FeatureCacheIdentity(
            symbol=symbol,
            timeframe=timeframe,
            # Hashed from the bar values in a fixed column order, so the
            # fingerprint is a statement about the data and not about how the
            # file happened to be written.
            data_fingerprint=data_fingerprint(bars.loc[:, list(_OHLCV_COLUMNS)]),
            feature_set=entry.name,
            code_fingerprint=self._code_fingerprint(),
        )
        cached = self._feature_cache.load(identity, feature_columns=entry.columns)
        if cached is not None:
            return cached.reset_index(drop=True), True

        features = entry.build(bars)
        frame = bars.copy()
        for column in entry.columns:
            frame[column] = features[column]
        self._feature_cache.save(identity, frame, feature_columns=entry.columns)
        return frame, False

    def _symbol_frame(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        entry: FeatureSet,
        horizon: int,
        source: str,
    ) -> LabelledFrame:
        """Featurise and label one symbol's cached bars.

        Rows are dropped here, before any alignment, so one symbol's warm-up
        cannot shrink another symbol's calendar.
        """
        frame, features_from_cache = self._feature_frame(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            entry=entry,
            source=source,
        )
        dataset = frame.copy()
        forward = forward_returns(dataset["close"], horizon)
        # Kept so predict can hand realised returns to evaluate_signal. It is
        # never a training column: train_model selects the declared features
        # only, so this cannot leak into a model.
        dataset["forward_return"] = forward
        dataset["target"] = np.sign(forward).astype(float)

        rows_before = len(dataset)
        dataset = dataset.dropna(
            subset=[*entry.columns, "target", "forward_return"]
        ).reset_index(drop=True)
        dropped_rows = rows_before - len(dataset)
        if dataset.empty:
            raise ValueError(f"no rows remain for {symbol} after dropping features")
        return LabelledFrame(
            frame=dataset,
            dropped_rows=dropped_rows,
            features_from_cache=features_from_cache,
        )

    @staticmethod
    def _panel_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
        """Normalise the panel's symbol list: trimmed, unique, sorted."""
        cleaned = [symbol.strip() for symbol in symbols]
        if not cleaned or any(not symbol for symbol in cleaned):
            raise ValueError("at least one symbol is required")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"duplicate symbols: {sorted(cleaned)}")
        ordered = tuple(sorted(set(cleaned)))
        if len(ordered) < 2:
            raise ValueError(
                "a panel needs at least two distinct symbols; use build_dataset for one"
            )
        return ordered

    @staticmethod
    def _enum(enum_cls: type[_EnumT], value: str, label: str) -> _EnumT:
        """Coerce an LLM-supplied option, naming the field in the error."""
        try:
            return enum_cls(value)
        except ValueError as exc:
            raise ValueError(f"unknown {label}: {value}") from exc

    @staticmethod
    def _calendar_summary(
        alignment: PanelAlignment,
        mode: CalendarMode,
        policy: MissingPolicy,
        max_staleness: int | None,
    ) -> dict[str, Any]:
        """Render the alignment's audit trail as a JSON-safe report."""
        return {
            "mode": mode.value,
            "missing_policy": policy.value,
            "max_staleness": max_staleness,
            "n_dates": int(alignment.calendar.n_dates),
            "start_date": alignment.calendar.start.isoformat(),
            "end_date": alignment.calendar.end.isoformat(),
            "n_dropped_dates": len(alignment.dropped_dates),
            "dropped_dates": [
                date.isoformat()
                for date in alignment.dropped_dates[:_MAX_REPORTED_DATES]
            ],
            "coverage": [
                {
                    "symbol": item.symbol,
                    "n_present": item.n_present,
                    "n_calendar_dates": item.n_calendar_dates,
                    "n_missing": item.n_missing,
                    "coverage": item.coverage,
                    "missing": [
                        date.isoformat() for date in item.missing[:_MAX_REPORTED_DATES]
                    ],
                }
                for item in alignment.coverage
            ],
            "forward_filled": dict(alignment.forward_filled),
        }

    @staticmethod
    def _validated_parameters(
        kind: ModelKind, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Reject parameters the model does not declare.

        Pydantic ignores unknown fields by default, which would let a typo
        silently train with defaults; on an LLM-facing surface that has to be
        an error instead.
        """
        allowed = set(kind.config_cls.model_fields)
        unknown = sorted(set(parameters) - allowed)
        if unknown:
            raise ValueError(
                f"unknown parameters for {kind.name}: {', '.join(unknown)}"
            )
        return dict(parameters)

    @staticmethod
    def _dataset_id(
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        feature_set: str,
        horizon: int,
        source: str,
    ) -> str:
        payload = json.dumps(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "feature_set": feature_set,
                "horizon": horizon,
                "source": source,
            },
            sort_keys=True,
        )
        return f"ds_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"

    @staticmethod
    def _panel_id(
        symbols: Sequence[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        feature_set: str,
        horizon: int,
        source: str,
        calendar_mode: str,
        missing_policy: str | None,
        max_staleness: int | None,
    ) -> str:
        """Fingerprint a panel request in the same ``ds_`` id space as datasets.

        ``kind`` keeps a panel's id distinct from a single-symbol dataset built
        from the same bar cache, so the two never collide in the store. The
        missing-value options may be ``None`` when the calendar mode makes them
        inert, so a request that cannot change the rows cannot change the id.
        """
        payload = json.dumps(
            {
                "kind": "panel",
                "symbols": list(symbols),
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "feature_set": feature_set,
                "horizon": horizon,
                "source": source,
                "calendar_mode": calendar_mode,
                "missing_policy": missing_policy,
                "max_staleness": max_staleness,
            },
            sort_keys=True,
        )
        return f"ds_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"

    @staticmethod
    def _align_length(
        signal: list[float], forward_returns: list[float]
    ) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "signal": pd.Series(signal, dtype=float),
                "forward": pd.Series(forward_returns, dtype=float),
            }
        )
        return frame.replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def _finite(value: object) -> float | int | None:
        """Copy a finite number, mapping missing or non-finite inputs to None."""
        if value is None:
            return None
        number = float(value)  # type: ignore[arg-type]
        return number if math.isfinite(number) else None

    @staticmethod
    def _feature_set_summary(entry: FeatureSet) -> dict[str, Any]:
        return {
            "name": entry.name,
            "description": entry.description,
            "n_columns": len(entry.columns),
            "columns": list(entry.columns),
            "tags": list(entry.tags),
        }

    @staticmethod
    def _model_kind_summary(entry: ModelKind) -> dict[str, Any]:
        defaults = entry.config_cls().model_dump()
        return {
            "name": entry.name,
            "description": entry.description,
            "framework": entry.framework,
            "default_parameters": defaults,
        }

    @staticmethod
    def _evaluation_id(
        signal: list[float],
        forward_returns: list[float],
        horizon: int,
        window: int,
        n_quantiles: int,
    ) -> str:
        payload = json.dumps(
            {
                "signal": [float(value) for value in signal],
                "forward": [float(value) for value in forward_returns],
                "horizon": horizon,
                "window": window,
                "n_quantiles": n_quantiles,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"eval_{digest[:16]}"
