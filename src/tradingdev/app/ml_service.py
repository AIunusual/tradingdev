"""Application orchestration for ML discovery, datasets, models and evaluations."""

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
)
from tradingdev.adapters.storage.ml_datasets import MLDatasetStore
from tradingdev.app.data_service import DataService
from tradingdev.app.ml_training import MLTrainingService
from tradingdev.domain.data.cache import DataIdentity
from tradingdev.domain.data.crawlers.registry import describe_sources
from tradingdev.domain.data.schemas import DataConfig, MarketDataRequest
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
from tradingdev.domain.ml.evaluation.performance import (
    periods_per_year,
    return_metrics,
)
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
    parameter_axes,
    rank_entries,
    rankable_metrics,
    signal_fingerprint,
    surface_caveats,
)
from tradingdev.domain.ml.models.registry import (
    ModelKind,
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
)
from tradingdev.domain.ml.report import ML_SAMPLE_PROVENANCE_NOTE
from tradingdev.domain.ml.rolling_gen import (
    get_rolling_scheme,
    rolling_scheme_names,
)
from tradingdev.domain.ml.validation import (
    get_validation_scheme,
    validation_scheme_names,
)
from tradingdev.domain.performance.sampling import bar_duration_ns

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tradingdev.domain.ml.calendar import PanelAlignment

_EnumT = TypeVar("_EnumT", bound=Enum)


@dataclass(frozen=True)
class LabelledFrame:
    """One symbol's labelled rows, and whether its features were recomputed."""

    frame: pd.DataFrame
    dropped_rows: int
    features_from_cache: bool


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


# Filesystem IDs must match literal shapes to prevent path traversal.
_MODEL_ID = re.compile(r"^[0-9a-f]{16}$")
_OHLCV_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
# Keep exact missing-date counts, but cap the response listing.
_MAX_REPORTED_DATES = 50

_ML_GUIDANCE = {
    "datasets": (
        "Datasets use cached market bars; acquire missing data with ensure_data. "
        "Features and labels are computed separately per symbol before calendar "
        "alignment. Labels are sign(close[t+horizon]/close[t]-1); warm-up and "
        "unlabelled tail rows are dropped and reported. Identical request, data "
        "and feature implementation share an immutable dataset ID."
    ),
    "panels": (
        "intersection keeps dates all symbols traded; union includes every date "
        "and applies missing_policy: drop, forward_fill or error. Forward fill "
        "carries prior observations for at most max_staleness dates (None means "
        "unbounded). Inspect coverage, dropped dates and forward-filled counts. "
        "Forward-filled panels cannot use purged validation; use intersection "
        "or missing_policy=drop."
    ),
    "training": (
        "Model parameters override the selected kind's declared defaults; unknown "
        "names are rejected. Artifact identity binds parameters, ordered features "
        "and model bytes. Native feature-importance scales differ by framework. "
        "The saved artifact is fitted on all dataset rows; in_sample_accuracy "
        "does not measure out-of-sample performance."
    ),
    "validation": (
        "Each fold fits a separate model after purging overlapping label intervals "
        "and applying embargo. Symbols on one date stay in one fold; "
        "min_train_bars counts distinct dates while fold counts describe rows. "
        "Fold scores estimate the training procedure, not out-of-sample "
        "predictions of the saved all-row artifact. purged_kfold can train on "
        "later dates and is retrospective, not a past-only deployment simulation."
    ),
    "rolling_schemes": (
        "Declarative rolling retraining geometry for bundled ML strategies, off "
        "by default and meant for expert use. Enabling a strategy's rolling_gen "
        "block replaces its classic retrainer for that run; leaving it off keeps "
        "the classic loop untouched. roll_sd slides a fixed-length training "
        "window, roll_ex keeps the first bar and expands. A training window "
        "always ends strictly before the bar it scores. Unset step, window_size, "
        "min_train_bars and predict_lookback follow the strategy's own "
        "retrain_interval and lookback."
    ),
    "evaluation": (
        "evaluate_signal loads the verified model and single-symbol dataset by ID "
        "and evaluates every saved row internally, independent of predict previews. "
        "Row order, realised returns and horizon come from that dataset. Window "
        "must be at least 2, quantiles 2..10, with enough finite observations. "
        "Both Pearson and Spearman IC families, hit rate, turnover and prediction "
        "strata are reported; ties can produce fewer strata than requested. "
        "A metric that comes out NaN or Inf is recorded as null rather than "
        "dropped, so a failed run keeps its place in the parameter grid. "
        "Annualised simulated metrics require verified continuous crypto "
        "holding intervals; other calendars remain unavailable. "
        "Training-data evaluation remains in-sample. "
        "Panel-trained models may be evaluated on a separate compatible "
        "single-symbol dataset; panel datasets themselves are refused."
    ),
    "simulated_returns": (
        "Position thresholds use earlier signals only; positions open after "
        "warm-up on a non-overlapping horizon-bar grid. Returns exclude costs "
        "and are holding-interval outcomes, not mark-to-market bar returns. "
        "Quantile strata describe the entire sample and do not drive positions."
    ),
    "ranking": (
        "record=True replaces only the same model/dataset/evaluation context. "
        "leaderboard metrics have declared directions; never assume higher is "
        "better. Filter by dataset_id to compare one verified dataset version. "
        "Each row reports the window, horizon and n_quantiles it was measured "
        "under, the candidate's fitted parameters and its signal_fingerprint; "
        "comparing that fingerprint with a fresh predict shows whether a re-run "
        "reproduced the recorded signal. allow_mixed permits incompatible "
        "contexts but does not make them comparable; allow_parameter_axes "
        "relaxes window, horizon and n_quantiles within the saved identity, "
        "without verifying shared source bars across horizons. This ranks a "
        "sensitivity surface and must not be read as one comparable "
        "measurement. A candidate with no finite score is still recorded, and "
        "its skipped row carries its window, horizon and n_quantiles, so the "
        "grid keeps the point where it failed instead of joining its "
        "neighbours. Ranking and selection caveats cover all candidates, even "
        "when only one page is returned; a winner is not a profit guarantee."
    ),
    "reports": (
        "After inspecting saved leaderboard candidates, discover report sections "
        "with get_report_sections and select their entry IDs with generate_report. "
        "ML reports snapshot the selected evidence without rerunning predictions "
        "or training. Keep unavailable candidates and comparison caveats; "
        "including runs does not prove a model-to-run relationship. "
        f"{ML_SAMPLE_PROVENANCE_NOTE}"
    ),
}


class MLService:
    """ML research over immutable datasets and disposable feature caches.

    Recording replaces only the same candidate and measurement context.
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
        # Allow explicit feature-version identity; otherwise derive it from source.
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
        # Discovery and training share the scheme registry.
        validation_schemes = [
            {"name": scheme.name, "description": scheme.description}
            for scheme in (
                get_validation_scheme(name) for name in validation_scheme_names()
            )
        ]
        # Listed from the same registry a strategy resolves, so the menu and the
        # executable geometry cannot drift apart.
        rolling_schemes = [
            {
                "name": scheme.name,
                "description": scheme.description,
                "rtype": scheme.rtype.value,
                "default_step": scheme.step,
            }
            for scheme in (get_rolling_scheme(name) for name in rolling_scheme_names())
        ]
        return {
            "success": True,
            "schema_version": 1,
            "feature_sets": feature_sets,
            "model_kinds": model_kinds,
            "validation_schemes": validation_schemes,
            "rolling_schemes": rolling_schemes,
            "processor_types": _PROCESSOR_TYPES,
            "ic_methods": ["pearson", "spearman"],
            "n_quantiles_range": [2, 10],
            "calendar_modes": [mode.value for mode in CalendarMode],
            "missing_policies": [policy.value for policy in MissingPolicy],
            "leaderboard_metrics": list(rankable_metrics()),
            "guidance": dict(_ML_GUIDANCE),
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
        """Evaluate one asset's chronological predictions against forward returns.

        Inputs must be equal-length finite arrays sufficient for the IC window
        and quantile count. A supplied dataset ID verifies every saved return,
        row order and horizon, even without recording; omission of the horizon
        adopts the saved value, or one bar without a dataset. Recording requires
        a dataset ID. ``model_id`` is a caller label, not proof of signal origin.

        Returns a ``SignalEvaluationResponse`` payload; invalid inputs raise
        ``ValueError``.
        """
        if len(signal) != len(forward_returns):
            raise ValueError(
                "signal and forward_returns must have equal length "
                f"(got {len(signal)} vs {len(forward_returns)})"
            )
        aligned = self._align_length(signal, forward_returns)
        if len(aligned) != len(signal):
            raise ValueError(
                "signal and forward_returns must contain only finite values"
            )
        if len(aligned) < max(window, n_quantiles):
            raise ValueError(
                f"not enough finite observations after alignment "
                f"({len(aligned)}); need at least {max(window, n_quantiles)}"
            )
        if window < 2:
            raise ValueError("window must be at least 2")
        if not 2 <= n_quantiles <= 10:
            raise ValueError("n_quantiles must be between 2 and 10")

        manifest: dict[str, Any] | None = None
        annual_periods: float | None = None
        if record or dataset_id is not None:
            horizon, manifest, annual_periods = self._recording_horizon(
                dataset_id,
                horizon,
                forward_returns=forward_returns,
                n_observations=int(len(aligned)),
            )
        elif horizon is None:
            horizon = 1
        if horizon < 1:
            raise ValueError("horizon must be at least 1")

        pred = pd.Series(signal, dtype=float)
        fwd = pd.Series(forward_returns, dtype=float)

        raw_metrics = signal_metrics(pred, fwd, window=window)
        metrics = {key: self._finite(value) for key, value in raw_metrics.items()}

        strata_df = quantile_strata(pred, fwd, n_bins=n_quantiles)
        strata = [record.model_dump() for record in strata_records(strata_df)]
        strata_stats = strata_statistics(strata_df)

        simulated = simulate_strategy_returns(
            pred, fwd, n_quantiles=n_quantiles, horizon=horizon
        )
        simulated_summary = {
            "long_short_mean": self._finite(simulated.long_short_mean),
            "long_only_mean": self._finite(simulated.long_only_mean),
            "long_short_cumulative": self._finite(simulated.long_short_cumulative),
            "long_only_cumulative": self._finite(simulated.long_only_cumulative),
            "quantile_means": [
                self._finite(value) for value in list(simulated.quantile_means)
            ],
        }

        # The book is long/short, and its benchmark is holding the same asset
        # over the same holding intervals, so alpha/beta describe the signal's
        # exposure rather than an unrelated index.
        # Off-grid zeros are placeholders, not additional holding intervals.
        # Keep cash returns on the grid and align the benchmark to that grid.
        decisions = pred.index[::horizon]
        performance = return_metrics(
            simulated.long_short.loc[decisions],
            fwd.loc[decisions],
            periods_per_year=annual_periods,
        )
        metrics.update(performance.as_metrics())

        evaluation_id = self._evaluation_id(
            signal, forward_returns, horizon, window, n_quantiles
        )
        # Hash every signal, modelled or not: a re-run can only be checked
        # against the recorded candidate if the recorded row carries this value.
        fingerprint = signal_fingerprint(signal)
        payload: dict[str, Any] = {
            "success": True,
            "evaluation_id": evaluation_id,
            "signal_fingerprint": fingerprint,
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
                fingerprint=fingerprint,
                manifest=manifest,
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
        forward_returns: list[float],
    ) -> tuple[int, dict[str, Any], float | None]:
        """Verify evaluation rows and resolve their horizon and annualisation."""
        if not dataset_id:
            raise ValueError(
                "recording an evaluation requires dataset_id, so the record "
                "knows what it was measured on"
            )
        frame, manifest = self._load_dataset(dataset_id)
        if manifest.get("kind") == "panel":
            raise ValueError(
                "evaluate_signal requires a single-symbol dataset; predict on "
                "a separately built single-symbol dataset to evaluate a panel model"
            )
        dataset_horizon = int(manifest["horizon"])
        dates = pd.to_datetime(frame["timestamp"], utc=True)
        if not dates.is_monotonic_increasing or not dates.is_unique:
            raise ValueError("evaluation requires strictly chronological dataset rows")
        if "label_end" not in frame.columns:
            raise ValueError(
                "dataset has no label intervals; rebuild it for evaluation"
            )
        entries = frame.iloc[::dataset_horizon]
        starts = pd.to_datetime(entries["timestamp"], utc=True).reset_index(drop=True)
        ends = pd.to_datetime(entries["label_end"], utc=True).reset_index(drop=True)
        if (ends.shift(1).iloc[1:] > starts.iloc[1:]).any():
            raise ValueError("dataset holding intervals overlap on the simulation grid")
        rows = int(manifest["rows"])
        if n_observations != rows:
            raise ValueError(
                f"signal covers {n_observations} observations but "
                f"{dataset_id} has {rows} rows; the record would not describe "
                "this dataset"
            )
        if not np.array_equal(
            np.asarray(forward_returns, dtype=float),
            frame["forward_return"].to_numpy(dtype=float),
        ):
            raise ValueError(
                "forward_returns do not match the stored dataset in row order"
            )
        if horizon is not None and horizon != dataset_horizon:
            raise ValueError(
                f"horizon {horizon} does not match the dataset's horizon "
                f"{dataset_horizon}; omit it to adopt the dataset's, or record "
                "against the dataset the signal was computed on"
            )
        annual_periods = self._dataset_periods_per_year(entries, manifest)
        return dataset_horizon, manifest, annual_periods

    @staticmethod
    def _dataset_periods_per_year(
        intervals: pd.DataFrame, manifest: dict[str, Any]
    ) -> float | None:
        """Annualise only verified continuous intervals in a known crypto market."""
        source = next(
            (
                item
                for item in describe_sources()
                if item["source"] == manifest.get("source")
            ),
            None,
        )
        if source is None or set(source["asset_classes"]) != {"crypto"}:
            return None
        timeframe = str(manifest["timeframe"])
        horizon = int(manifest["horizon"])
        duration = bar_duration_ns(timeframe)
        if duration is None:
            return None
        starts = pd.to_datetime(intervals["timestamp"], utc=True).reset_index(drop=True)
        ends = pd.to_datetime(intervals["label_end"], utc=True).reset_index(drop=True)
        expected = pd.Timedelta(duration * horizon, unit="ns")
        if not (ends - starts).eq(expected).all():
            return None
        if (
            not ends.iloc[:-1]
            .reset_index(drop=True)
            .equals(starts.iloc[1:].reset_index(drop=True))
        ):
            return None
        return periods_per_year(timeframe, horizon)

    def _record_evaluation(
        self,
        metrics: dict[str, Any],
        *,
        fingerprint: str,
        manifest: dict[str, Any] | None,
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
        if manifest is None:
            raise ValueError(
                "recording an evaluation requires the dataset manifest it was "
                "measured against"
            )
        if model_id is not None and not _MODEL_ID.match(model_id):
            raise ValueError(f"malformed model_id: {model_id}")

        context = EvaluationContext(
            symbol=self._manifest_symbol(manifest),
            timeframe=str(manifest["timeframe"]),
            start_date=str(manifest["start_date"]),
            end_date=str(manifest["end_date"]),
            horizon=horizon,
            window=window,
            n_quantiles=n_quantiles,
            feature_set=str(manifest["feature_set"]),
            code_fingerprint=str(manifest["code_fingerprint"]),
            data_fingerprint=str(manifest["content_hash"]),
        )
        # Every measured metric is stored, including the ones that failed. A
        # NaN or Inf becomes an explicit None so the candidate keeps its
        # coordinates on the parameter grid: dropping the key would let a later
        # surface read 10 and 30 as adjacent after 20 failed. Non-numeric values
        # are not metrics at all and are still dropped.
        stored_metrics: dict[str, float | None] = {}
        for key, value in metrics.items():
            if value is None:
                stored_metrics[key] = None
            elif isinstance(value, (int, float)):
                number = float(value)
                stored_metrics[key] = number if math.isfinite(number) else None
        label, parameters = self._candidate_identity(model_id, fingerprint)
        entry = LeaderboardEntry(
            entry_id=build_entry_id(
                model_id=model_id or "",
                dataset_id=dataset_id,
                context=context,
                # A model ID already names the candidate, so folding the hash
                # in would change every existing entry_id. It is stored in its
                # own field instead. Without a model ID the hash is the only
                # thing separating two ad-hoc signals on the same dataset.
                signal_fingerprint="" if model_id else fingerprint,
            ),
            label=label,
            context=context,
            metrics=stored_metrics,
            n_observations=n_observations,
            model_id=model_id or "",
            dataset_id=dataset_id,
            parameters=parameters,
            signal_fingerprint=fingerprint,
        )
        store = LeaderboardStore(self._workspace.ml_leaderboard)
        replaced = store.record(entry)
        return {
            "entry_id": entry.entry_id,
            "replaced": replaced,
            "context": context.describe(),
            "n_candidates": len(store.entries()),
            "signal_fingerprint": fingerprint,
        }

    @staticmethod
    def _manifest_symbol(manifest: dict[str, Any]) -> str:
        """The instrument label of a dataset, including panels."""
        symbol = manifest.get("symbol")
        if symbol:
            return str(symbol)
        symbols = manifest.get("symbols") or []
        return ",".join(str(item) for item in symbols)

    def _candidate_identity(
        self, model_id: str | None, fingerprint: str
    ) -> tuple[str, dict[str, Any]]:
        """A readable name for a recorded candidate and its fitted settings.

        The parameters come from the artifact itself, so a row describes the
        model that produced it even if the artifact is later retrained.
        """
        if not model_id:
            label = (
                f"ad-hoc signal ({fingerprint[:8]})" if fingerprint else "ad-hoc signal"
            )
            return label, {}
        try:
            artifact = ModelArtifactStore(self._workspace.models).load(model_id)
        except FileNotFoundError, ValueError:
            return model_id, {}
        return (
            f"{artifact.model_name} ({artifact.framework})",
            dict(artifact.parameters),
        )

    def leaderboard(
        self,
        metric: str,
        *,
        dataset_id: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        horizon: int | None = None,
        feature_set: str | None = None,
        top_k: int | None = None,
        allow_mixed: bool = False,
        allow_parameter_axes: bool = False,
    ) -> dict[str, Any]:
        """Rank recorded evaluations using the metric's declared direction.

        Filters narrow candidates by dataset or measurement conditions. A
        supplied dataset is verified against its records. Mixed contexts raise
        ``ValueError`` unless ``allow_mixed`` is enabled; that opt-in does not
        make their measurements comparable. ``allow_parameter_axes`` permits
        the sweepable settings (window, horizon, n_quantiles) to differ within
        the saved identity; cross-horizon source-bar equivalence is unverified.
        Invalid metrics, datasets or empty results also fail.
        Returns a ``LeaderboardResponse`` payload.
        """
        manifest = self._load_dataset(dataset_id)[1] if dataset_id is not None else None
        store = LeaderboardStore(self._workspace.ml_leaderboard)
        entries = store.entries(
            dataset_id=dataset_id,
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
            feature_set=feature_set,
        )
        if manifest is not None:
            for entry in entries:
                expected = EvaluationContext(
                    symbol=self._manifest_symbol(manifest),
                    timeframe=str(manifest["timeframe"]),
                    start_date=str(manifest["start_date"]),
                    end_date=str(manifest["end_date"]),
                    horizon=int(manifest["horizon"]),
                    window=entry.context.window,
                    n_quantiles=entry.context.n_quantiles,
                    feature_set=str(manifest["feature_set"]),
                    code_fingerprint=str(manifest["code_fingerprint"]),
                    data_fingerprint=str(manifest["content_hash"]),
                )
                if entry.context != expected or entry.n_observations != int(
                    manifest["rows"]
                ):
                    raise ValueError(
                        "recorded evaluation context does not match dataset_id "
                        f"{dataset_id}: {entry.entry_id}"
                    )
        if not entries:
            raise ValueError(
                "no recorded evaluations; call evaluate_signal with record=True first"
            )
        axes = parameter_axes(entries)
        if not allow_mixed:
            mismatch = comparability_error(
                entries, allow_parameter_axes=allow_parameter_axes
            )
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
                "horizon": item.entry.context.horizon,
                "window": item.entry.context.window,
                "n_quantiles": item.entry.context.n_quantiles,
                "parameters": dict(item.entry.parameters),
                "signal_fingerprint": item.entry.signal_fingerprint,
                "value": item.value,
                "tied": item.tied,
                "n_observations": item.entry.n_observations,
                "summary": {
                    key: None if value is None else float(value)
                    for key, value in summarize_signal_metrics(
                        dict(item.entry.metrics)
                    ).items()
                    if value is None or isinstance(value, (int, float))
                },
            }
            for item in result.ranked
        ]
        contexts = sorted({entry.context.describe() for entry in entries})
        caveats = list(result.caveats)
        caveats.extend(surface_caveats(axes))
        return {
            "success": True,
            "metric": result.metric,
            "direction": result.direction,
            "n_candidates": len(entries),
            "n_ranked": len(result.ranked),
            "context": contexts[0] if len(contexts) == 1 else "mixed",
            "parameter_axes": axes,
            "ranked": ranked,
            "skipped": [
                {
                    "entry_id": item.entry.entry_id,
                    "label": item.entry.label,
                    "reason": item.reason,
                    # A candidate with no score is still a point on the grid:
                    # without these coordinates a failed run leaves a hole that
                    # reads as two adjacent settings.
                    "horizon": item.entry.context.horizon,
                    "window": item.entry.context.window,
                    "n_quantiles": item.entry.context.n_quantiles,
                }
                for item in result.skipped
            ],
            "caveats": caveats,
        }

    def leaderboard_page(
        self,
        metric: str,
        *,
        dataset_id: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        horizon: int | None = None,
        feature_set: str | None = None,
        offset: int = 0,
        limit: int = 20,
        allow_mixed: bool = False,
        allow_parameter_axes: bool = False,
    ) -> dict[str, Any]:
        """Page output only after ranking and comparing the full candidate set."""
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        result = self.leaderboard(
            metric,
            dataset_id=dataset_id,
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
            feature_set=feature_set,
            allow_mixed=allow_mixed,
            allow_parameter_axes=allow_parameter_axes,
        )
        ranked, skipped = result["ranked"], result["skipped"]
        total = max(len(ranked), len(skipped))
        if offset > total:
            raise ValueError(f"offset exceeds leaderboard row count ({total})")
        end = min(offset + limit, total)
        return {
            **result,
            "ranked": ranked[offset:end],
            "skipped": skipped[offset:end],
            "n_skipped": len(skipped),
            "offset": offset,
            "limit": limit,
            "next_offset": end if end < total else None,
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
        market_type: str = "futures/um",
    ) -> dict[str, Any]:
        """Build immutable labelled features from existing cached OHLCV bars.

        Targets are signs of forward returns over ``horizon`` source bars.
        Drop incomplete feature/label rows here; feature builders preserve rows.
        Unknown features, non-positive horizons and missing market data fail.
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
            market_type=market_type,
        )
        dataset = labelled.frame
        dropped_rows = labelled.dropped_rows

        feature_columns = list(entry.columns)
        parquet_path, manifest = MLDatasetStore(self._workspace.ml_datasets).save(
            dataset,
            {
                "kind": "single",
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "source": source,
                "market_type": self._market_identity(
                    source,
                    market_type,
                    timeframe,
                    start_date,
                    end_date,
                    symbol,
                ).market_type,
                "feature_set": feature_set,
                "feature_columns": feature_columns,
                "code_fingerprint": self._code_fingerprint(),
                "label_column": "target",
                "horizon": horizon,
                "dropped_rows": int(dropped_rows),
            },
        )
        dataset_id = manifest["dataset_id"]
        content_hash = manifest["content_hash"]
        return {
            "success": True,
            "dataset": {
                "dataset_id": dataset_id,
                "source": manifest["source"],
                "market_type": manifest["market_type"],
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
                # Cache-hit status must not change dataset identity.
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
        market_type: str = "futures/um",
        calendar_mode: str = "intersection",
        missing_policy: str = "drop",
        max_staleness: int | None = None,
    ) -> dict[str, Any]:
        """Build a shared-calendar panel from at least two distinct symbols.

        Compute each symbol's features and labels before alignment. Intersection
        keeps common dates; union applies drop, error or forward-fill handling.
        Forward fill carries original values and label intervals for at most
        ``max_staleness`` calendar positions (``None`` is unbounded). Retain only
        dates every symbol supplies. Symbol order does not affect identity.

        Invalid symbols, features, calendar options or cached data fail.
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
                market_type=market_type,
            )
            dropped_rows += labelled.dropped_rows
            # A panel is cached only when every symbol is a hit.
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

        calendar = self._calendar_summary(alignment, mode, policy, max_staleness)
        feature_columns = list(entry.columns)
        # Inactive options are normalized so identical panel requests share identity.
        inert = mode is CalendarMode.INTERSECTION
        calendar["missing_policy"] = "drop" if inert else policy.value
        calendar["max_staleness"] = None if inert else max_staleness
        parquet_path, manifest = MLDatasetStore(self._workspace.ml_datasets).save(
            panel,
            {
                "kind": "panel",
                "symbols": list(ordered),
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "source": source,
                "market_type": self._market_identity(
                    source,
                    market_type,
                    timeframe,
                    start_date,
                    end_date,
                    symbol,
                ).market_type,
                "feature_set": feature_set,
                "feature_columns": feature_columns,
                "code_fingerprint": self._code_fingerprint(),
                "label_column": "target",
                "horizon": horizon,
                "n_dates": int(alignment.calendar.n_dates),
                "dropped_rows": int(dropped_rows),
                "calendar": calendar,
            },
        )
        dataset_id = manifest["dataset_id"]
        content_hash = manifest["content_hash"]
        return {
            "success": True,
            "panel": {
                "dataset_id": dataset_id,
                "source": manifest["source"],
                "market_type": manifest["market_type"],
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
        """Fit a registered direction model and persist its pickled artifact.

        The final artifact fits all dataset rows; its training-data predictions
        are in-sample. Optional validation fits separate fold models and returns
        scores, not out-of-fold predictions. Walk-forward trains on the past;
        K-fold is retrospective. ``None`` skips validation.

        Purging uses saved label intervals and horizon; ``embargo`` adds a gap.
        ``min_train_bars`` counts distinct dates and defaults to half a block,
        leaving room for the first walk-forward fold's purge. Artifact identity
        binds effective parameters, ordered features and model bytes.

        Invalid IDs, model kinds, parameters or fold geometry raise ``ValueError``.
        """
        return MLTrainingService(self._workspace).train_model(
            dataset_id=dataset_id,
            model_kind=model_kind,
            parameters=parameters,
            validation=validation,
            n_folds=n_folds,
            embargo=embargo,
            min_train_bars=min_train_bars,
        )

    def predict(self, *, model_id: str, dataset_id: str) -> dict[str, Any]:
        """Predict with a verified stored model and dataset in saved row order.

        Verify blob integrity and exact ordered features before unpickling.
        Workspace model files must be trusted: hashing is not a sandbox.
        Return signals and saved forward returns; prediction does not establish
        out-of-sample provenance. Invalid IDs, blobs or feature contracts fail.
        """
        if not _MODEL_ID.match(model_id):
            raise ValueError(f"malformed model_id: {model_id}")
        frame, manifest = self._load_dataset(dataset_id)

        store = ModelArtifactStore(self._workspace.models)
        if not store.exists(model_id):
            raise ValueError(f"unknown model_id: {model_id}")
        try:
            artifact, model_bytes = store.load_with_bytes(model_id)
        except FileNotFoundError as exc:
            raise ValueError(f"model blob missing for id: {model_id}") from exc

        expected = tuple(manifest["feature_columns"])
        if tuple(artifact.feature_names) != expected:
            raise ValueError(
                "feature contract mismatch: model was trained on "
                f"{list(artifact.feature_names)} but the dataset supplies "
                f"{list(expected)}"
            )

        model = pickle.loads(model_bytes)
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
                # Covers every saved row, not a page, so a re-run can be
                # compared with the fingerprint stored on the leaderboard.
                "signal_fingerprint": signal_fingerprint(signal),
            },
        }

    def predict_page(
        self,
        *,
        model_id: str,
        dataset_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Expose a bounded preview; complete arrays stay in backend computation."""
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer between 1 and 500")
        prediction = self.predict(model_id=model_id, dataset_id=dataset_id)[
            "prediction"
        ]
        total = int(prediction["n_rows"])
        if offset > total:
            raise ValueError(f"offset exceeds prediction row count ({total})")
        end = min(offset + limit, total)
        return {
            "success": True,
            "prediction": {
                **prediction,
                "signal": prediction["signal"][offset:end],
                "forward_returns": prediction["forward_returns"][offset:end],
                "offset": offset,
                "limit": limit,
                "next_offset": end if end < total else None,
            },
        }

    def evaluate_prediction(
        self,
        *,
        model_id: str,
        dataset_id: str,
        window: int = 60,
        n_quantiles: int = 5,
        record: bool = False,
    ) -> dict[str, Any]:
        """Evaluate the entire verified model/dataset pair without an array relay.

        The dataset fixes row order, realised returns and the label horizon.
        Model identity establishes which saved model produced the signal; it
        does not establish that the evaluation rows were held out of training.
        """
        prediction = self.predict(model_id=model_id, dataset_id=dataset_id)[
            "prediction"
        ]
        result = self.evaluate_signal(
            prediction["signal"],
            prediction["forward_returns"],
            horizon=prediction["horizon"],
            window=window,
            n_quantiles=n_quantiles,
            record=record,
            model_id=model_id,
            dataset_id=dataset_id,
        )
        return {**result, "model_id": model_id, "dataset_id": dataset_id}

    def _load_dataset(self, dataset_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Read the exact verified bytes of an immutable dataset version."""
        return MLDatasetStore(self._workspace.ml_datasets).load(dataset_id)

    @staticmethod
    def _feature_set(name: str) -> FeatureSet:
        """Resolve a feature set by name, as a ``ValueError`` for the tools."""
        try:
            return get_feature_set(name)
        except KeyError as exc:
            raise ValueError(f"unknown feature set: {name}") from exc

    @staticmethod
    def _market_identity(
        source: str,
        market_type: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        symbol: str,
    ) -> DataIdentity:
        return DataIdentity.market(
            DataConfig(source=source, market_type=market_type),
            MarketDataRequest(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            ),
        )

    def _cached_bars(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        source: str,
        market_type: str,
    ) -> pd.DataFrame:
        """Load one symbol's cached OHLCV bars, or say what is missing."""
        loaded = self._data_service.load_cached(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            source=source,
            market_type=market_type,
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
        market_type: str,
    ) -> tuple[pd.DataFrame, bool]:
        """Return bars plus features and a cache-hit flag.

        Cache features before labelling so the same matrix serves all horizons.
        """
        bars = self._cached_bars(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            source=source,
            market_type=market_type,
        )
        identity = FeatureCacheIdentity(
            symbol=symbol,
            timeframe=timeframe,
            # Hash bar values in fixed column order, independent of file encoding.
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
        market_type: str,
    ) -> LabelledFrame:
        """Label one symbol before panel alignment, dropping incomplete rows."""
        frame, features_from_cache = self._feature_frame(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            entry=entry,
            source=source,
            market_type=market_type,
        )
        dataset = frame.copy()
        forward = forward_returns(dataset["close"], horizon)
        # Keep returns for evaluation; training selects declared features only.
        dataset["forward_return"] = forward
        dataset["target"] = np.sign(forward).astype(float)
        dataset["label_start"] = dataset["timestamp"]
        dataset["label_end"] = dataset["timestamp"].shift(-horizon)

        rows_before = len(dataset)
        dataset = dataset.dropna(
            subset=[*entry.columns, "target", "forward_return", "label_end"]
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
