"""Execute a declared rolling scheme: causal retraining plus per-bar scoring.

The engine is a sibling of :class:`~tradingdev.domain.ml.retrainer.RollingRetrainer`,
not a replacement.  It is reached only when a strategy enables ``rolling_gen``;
the classic loop stays untouched for every run that does not.

Two rules from the scoring-history work are structural here rather than
remembered:

* history is cut with :func:`align_scoring_history`, so the scored frame is
  never concatenated in front of itself;
* a training window ends at the bar being scored, never after it, and the
  prediction is read back by the scored bar's own row label rather than by
  position in the returned probabilities.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.rolling_gen.schedule import RollWindow, enumerate_rolls
from tradingdev.domain.ml.rolling_gen.scheme import (
    RollingGeometry,
    RollingScheme,
    resolve_geometry,
)
from tradingdev.domain.ml.scoring_history import align_scoring_history
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from tradingdev.domain.ml.schemas import XGBoostModelConfig

logger = setup_logger(__name__)

_PROBABILITY_COLUMNS = ("p_long", "p_short")


@dataclass(frozen=True)
class RollTrace:
    """What one roll actually did, next to the window it was declared to cover.

    ``window`` is the same :class:`RollWindow` the schedule enumerates, so a
    trace can be compared with a preview by equality instead of by recomputing
    bounds. ``trained`` separates "fitted on this window" from "kept the previous
    model because the window was too small", which is the difference between a
    model that saw this data and one that did not.
    """

    window: RollWindow
    n_train_rows: int
    """Labelled rows the window offered; 0 when it was too short to try."""

    trained: bool
    """Whether a model was fitted, or the previous one kept."""

    bars_scored: int
    """Bars this roll's model actually scored."""


@dataclass(frozen=True)
class RollingOutcome:
    """Probabilities, the geometry that produced them, and their provenance."""

    probabilities: pd.DataFrame
    bar_roll: tuple[int, ...]
    """Roll index per scored bar; -1 where the bar could not be scored."""

    rolls: tuple[RollTrace, ...]
    scheme_name: str
    geometry: RollingGeometry
    """The effective geometry: unset scheme fields already resolved."""

    label_horizon: int
    """Bars a label looks ahead; the engine refuses to train past it."""

    initial_model_available: bool
    """Whether fit() used only history preceding the scored frame."""


@dataclass(frozen=True)
class _RetrainOutcome:
    """A retrain's model and what it was fitted on."""

    model: XGBoostDirectionModel | None
    n_train_rows: int
    trained: bool


def trace_payload(outcome: RollingOutcome) -> dict[str, object]:
    """A JSON-ready record of the run, for storage beside the run metrics.

    One entry per roll rather than per bar: a reader can resolve any bar's model
    by its roll, and the record stays proportional to the retrain count instead
    of the bar count. The geometry reported is the effective one, so a run that
    left ``step`` unset still records the cadence it actually used.
    """
    scored = sum(1 for roll in outcome.bar_roll if roll >= 0)
    return {
        "scheme": outcome.scheme_name,
        "rtype": outcome.geometry.rtype.value,
        "step": outcome.geometry.step,
        "window_size": outcome.geometry.window_size,
        "min_train_bars": outcome.geometry.min_train_bars,
        "predict_lookback": outcome.geometry.predict_lookback,
        "label_horizon": outcome.label_horizon,
        "initial_model_available": outcome.initial_model_available,
        "bars": len(outcome.bar_roll),
        "bars_scored": scored,
        # Positions that could not be scored at all. With the roll intervals,
        # this is what makes a bar's model recoverable without storing one
        # entry per bar: a bar is scored by the roll whose interval holds it
        # unless it appears here.
        "unscored_bars": [bar for bar, roll in enumerate(outcome.bar_roll) if roll < 0],
        "rolls": [
            {
                "index": trace.window.index,
                "retrain_bar": trace.window.retrain_bar,
                "train_start": trace.window.train_start,
                "train_end": trace.window.train_end,
                "test_start": trace.window.test_start,
                "test_end": trace.window.test_end,
                "n_train_rows": trace.n_train_rows,
                "trained": trace.trained,
                "bars_scored": trace.bars_scored,
            }
            for trace in outcome.rolls
        ],
    }


class FeatureTransformer(Protocol):
    """The part of a feature engineer the engine relies on.

    Both the direction and the risk feature engineers satisfy it, which is why
    one engine can serve both strategies without importing either.
    """

    def transform(
        self, df: pd.DataFrame, *, include_target: bool = True
    ) -> pd.DataFrame: ...


@dataclass(frozen=True)
class _Retrainer:
    """The immutable inputs of one rolling run's retraining step."""

    combined: pd.DataFrame
    geometry: RollingGeometry
    scheme_name: str
    fe: FeatureTransformer
    model_config: XGBoostModelConfig
    model_factory: Callable[[XGBoostModelConfig], XGBoostDirectionModel]
    total_retrains: int
    total_bars: int
    label_horizon: int

    def matured(self, labelled: pd.DataFrame, combined_idx: int) -> pd.DataFrame:
        """Keep only rows whose label was realised by the bar being scored.

        A label at row ``r`` reads the close ``label_horizon`` bars later, so it
        is realised within the supplied window once
        ``r + label_horizon < combined_idx``. Rows past that
        point are dropped here rather than trusted to the feature engineer: an
        engineer that fills its tail would otherwise train the model on outcomes
        that had not happened yet, which is the shape of issue #24's tail-label
        defect. Filtering the output instead of shortening the input keeps this a
        no-op for an engineer that already leaves those rows unlabelled.
        """
        last_realisable = combined_idx - 1 - self.label_horizon
        if last_realisable < 0:
            return labelled.iloc[:0]
        if "timestamp" not in labelled.columns:
            msg = (
                "training rows lost their time identity, so label maturity "
                "cannot be checked; keep the timestamp column in transform()"
            )
            raise ValueError(msg)
        cutoff = self.combined["timestamp"].iloc[last_realisable]
        kept: pd.DataFrame = labelled.loc[labelled["timestamp"] <= cutoff]
        return kept

    def retrain(
        self,
        combined_idx: int,
        *,
        fallback: XGBoostDirectionModel | None,
        retrain_count: int,
        bar: int,
    ) -> _RetrainOutcome:
        """Train on the declared window, or keep the previous model.

        A window that holds too few labelled rows is skipped rather than
        stretched: extending it forward to reach the minimum would train on the
        bar being scored.
        """
        logger.info(
            "[rolling_gen:%s] Retrain %d/%d at bar %d/%d",
            self.scheme_name,
            retrain_count,
            self.total_retrains,
            bar,
            self.total_bars,
        )
        start, stop = self.geometry.train_window(combined_idx)
        window = self.combined.iloc[start:stop].copy()
        if len(window) < 2:
            return _RetrainOutcome(fallback, 0, trained=False)
        labelled = self.matured(
            self.fe.transform(window, include_target=True),
            combined_idx,
        )
        if len(labelled) < self.geometry.min_train_bars:
            logger.info(
                "[rolling_gen:%s] %d labelled bars at bar %d is below "
                "min_train_bars=%d; keeping the previous model",
                self.scheme_name,
                len(labelled),
                bar,
                self.geometry.min_train_bars,
            )
            return _RetrainOutcome(fallback, len(labelled), trained=False)
        model = self.model_factory(self.model_config)
        model.train(labelled)
        return _RetrainOutcome(model, len(labelled), trained=True)

    def score(
        self,
        combined_idx: int,
        model: XGBoostDirectionModel,
        probabilities: npt.NDArray[np.float64],
        *,
        bar: int,
    ) -> bool:
        """Write this bar's class probabilities; report whether it scored.

        A bar whose features did not survive the feature warm-up is left
        unscored.  Borrowing the previous bar's probability would answer a
        question about a bar that has no answer.
        """
        start, stop = self.geometry.predict_context(combined_idx)
        context = self.combined.iloc[start:stop].copy()
        features = self.fe.transform(context, include_target=False)
        if combined_idx not in features.index:
            return False
        proba = model.predict_proba(features.loc[[combined_idx]])
        # A class the model never saw keeps probability 0 rather than a guess.
        probabilities[bar, 0] = float(proba[1].iloc[0]) if 1 in proba.columns else 0.0
        probabilities[bar, 1] = float(proba[-1].iloc[0]) if -1 in proba.columns else 0.0
        return True


def rolling_outcome(
    train_data: pd.DataFrame,
    test_df: pd.DataFrame,
    scheme: RollingScheme,
    *,
    model: XGBoostDirectionModel,
    model_config: XGBoostModelConfig,
    fe: FeatureTransformer,
    lookback: int,
    retrain_interval: int,
    label_horizon: int,
    model_factory: Callable[[XGBoostModelConfig], XGBoostDirectionModel] = (
        XGBoostDirectionModel
    ),
) -> RollingOutcome:
    """Score every bar of ``test_df`` and report which roll scored it.

    Args:
        train_data: History kept by ``fit()``; rows overlapping the scored frame
            are dropped before concatenation.
        test_df: Bars to score, in chronological order.
        scheme: The declared geometry.
        model: The model trained on train_data by ``fit()``. It may be reused
            only when all of that training frame precedes the scoring frame.
        model_config: Configuration handed to each retrained model.
        fe: Feature engineer producing training features and scoring context.
        lookback: The strategy's feature lookback, used for unset scheme fields.
        retrain_interval: The strategy's classic cadence, used for an unset step.
        label_horizon: Bars a label looks ahead. A training row whose label reads
            past the bar being scored is dropped before fitting.
        model_factory: Builds a fresh untrained model; overridable for tests.

    Returns:
        The per-bar probabilities (a bar whose features could not be computed is
        NaN in both columns), the roll that scored each bar, and one trace entry
        per roll describing what it actually trained on.

    Raises:
        ValueError: If either frame lacks a usable time identity, the scheme
            cannot form a usable window, or ``label_horizon`` is not positive.
    """
    if label_horizon < 1:
        msg = f"label_horizon must be at least 1 bar, got {label_horizon}"
        raise ValueError(msg)
    history, scored = align_scoring_history(train_data, test_df)
    geometry = resolve_geometry(
        scheme,
        lookback=lookback,
        retrain_interval=retrain_interval,
    )
    combined = pd.concat([history, scored], ignore_index=True)
    test_offset = len(history)
    total_bars = len(scored)
    # The declared schedule names the windows; the loop below records what each
    # one actually did. Both come from the same geometry, and the traces carry
    # the schedule's own RollWindow so a trace can be compared by equality.
    schedule = enumerate_rolls(
        scheme,
        n_scored_bars=total_bars,
        history_bars=test_offset,
        lookback=lookback,
        retrain_interval=retrain_interval,
    )
    bar_roll = [-1] * total_bars
    retrains: dict[int, _RetrainOutcome] = {}

    probabilities = np.full((total_bars, len(_PROBABILITY_COLUMNS)), np.nan)
    scored_mask = np.zeros(total_bars, dtype=bool)
    # Clipping the frame cannot remove future information from an already fitted
    # model. An overlapping replay must wait for its first successful refit.
    initial_model_available = bool(len(history) and len(history) == len(train_data))
    current_model = model if initial_model_available else None
    model_was_available = current_model is not None
    total_retrains = (total_bars - 1) // geometry.step + 1
    retrainer = _Retrainer(
        combined=combined,
        geometry=geometry,
        scheme_name=scheme.name,
        fe=fe,
        model_config=model_config,
        model_factory=model_factory,
        total_retrains=total_retrains,
        total_bars=total_bars,
        label_horizon=label_horizon,
    )

    t_start = time.monotonic()
    logger.info(
        "[rolling_gen:%s] Starting: %d bars, %s, step=%d, %d retrains, "
        "window=%s, min_train_bars=%d",
        scheme.name,
        total_bars,
        geometry.rtype.value,
        geometry.step,
        total_retrains,
        "expanding" if geometry.window_size is None else geometry.window_size,
        geometry.min_train_bars,
    )

    retrain_count = 0
    for bar in range(total_bars):
        combined_idx = test_offset + bar
        roll_index = bar // geometry.step
        if bar % geometry.step == 0:
            retrain_count += 1
            retrain = retrainer.retrain(
                combined_idx,
                fallback=current_model,
                retrain_count=retrain_count,
                bar=bar,
            )
            current_model = retrain.model
            retrains[roll_index] = retrain
        if current_model is not None:
            model_was_available = True
            scored_mask[bar] = retrainer.score(
                combined_idx,
                current_model,
                probabilities,
                bar=bar,
            )
        if scored_mask[bar]:
            bar_roll[bar] = roll_index

    elapsed = time.monotonic() - t_start
    logger.info(
        "[rolling_gen:%s] Complete: %d bars in %.1fs, %d scored",
        scheme.name,
        total_bars,
        elapsed,
        int(scored_mask.sum()),
    )
    if total_bars and not scored_mask.any():
        if not model_was_available:
            raise ValueError(
                "rolling replay has no causal model: its initial training frame "
                "overlaps scoring and no scheduled window could fit a model; "
                "provide earlier training history or revise the rolling geometry"
            )
        # Every bar lost its features, so every signal would be a silent 0 and
        # the run would look like a strategy that simply never trades.
        msg = (
            f"rolling scheme {scheme.name!r} scored none of {total_bars} bars: "
            f"predict_lookback={geometry.predict_lookback} is shorter than the "
            "feature warm-up, so the feature engineer drops every scored bar; "
            "raise rolling_gen.predict_lookback"
        )
        raise ValueError(msg)
    scored_per_roll = Counter(roll for roll in bar_roll if roll >= 0)
    traces: list[RollTrace] = []
    for window in schedule:
        record = retrains.get(window.index)
        if record is None:
            msg = (
                f"rolling scheme {scheme.name!r} enumerated {len(schedule)} rolls "
                f"but never retrained for roll {window.index}"
            )
            raise ValueError(msg)
        traces.append(
            RollTrace(
                window=window,
                n_train_rows=record.n_train_rows,
                trained=record.trained,
                bars_scored=scored_per_roll.get(window.index, 0),
            )
        )
    return RollingOutcome(
        probabilities=pd.DataFrame(probabilities, columns=list(_PROBABILITY_COLUMNS)),
        bar_roll=tuple(bar_roll),
        rolls=tuple(traces),
        scheme_name=scheme.name,
        geometry=geometry,
        label_horizon=label_horizon,
        initial_model_available=initial_model_available,
    )


def rolling_probabilities(
    train_data: pd.DataFrame,
    test_df: pd.DataFrame,
    scheme: RollingScheme,
    *,
    model: XGBoostDirectionModel,
    model_config: XGBoostModelConfig,
    fe: FeatureTransformer,
    lookback: int,
    retrain_interval: int,
    label_horizon: int,
    model_factory: Callable[[XGBoostModelConfig], XGBoostDirectionModel] = (
        XGBoostDirectionModel
    ),
) -> pd.DataFrame:
    """The probabilities alone, for callers that do not need the provenance."""
    return rolling_outcome(
        train_data,
        test_df,
        scheme,
        model=model,
        model_config=model_config,
        fe=fe,
        lookback=lookback,
        retrain_interval=retrain_interval,
        label_horizon=label_horizon,
        model_factory=model_factory,
    ).probabilities


def signals_from_probabilities(
    probabilities: pd.DataFrame,
    *,
    threshold: float,
    cooldown: int,
) -> npt.NDArray[np.intp]:
    """Turn per-bar probabilities into ``1``/``-1``/``0`` signals.

    The cooldown is applied here, after every bar has been scored, so it can
    only suppress entries and never change which bars trained which model.
    """
    total = len(probabilities)
    signals = np.zeros(total, dtype=int)
    p_long = probabilities["p_long"].to_numpy()
    p_short = probabilities["p_short"].to_numpy()
    last_trade_bar = -cooldown
    for bar in range(total):
        if bar - last_trade_bar < cooldown:
            continue
        if np.isnan(p_long[bar]) or np.isnan(p_short[bar]):
            continue
        if p_long[bar] >= threshold and p_long[bar] >= p_short[bar]:
            signals[bar] = 1
            last_trade_bar = bar
        elif p_short[bar] >= threshold:
            signals[bar] = -1
            last_trade_bar = bar
    return signals


def run_rolling(
    train_data: pd.DataFrame,
    test_df: pd.DataFrame,
    scheme: RollingScheme,
    *,
    model: XGBoostDirectionModel,
    model_config: XGBoostModelConfig,
    fe: FeatureTransformer,
    threshold: float,
    cooldown: int,
    lookback: int,
    retrain_interval: int,
    label_horizon: int,
    model_factory: Callable[[XGBoostModelConfig], XGBoostDirectionModel] = (
        XGBoostDirectionModel
    ),
) -> npt.NDArray[np.intp]:
    """Score ``test_df`` with the scheme and map probabilities to signals.

    Same arguments as :func:`rolling_probabilities`, plus the decision rule the
    direction strategy applies to each bar's probabilities.
    """
    return signals_from_probabilities(
        rolling_probabilities(
            train_data,
            test_df,
            scheme,
            model=model,
            model_config=model_config,
            fe=fe,
            lookback=lookback,
            retrain_interval=retrain_interval,
            label_horizon=label_horizon,
            model_factory=model_factory,
        ),
        threshold=threshold,
        cooldown=cooldown,
    )
