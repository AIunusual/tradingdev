"""ML MCP tools: enumerate the safe ML vocabulary and work through a signal.

``get_ml_contract`` is pure discovery of the audited, enumerable ML options and
``evaluate_signal`` runs the signal-evaluation suite on caller-supplied arrays;
both are read-only and idempotent.

``build_dataset`` and ``train_model`` write into the workspace and are
idempotent, so repeating either yields the same id. ``predict`` is read-only
and closes the loop: it scores a stored model on a stored dataset and returns
the signal plus its realised forward returns, ready for ``evaluate_signal``.

``evaluate_signal`` stays read-only unless ``record=True`` is passed, in which
case it stores the result for later ranking. ``leaderboard`` ranks those stored
results, refusing to order candidates that were not measured on the same
conditions or on a metric that has no declared direction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.types import ToolAnnotations

from tradingdev.app.contracts.common import ErrorResponse
from tradingdev.app.contracts.ml import (
    BuildDatasetResponse,
    BuildPanelResponse,
    LeaderboardResponse,
    MlContractError,
    MlContractResponse,
    PredictionResponse,
    SignalEvaluationResponse,
    TrainModelResponse,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.ml_service import MLService


def register(mcp: FastMCP, service: MLService) -> None:
    """Register the ML tools against a shared service instance."""

    read_only = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )

    @mcp.tool(annotations=read_only)
    def get_ml_contract() -> MlContractResponse | MlContractError:
        """List the enumerable ML vocabulary: feature sets, model kinds, split schemes.

        This is the menu an LLM uses to make ML choices without writing code.
        It exposes audited feature-set names and their column contracts, the
        registered direction-model kinds with default hyperparameters, the
        leak-aware validation schemes, the available processor types, and the
        supported IC methods and quantile range. Read-only; no data is fetched.
        """
        result = service.get_contract()
        if result.get("success") is False:
            return MlContractError.model_validate(result)
        return MlContractResponse.model_validate(result)

    # Recording an evaluation persists a row, so this tool is not read-only in
    # the general case; it writes nothing unless `record=True` is passed, and
    # recording the same candidate twice rewrites one row rather than adding a
    # second, which keeps it idempotent.
    write_artifact = ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )

    @mcp.tool(annotations=write_artifact)
    def evaluate_signal(
        signal: list[float],
        forward_returns: list[float],
        horizon: int | None = None,
        window: int = 60,
        n_quantiles: int = 5,
        record: bool = False,
        model_id: str | None = None,
        dataset_id: str | None = None,
    ) -> SignalEvaluationResponse | ErrorResponse:
        """Evaluate a prediction against realised forward returns.

        Computes the full signal-quality suite without any backtest: IC and Rank
        IC means/IR/t-stat (with overlap-corrected sample size), hit rate,
        turnover, prediction quantile strata (monotonicity and long/short
        spread), and a simulated long/short book built from the signal ranks.
        Both the Pearson and Spearman families are returned. Provide a signal
        and its realised forward returns as equal-length numeric arrays. Returns
        a validation error on length mismatch, non-finite values, or too few
        finite bars for the requested window/quantiles.

        Pass record=True together with the dataset_id the signal was computed on
        to store the result for leaderboard ranking; model_id names the model
        that produced the signal, when there is one. Recording is idempotent:
        re-evaluating the same model on the same data rewrites its row instead
        of adding a second candidate. A recorded signal must cover the whole
        dataset, and its horizon must match the dataset's (omit horizon to
        adopt it), so a record can never claim data it was not measured on.
        """
        try:
            result = service.evaluate_signal(
                signal,
                forward_returns,
                horizon=horizon,
                window=window,
                n_quantiles=n_quantiles,
                record=record,
                model_id=model_id,
                dataset_id=dataset_id,
            )
        except (ValueError, TypeError) as exc:
            return ErrorResponse(success=False, error=str(exc), code="invalid_signal")
        if result.get("success") is False:
            return ErrorResponse.model_validate(result)
        return SignalEvaluationResponse.model_validate(result)

    @mcp.tool(annotations=write_artifact)
    def build_dataset(
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        feature_set: str,
        horizon: int = 1,
        source: str = "binance_vision",
    ) -> BuildDatasetResponse | ErrorResponse:
        """Build a labelled feature dataset from cached OHLCV bars.

        Pick a feature set from get_ml_contract; it supplies the columns and
        their contract. The label is the sign of the realised forward return
        over `horizon` bars (-1, 0, 1). Incomplete rows from indicator
        warm-up and the unlabelled tail are dropped and reported as
        dropped_rows. Requires cached market data for the whole range: this
        tool never downloads, so run ensure_data first if the range is not
        cached. Returns a dataset_id to pass to train_model. Deterministic:
        the same inputs always yield the same dataset_id.
        """
        try:
            result = service.build_dataset(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                feature_set=feature_set,
                horizon=horizon,
                source=source,
            )
        except (ValueError, KeyError) as exc:
            return ErrorResponse(
                success=False, error=str(exc), code="invalid_dataset_request"
            )
        if result.get("success") is False:
            return ErrorResponse.model_validate(result)
        return BuildDatasetResponse.model_validate(result)

    @mcp.tool(annotations=write_artifact)
    def build_panel(
        symbols: list[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        feature_set: str,
        horizon: int = 1,
        source: str = "binance_vision",
        calendar_mode: str = "intersection",
        missing_policy: str = "drop",
        max_staleness: int | None = None,
    ) -> BuildPanelResponse | ErrorResponse:
        """Build a multi-symbol panel dataset on one shared trading calendar.

        Use this instead of build_dataset when the model should see several
        symbols at once. Each symbol is featurised and labelled on its own bars
        first, then every frame is put on a common calendar: a date enters the
        panel only when all symbols can supply it, so there are no
        half-populated rows.

        calendar_mode: "intersection" (default) keeps only dates every symbol
        traded; "union" keeps every date any symbol traded and then applies
        missing_policy. missing_policy: "drop" removes dates a symbol cannot
        supply, "forward_fill" carries its last bar (bounded by max_staleness
        dates; None is unbounded), "error" refuses the panel on any gap.

        The response reports the resulting calendar: per-symbol coverage, the
        dates that were dropped, and how many rows were forward-filled, so a
        panel dominated by stale carries is visible instead of silent. Requires
        cached data for every symbol: run ensure_data first. Deterministic; the
        same inputs always yield the same dataset_id, which train_model and
        predict accept like any other dataset.
        """
        try:
            result = service.build_panel(
                symbols=symbols,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                feature_set=feature_set,
                horizon=horizon,
                source=source,
                calendar_mode=calendar_mode,
                missing_policy=missing_policy,
                max_staleness=max_staleness,
            )
        except (ValueError, KeyError) as exc:
            return ErrorResponse(
                success=False, error=str(exc), code="invalid_panel_request"
            )
        if result.get("success") is False:
            return ErrorResponse.model_validate(result)
        return BuildPanelResponse.model_validate(result)

    @mcp.tool(annotations=write_artifact)
    def train_model(
        dataset_id: str,
        model_kind: str,
        parameters: dict[str, object] | None = None,
        validation: str | None = None,
        n_folds: int = 5,
        embargo: int = 0,
        min_train_bars: int | None = None,
    ) -> TrainModelResponse | ErrorResponse:
        """Train a registered direction model on a dataset built by build_dataset.

        Choose model_kind from get_ml_contract; parameters override that kind's
        defaults and must be names it declares, so a typo is rejected instead of
        silently training with defaults. The trained estimator is persisted
        through the model artifact registry and the returned model_id is a
        fingerprint of the hyper-parameters, the feature contract and the model
        blob. feature_importance is gain-based and sorted descending.
        in_sample_accuracy is measured on the training rows and is NOT an
        out-of-sample estimate; use evaluate_signal for that judgement.

        Pass validation (a scheme name from get_ml_contract) to also fit the
        same kind once per fold on purged training bars and score it on that
        fold's unseen test window. The purge is the dataset's own label
        horizon - a bar whose label is decided by prices inside the test window
        is removed from training - and embargo adds further bars around each
        test window for serially correlated features. The returned validation
        block reports, per fold, how many bars were purged and embargoed and
        what the fold scored, plus their average. Note the artifact is still
        fit on every bar: the folds estimate this way of training, not that
        particular fitted object.
        """
        try:
            result = service.train_model(
                dataset_id=dataset_id,
                model_kind=model_kind,
                parameters=parameters,
                validation=validation,
                n_folds=n_folds,
                embargo=embargo,
                min_train_bars=min_train_bars,
            )
        except (ValueError, KeyError) as exc:
            return ErrorResponse(
                success=False, error=str(exc), code="invalid_training_request"
            )
        if result.get("success") is False:
            return ErrorResponse.model_validate(result)
        return TrainModelResponse.model_validate(result)

    @mcp.tool(annotations=read_only)
    def predict(model_id: str, dataset_id: str) -> PredictionResponse | ErrorResponse:
        """Score a trained model on a dataset to produce a signal.

        Loads the model artifact by model_id (re-verifying its blob hash, so a
        tampered or corrupted model is refused), checks the dataset's feature
        contract matches the one the model was trained on, and returns the
        per-row signal together with the realised forward_returns aligned to it.
        Feed that pair straight into evaluate_signal to judge the signal
        out-of-sample. Read-only and deterministic: the same ids always yield
        the same signal.
        """
        try:
            result = service.predict(model_id=model_id, dataset_id=dataset_id)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            return ErrorResponse(
                success=False, error=str(exc), code="invalid_prediction_request"
            )
        if result.get("success") is False:
            return ErrorResponse.model_validate(result)
        return PredictionResponse.model_validate(result)

    @mcp.tool(annotations=read_only)
    def leaderboard(
        metric: str,
        symbol: str | None = None,
        timeframe: str | None = None,
        horizon: int | None = None,
        feature_set: str | None = None,
        top_k: int | None = None,
        allow_mixed: bool = False,
    ) -> LeaderboardResponse | ErrorResponse:
        """Rank evaluations recorded by evaluate_signal(record=True).

        Choose metric from the leaderboard_metrics list in get_ml_contract: only
        metrics with a declared direction can order candidates, so "higher is
        better" is never assumed (turnover and IC std improve when they shrink).

        By default candidates are only ranked together when they were measured
        on the same instrument, bar size, date range, horizon, evaluation
        settings, feature set and feature code - a Rank IC earned on one symbol
        says nothing about one earned on another, and one earned before a
        feature-code fix is not the same candidate as one earned after it. Pass
        allow_mixed=True to override at your own risk, or use
        symbol/timeframe/horizon/feature_set to narrow the field first.

        Read the caveats before trusting the winner: picking the best of N
        candidates is N comparisons, so the top of a long list is the most
        optimistic number in it, and the caveats state the t-statistic the
        winner would have to clear to be called signal rather than selection.
        """
        try:
            result = service.leaderboard(
                metric,
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                feature_set=feature_set,
                top_k=top_k,
                allow_mixed=allow_mixed,
            )
        except (ValueError, TypeError) as exc:
            return ErrorResponse(
                success=False, error=str(exc), code="invalid_leaderboard_request"
            )
        if result.get("success") is False:
            return ErrorResponse.model_validate(result)
        return LeaderboardResponse.model_validate(result)
