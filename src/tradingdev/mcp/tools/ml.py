"""ML discovery and bounded research tools over verified saved identities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from mcp_types import ToolAnnotations

from tradingdev.app.contracts.common import ErrorResponse
from tradingdev.app.contracts.ml import (
    BuildDatasetResponse,
    BuildPanelResponse,
    LeaderboardPageResponse,
    MlContractError,
    MlContractResponse,
    PredictionEvaluationResponse,
    PredictionResponse,
    RuleEvaluationResponse,
    TrainModelResponse,
)

if TYPE_CHECKING:
    from mcp.server import MCPServer

    from tradingdev.app.ml_service import MLService


def register(mcp: MCPServer, service: MLService) -> None:
    """Register the ML tools against a shared service instance."""

    read_only = ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @mcp.tool(annotations=read_only)
    def get_ml_contract() -> MlContractResponse | MlContractError:
        """Discover features, models, validation schemes and ranking metrics.

        Read guidance for dataset preparation, model validation, evaluation semantics
        and ranking limitations before selecting a workflow. Does not fetch data.
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
        read_only_hint=False,
        destructive_hint=True,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @mcp.tool(annotations=write_artifact)
    def evaluate_signal(
        dataset_id: str,
        model_id: str | None = None,
        column: str | None = None,
        threshold: float | None = None,
        direction: Literal["above", "below"] = "above",
        window: int = 60,
        n_quantiles: int = 5,
        record: bool = False,
    ) -> PredictionEvaluationResponse | RuleEvaluationResponse | ErrorResponse:
        """Evaluate a saved dataset with a model or a threshold rule.

        model_id evaluates every row with that model; without it, column+threshold
        build a rule signal and record it as a rule candidate. record=True adds
        the result to the leaderboard. Read get_ml_contract guidance.
        """
        try:
            if model_id is not None:
                if column is not None or threshold is not None:
                    raise ValueError(
                        "a model evaluation cannot also declare a rule column or "
                        "threshold; pass model_id alone, or column and threshold "
                        "alone for a rule"
                    )
                result = service.evaluate_prediction(
                    model_id=model_id,
                    dataset_id=dataset_id,
                    window=window,
                    n_quantiles=n_quantiles,
                    record=record,
                )
            elif column is not None and threshold is not None:
                result = service.evaluate_rule(
                    dataset_id=dataset_id,
                    column=column,
                    threshold=threshold,
                    direction=direction,
                    window=window,
                    n_quantiles=n_quantiles,
                    record=record,
                )
            else:
                raise ValueError(
                    "pass model_id to evaluate a saved model, or column and "
                    "threshold to evaluate a rule"
                )
        except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
            return ErrorResponse(success=False, error=str(exc), code="invalid_signal")
        if result.get("success") is False:
            return ErrorResponse.model_validate(result)
        if model_id is not None:
            return PredictionEvaluationResponse.model_validate(result)
        return RuleEvaluationResponse.model_validate(result)

    @mcp.tool(annotations=write_artifact)
    def build_dataset(
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        feature_set: str,
        horizon: int = 1,
        source: str = "binance_vision",
        market_type: str = "futures/um",
    ) -> BuildDatasetResponse | ErrorResponse:
        """Build a saved single-symbol feature/label dataset from cached bars.

        Choose features via get_ml_contract; source/market via list_data_sources.
        Requires queries acquired >=7 days after the interval ended; never downloads.
        Returns dropped rows and dataset_id for training/evaluation.
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
                market_type=market_type,
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
        market_type: str = "futures/um",
        calendar_mode: str = "intersection",
        missing_policy: str = "drop",
        max_staleness: int | None = None,
    ) -> BuildPanelResponse | ErrorResponse:
        """Build an immutable multi-symbol dataset on a shared calendar.

        Read get_ml_contract guidance for calendar_mode, missing_policy and
        max_staleness. Inspect returned coverage/drop/fill counts before training.
        Evaluation of a panel-trained model requires a single-symbol dataset.
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
                market_type=market_type,
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
        """Fit a registered model and save its verified artifact identity.

        Choose model_kind/validation from get_ml_contract; unknown parameters fail.
        Optional purged folds estimate the training procedure, while the saved model
        still fits every row. Training-row accuracy/predictions remain in-sample.
        Read contract guidance for panel restrictions and validation interpretation.
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
    def predict(
        model_id: str,
        dataset_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> PredictionResponse | ErrorResponse:
        """Preview a verified model's predictions in saved dataset row order.

        limit is 1..500; n_rows is the full count and next_offset continues the
        preview. evaluate_signal takes the same IDs and uses all rows internally.
        Training-data predictions remain in-sample. Read get_ml_contract guidance.
        """
        try:
            result = service.predict_page(
                model_id=model_id, dataset_id=dataset_id, offset=offset, limit=limit
            )
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
        dataset_id: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        horizon: int | None = None,
        feature_set: str | None = None,
        family: str | None = None,
        offset: int = 0,
        limit: int = 20,
        allow_mixed: bool = False,
        allow_parameter_axes: bool = False,
    ) -> LeaderboardPageResponse | ErrorResponse:
        """Page recorded evaluations ranked with the metric's declared direction.

        Choose the metric and filters from get_ml_contract guidance. Omitting
        family ranks every method together and reports the mix. Follow
        next_offset; inspect caveats.
        """
        try:
            result = service.leaderboard_page(
                metric,
                dataset_id=dataset_id,
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                feature_set=feature_set,
                family=family,
                offset=offset,
                limit=limit,
                allow_mixed=allow_mixed,
                allow_parameter_axes=allow_parameter_axes,
            )
        except (ValueError, TypeError) as exc:
            return ErrorResponse(
                success=False, error=str(exc), code="invalid_leaderboard_request"
            )
        if result.get("success") is False:
            return ErrorResponse.model_validate(result)
        return LeaderboardPageResponse.model_validate(result)
