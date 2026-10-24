"""Complete, hashed execution requests independent of jobs and storage."""

from __future__ import annotations

import datetime as dt  # noqa: TC003
import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from tradingdev.domain.backtest.schemas import BacktestRunConfig, ParallelConfig
from tradingdev.domain.data.requirements import DataRequirement
from tradingdev.domain.data.schemas import DataConfig
from tradingdev.domain.optimization.grid_search import metric_direction
from tradingdev.domain.performance.catalog import METRIC_CATALOG
from tradingdev.domain.performance.sampling import (
    DAILY_EQUITY_METRICS,
    daily_observation_unavailable_reason,
)
from tradingdev.domain.strategies.capabilities import ExecutionKind as ExecutionKind
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.shared.utils.json_values import strict_json_value


class ManifestError(ValueError):
    """An execution request is invalid or differs from its recorded digest."""


def _json_value(value: object, *, allow_dates: bool = False) -> JsonValue:
    """Translate the shared strict input policy into a manifest error."""
    try:
        return strict_json_value(value, allow_dates=allow_dates)
    except ValueError as exc:
        raise ManifestError(f"Invalid execution JSON: {exc}") from exc


def _resolve_config(value: object) -> dict[str, JsonValue]:
    copied = _json_value(value, allow_dates=True)
    if not isinstance(copied, dict):
        raise ManifestError("Execution config must be an object")
    try:
        config = BacktestRunConfig.model_validate(copied)
        parallel_value = copied.get("parallel")
        parallel = ParallelConfig.model_validate(
            {} if parallel_value is None else parallel_value
        )
    except ValidationError as exc:
        raise ManifestError(f"Invalid execution config: {exc}") from exc
    resolved = config.model_dump(mode="python")
    resolved["parallel"] = parallel.model_dump(mode="python")
    result = _json_value(resolved, allow_dates=True)
    assert isinstance(result, dict)
    return result


class OptimizationSpec(BaseModel):
    """The complete search and estimation policy for one optimization."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    param_ranges: dict[str, list[JsonValue]]
    optimization_metric: str = Field(min_length=1)
    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date
    direction: Literal["maximize", "minimize"]
    trial_timeout_seconds: int = Field(default=300, gt=0, strict=True)

    @model_validator(mode="before")
    @classmethod
    def resolve_new_objective(cls, value: object) -> object:
        """Resolve omitted direction once; saved policies retain their meaning."""
        if isinstance(value, dict) and "direction" not in value:
            metric_id = value.get("optimization_metric")
            if isinstance(metric_id, str):
                return {**value, "direction": metric_direction(metric_id)}
        return value

    def validate_new_request(self, backtest: dict[str, JsonValue]) -> None:
        """Validate applicability only when creating a new execution request."""
        direction = metric_direction(self.optimization_metric)
        definition = METRIC_CATALOG[self.optimization_metric]
        if self.direction != direction:
            raise ManifestError(
                f"Metric '{self.optimization_metric}' requires direction '{direction}'"
            )
        if backtest["mode"] not in definition.modes:
            raise ManifestError(
                f"Metric '{self.optimization_metric}' is not applicable to "
                f"mode '{backtest['mode']}'"
            )
        if (
            definition.requires_annualization
            and backtest.get("periods_per_year") is None
        ):
            raise ManifestError(
                f"Metric '{self.optimization_metric}' requires periods_per_year"
            )
        if self.optimization_metric in DAILY_EQUITY_METRICS:
            frequency = backtest["timeframe"]
            assert isinstance(frequency, str)
            reason = daily_observation_unavailable_reason(frequency)
            if reason is not None:
                raise ManifestError(
                    f"Metric '{self.optimization_metric}' requires daily or finer "
                    f"bars; timeframe '{frequency}': {reason}"
                )

    @field_validator("param_ranges", mode="before")
    @classmethod
    def finite_nonempty_ranges(cls, value: object) -> dict[str, list[JsonValue]]:
        """Fix parameter traversal order while preserving each candidate order."""
        copied = _json_value(value)
        if not isinstance(copied, dict) or not copied:
            raise ManifestError("param_ranges must be a non-empty object")
        ranges: dict[str, list[JsonValue]] = {}
        for name in sorted(copied):
            values = copied[name]
            if not isinstance(values, list) or not values:
                raise ManifestError(f"param_ranges['{name}'] must be a non-empty list")
            ranges[name] = values
        return ranges

    @field_validator(
        "train_start", "train_end", "test_start", "test_end", mode="before"
    )
    @classmethod
    def calendar_day(cls, value: object) -> dt.date:
        """Reject timestamps, since search boundaries are inclusive dates."""
        if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
            return value
        if isinstance(value, str):
            try:
                parsed = dt.date.fromisoformat(value)
            except ValueError as exc:
                raise ManifestError("Optimization dates must use YYYY-MM-DD") from exc
            if parsed.isoformat() == value:
                return parsed
        raise ManifestError("Optimization dates must use YYYY-MM-DD")

    @model_validator(mode="after")
    def ordered_periods(self) -> Self:
        """Keep training and test calendar days ordered and disjoint."""
        if not (self.train_start < self.train_end < self.test_start < self.test_end):
            raise ManifestError(
                "Dates must satisfy: train_start < train_end < test_start < test_end "
                "(inclusive calendar days, without overlap)"
            )
        return self

    @property
    def total_combinations(self) -> int:
        """Derive search size without persisting duplicate state."""
        return math.prod(len(values) for values in self.param_ranges.values())


class ExecutionManifest(BaseModel):
    """A versioned execution request with a digest over all effective settings.

    Pydantic freezes field assignment, not nested dictionaries and lists. Consumers
    must verify before use and mutate only the independent ``config_copy()`` value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[5]
    kind: ExecutionKind
    config: dict[str, JsonValue]
    strategy_execution: StrategyExecution
    optimization: OptimizationSpec | None = None
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("config", mode="before")
    @classmethod
    def persisted_config(cls, value: object) -> dict[str, JsonValue]:
        """Decode saved values without consulting current execution defaults."""
        copied = _json_value(value)
        if not isinstance(copied, dict):
            raise ManifestError("Execution config must be an object")
        return copied

    @field_validator("optimization", mode="before")
    @classmethod
    def complete_saved_search(cls, value: object) -> object:
        """Persist every policy field, even when a new request uses defaults."""
        if isinstance(value, dict):
            missing = OptimizationSpec.model_fields.keys() - value.keys()
            if missing:
                raise ManifestError(
                    "Saved optimization policy is missing fields: "
                    + ", ".join(sorted(missing))
                )
        return value

    @model_validator(mode="after")
    def valid_manifest(self) -> Self:
        """A decoded manifest must satisfy its mode and recorded digest."""
        self.verify()
        return self

    @classmethod
    def create(
        cls,
        *,
        kind: ExecutionKind,
        config: dict[str, Any],
        strategy_execution: StrategyExecution,
        optimization: OptimizationSpec | None = None,
    ) -> Self:
        """Resolve an independent request and compute its canonical SHA-256."""
        try:
            resolved = _resolve_config(config)
            strategy = StrategyExecution.model_validate(
                strategy_execution.model_dump(mode="python")
            )
            search = (
                OptimizationSpec.model_validate(optimization.model_dump(mode="python"))
                if optimization is not None
                else None
            )
            if search is not None:
                backtest = resolved["backtest"]
                assert isinstance(backtest, dict)
                search.validate_new_request(backtest)
            payload = {
                "schema_version": 5,
                "kind": kind,
                "config": resolved,
                "strategy_execution": strategy.model_dump(mode="python"),
                "optimization": search.model_dump(mode="python") if search else None,
            }
            return cls(
                schema_version=5,
                kind=kind,
                config=resolved,
                strategy_execution=strategy,
                optimization=search,
                manifest_hash=cls._digest(payload),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ManifestError):
                raise
            raise ManifestError(f"Invalid execution manifest: {exc}") from exc

    @staticmethod
    def _digest(payload: dict[str, Any]) -> str:
        finite_payload = _json_value(payload, allow_dates=True)
        encoded = json.dumps(
            finite_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def verify(self, expected_hash: str | None = None) -> None:
        """Reject invalid mode, nested mutations, and a mismatching job digest."""
        _json_value(self.config)
        _json_value(self.strategy_execution.constructor_kwargs)
        if self.optimization is not None:
            _json_value(self.optimization.param_ranges)
        is_walk_forward = self.config.get("validation") is not None
        if is_walk_forward != (self.kind == "walk_forward"):
            raise ManifestError("Execution kind does not match config.validation")
        if (self.optimization is not None) != (self.kind == "optimization"):
            raise ManifestError(
                "Only optimization execution requires an optimization spec"
            )
        if expected_hash is not None and expected_hash != self.manifest_hash:
            raise ManifestError("Execution manifest does not match the expected hash")
        payload = self.model_dump(mode="python", exclude={"manifest_hash"})
        if self._digest(payload) != self.manifest_hash:
            raise ManifestError("Execution manifest hash does not match its content")

    def config_copy(self) -> dict[str, Any]:
        """Return an independent execution config after checking its integrity."""
        self.verify()
        return deepcopy(self.config)

    def config_for_execution(self) -> dict[str, Any]:
        """Reject saved settings that current execution would change implicitly.

        Historical manifests remain readable through ``config_copy()`` even when
        their settings no longer match the current runtime. An execution boundary
        must call this method before allowing current models to interpret them.
        Canonical JSON comparison also distinguishes booleans, integers and floats.
        """
        self.verify()
        resolved = _resolve_config(self.config)
        if self._digest(resolved) != self._digest(self.config):
            raise ManifestError(
                "Saved execution settings differ from the current runtime schema; "
                "submit a new job"
            )
        data = self.config.get("data")
        if not isinstance(data, dict):
            raise ManifestError("Saved execution requires explicit data settings")
        try:
            data_config = DataConfig.model_validate(data).model_dump(mode="python")
            data_config["requirements"] = DataRequirement.model_validate(
                data.get("requirements")
            ).model_dump(mode="python")
        except ValidationError as exc:
            raise ManifestError(f"Invalid saved data settings: {exc}") from exc
        if self._digest(data_config) != self._digest(data):
            raise ManifestError(
                "Saved data settings differ from the current runtime schema; "
                "submit a new job"
            )
        return deepcopy(self.config)
