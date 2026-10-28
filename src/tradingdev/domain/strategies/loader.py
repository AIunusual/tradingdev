"""Strategy loading for bundled and generated strategies."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from tradingdev.domain.backtest.schemas import ParallelConfig
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.catalog import BundledStrategyCatalog
from tradingdev.domain.strategies.execution import (
    StrategyExecution,
    finite_strategy_value,
    merge_strategy_parameters,
)
from tradingdev.shared.paths import resolve_workspace_root
from tradingdev.shared.utils.json_values import strict_json_value

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine


class StrategyLoader:
    """Load bundled or workspace-generated strategies through one contract."""

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        catalog: BundledStrategyCatalog | None = None,
    ) -> None:
        self._workspace_root = resolve_workspace_root(workspace_root)
        self._generated_root = self._workspace_root / "generated_strategies"
        self._catalog = catalog or BundledStrategyCatalog()

    def create_from_config(
        self,
        raw_config: dict[str, Any],
        engine: BaseBacktestEngine | None,
        parallel_config: ParallelConfig | None = None,
    ) -> BaseStrategy:
        """Resolve and build using the same constructor contract as execution."""
        strategy_cfg = raw_config["strategy"]
        if not isinstance(strategy_cfg, dict):
            msg = "strategy config must be a mapping"
            raise ValueError(msg)

        if parallel_config is None:
            raw_parallel = raw_config.get("parallel")
            parallel_config = ParallelConfig.model_validate(
                {} if raw_parallel is None else raw_parallel
            )
        execution = self.resolve_execution(strategy_cfg)
        return self.create_from_execution(
            strategy_cfg, execution, engine, parallel_config
        )

    def resolve_execution(self, strategy_cfg: dict[str, Any]) -> StrategyExecution:
        """Capture effective settings without constructing a strategy instance."""
        cls = self.load_class(strategy_cfg)
        bundled = strategy_cfg["id"] in self._bundled_class_by_id()
        params = strategy_cfg.get("parameters", {})
        if not isinstance(params, dict):
            raise ValueError("strategy.parameters must be a mapping")
        signature = inspect.signature(cls)
        kwargs: dict[str, Any]
        if bundled:
            kwargs = {}
            if "config" in signature.parameters:
                model = self._bundled_config_model(cls).model_validate(params)
                kwargs["config"] = model.model_dump(mode="python")
            if "fit_config" in signature.parameters and "fit" in strategy_cfg:
                fit = strategy_cfg["fit"]
                if not isinstance(fit, dict):
                    raise ValueError("strategy.fit must be a mapping")
                fit_model = self._bundled_config_model(cls, suffix="FitConfig")
                kwargs["fit_config"] = fit_model.model_validate(fit).model_dump(
                    mode="python"
                )
        else:
            kwargs = deepcopy(params)
        injected = {"backtest_engine", "parallel_config"}
        if injected.intersection(kwargs):
            raise ValueError("Strategy parameters cannot override injected settings")
        for name, parameter in signature.parameters.items():
            if name in injected or name in kwargs:
                continue
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if parameter.default is not inspect.Parameter.empty:
                kwargs[name] = parameter.default
        signature.bind(
            **kwargs,
            **{name: None for name in injected if name in signature.parameters},
        )
        execution = StrategyExecution(
            kind="bundled" if bundled else "generated",
            constructor_kwargs=kwargs,
        )
        self._execution_kwargs(cls, execution, None, None, None)
        return execution

    def create_from_execution(
        self,
        strategy_cfg: dict[str, Any],
        execution: StrategyExecution,
        engine: BaseBacktestEngine | None,
        parallel_config: ParallelConfig | None = None,
        *,
        parameter_overrides: dict[str, Any] | None = None,
    ) -> BaseStrategy:
        """Construct from explicit settings; never fill missing strategy defaults."""
        cls = self._execution_class(strategy_cfg, execution)
        kwargs = self._execution_kwargs(
            cls, execution, engine, parallel_config, parameter_overrides
        )
        return cls(**kwargs)

    def override_execution(
        self,
        strategy_cfg: dict[str, Any],
        execution: StrategyExecution,
        parameters: dict[str, Any],
    ) -> StrategyExecution:
        """Freeze one experiment using the search grid's strict nested overlay."""
        overrides = strict_json_value(parameters)
        if not isinstance(overrides, dict):
            raise ValueError("Strategy parameters must be a mapping")
        kwargs = deepcopy(execution.constructor_kwargs)
        target = kwargs.get("config") if execution.kind == "bundled" else kwargs
        if not isinstance(target, dict):
            raise ValueError("Strategy does not expose searchable parameters")
        target.update(merge_strategy_parameters(target, overrides))
        selected = StrategyExecution(kind=execution.kind, constructor_kwargs=kwargs)
        self.validate_parameter_overrides(strategy_cfg, selected, {})
        return selected

    def validate_parameter_overrides(
        self,
        strategy_cfg: dict[str, Any],
        execution: StrategyExecution,
        parameter_overrides: dict[str, Any],
    ) -> None:
        """Validate one search combination without running its constructor."""
        cls = self._execution_class(strategy_cfg, execution)
        self._execution_kwargs(cls, execution, None, None, parameter_overrides)

    def validate_parameter_grid(
        self,
        strategy_cfg: dict[str, Any],
        execution: StrategyExecution,
        combinations: Iterable[dict[str, Any]],
    ) -> None:
        """Check a lazy grid using one resolved class, without instantiation."""
        cls = self._execution_class(strategy_cfg, execution)
        models = self._execution_models(cls, execution)
        signature = inspect.signature(cls)
        for parameters in combinations:
            self._execution_kwargs(
                cls,
                execution,
                None,
                None,
                parameters,
                models=models,
                signature=signature,
            )

    def _execution_class(
        self, strategy_cfg: dict[str, Any], execution: StrategyExecution
    ) -> type[BaseStrategy]:
        bundled = strategy_cfg.get("id") in self._bundled_class_by_id()
        if bundled != (execution.kind == "bundled"):
            raise ValueError("Strategy execution kind does not match the strategy")
        return self.load_class(strategy_cfg)

    def _execution_kwargs(
        self,
        cls: type[BaseStrategy],
        execution: StrategyExecution,
        engine: BaseBacktestEngine | None,
        parallel_config: ParallelConfig | None,
        parameter_overrides: dict[str, Any] | None,
        *,
        models: dict[str, type[BaseModel]] | None = None,
        signature: inspect.Signature | None = None,
    ) -> dict[str, Any]:
        copied = finite_strategy_value(execution.constructor_kwargs)
        assert isinstance(copied, dict)
        kwargs: dict[str, Any] = copied
        if parameter_overrides is not None:
            target = kwargs.get("config") if execution.kind == "bundled" else kwargs
            if not isinstance(target, dict):
                raise ValueError("Strategy does not expose searchable parameters")
            target.update(merge_strategy_parameters(target, parameter_overrides))
        signature = signature or inspect.signature(cls)
        injected = {"backtest_engine", "parallel_config"}
        if injected.intersection(kwargs):
            raise ValueError("Strategy execution cannot override injected settings")
        for name, parameter in signature.parameters.items():
            if name in injected or parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if name not in kwargs:
                raise ValueError(
                    "Strategy execution is missing explicit constructor setting "
                    f"{name!r}"
                )
        for name, model in (
            models if models is not None else self._execution_models(cls, execution)
        ).items():
            kwargs[name] = self._restore_model(model, kwargs[name])
        if "backtest_engine" in signature.parameters:
            kwargs["backtest_engine"] = engine
        if "parallel_config" in signature.parameters:
            kwargs["parallel_config"] = parallel_config
        signature.bind(**kwargs)
        return kwargs

    def _execution_models(
        self, cls: type[BaseStrategy], execution: StrategyExecution
    ) -> dict[str, type[BaseModel]]:
        if execution.kind != "bundled":
            return {}
        models: dict[str, type[BaseModel]] = {}
        if "config" in execution.constructor_kwargs:
            models["config"] = self._bundled_config_model(cls)
        if execution.constructor_kwargs.get("fit_config") is not None:
            models["fit_config"] = self._bundled_config_model(cls, suffix="FitConfig")
        return models

    @staticmethod
    def _restore_model(model: type[BaseModel], values: object) -> BaseModel:
        """Reject schema drift that adds, removes, or changes recorded values."""
        before = finite_strategy_value(values)
        restored = model.model_validate(deepcopy(before))
        after = finite_strategy_value(restored.model_dump(mode="python"))
        if not StrategyLoader._unchanged_model_values(before, after):
            raise ValueError(
                f"Recorded {model.__name__} settings no longer match its model; "
                "submit a new execution"
            )
        return restored

    @staticmethod
    def _unchanged_model_values(before: object, after: object) -> bool:
        """Allow only identical JSON values or lossless integer-to-float leaves."""
        if isinstance(before, dict) and isinstance(after, dict):
            return before.keys() == after.keys() and all(
                StrategyLoader._unchanged_model_values(value, after[key])
                for key, value in before.items()
            )
        if isinstance(before, list) and isinstance(after, list):
            return len(before) == len(after) and all(
                StrategyLoader._unchanged_model_values(left, right)
                for left, right in zip(before, after, strict=True)
            )
        if type(before) is int and type(after) is float:
            return before == after
        return type(before) is type(after) and before == after

    def load_class(self, strategy_cfg: dict[str, Any]) -> type[BaseStrategy]:
        """Resolve a strategy class from bundled metadata or generated source."""
        strategy_id = self._required_strategy_string(strategy_cfg, "id")
        bundled = self._bundled_class_by_id()
        bundled_class = bundled.get(strategy_id)
        if bundled_class is not None:
            module_name, class_name = bundled_class
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            if not issubclass(cls, BaseStrategy):
                msg = f"{class_name} must inherit from BaseStrategy"
                raise TypeError(msg)
            return cast("type[BaseStrategy]", cls)

        class_name = self._required_strategy_string(strategy_cfg, "class_name")
        source_value = self._required_strategy_string(strategy_cfg, "source_path")
        source_path = Path(str(source_value))
        self._reject_unknown_bundled_source(strategy_id, strategy_cfg, bundled)
        source_hash = strategy_cfg.get("source_hash")
        if source_hash is not None and not isinstance(source_hash, str):
            msg = "strategy.source_hash must be a string"
            raise ValueError(msg)
        module = self._load_module(
            self._resolve_generated_source(source_path), source_hash=source_hash
        )
        cls = getattr(module, class_name, None)
        if cls is None:
            msg = f"Class {class_name!r} not found"
            raise ValueError(msg)
        if not issubclass(cls, BaseStrategy):
            msg = f"{class_name} must inherit from BaseStrategy"
            raise TypeError(msg)
        return cast("type[BaseStrategy]", cls)

    def _bundled_class_by_id(self) -> dict[str, tuple[str, str]]:
        """Discover bundled strategy classes from the bundled catalog."""
        return {
            entry.strategy_id: (entry.module_name, entry.class_name)
            for entry in self._catalog.entries()
        }

    def _reject_unknown_bundled_source(
        self,
        strategy_id: str,
        strategy_cfg: dict[str, Any],
        bundled: dict[str, tuple[str, str]],
    ) -> None:
        source_value = strategy_cfg.get("source_path")
        if not isinstance(source_value, str) or not source_value:
            return
        if not self._is_bundled_source(Path(source_value)):
            return
        known = ", ".join(sorted(bundled))
        msg = (
            f"Bundled strategy id {strategy_id!r} is not recognized. "
            f"Known bundled strategy ids: {known}"
        )
        raise ValueError(msg)

    def _is_bundled_source(self, source_path: Path) -> bool:
        return self._catalog.contains_source(source_path)

    def _bundled_config_model(
        self,
        cls: type[BaseStrategy],
        *,
        suffix: str = "Config",
    ) -> type[BaseModel]:
        module_name = cls.__module__.removesuffix(".strategy") + ".config"
        module = importlib.import_module(module_name)
        exact_name = f"{cls.__name__}{suffix}"
        exact = getattr(module, exact_name, None)
        if (
            inspect.isclass(exact)
            and issubclass(exact, BaseModel)
            and exact is not BaseModel
        ):
            return exact

        candidates = [
            value
            for name, value in vars(module).items()
            if name.endswith(suffix)
            and inspect.isclass(value)
            and issubclass(value, BaseModel)
            and value is not BaseModel
        ]
        if len(candidates) != 1:
            msg = (
                f"Expected one {suffix} model in {module_name} for {cls.__name__}, "
                f"found {len(candidates)}"
            )
            raise ValueError(msg)
        return candidates[0]

    def _required_strategy_string(
        self,
        strategy_cfg: dict[str, Any],
        field: str,
    ) -> str:
        value = strategy_cfg.get(field)
        if not isinstance(value, str) or not value:
            msg = f"strategy.{field} is required"
            raise ValueError(msg)
        return value

    def _resolve_generated_source(self, source_path: Path) -> Path:
        candidate = source_path
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        resolved = candidate.resolve()
        allowed_root = self._generated_root.resolve()
        if not resolved.is_relative_to(allowed_root):
            msg = f"Generated strategy must live under {allowed_root}"
            raise ValueError(msg)
        if not resolved.exists():
            msg = f"Strategy source not found: {resolved}"
            raise FileNotFoundError(msg)
        return resolved

    def _load_module(
        self, source_path: Path, *, source_hash: str | None = None
    ) -> ModuleType:
        spec = importlib.util.spec_from_file_location(
            f"_tradingdev_generated_{source_path.stem}",
            source_path,
        )
        if spec is None or spec.loader is None:
            msg = f"Cannot load module from {source_path}"
            raise ImportError(msg)
        module = importlib.util.module_from_spec(spec)
        # LLM repair loops can rewrite same-size source within one timestamp
        # unit. Read and compile the current source directly so a stale .pyc
        # never changes which revision gets validated or executed.
        source = source_path.read_bytes()
        if (
            source_hash is not None
            and hashlib.sha256(source).hexdigest() != source_hash
        ):
            msg = (
                "Strategy revision source has changed; save and validate a new revision"
            )
            raise ValueError(msg)
        code = compile(source, str(source_path), "exec")
        exec(code, module.__dict__)  # noqa: S102
        return module
