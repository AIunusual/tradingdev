"""Strategy loading for bundled and generated strategies."""

from __future__ import annotations

import base64
import hashlib
import importlib
import inspect
import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from tradingdev.adapters.execution.container_runtime import DockerRuntime
from tradingdev.adapters.execution.generated_strategy import (
    inspect_parameters,
    open_channel,
    open_strategy,
)
from tradingdev.domain.backtest.schemas import ParallelConfig
from tradingdev.domain.randomness import execution_randomness, get_seed
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.catalog import BundledStrategyCatalog
from tradingdev.domain.strategies.execution import (
    GeneratedProgram,
    StrategyExecution,
    finite_strategy_value,
    merge_strategy_parameters,
)
from tradingdev.domain.strategies.signals import strategy_callback
from tradingdev.shared.paths import resolve_workspace_root
from tradingdev.shared.utils.json_values import strict_json_value

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine


class StrategyFactory:
    """Own strategy lifetimes; generated imports and instances are always isolated."""

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        catalog: BundledStrategyCatalog | None = None,
        runtime: DockerRuntime | None = None,
    ) -> None:
        self._workspace_root = resolve_workspace_root(workspace_root)
        self._generated_root = self._workspace_root / "generated_strategies"
        self._catalog = catalog or BundledStrategyCatalog()
        self._runtime = runtime

    @contextmanager
    def open_from_config(
        self,
        raw_config: dict[str, Any],
        engine: BaseBacktestEngine | None,
        parallel_config: ParallelConfig | None = None,
    ) -> Iterator[BaseStrategy]:
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
        with (
            execution_randomness(raw_config.get("random_seed")),
            self.open_from_execution(
                strategy_cfg, execution, engine, parallel_config
            ) as strategy,
        ):
            yield strategy

    def resolve_execution(self, strategy_cfg: dict[str, Any]) -> StrategyExecution:
        """Capture effective settings without constructing a strategy instance."""
        strategy_id = self._required_strategy_string(strategy_cfg, "id")
        bundled = strategy_id in self._bundled_class_by_id()
        params = strategy_cfg.get("parameters", {})
        if not isinstance(params, dict):
            raise ValueError("strategy.parameters must be a mapping")
        if not bundled:
            runtime = self._generated_runtime()
            spec = runtime.resolve()
            source, class_name = self._generated_source(
                strategy_cfg, max_bytes=spec.limits.message_bytes // 2
            )
            program = GeneratedProgram(
                source_hash=hashlib.sha256(source).hexdigest(),
                class_name=class_name,
                runtime=spec,
            )
            return StrategyExecution(
                kind="generated",
                program=program,
                constructor_kwargs=inspect_parameters(
                    runtime, program, source, params, capture=True
                ),
            )
        cls = self._load_bundled_class(strategy_cfg)
        signature = inspect.signature(cls)
        kwargs: dict[str, Any] = {}
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
            kind="bundled",
            constructor_kwargs=kwargs,
        )
        self._execution_kwargs(cls, execution, None, None, None)
        return execution

    @contextmanager
    def open_from_execution(
        self,
        strategy_cfg: dict[str, Any],
        execution: StrategyExecution,
        engine: BaseBacktestEngine | None,
        parallel_config: ParallelConfig | None = None,
        *,
        parameter_overrides: dict[str, Any] | None = None,
        classify_callback_error: bool = False,
    ) -> Iterator[BaseStrategy]:
        """Construct from explicit settings; never fill missing strategy defaults."""
        self._check_kind(strategy_cfg, execution)
        if execution.program is not None:
            program = execution.program
            source, _ = self._generated_source(strategy_cfg, program=program)
            values = execution.constructor_kwargs
            if parameter_overrides is not None:
                values = merge_strategy_parameters(values, parameter_overrides)
            try:
                seed = get_seed()
            except RuntimeError:
                seed = None
            with open_strategy(
                self._generated_runtime(), program, source, values, seed
            ) as strategy:
                yield strategy
            return
        cls = self._load_bundled_class(strategy_cfg)
        kwargs = self._execution_kwargs(
            cls, execution, engine, parallel_config, parameter_overrides
        )
        if classify_callback_error:
            with strategy_callback():
                instance = cls(**kwargs)
        else:
            instance = cls(**kwargs)
        yield instance

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
        selected = StrategyExecution(
            kind=execution.kind, constructor_kwargs=kwargs, program=execution.program
        )
        self.validate_parameter_overrides(strategy_cfg, selected, {})
        return selected

    def validate_execution(
        self,
        strategy_cfg: dict[str, Any],
        execution: StrategyExecution,
    ) -> None:
        """Revalidate captured settings without construction or new defaults."""
        self.validate_parameter_overrides(strategy_cfg, execution, {})

    def validate_parameter_overrides(
        self,
        strategy_cfg: dict[str, Any],
        execution: StrategyExecution,
        parameter_overrides: dict[str, Any],
    ) -> None:
        """Validate one search combination without running its constructor."""
        self._check_kind(strategy_cfg, execution)
        if execution.program is not None:
            program = execution.program
            source, _ = self._generated_source(strategy_cfg, program=program)
            inspect_parameters(
                self._generated_runtime(),
                program,
                source,
                merge_strategy_parameters(
                    execution.constructor_kwargs, parameter_overrides
                ),
                capture=False,
            )
            return
        cls = self._load_bundled_class(strategy_cfg)
        self._execution_kwargs(cls, execution, None, None, parameter_overrides)

    def validate_parameter_grid(
        self,
        strategy_cfg: dict[str, Any],
        execution: StrategyExecution,
        combinations: Iterable[dict[str, Any]],
    ) -> None:
        """Check a lazy grid using one resolved class, without instantiation."""
        self._check_kind(strategy_cfg, execution)
        if execution.program is not None:
            program = execution.program
            source, _ = self._generated_source(strategy_cfg, program=program)
            encoded: str | None = base64.b64encode(source).decode("ascii")
            with open_channel(self._generated_runtime(), program) as channel:
                for parameters in combinations:
                    channel.request(
                        "validate",
                        source=encoded,
                        kwargs=merge_strategy_parameters(
                            execution.constructor_kwargs, parameters
                        ),
                    )
                    encoded = None
            return
        cls = self._load_bundled_class(strategy_cfg)
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

    def _check_kind(
        self, strategy_cfg: dict[str, Any], execution: StrategyExecution
    ) -> None:
        bundled = strategy_cfg.get("id") in self._bundled_class_by_id()
        if bundled != (execution.kind == "bundled"):
            raise ValueError("Strategy execution kind does not match the strategy")

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
        if not StrategyFactory._unchanged_model_values(before, after):
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
                StrategyFactory._unchanged_model_values(value, after[key])
                for key, value in before.items()
            )
        if isinstance(before, list) and isinstance(after, list):
            return len(before) == len(after) and all(
                StrategyFactory._unchanged_model_values(left, right)
                for left, right in zip(before, after, strict=True)
            )
        if type(before) is int and type(after) is float:
            return before == after
        return type(before) is type(after) and before == after

    def _load_bundled_class(self, strategy_cfg: dict[str, Any]) -> type[BaseStrategy]:
        """Import only a class selected by the installed bundled catalog."""
        strategy_id = self._required_strategy_string(strategy_cfg, "id")
        entry = self._bundled_class_by_id().get(strategy_id)
        if entry is None:
            raise ValueError("Generated strategy classes cannot be loaded on the host")
        module_name, class_name = entry
        cls = getattr(importlib.import_module(module_name), class_name)
        if not inspect.isclass(cls) or not issubclass(cls, BaseStrategy):
            raise TypeError(f"{class_name} must inherit from BaseStrategy")
        return cls

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

    def _generated_runtime(self) -> DockerRuntime:
        if self._runtime is None:
            configured = os.environ.get("TRADINGDEV_RUNTIME_CONTROL_ROOT")
            root = (
                Path(configured)
                if configured
                else self._workspace_root / ".strategy-runtime"
            )
            self._runtime = DockerRuntime(root)
        return self._runtime

    def _generated_source(
        self,
        strategy_cfg: dict[str, Any],
        *,
        program: GeneratedProgram | None = None,
        max_bytes: int | None = None,
    ) -> tuple[bytes, str]:
        strategy_id = self._required_strategy_string(strategy_cfg, "id")
        class_name = self._required_strategy_string(strategy_cfg, "class_name")
        source_path = Path(self._required_strategy_string(strategy_cfg, "source_path"))
        self._reject_unknown_bundled_source(
            strategy_id, strategy_cfg, self._bundled_class_by_id()
        )
        declared_hash = strategy_cfg.get("source_hash")
        if declared_hash is not None and not isinstance(declared_hash, str):
            raise ValueError("strategy.source_hash must be a string")
        if program is not None:
            if class_name != program.class_name or declared_hash not in (
                None,
                program.source_hash,
            ):
                raise ValueError(
                    "Strategy source identity differs from its fixed execution"
                )
            max_bytes = program.runtime.limits.message_bytes // 2
            declared_hash = program.source_hash
        if max_bytes is None:
            raise ValueError(
                "A bounded runtime is required before reading generated source"
            )
        with self._resolve_generated_source(source_path).open("rb") as stream:
            source = stream.read(max_bytes + 1)
        if len(source) > max_bytes:
            raise ValueError("Generated source exceeds the isolated message capacity")
        if (
            declared_hash is not None
            and hashlib.sha256(source).hexdigest() != declared_hash
        ):
            raise ValueError(
                "Strategy revision source has changed; save and validate a new revision"
            )
        return source, class_name
