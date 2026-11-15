"""Execute generated imports and methods exclusively inside a private container."""

from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import os
import sys
from contextlib import ExitStack
from types import ModuleType
from typing import Any

from tradingdev.adapters.execution.frame_codec import decode_frame, encode_frame
from tradingdev.adapters.execution.strategy_protocol import (
    StrategyRequest,
    StrategyResponse,
)
from tradingdev.domain.randomness import execution_randomness
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.execution import finite_strategy_value
from tradingdev.domain.strategies.signals import (
    SignalContractError,
    StrategyCallbackError,
    generate_checked_signals,
    strategy_callback,
)


def _load_class(request: StrategyRequest) -> type[BaseStrategy]:
    if request.source is None:
        raise ValueError("The first request must supply the fixed strategy source")
    source = base64.b64decode(request.source, validate=True)
    if hashlib.sha256(source).hexdigest() != request.source_hash:
        raise ValueError("Strategy source does not match its fixed identity")
    name = f"_tradingdev_generated_{request.source_hash}"
    module = ModuleType(name)
    module.__file__ = "/strategy.py"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102
    cls = getattr(module, request.class_name, None)
    if not inspect.isclass(cls) or not issubclass(cls, BaseStrategy):
        raise TypeError(f"{request.class_name!r} must inherit from BaseStrategy")
    return cls


def _constructor_kwargs(
    cls: type[BaseStrategy],
    values: dict[str, Any],
    *,
    capture: bool,
) -> dict[str, Any]:
    signature = inspect.signature(cls)
    reserved = {"backtest_engine", "parallel_config"}
    if reserved.intersection(signature.parameters) or reserved.intersection(values):
        raise ValueError(
            "Generated strategies cannot accept backtest_engine or parallel_config; "
            "remove application service injections and compute signals "
            "from supplied data"
        )
    copied = finite_strategy_value(values)
    assert isinstance(copied, dict)
    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name not in copied:
            if capture and parameter.default is not inspect.Parameter.empty:
                copied[name] = finite_strategy_value(parameter.default)
            elif not capture:
                raise ValueError(
                    "Strategy execution is missing explicit constructor setting "
                    f"{name!r}"
                )
    signature.bind(**copied)
    return copied


def serve(message_bytes: int) -> None:
    """Keep learned state and RNG streams for this one bounded strategy session."""
    if sys.platform != "linux" or os.geteuid() != 65534 or os.getppid() != 1:
        raise RuntimeError("Generated Python requires the isolated strategy runtime")
    # Strategy prints belong to diagnostics, never to the response protocol.
    wire = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    identity: tuple[str, str] | None = None
    cls: type[BaseStrategy] | None = None
    strategy: BaseStrategy | None = None
    with wire, ExitStack() as stack:
        while line := sys.stdin.buffer.readline(message_bytes + 2):
            if len(line) > message_bytes + 1 or not line.endswith(b"\n"):
                raise ValueError("Strategy request exceeds its message limit")
            request = StrategyRequest.model_validate_json(line)
            response = StrategyResponse(
                request_id=request.request_id,
                operation=request.operation,
                source_hash=request.source_hash,
                class_name=request.class_name,
            )
            try:
                selected_identity = (request.source_hash, request.class_name)
                if identity is None:
                    if request.operation not in {"capture", "validate", "construct"}:
                        raise ValueError("The first request must select a strategy")
                    cls = _load_class(request)
                    identity = selected_identity
                elif identity != selected_identity or request.source is not None:
                    raise ValueError("An execution session cannot change its strategy")
                assert cls is not None
                if request.operation in {"capture", "validate", "construct"}:
                    if strategy is not None:
                        raise ValueError("A session can construct only one strategy")
                    kwargs = _constructor_kwargs(
                        cls, request.kwargs, capture=request.operation == "capture"
                    )
                    response.value = finite_strategy_value(kwargs)
                    if request.operation == "construct":
                        stack.enter_context(execution_randomness(request.random_seed))
                        with strategy_callback():
                            strategy = cls(**kwargs)
                else:
                    if strategy is None:
                        raise ValueError("The strategy has not been constructed")
                    if request.operation in {"fit", "signals"}:
                        if request.frame is None:
                            raise ValueError("A frame is required")
                        frame = decode_frame(request.frame)
                        if request.operation == "fit":
                            strategy.fit(frame)
                        else:
                            result = generate_checked_signals(
                                strategy, frame, classify_callback_error=True
                            )
                            response.frame = encode_frame(result)
                    elif request.operation == "parameters":
                        with strategy_callback():
                            parameters = strategy.get_parameters()
                        response.value = finite_strategy_value(parameters)
                    elif request.operation == "trace":
                        with strategy_callback():
                            trace = strategy.rolling_trace()
                        response.value = finite_strategy_value(trace)
            except SignalContractError as error:
                response.error = str(error)
                response.diagnostics = list(error.diagnostics)
                response.signal_analysis = error.analysis
            except StrategyCallbackError as error:
                response.error = str(error)
                response.error_kind = "strategy_callback"
            except (
                BaseException
            ) as error:  # Generated SystemExit is an operation failure.
                response.error = f"{type(error).__name__}: {error}"
            encoded = response.model_dump_json().encode("utf-8")
            if len(encoded) > message_bytes:
                raise ValueError("Strategy response exceeds its message limit")
            wire.write(encoded + b"\n")
            if response.error is not None:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message-bytes", type=int, required=True)
    arguments = parser.parse_args()
    serve(arguments.message_bytes)


if __name__ == "__main__":
    main()
