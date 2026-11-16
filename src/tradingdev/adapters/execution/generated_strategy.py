"""A managed strategy proxy whose generated Python never enters the host process."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from tradingdev.adapters.execution.frame_codec import decode_frame, encode_frame
from tradingdev.adapters.execution.strategy_protocol import (
    Operation,
    StrategyRequest,
    StrategyResponse,
)
from tradingdev.domain.runtime import IsolatedRuntimeError
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.execution import (
    GeneratedProgram,
    finite_strategy_value,
)
from tradingdev.domain.strategies.signals import SignalContractError, check_signal_frame

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

    from tradingdev.adapters.execution.container_runtime import (
        ContainerSession,
        DockerRuntime,
    )


class StrategyChannel:
    def __init__(self, session: ContainerSession, program: GeneratedProgram) -> None:
        self.session = session
        self.program = program

    def request(self, operation: Operation, **arguments: Any) -> StrategyResponse:
        request = StrategyRequest(
            request_id=uuid4().hex,
            operation=operation,
            source_hash=self.program.source_hash,
            class_name=self.program.class_name,
            **arguments,
        )
        response = StrategyResponse.model_validate_json(
            self.session.exchange(request.model_dump_json().encode("utf-8"))
        )
        if (
            response.request_id != request.request_id
            or response.operation != request.operation
            or response.source_hash != request.source_hash
            or response.class_name != request.class_name
        ):
            raise IsolatedRuntimeError(
                "invalid_runtime_response",
                "Strategy response identity differs from its request",
            )
        if response.diagnostics:
            raise SignalContractError(
                response.diagnostics, analysis=response.signal_analysis
            )
        if response.error is not None:
            raise IsolatedRuntimeError(
                "strategy_execution_failed",
                response.error,
                strategy_callback_failed=response.error_kind == "strategy_callback",
            )
        return response


class IsolatedStrategy(BaseStrategy):
    """Keep fit state and RNG streams in the session owned by the caller's context."""

    def __init__(self, channel: StrategyChannel) -> None:
        self._channel = channel

    def fit(self, df: pd.DataFrame) -> None:
        self._channel.request("fit", frame=encode_frame(df))

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        response = self._channel.request("signals", frame=encode_frame(df))
        if response.frame is None:
            raise IsolatedRuntimeError(
                "invalid_runtime_response", "Strategy response has no signals"
            )
        result = decode_frame(response.frame)
        diagnostics = check_signal_frame(df, df, result)
        if diagnostics:
            raise SignalContractError(diagnostics)
        return result

    def get_parameters(self) -> dict[str, Any]:
        value = finite_strategy_value(self._channel.request("parameters").value)
        if not isinstance(value, dict):
            raise ValueError("Strategy parameters must be a JSON object")
        return value

    def rolling_trace(self) -> dict[str, Any] | None:
        value = finite_strategy_value(self._channel.request("trace").value)
        if value is not None and not isinstance(value, dict):
            raise ValueError("Strategy rolling trace must be a JSON object or null")
        return value


@contextmanager
def open_channel(
    runtime: DockerRuntime, program: GeneratedProgram
) -> Iterator[StrategyChannel]:
    runtime.recover_abandoned()
    command = (
        "/app/.venv/bin/python",
        "-I",
        "-u",
        "-m",
        "tradingdev.adapters.execution.strategy_worker",
        "--message-bytes",
        str(program.runtime.limits.message_bytes),
    )
    with runtime.open(program.runtime, command) as session:
        yield StrategyChannel(session, program)


def inspect_parameters(
    runtime: DockerRuntime,
    program: GeneratedProgram,
    source: bytes,
    values: dict[str, Any],
    *,
    capture: bool,
) -> dict[str, Any]:
    with open_channel(runtime, program) as channel:
        response = channel.request(
            "capture" if capture else "validate",
            source=base64.b64encode(source).decode("ascii"),
            kwargs=values,
        )
        value = finite_strategy_value(response.value)
        if not isinstance(value, dict):
            raise ValueError("Strategy constructor settings must be a JSON object")
        return value


@contextmanager
def open_strategy(
    runtime: DockerRuntime,
    program: GeneratedProgram,
    source: bytes,
    values: dict[str, Any],
    random_seed: int | None,
) -> Iterator[BaseStrategy]:
    with open_channel(runtime, program) as channel:
        channel.request(
            "construct",
            source=base64.b64encode(source).decode("ascii"),
            kwargs=values,
            random_seed=random_seed,
        )
        yield IsolatedStrategy(channel)
