"""Check strategy output against the exact frame supplied to the strategy."""

from __future__ import annotations

from concurrent.futures import BrokenExecutor
from contextlib import contextmanager
from copy import deepcopy
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from tradingdev.domain.runtime import IsolatedRuntimeError
from tradingdev.domain.strategies.schemas import StrategyDiagnostic

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tradingdev.domain.strategies.base import BaseStrategy

_MARKET_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


class SignalContractError(ValueError):
    """Reject invalid strategy output before handing any rows to an engine."""

    def __init__(
        self,
        diagnostics: list[StrategyDiagnostic],
        *,
        analysis: dict[str, Any] | None = None,
    ) -> None:
        self.diagnostics = tuple(diagnostics)
        self.analysis = analysis or {}
        super().__init__(
            "; ".join(f"{item.code}: {item.message}" for item in diagnostics)
        )


class StrategyCallbackError(RuntimeError):
    """An attributable exception raised by a bundled strategy callback."""


@contextmanager
def strategy_callback() -> Iterator[None]:
    """Translate ordinary callback errors at their source, excluding system faults."""
    try:
        yield
    except Exception as exc:
        # A trusted strategy callback may call into another engine or runtime.
        # An explicitly typed runtime failure keeps its original classification.
        if isinstance(
            exc,
            (
                IsolatedRuntimeError,
                BrokenExecutor,
                OSError,
                MemoryError,
                SignalContractError,
            ),
        ):
            raise
        raise StrategyCallbackError(f"{type(exc).__name__}: {exc}") from exc


def _same_nested_value(left: object, right: object) -> bool:
    """Compare ordinary nested feature values, including arrays and missing scalars."""
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        if left.dtype.hasobject:
            return all(
                _same_nested_value(a, b)
                for a, b in zip(left.flat, right.flat, strict=True)
            )
        try:
            return bool(np.array_equal(left, right, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_nested_value(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _same_nested_value(a, b) for a, b in zip(left, right, strict=True)
        )
    try:
        return bool(left == right) or bool(
            pd.isna(cast("Any", left)) and pd.isna(cast("Any", right))
        )
    except TypeError, ValueError:
        return False


def _same_input(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    if not (
        after.index.identical(before.index)
        and after.columns.identical(before.columns)
        and after.dtypes.equals(before.dtypes)
    ):
        return False
    for position, dtype in enumerate(before.dtypes):
        left, right = before.iloc[:, position], after.iloc[:, position]
        if pd.api.types.is_object_dtype(dtype):
            if not all(
                _same_nested_value(a, b) for a, b in zip(left, right, strict=True)
            ):
                return False
        elif not left.equals(right):
            return False
    return True


def check_signal_frame(
    before: pd.DataFrame, after: pd.DataFrame, result: object
) -> list[StrategyDiagnostic]:
    """Inspect final input/output values; do not infer trading intent or causality."""
    diagnostics: list[StrategyDiagnostic] = []

    def reject(code: str, message: str, fix: str) -> None:
        diagnostics.append(
            StrategyDiagnostic(code=code, phase="contract", message=message, fix=fix)
        )

    if not _same_input(before, after):
        reject(
            "input_mutated",
            "generate_signals must not mutate input",
            "Start with result = df.copy() and mutate result only.",
        )
    if not isinstance(result, pd.DataFrame):
        reject(
            "signals_not_dataframe",
            "generate_signals must return DataFrame",
            "Return the copied DataFrame with a signal column.",
        )
        return diagnostics
    if result is after:
        reject(
            "output_aliases_input",
            "generate_signals must return a separate DataFrame",
            "Return a copy even when the supplied frame already contains signals.",
        )
    if not result.columns.is_unique:
        reject(
            "duplicate_output_columns",
            "Strategy output columns must be unique",
            "Return one signal column and uniquely named indicator columns.",
        )
        return diagnostics
    if len(result) != len(before):
        reject(
            "signal_row_count_mismatch",
            f"Expected {len(before)} output rows, received {len(result)}",
            "Preserve every input row; emit zero during warm-up, not fewer rows.",
        )
    if not result.index.identical(before.index):
        reject(
            "signal_index_mismatch",
            "Strategy output index differs from the input index",
            "Preserve the input index, order, labels and metadata; do not reset it.",
        )
    for column in _MARKET_COLUMNS:
        if column not in before.columns:
            continue
        if column not in result.columns or not result[column].equals(before[column]):
            reject(
                "signal_market_data_changed",
                f"Strategy output must preserve the original {column} column",
                "Keep market columns unchanged; use separate columns for features.",
            )
    if "signal" not in result.columns:
        reject(
            "missing_signal_column",
            "result must include signal column",
            "Add result['signal'] with values -1, 0, or 1.",
        )
        return diagnostics
    signals = result["signal"]
    real_values = signals.dtype.kind in "biuf" or all(
        isinstance(value, Real) for value in signals
    )
    if not real_values or not signals.isin((-1, 0, 1)).all():
        reject(
            "invalid_signal_values",
            "Every row must have a finite real signal equal to -1, 0, or 1",
            "Use zero for warm-up; never return missing or infinite signals.",
        )
    return diagnostics


def copy_signal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Own arrays, nested feature values and mutable Index array views."""
    copied = frame.copy(deep=True)
    # DataFrame.copy deliberately shares the otherwise immutable Index data.
    copied.index = frame.index.copy(deep=True)
    copied.columns = frame.columns.copy(deep=True)
    # Pandas also shares Python objects inside object-dtype cells, even for a
    # deep copy. Detach ordinary nested feature values as well as numeric arrays.
    for position, dtype in enumerate(frame.dtypes):
        if pd.api.types.is_object_dtype(dtype):
            copied.isetitem(
                position,
                pd.Series(
                    [deepcopy(value) for value in frame.iloc[:, position]],
                    index=copied.index,
                    dtype=object,
                ).array,
            )
    return copied


def generate_checked_signals(
    strategy: BaseStrategy,
    frame: pd.DataFrame,
    *,
    classify_callback_error: bool = False,
) -> pd.DataFrame:
    """Execute once, then reject mutation, misalignment or invalid signal values."""
    # A failed candidate must not poison the captured data used by later candidates.
    supplied = copy_signal_frame(frame)
    if classify_callback_error:
        with strategy_callback():
            result = strategy.generate_signals(supplied)
    else:
        result = strategy.generate_signals(supplied)
    diagnostics = check_signal_frame(frame, supplied, result)
    if diagnostics:
        raise SignalContractError(diagnostics, analysis=signal_analysis(result))
    return result


def signal_analysis(result: object) -> dict[str, Any]:
    """Preserve available observations even when signal validation rejects output."""
    if not isinstance(result, pd.DataFrame):
        return {}
    analysis: dict[str, Any] = {"rows": int(len(result))}
    if not result.columns.is_unique or "signal" not in result.columns:
        return analysis
    signals = result["signal"]
    try:
        distribution = {
            str(key): int(value)
            for key, value in signals.value_counts(dropna=False).to_dict().items()
        }
        transitions = int(signals.fillna(0).ne(signals.fillna(0).shift()).sum() - 1)
        active = int(signals.isin([-1, 1]).sum())
    except TypeError, ValueError:
        # Invalid, non-scalar signal values must not erase the original finding.
        return analysis
    analysis.update(
        signal_distribution=distribution,
        nan_count=int(signals.isna().sum()),
        transition_count=max(transitions, 0),
        active_signal_ratio=active / max(len(result), 1),
        first_timestamp=(
            str(result["timestamp"].iloc[0])
            if "timestamp" in result.columns and not result.empty
            else None
        ),
        last_timestamp=(
            str(result["timestamp"].iloc[-1])
            if "timestamp" in result.columns and not result.empty
            else None
        ),
    )
    return analysis
