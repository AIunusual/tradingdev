"""Signal checks reject invalid real outputs as well as synthetic fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.contract import SignalContractChecker
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.strategies.signals import (
    SignalContractError,
    generate_checked_signals,
)

if TYPE_CHECKING:
    from pathlib import Path


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC"),
            "close": np.arange(8, dtype=float) + 100,
            "feature": [{"values": [i]} for i in range(8)],
        },
        index=pd.Index(np.arange(100, 108), name="source_row"),
    )


class _FaultyStrategy(BaseStrategy):
    def __init__(self, fault: str) -> None:
        self.fault = fault
        self.calls = 0

    def get_parameters(self) -> dict[str, Any]:
        return {}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.calls += 1
        result = df.assign(signal=0.0)
        if self.fault == "nan":
            result["signal"] = np.nan
        elif self.fault in {"partial_nan", "infinity", "string", "complex", "null"}:
            values: list[Any] = [0] * len(df)
            values[-1] = {
                "partial_nan": np.nan,
                "infinity": np.inf,
                "string": "1",
                "complex": 1 + 0j,
                "null": pd.NA,
            }[self.fault]
            result["signal"] = pd.Series(values, index=df.index, dtype=object)
        elif self.fault == "short":
            result = result.iloc[:-1].copy()
        elif self.fault == "long":
            result = pd.concat([result, result.iloc[[-1]]])
        elif self.fault == "empty":
            result = result.iloc[:0].copy()
        elif self.fault == "index":
            result.index = result.index + 1000
        elif self.fault == "index_name":
            result.index = result.index.rename("different")
        elif self.fault == "reverse":
            result = result.iloc[::-1].copy()
        elif self.fault == "timestamps":
            result["timestamp"] = result["timestamp"].iloc[::-1].to_numpy()
        elif self.fault == "prices":
            result["close"] += 1
        elif self.fault == "missing_market":
            result = result.drop(columns="timestamp")
        elif self.fault == "missing_signal":
            result = result.drop(columns="signal")
        elif self.fault == "duplicate_signal":
            result = pd.concat([result, result[["signal"]]], axis=1)
        elif self.fault == "mutate":
            df["close"] += 100
        elif self.fault == "mutate_index":
            df.index.name = "changed"
        elif self.fault in {"mutate_index_array", "raise_after_index_mutation"}:
            df.index.array[0] = 999
            if self.fault == "raise_after_index_mutation":
                raise RuntimeError("strategy failed")
        elif self.fault in {"mutate_columns_array", "raise_after_columns_mutation"}:
            df.columns.array[0] = "changed"
            if self.fault == "raise_after_columns_mutation":
                raise RuntimeError("strategy failed")
        elif self.fault in {"mutate_object", "raise_after_object_mutation"}:
            nested = cast("dict[str, list[int]]", df.at[df.index[0], "feature"])
            nested["values"][0] = 999
            if self.fault == "raise_after_object_mutation":
                raise RuntimeError("strategy failed")
        elif self.fault == "raise_after_mutation":
            df["close"] += 100
            raise RuntimeError("strategy failed")
        elif self.fault == "alias":
            df["signal"] = 0
            return df
        elif self.fault == "not_frame":
            return cast("pd.DataFrame", {"signal": [0]})
        else:
            raise AssertionError(self.fault)
        return result


@pytest.mark.parametrize(
    ("fault", "code"),
    [
        (fault, "invalid_signal_values")
        for fault in ("nan", "partial_nan", "infinity", "string", "complex", "null")
    ]
    + [
        ("short", "signal_row_count_mismatch"),
        ("long", "signal_row_count_mismatch"),
        ("empty", "signal_row_count_mismatch"),
        ("index", "signal_index_mismatch"),
        ("index_name", "signal_index_mismatch"),
        ("reverse", "signal_index_mismatch"),
        ("timestamps", "signal_market_data_changed"),
        ("prices", "signal_market_data_changed"),
        ("missing_market", "signal_market_data_changed"),
        ("missing_signal", "missing_signal_column"),
        ("duplicate_signal", "duplicate_output_columns"),
        ("mutate", "input_mutated"),
        ("mutate_index", "input_mutated"),
        ("alias", "output_aliases_input"),
        ("not_frame", "signals_not_dataframe"),
    ],
)
def test_same_fault_is_rejected_in_program_checks_and_actual_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, code: str
) -> None:
    strategy = _FaultyStrategy(fault)
    loader = StrategyLoader(workspace_root=tmp_path)
    monkeypatch.setattr(loader, "create_from_config", lambda *a, **kw: strategy)
    checked = SignalContractChecker(loader).check_config(
        {
            "strategy": {"class_name": "Fixture"},
            "backtest": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-08",
                "init_cash": 10000,
            },
        },
        fixture_rows=80,
    )
    assert code in {item.code for item in checked["diagnostics"]}
    frame = _frame()
    before = frame.copy(deep=True)
    with pytest.raises(SignalContractError) as caught:
        generate_checked_signals(strategy, frame)
    assert code in {item.code for item in caught.value.diagnostics}
    assert strategy.calls == 2
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    "fault",
    [
        "raise_after_mutation",
        "raise_after_index_mutation",
        "raise_after_columns_mutation",
        "raise_after_object_mutation",
    ],
)
def test_strategy_exception_cannot_poison_the_next_candidates_market_data(
    fault: str,
) -> None:
    frame = _frame()
    before = frame.copy(deep=True)
    with pytest.raises(RuntimeError, match="strategy failed"):
        generate_checked_signals(_FaultyStrategy(fault), frame)
    pd.testing.assert_frame_equal(frame, before)
    assert frame.index.tolist() == list(range(100, 108))
    assert frame.columns.tolist() == ["timestamp", "close", "feature"]
    assert frame.at[100, "feature"] == {"values": [0]}


@pytest.mark.parametrize(
    "fault", ["mutate_index_array", "mutate_columns_array", "mutate_object"]
)
def test_nested_or_axis_mutation_cannot_change_input_or_comparison_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    from tradingdev.domain.strategies import contract

    frame = _frame()
    monkeypatch.setattr(contract, "_fixture_df", lambda _rows: _frame())
    # Exercise both the fixture comparison snapshot and the actual input copy.
    test_same_fault_is_rejected_in_program_checks_and_actual_generation(
        tmp_path, monkeypatch, fault, "input_mutated"
    )
    with pytest.raises(SignalContractError, match="input_mutated"):
        generate_checked_signals(_FaultyStrategy(fault), frame)
    assert frame.index.tolist() == list(range(100, 108))
    assert frame.columns.tolist() == ["timestamp", "close", "feature"]
    assert frame.at[100, "feature"] == {"values": [0]}


@pytest.mark.parametrize("dtype", ["int64", "float64", "Int64", "object"])
def test_valid_states_preserve_nondefault_rows_and_keep_feature_nan(
    dtype: str,
) -> None:
    class ValidStrategy(BaseStrategy):
        def get_parameters(self) -> dict[str, Any]:
            return {}

        def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
            result = df.assign(indicator=np.nan)
            result["signal"] = pd.Series(
                [0, 0, 1, 1, 0, -1, -1, 0], index=df.index, dtype=dtype
            )
            return result

    frame = _frame()
    before = frame.copy(deep=True)
    result = generate_checked_signals(ValidStrategy(), frame)
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result["indicator"].isna().all()
    pd.testing.assert_frame_equal(result[frame.columns], before)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("container", ["dict", "list"])
@pytest.mark.parametrize("mutate", [False, True])
def test_nested_array_features_are_compared_by_value_without_false_mutation(
    container: str,
    mutate: bool,
) -> None:
    class NestedArrayStrategy(BaseStrategy):
        def get_parameters(self) -> dict[str, Any]:
            return {}

        def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
            if mutate:
                value = cast("Any", df.at[0, "feature"])
                array = value["array"] if container == "dict" else value[0]
                assert isinstance(array, np.ndarray)
                array[0] = 999
            return df.assign(signal=0)

    array = np.array([1.0, np.nan, 2.0])
    nested = (
        {"array": array, "missing": pd.NA} if container == "dict" else [array, pd.NA]
    )
    frame = pd.DataFrame({"close": [100.0], "feature": [nested]})
    if mutate:
        with pytest.raises(SignalContractError, match="input_mutated"):
            generate_checked_signals(NestedArrayStrategy(), frame)
    else:
        result = generate_checked_signals(NestedArrayStrategy(), frame)
        assert result["signal"].tolist() == [0]
    np.testing.assert_array_equal(array, [1.0, np.nan, 2.0])
