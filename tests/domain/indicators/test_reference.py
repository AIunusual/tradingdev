"""Indicator discovery follows actual exports and includes the checks authors need."""

from __future__ import annotations

import inspect
from dataclasses import fields
from typing import TYPE_CHECKING

import pytest

from tradingdev.domain import indicators
from tradingdev.domain.indicators import technical
from tradingdev.domain.indicators.reference import indicator_reference

if TYPE_CHECKING:
    from types import FunctionType

    import pandas as pd


def _exported_fixture(close: pd.Series, length: int = 7) -> pd.Series:
    """A newly exported callable discovered without editing a second API list."""
    raise AssertionError("Reference discovery must never execute indicators")


def test_catalog_tracks_public_exports_without_executing_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(indicators, "fixture", _exported_fixture, raising=False)
    monkeypatch.setattr(indicators, "__all__", [*indicators.__all__, "fixture"])
    payload = indicator_reference()
    assert payload["selected"] is None
    names = {row["name"] for row in payload["catalog"]}
    assert names == {
        name
        for name in indicators.__all__
        if inspect.isfunction(getattr(indicators, name))
    }
    assert "fixture" in names
    selected = indicator_reference("fixture")["selected"]
    assert "length: 'int' = 7" in selected["signature"]
    assert "Reference discovery must never execute" in selected["source"]


@pytest.mark.parametrize(
    ("name", "result_type"),
    [
        ("macd", technical.MACDResult),
        ("bollinger_bands", technical.BollingerBands),
        ("adx", technical.ADXResult),
        ("stochastic", technical.StochasticResult),
    ],
)
def test_selected_api_exposes_named_results_and_transitive_input_checks(
    name: str,
    result_type: type,
) -> None:
    payload = indicator_reference(name)
    selected = payload["selected"]
    assert selected["name"] == name
    assert selected["result_fields"] == [field.name for field in fields(result_type)]
    helpers = {row["name"]: row["source"] for row in selected["supporting_source"]}
    assert {"_validate_period", "_as_float", "_series", "_empty_like"} <= set(helpers)
    assert "Integral" in helpers["_validate_period"]
    assert "100000" in helpers["_validate_period"]
    assert "np.isinf" in helpers["_as_float"]
    if name in {"adx", "stochastic"}:
        assert "_as_float_ohlc" in helpers
        assert "identical indexes" in helpers["_as_float_ohlc"]
    assert "missing-value handling follow TA-Lib" in payload["module_contract"]


@pytest.mark.parametrize(
    "name", ["_as_float", "MACDResult", "indicator_column", "../technical.py"]
)
def test_unknown_or_nonfunction_selection_cannot_read_arbitrary_source(
    name: str,
) -> None:
    with pytest.raises(ValueError, match="Unknown indicator"):
        indicator_reference(name)


def test_missing_source_fails_instead_of_issuing_partial_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = inspect.getsource

    def unavailable(value: FunctionType) -> str:
        if value is technical._as_float:
            raise OSError("source unavailable")
        return original(value)

    monkeypatch.setattr(inspect, "getsource", unavailable)
    with pytest.raises(OSError, match="source unavailable"):
        indicator_reference("atr")


def test_named_result_fields_accept_resolved_class_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        technical.macd,
        "__annotations__",
        {**technical.macd.__annotations__, "return": technical.MACDResult},
    )
    assert indicator_reference("macd")["selected"]["result_fields"] == [
        "macd",
        "signal",
        "histogram",
    ]
