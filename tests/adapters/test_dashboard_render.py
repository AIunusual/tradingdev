"""Exercise real Streamlit chart rendering and Plotly's serialized data."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest
from streamlit.testing.v1 import AppTest

from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.result import BacktestResult

if TYPE_CHECKING:
    from pathlib import Path


def _pipeline(mode: str) -> PipelineResult:
    timestamps = np.arange(np.datetime64("2024-01-15"), np.datetime64("2024-02-24"))
    capital = 1000.0 if mode == "signal" else 0.0
    result = BacktestResult(
        metrics={"total_trades": 3, "total_volume": 1405.0},
        equity_curve=capital + np.arange(len(timestamps), dtype=float),
        timestamps=timestamps,
        init_cash=capital if mode == "signal" else None,
        mode=mode,
        trades=[
            {
                "entry_idx": entry,
                "exit_idx": exit_,
                "size_quote": 200.0,
                "exit_notional": 200.0 + pnl,
                "net_pnl": pnl,
                "status": status,
            }
            for entry, exit_, pnl, status in [
                (1, 2, -10.0, "closed"),
                (3, 4, -5.0, "closed"),
                (20, 21, 20.0, "closed"),
                (39, 39, 10.0, "open"),
            ]
        ],
    )
    return PipelineResult(
        mode="simple",
        backtest_result=result,
        config_snapshot={
            "strategy": {
                "id": "fixture",
                "parameters": {"monthly_volume_target": 750.0},
            },
            "backtest": {
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-15",
                "end_date": "2024-02-24",
                "init_cash": result.init_cash,
                "mode": mode,
            },
        },
    )


def _app(tmp_path: Path, pipeline: PipelineResult) -> AppTest:
    # from_file keeps the script inside the immediately removed test directory;
    # the renderer uses in-memory results and never opens a user workspace.
    script = tmp_path / "dashboard.py"
    script.write_text(
        "import streamlit as st\n"
        "from tradingdev.adapters.dashboard.app import _render_simple\n"
        "_render_simple(st.session_state['pipeline'], "
        "{'strategy_id': 'fixture', 'run_id': 'fixture'})\n",
        encoding="utf-8",
    )
    app = AppTest.from_file(script, default_timeout=15)
    app.session_state["pipeline"] = pipeline
    app.run()
    return app


def _charts(app: AppTest) -> list[dict[str, Any]]:
    assert not app.exception
    return [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]


def _numbers(value: Any) -> npt.NDArray[np.float64]:
    """Read both JSON arrays and Plotly's typed-array wire representation."""
    if isinstance(value, dict):
        return np.frombuffer(
            base64.b64decode(value["bdata"]), dtype=value["dtype"]
        ).astype(float)
    return np.asarray(value, dtype=float)


@pytest.mark.parametrize("mode", ["signal", "volume"])
def test_charts_serialize_recorded_data_and_survive_filter_reruns(
    tmp_path: Path, mode: str
) -> None:
    app = _app(tmp_path, _pipeline(mode))
    charts = _charts(app)

    assert [chart["data"][0]["type"] for chart in charts] == [
        "scatter",
        "histogram",
        "bar",
        "bar",
        "scatter",
    ]
    pnl = charts[0]["data"][0]
    assert pnl["x"][0].startswith("2024-01-15")
    assert pnl["x"][-1].startswith("2024-02-23")
    expected = np.arange(40, dtype=float)
    if mode == "signal":
        expected /= 10
    np.testing.assert_allclose(_numbers(pnl["y"]), expected)
    # The open position contributes volume, but never a realized trade outcome.
    outcomes = [-5, -2.5, 10] if mode == "signal" else [-10, -5, 20]
    np.testing.assert_allclose(_numbers(charts[1]["data"][0]["x"]), outcomes)
    np.testing.assert_array_equal(charts[2]["data"][0]["x"], ["2"])
    np.testing.assert_array_equal(_numbers(charts[2]["data"][0]["y"]), [1])
    np.testing.assert_array_equal(charts[3]["data"][0]["x"], ["2024-01", "2024-02"])
    np.testing.assert_allclose(_numbers(charts[3]["data"][0]["y"]), [785, 620])
    mdd = _numbers(charts[4]["data"][0]["y"])
    assert np.isnan(mdd[:29]).all()
    np.testing.assert_allclose(mdd[29:], 0)
    if mode == "volume":
        assert charts[3]["layout"]["shapes"][0]["y0"] == 750.0

    app.radio[0].set_value("Absolute").run()
    np.testing.assert_allclose(_numbers(_charts(app)[0]["data"][0]["y"]), np.arange(40))
    app.selectbox[0].set_value("2024-01").run()
    january = _charts(app)
    assert len(january) == 5
    np.testing.assert_allclose(_numbers(january[0]["data"][0]["y"]), np.arange(17))
    np.testing.assert_allclose(_numbers(january[1]["data"][0]["x"]), [-10, -5])


def test_empty_trade_ledger_renders_equity_and_explicit_empty_states(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline("signal")
    assert pipeline.backtest_result is not None
    pipeline.backtest_result.trades.clear()

    app = _app(tmp_path, pipeline)

    assert len(_charts(app)) == 2
    assert [element.value for element in app.info] == [
        "No closed trades in selected period.",
        "No losing streaks in selected period.",
        "No trade data with timestamps.",
    ]


def test_missing_backtest_result_stops_before_chart_rendering(tmp_path: Path) -> None:
    pipeline = _pipeline("signal")
    pipeline.backtest_result = None

    app = _app(tmp_path, pipeline)

    assert not _charts(app)
    assert [element.value for element in app.error] == [
        "No backtest result in pipeline."
    ]
