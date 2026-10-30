"""Exercise real Streamlit chart rendering and Plotly's serialized data."""

from __future__ import annotations

import base64
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest

from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.result import BacktestResult

if TYPE_CHECKING:
    from collections.abc import Iterator
    from multiprocessing.process import BaseProcess
    from multiprocessing.synchronize import Event

    from streamlit.testing.v1 import AppTest


def _run_isolated(script: str, *, timeout: float = 90) -> None:
    """Keep AppTest's process-wide runtime and __main__ changes out of pytest."""
    # The scenario deadline covers cold imports and up to three AppTest runs.
    # Each individual AppTest.run still has its original 15-second limit.
    # Retain the directory if an owned process cannot be stopped. It must not
    # belong to tmp_path, whose fixture would otherwise remove it regardless.
    root = Path(mkdtemp(prefix="dashboard-app-test-"))
    process: subprocess.Popen[str] | None = None
    try:
        temporary = root / "temporary"
        temporary.mkdir()
        environment = os.environ.copy()
        environment.update(
            TMPDIR=str(temporary),
            MPLCONFIGDIR=str(root / "matplotlib"),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=os.pathsep.join(
                filter(
                    None,
                    [
                        str(Path(__file__).resolve().parents[2]),
                        environment.get("PYTHONPATH"),
                    ],
                )
            ),
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(timeout=timeout)
        if process.returncode:
            raise subprocess.CalledProcessError(
                process.returncode, process.args, output=stdout, stderr=stderr
            )
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
            except Exception as error:
                raise RuntimeError(
                    f"dashboard process cleanup failed; workspace retained at {root}"
                ) from error
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        shutil.rmtree(root)


def _run_dashboard_checks(check_name: str, *arguments: str) -> None:
    script = (
        "from pathlib import Path\n"
        "from tests.adapters import test_dashboard_render as checks\n"
        f"checks.{check_name}(Path.cwd(), *{arguments!r})\n"
    )
    try:
        _run_isolated(script)
    except subprocess.CalledProcessError as error:
        pytest.fail(error.stdout + error.stderr)


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
    from streamlit.testing.v1 import AppTest

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


def _check_charts_serialize_recorded_data_and_survive_filter_reruns(
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
    assert np.isnan(mdd[:30]).all()
    np.testing.assert_allclose(mdd[30:], 0)
    if mode == "volume":
        assert charts[3]["layout"]["shapes"][0]["y0"] == 750.0

    app.radio[0].set_value("Absolute").run()
    np.testing.assert_allclose(_numbers(_charts(app)[0]["data"][0]["y"]), np.arange(40))
    app.selectbox[0].set_value("2024-01").run()
    january = _charts(app)
    assert len(january) == 5
    np.testing.assert_allclose(_numbers(january[0]["data"][0]["y"]), np.arange(17))
    np.testing.assert_allclose(_numbers(january[1]["data"][0]["x"]), [-10, -5])


def _check_empty_trade_ledger_renders_equity_and_explicit_empty_states(
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


def _check_missing_backtest_result_stops_before_chart_rendering(tmp_path: Path) -> None:
    pipeline = _pipeline("signal")
    pipeline.backtest_result = None

    app = _app(tmp_path, pipeline)

    assert not _charts(app)
    assert [element.value for element in app.error] == [
        "No backtest result in pipeline."
    ]


@pytest.mark.parametrize("mode", ["signal", "volume"])
def test_charts_serialize_recorded_data_and_survive_filter_reruns(mode: str) -> None:
    _run_dashboard_checks(
        "_check_charts_serialize_recorded_data_and_survive_filter_reruns", mode
    )


def test_empty_trade_ledger_renders_equity_and_explicit_empty_states() -> None:
    _run_dashboard_checks(
        "_check_empty_trade_ledger_renders_equity_and_explicit_empty_states"
    )


def test_missing_backtest_result_stops_before_chart_rendering() -> None:
    _run_dashboard_checks("_check_missing_backtest_result_stops_before_chart_rendering")


def _check_drawdown_evicts_expired_peak_for_sparse_weekly_data(tmp_path: Path) -> None:
    pipeline = _pipeline("signal")
    assert pipeline.backtest_result is not None
    result = pipeline.backtest_result
    result.trades.clear()
    result.timestamps = np.arange(
        np.datetime64("2024-01-01"),
        np.datetime64("2024-03-01"),
        np.timedelta64(7, "D"),
    )
    result.equity_curve = np.array(
        [100.0, 50.0, 60.0, 70.0, 80.0, 90.0, 70.0, 80.0, 85.0]
    )
    pipeline.config_snapshot["backtest"]["timeframe"] = "1w"
    charts = _charts(_app(tmp_path, pipeline))
    mdd = _numbers(charts[-1]["data"][0]["y"])
    np.testing.assert_allclose(
        mdd, [np.nan] * 5 + [0.0, 20.0, 20.0, 20.0], equal_nan=True
    )


def test_drawdown_evicts_expired_peak_for_sparse_weekly_data() -> None:
    _run_dashboard_checks("_check_drawdown_evicts_expired_peak_for_sparse_weekly_data")


@contextmanager
def _spawn_workspace() -> Iterator[tuple[Path, list[BaseProcess]]]:
    # Keep ownership outside tmp_path so a fixture cannot remove evidence
    # while a child is still using it after a failed cleanup.
    root = Path(mkdtemp(prefix="dashboard-spawn-"))
    workers: list[BaseProcess] = []
    try:
        yield root, workers
    finally:
        failures: list[Exception] = []
        for worker in workers:
            try:
                if worker.pid is not None:
                    if worker.is_alive():
                        worker.terminate()
                    worker.join(5)
                    if worker.is_alive():
                        worker.kill()
                        worker.join(5)
                    if worker.is_alive():
                        raise RuntimeError("dashboard spawn child is still running")
                worker.close()
            except Exception as error:
                failures.append(error)
        if failures:
            raise ExceptionGroup(
                f"dashboard spawn cleanup failed; workspace retained at {root}",
                failures,
            )
        shutil.rmtree(root)


def test_dashboard_render_does_not_break_later_spawn() -> None:
    original_main = sys.modules["__main__"]
    context = multiprocessing.get_context("spawn")
    try:
        _run_dashboard_checks(
            "_check_charts_serialize_recorded_data_and_survive_filter_reruns", "signal"
        )
        with _spawn_workspace() as (root, workers):
            marker = root / "spawned.txt"
            worker = context.Process(target=Path.write_text, args=(marker, "spawned"))
            workers.append(worker)
            worker.start()
            worker.join(15)
            assert worker.exitcode == 0
            assert marker.read_text() == "spawned"
            assert sys.modules["__main__"] is original_main
        assert not root.exists()
    finally:
        sys.modules["__main__"] = original_main


def _hold_spawn_workspace(root: Path, ready: Event, release: Event) -> None:
    (root / "spawned.txt").write_text("spawned", encoding="utf-8")
    ready.set()
    release.wait(60)


@pytest.mark.parametrize("operation", ["terminate", "join", "kill"])
def test_spawn_workspace_is_retained_when_child_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    root: Path | None = None
    worker: BaseProcess | None = None

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError(f"injected {operation} failure")

    try:
        with monkeypatch.context() as patch:
            with (
                pytest.raises(ExceptionGroup, match="workspace retained at") as failure,
                _spawn_workspace() as (root, workers),
            ):
                worker = context.Process(
                    target=_hold_spawn_workspace, args=(root, ready, release)
                )
                workers.append(worker)
                worker.start()
                assert ready.wait(15)
                # Keep the real child alive until the selected cleanup
                # operation fails, including the kill escalation path.
                if operation in {"join", "kill"}:
                    patch.setattr(worker, "terminate", lambda: None)
                if operation == "kill":
                    patch.setattr(worker, "join", lambda timeout: None)
                patch.setattr(worker, operation, fail)
            assert worker is not None and worker.is_alive()
            assert root is not None and root.is_dir()
            assert (root / "spawned.txt").read_text() == "spawned"
            assert str(root) in str(failure.value)
            assert isinstance(failure.value.exceptions[0], OSError)
    finally:
        # Undo the injected failure before stopping the retained real child.
        if root is not None and root.exists() and worker is not None:
            release.set()
            worker.join(5)
            if worker.is_alive():
                worker.kill()
                worker.join(5)
            assert not worker.is_alive(), f"spawn workspace retained at {root}"
            worker.close()
            shutil.rmtree(root)


@pytest.mark.parametrize("outcome", ["normal", "failure", "timeout"])
def test_isolated_dashboard_process_and_workspace_are_cleaned_on_every_exit(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    original_main = sys.modules["__main__"]
    started: list[tuple[subprocess.Popen[str], Path]] = []
    real_popen = subprocess.Popen

    def record_process(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        process: subprocess.Popen[str] = real_popen(*args, **kwargs)
        started.append((process, Path(kwargs["cwd"])))
        return process

    monkeypatch.setattr(subprocess, "Popen", record_process)
    script = (
        "from pathlib import Path\n"
        "import sys\n"
        "import time\n"
        "from tests.adapters.test_dashboard_render import _app, _pipeline\n"
        "original_main = sys.modules['__main__']\n"
        "_app(Path.cwd(), _pipeline('signal'))\n"
        "assert sys.modules['__main__'] is not original_main\n"
        "print('AppTest completed', flush=True)\n"
    )
    if outcome == "normal":
        _run_isolated(script)
    elif outcome == "failure":
        with pytest.raises(subprocess.CalledProcessError) as failure:
            _run_isolated(script + "raise RuntimeError('injected dashboard failure')\n")
        assert "injected dashboard failure" in failure.value.stderr
        assert "AppTest completed" in failure.value.stdout
    else:
        with pytest.raises(subprocess.TimeoutExpired) as timeout:
            _run_isolated(script + "time.sleep(60)\n", timeout=15)
        assert timeout.value.stdout is not None
        assert b"AppTest completed" in timeout.value.stdout
    assert sys.modules["__main__"] is original_main
    assert len(started) == 1
    process, root = started[0]
    assert process.poll() is not None
    assert process.stdout is not None and process.stdout.closed
    assert process.stderr is not None and process.stderr.closed
    assert not root.exists()


def test_dashboard_workspace_is_retained_when_process_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[tuple[subprocess.Popen[str], Path]] = []
    real_popen = subprocess.Popen

    def cannot_kill() -> None:
        raise OSError("injected process cleanup failure")

    try:
        with monkeypatch.context() as patch:

            def record_process(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
                process: subprocess.Popen[str] = real_popen(*args, **kwargs)
                started.append((process, Path(kwargs["cwd"])))
                patch.setattr(process, "kill", cannot_kill)
                return process

            patch.setattr(subprocess, "Popen", record_process)
            with pytest.raises(RuntimeError, match="workspace retained at") as failure:
                _run_isolated("import time; time.sleep(60)", timeout=0.1)
            assert isinstance(failure.value.__cause__, OSError)
            assert len(started) == 1
            process, root = started[0]
            assert process.poll() is None
            assert root.is_dir()
    finally:
        # The injected failure is removed before reaping the real child.
        for process, root in started:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            shutil.rmtree(root)
