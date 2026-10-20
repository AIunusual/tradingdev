"""Native tear sheets preserve saved returns and isolate renderer resources."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
from textwrap import dedent
from threading import current_thread, main_thread
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from quantstats import reports, stats, utils

from tradingdev.adapters.reporting import quantstats_report
from tradingdev.adapters.reporting.quantstats_report import (
    QuantStatsReportError,
    build_quantstats_report,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _scope(returns: list[float] | None = None) -> dict[str, Any]:
    if returns is None:
        returns = np.random.default_rng(42).normal(0.001, 0.02, 90).tolist()
    return {
        "kind": "backtest",
        "mode": "signal",
        "metadata": {
            "settings": {
                "frequency": "1D",
                "periods_per_year": 365.0,
                "risk_free_rate": 0.0,
                "required_return": 0.03,
            }
        },
        "observations": {
            "timestamps": pd.date_range("2024-01-01", periods=len(returns), tz="UTC")
            .astype(str)
            .tolist(),
            "returns": returns,
            "equity_curve": (10000 * np.cumprod(1 + np.asarray(returns))).tolist(),
            "init_cash": 10000.0,
        },
    }


def _native_metric(html: str, label: str) -> str:
    match = re.search(rf"<td[^>]*>{re.escape(label)}</td>\s*<td[^>]*>(.*?)</td>", html)
    assert match is not None, label
    return match.group(1)


@pytest.mark.parametrize("periods", [365.0, 3.5])
def test_native_html_is_deterministic_offline_and_restores_plot_state(
    monkeypatch: pytest.MonkeyPatch, periods: float
) -> None:
    def forbid_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Native report must not fetch remote data")

    monkeypatch.setattr(utils, "download_returns", forbid_network)
    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    scope = _scope()
    scope["metadata"]["settings"]["risk_free_rate"] = 0.02
    scope["metadata"]["settings"]["periods_per_year"] = periods
    original = copy.deepcopy(scope)
    original_backend = mpl.get_backend(auto_select=False)
    plt.switch_backend("svg")
    existing = plt.figure()
    before_figures = set(plt.get_fignums())
    try:
        with mpl.rc_context({"axes.facecolor": "red", "svg.hashsalt": "caller"}):
            first = build_quantstats_report(scope)
            assert mpl.rcParams["axes.facecolor"] == "red"
            assert mpl.rcParams["svg.hashsalt"] == "caller"
        with mpl.rc_context({"axes.facecolor": "green", "svg.hashsalt": "different"}):
            second = build_quantstats_report(scope)
            assert mpl.rcParams["axes.facecolor"] == "green"
        assert first == second
        assert first["available"] is True
        html = first["html"]
        assert "<html" in html and "<svg" in html
        assert f"Periods/Year: {periods:g}" in html
        assert "Cumulative Returns" in html
        assert "Monthly Returns" in html
        native_tables = re.findall(r"<table\b.*?</table>", html, flags=re.DOTALL)
        assert len(native_tables) >= 3
        assert re.findall(r"<td\b[^>]*>.*?</td>", first["tables_html"], re.DOTALL) == (
            re.findall(r"<td\b[^>]*>.*?</td>", "\n".join(native_tables), re.DOTALL)
        )
        assert "Return (%)" in first["tables_html"]
        assert "Cumulative (%)" in first["tables_html"]
        assert "Drawdown (%)" in first["tables_html"]
        assert "Cumulative Return" in first["tables_html"]
        assert not re.search(r"<(?:svg|script|html|body|aside)\b", first["tables_html"])
        assert "<dc:date>" not in html
        assert "qtpylib.io/favicon.ico" not in html
        assert not re.search(
            r"<(?:script|link|img)\b[^>]*(?:src|href)=[\"']https?://", html
        )
        raw_returns = np.asarray(scope["observations"]["returns"])
        total_return = float(np.prod(1 + raw_returns) - 1)
        assert _native_metric(html, "Cumulative Return") == f"{total_return * 100:.2f}%"
        excess = raw_returns - ((1.02 ** (1 / periods)) - 1)
        sharpe = float(excess.mean() / excess.std(ddof=1) * math.sqrt(periods))
        downside = math.sqrt(float(np.square(np.minimum(excess, 0)).mean()))
        sortino = float(excess.mean() / downside * math.sqrt(periods))
        assert float(_native_metric(html, "Sharpe")) == pytest.approx(sharpe, abs=0.005)
        assert float(_native_metric(html, "Sortino")) == pytest.approx(
            sortino, abs=0.005
        )
        metadata = first["metadata"]
        assert metadata["providers"]["quantstats"]
        assert metadata["providers"]["matplotlib"]
        assert len(metadata["input_sha256"]) == 64
        assert metadata["daily_observation_count"] == 90
        assert metadata["settings"]["native_rolling_risk_free_rate"] == 0
        assert metadata["settings"]["sortino_target"] == "risk_free_rate"
        assert metadata["settings"]["saved_required_return"] == 0.03
        assert metadata["settings"]["native_cagr_basis"] == "gross_compounded_returns"
        assert metadata["settings"]["native_eoy_return_basis"] == (
            "yearly_sum_of_daily_returns"
        )
        assert metadata["settings"]["native_eoy_cumulative_basis"] == (
            "yearly_compounded_returns"
        )
        assert scope == original
        assert set(plt.get_fignums()) == before_figures
        assert mpl.get_backend(auto_select=False) == "svg"
    finally:
        plt.close(existing)
        if original_backend is not None:
            plt.switch_backend(original_backend)


def test_native_report_renders_in_worker_with_unselected_backend(
    tmp_path: Path,
) -> None:
    config = tmp_path / "matplotlib"
    config.mkdir()
    matplotlibrc = config / "matplotlibrc"
    matplotlibrc.write_text("", encoding="utf-8")
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    source = tmp_path / "scope.json"
    source.write_text(json.dumps(_scope()), encoding="utf-8")
    environment = os.environ.copy()
    environment.pop("MPLBACKEND", None)
    environment.update(
        MPLCONFIGDIR=str(config),
        MATPLOTLIBRC=str(matplotlibrc),
        TMPDIR=str(temporary),
    )
    script = dedent(
        """
        import json
        import sys
        from concurrent.futures import ThreadPoolExecutor
        from threading import current_thread, main_thread

        import matplotlib as mpl
        from matplotlib import pyplot as plt

        from tradingdev.adapters.reporting.quantstats_report import (
            build_quantstats_report,
        )

        assert mpl.get_backend(auto_select=False) is None
        assert plt.get_fignums() == []
        mpl.rcParams["axes.facecolor"] = "orange"
        mpl.rcParams["svg.hashsalt"] = "caller"
        plt.ion()
        with open(sys.argv[1], encoding="utf-8") as source:
            scope = json.load(source)

        def render():
            assert current_thread() is not main_thread()
            assert mpl.get_backend(auto_select=False) is None
            return build_quantstats_report(scope)

        with ThreadPoolExecutor(max_workers=1) as executor:
            report = executor.submit(render).result(timeout=90)
        assert report["available"] is True
        assert "<svg" in report["html"]
        assert "Cumulative Return" in report["tables_html"]
        assert mpl.get_backend(auto_select=False) == "agg"
        assert mpl.rcParams["axes.facecolor"] == "orange"
        assert mpl.rcParams["svg.hashsalt"] == "caller"
        assert plt.isinteractive()
        assert plt.get_fignums() == []
        print("worker report rendered and plot state restored")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(source)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "worker report rendered and plot state restored" in completed.stdout
    assert list(temporary.iterdir()) == []


def test_utc_daily_compounding_preserves_first_return_and_missing_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = _scope([0.1, -0.2, 0.03, -0.01])
    scope["metadata"]["settings"]["frequency"] = "1h"
    scope["metadata"]["settings"]["periods_per_year"] = 360
    scope["observations"]["timestamps"] = [
        "2024-01-01T20:00:00-05:00",
        "2024-01-02T02:00:00+00:00",
        "2024-01-05T10:00:00+08:00",
        "2024-01-05T03:00:00+00:00",
    ]
    captured: list[float] = []

    def native(daily: pd.Series, periods: int | float, rate: float) -> str:
        assert daily.index.tolist() == [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-05"),
        ]
        captured.extend(daily.tolist())
        assert periods == 360
        assert rate == 0
        return "<html><body>native</body></html>"

    monkeypatch.setattr(quantstats_report, "_native_html", native)
    result = build_quantstats_report(scope)
    assert captured == pytest.approx([1.1 * 0.8 - 1, 1.03 * 0.99 - 1])
    assert result["metadata"]["daily_observation_count"] == 2
    assert result["metadata"]["start_date"] == "2024-01-02T00:00:00"
    assert result["metadata"]["settings"]["missing_dates"] == "not_filled"
    digest_input = {
        "dates": ["2024-01-02", "2024-01-05"],
        "returns": captured,
    }
    expected_digest = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert result["metadata"]["daily_returns_sha256"] == expected_digest


@pytest.mark.parametrize(
    ("section", "key", "value", "reason"),
    [
        ("scope", "kind", "fold_summary", "unsupported_scope"),
        ("scope", "mode", "volume", "unsupported_mode"),
        ("scope", "observations", None, "missing_observations"),
        ("observations", "timestamps", None, "missing_timestamps"),
        ("observations", "returns", None, "missing_returns"),
        ("observations", "returns", [None] * 3, "nonfinite_observations"),
        ("observations", "returns", [float("inf")] * 3, "nonfinite_observations"),
        ("observations", "returns", [1.1, 0.1, -0.1], "ambiguous_return_input"),
        ("observations", "returns", [1.0, 0.1, 0.1], "ambiguous_return_input"),
        ("observations", "returns", [0.1, -1.0, 0.1], "unsupported_return_domain"),
        ("observations", "equity_curve", [1.0], "unaligned_observations"),
        ("observations", "equity_curve", [1.0, 0.0, 1.0], "nonpositive_capital"),
        ("observations", "init_cash", 0, "nonpositive_capital"),
        ("observations", "timestamps", ["invalid"] * 3, "invalid_timestamps"),
        ("observations", "timestamps", ["2024-01-01"] * 3, "invalid_timestamps"),
        ("settings", "frequency", None, "unknown_bar_frequency"),
        ("settings", "frequency", "1W", "unsupported_daily_sampling"),
        ("settings", "periods_per_year", None, "missing_annualization"),
        ("settings", "periods_per_year", 0, "missing_annualization"),
        ("settings", "risk_free_rate", None, "invalid_risk_free_rate"),
        ("settings", "risk_free_rate", -1, "invalid_risk_free_rate"),
        (
            "settings",
            "risk_free_rate",
            -0.02,
            "unsupported_negative_risk_free_rate",
        ),
    ],
)
def test_unsupported_inputs_are_explicit_without_running_native_renderer(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    key: str,
    value: Any,
    reason: str,
) -> None:
    scope = _scope([0.01, -0.02, 0.03])
    target = (
        scope
        if section == "scope"
        else scope["metadata"]["settings"]
        if section == "settings"
        else scope["observations"]
    )
    target[key] = value

    def unexpected(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("Unsupported input must not reach native rendering")

    monkeypatch.setattr(quantstats_report, "_native_html", unexpected)
    result = build_quantstats_report(scope)
    assert result["available"] is False
    assert result["reason"] == reason
    assert result["message"]
    assert result["html"] is None
    assert result["tables_html"] is None


@pytest.mark.parametrize("returns", [[], [0.01]])
def test_empty_or_single_day_cannot_produce_a_native_report(
    returns: list[float],
) -> None:
    result = build_quantstats_report(_scope(returns))
    assert result["available"] is False
    assert result["reason"] == "insufficient_data"


def test_negative_risk_free_rate_is_rejected_because_provider_ignores_it() -> None:
    returns = pd.Series(
        [0.01, -0.02, 0.03], index=pd.date_range("2024-01-01", periods=3)
    )
    native_sharpe = cast("Callable[..., float]", stats.sharpe)
    native_sortino = cast("Callable[..., float]", stats.sortino)
    assert native_sharpe(returns, rf=-0.02, periods=365) == native_sharpe(
        returns, rf=0.0, periods=365
    )
    assert native_sortino(returns, rf=-0.02, periods=365) == native_sortino(
        returns, rf=0.0, periods=365
    )
    excess = returns - ((0.98 ** (1 / 365)) - 1)
    expected = float(excess.mean() / excess.std(ddof=1) * math.sqrt(365))
    assert native_sharpe(returns, rf=-0.02, periods=365) != pytest.approx(expected)
    scope = _scope()
    scope["metadata"]["settings"]["risk_free_rate"] = -0.02
    assert build_quantstats_report(scope)["available"] is False


@pytest.mark.parametrize(
    "failure",
    [
        None,
        RuntimeError("provider failure"),
        TimeoutError("timed out"),
        KeyboardInterrupt("interrupted"),
        SystemExit("exit requested"),
    ],
)
@pytest.mark.parametrize("backend", ["agg", "svg"])
@pytest.mark.parametrize("interactive", [False, True])
def test_worker_render_restores_plot_state_and_cleans_output_even_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException | None,
    backend: str,
    interactive: bool,
) -> None:
    outputs: list[Path] = []

    def native(*args: Any, **kwargs: Any) -> None:
        assert current_thread() is not main_thread()
        assert mpl.get_backend(auto_select=False) == "agg"
        assert not plt.isinteractive()
        output = Path(kwargs["output"])
        outputs.append(output)
        plt.figure()
        output.write_text("partial" if failure else "<html><body>native</body></html>")
        if failure is not None:
            raise failure

    monkeypatch.setattr(reports, "html", native)
    original_backend = mpl.get_backend(auto_select=False)
    plt.switch_backend(backend)
    existing = plt.figure()
    existing_canvas = existing.canvas
    figures_before = set(plt.get_fignums())
    try:
        with mpl.rc_context({"axes.facecolor": "orange", "interactive": interactive}):
            with ThreadPoolExecutor(max_workers=1) as executor:
                rendered = executor.submit(build_quantstats_report, _scope())
                if failure is None:
                    assert rendered.result(timeout=10)["available"] is True
                else:
                    expected = (
                        QuantStatsReportError
                        if isinstance(failure, Exception)
                        else type(failure)
                    )
                    with pytest.raises(expected, match=str(failure)) as error:
                        rendered.result(timeout=10)
                    if isinstance(failure, Exception):
                        assert error.value.__cause__ is failure
            assert mpl.get_backend(auto_select=False) == backend
            assert plt.isinteractive() is interactive
            assert mpl.rcParams["axes.facecolor"] == "orange"
        assert len(outputs) == 1
        assert not outputs[0].exists()
        assert not outputs[0].parent.exists()
        assert set(plt.get_fignums()) == figures_before
        assert existing.canvas is existing_canvas
        assert plt.figure(existing.number) is existing
    finally:
        plt.close(existing)
        if original_backend is not None:
            plt.switch_backend(original_backend)


@pytest.mark.parametrize("include_source", [False, True])
def test_download_keeps_escaped_source_and_complete_calculation_evidence(
    monkeypatch: pytest.MonkeyPatch, include_source: bool
) -> None:
    scope = _scope()
    injected = (
        '</script><table id="injected"><tr><td>fake</td></tr></table>'
        '<img src="https://invalid.example/x" onerror="alert(1)">'
    )
    if include_source:
        scope["scope_id"] = "test/" + injected
        scope["report_source"] = {
            "run_id": "run/" + injected,
            "artifacts": [{"path": injected, "sha256": "a" * 64}],
        }
    original = copy.deepcopy(scope)

    native_table = '<table id="metrics"><tr><td>Native content</td></tr></table>'

    def native(*args: Any) -> str:
        return '<html><body class="native">' + native_table + "</body></html>"

    monkeypatch.setattr(quantstats_report, "_native_html", native)
    result = build_quantstats_report(scope)
    metadata = result["metadata"]
    html = result["html"]
    assert html.index('id="tradingdev-report-source"') < html.index("Native content")
    assert metadata["input_sha256"] in html
    assert "期間報酬" in html and "不能當作逐筆成交統計" in html
    assert "最低要求報酬率為 3.00%" in html
    assert "滾動 Sharpe／Sortino 圖的無風險利率固定為 0.00%" in html
    assert "不是跨年累積報酬率" in html
    assert result["tables_html"] == native_table
    assert 'id="injected"' not in result["tables_html"]
    assert "tradingdev-report-source" not in result["tables_html"]
    match = re.search(
        r'<script id="quantstats-report-evidence" type="application/json">'
        r"(.*?)</script>",
        html,
        re.DOTALL,
    )
    assert match is not None
    assert json.loads(match.group(1)) == metadata
    assert scope == original
    if include_source:
        assert metadata["source"] == original["report_source"]
        assert metadata["scope_id"] == original["scope_id"]
        assert escape("run/" + injected) in html
        assert escape("test/" + injected) in html
        assert injected not in html
        metadata["source"]["artifacts"][0]["path"] = "changed copy"
        assert scope == original
    else:
        assert "source" not in metadata and "scope_id" not in metadata
        assert "執行編號" not in html and "結果範圍" not in html


def test_native_tables_preserve_exact_markup_without_surrounding_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = (
        '<TABLE class="metrics">\n<tr><td>Sharpe &amp; Sortino</td>'
        "<td>0.12</td></tr>\n</TABLE>"
    )
    second = "<table><tr><td><table><tr><td>3.45</td></tr></table></td></tr></table>"
    native = (
        "<html><body>\u2028\u2029\r\n"
        "<script>const text = '<table>fake</table>';</script>"
        "<!-- <table>comment</table> --><svg><text>plot</text></svg>\n"
        + first
        + "<p>Between tables</p>\n"
        + second
        + "</body></html>"
    )
    monkeypatch.setattr(quantstats_report, "_native_html", lambda *args: native)
    result = build_quantstats_report(_scope())
    assert result["tables_html"] == first + "\n" + second
    assert first in result["html"] and second in result["html"]


def test_incomplete_native_table_is_not_silently_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        quantstats_report,
        "_native_html",
        lambda *args: "<html><body><table><tr><td>partial</td></tr></body></html>",
    )
    with pytest.raises(ValueError, match="incomplete table"):
        build_quantstats_report(_scope())
