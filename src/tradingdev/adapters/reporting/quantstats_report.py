"""Native QuantStats tear sheets derived exclusively from saved observations."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
from copy import deepcopy
from html.parser import HTMLParser
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import TYPE_CHECKING, Any, TypeGuard, cast

import numpy as np
import pandas as pd

from tradingdev.adapters.presentation.html import escape, json_text
from tradingdev.domain.performance.sampling import daily_observation_unavailable_reason

if TYPE_CHECKING:
    from collections.abc import Callable

_RENDER_LOCK = RLock()
_RENDER_VERSION = 4
_MESSAGES = {
    "unsupported_scope": "各折統計摘要沒有可供 QuantStats 分析的單一報酬序列。",
    "unsupported_mode": (
        "固定交易量模式沒有投資組合報酬序列，無法產生 QuantStats 報告。"
    ),
    "missing_observations": "沒有保存原始觀測資料，無法產生 QuantStats 報告。",
    "missing_returns": "沒有保存報酬率序列，無法產生 QuantStats 報告。",
    "missing_timestamps": "沒有保存有效日期，無法產生 QuantStats 日報酬分析。",
    "invalid_timestamps": "保存日期必須有效且嚴格遞增，不能重複或重新排序。",
    "unaligned_observations": "保存日期、報酬率與權益序列長度不一致。",
    "missing_annualization": "沒有保存有效的每年計算期數，不能推測年化設定。",
    "unknown_bar_frequency": "沒有可辨識的 K 線頻率，不能推測為日報酬資料。",
    "unsupported_daily_sampling": "K 線頻率大於一天，不能建立逐日報酬分析。",
    "insufficient_data": "至少需要兩個有觀測資料的 UTC 日期才能產生 QuantStats 報告。",
    "nonfinite_observations": "報酬率或權益含缺值或非有限數值，不能補成零報酬。",
    "nonpositive_capital": "初始資金或保存權益不是正數，不能產生複利投資組合報告。",
    "unsupported_return_domain": "報酬率達到或低於 -100%，不能產生此複利報告。",
    "ambiguous_return_input": (
        "日報酬達到 100%，QuantStats 原生報告可能誤認為價格資料。"
    ),
    "invalid_risk_free_rate": "沒有保存有效的無風險利率，不能推測其值。",
    "unsupported_negative_risk_free_rate": (
        "此版 QuantStats 原生報告會忽略負無風險利率，故不產生可能誤導的報告。"
    ),
}


class QuantStatsReportError(Exception):
    """Native provider failed to render an otherwise eligible report."""


class _UnavailableInputError(Exception):
    """A known saved-data limitation, distinct from rendering failures."""


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite_number(value: Any) -> TypeGuard[int | float]:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _digest(value: Any) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _daily_returns(scope: dict[str, Any], settings: dict[str, Any]) -> pd.Series:
    if scope.get("kind") != "backtest":
        raise _UnavailableInputError("unsupported_scope")
    if scope.get("mode") != "signal":
        raise _UnavailableInputError("unsupported_mode")
    observations = scope.get("observations")
    if not isinstance(observations, dict):
        raise _UnavailableInputError("missing_observations")
    returns = observations.get("returns")
    if not isinstance(returns, list):
        raise _UnavailableInputError("missing_returns")
    timestamps = observations.get("timestamps")
    if not isinstance(timestamps, list):
        raise _UnavailableInputError("missing_timestamps")
    equity = observations.get("equity_curve")
    if (
        not isinstance(equity, list)
        or len(returns) != len(timestamps)
        or len(returns) != len(equity)
    ):
        raise _UnavailableInputError("unaligned_observations")
    if not returns:
        raise _UnavailableInputError("insufficient_data")
    frequency = settings.get("frequency")
    if not isinstance(frequency, str):
        raise _UnavailableInputError("unknown_bar_frequency")
    if reason := daily_observation_unavailable_reason(frequency):
        raise _UnavailableInputError(reason)
    if not all(isinstance(stamp, str) for stamp in timestamps):
        raise _UnavailableInputError("invalid_timestamps")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="raise"))
    except (ValueError, TypeError) as error:
        raise _UnavailableInputError("invalid_timestamps") from error
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise _UnavailableInputError("invalid_timestamps")
    if not all(_finite_number(value) for value in [*returns, *equity]):
        raise _UnavailableInputError("nonfinite_observations")
    initial_cash = observations.get("init_cash")
    if (
        not _finite_number(initial_cash)
        or initial_cash <= 0
        or any(value <= 0 for value in equity)
    ):
        raise _UnavailableInputError("nonpositive_capital")
    if any(value <= -1 for value in returns):
        raise _UnavailableInputError("unsupported_return_domain")
    series: pd.Series[float] = pd.Series(
        np.asarray(returns, dtype=float), index=dates, name="Strategy"
    )
    # Group only observed UTC days. Resampling would silently insert missing days.
    daily = series.add(1).groupby(dates.normalize()).prod().sub(1)
    daily.index = pd.DatetimeIndex(daily.index).tz_localize(None)
    if len(daily) < 2:
        raise _UnavailableInputError("insufficient_data")
    if not np.isfinite(daily.to_numpy()).all():
        raise _UnavailableInputError("nonfinite_observations")
    if daily.le(-1).any():
        raise _UnavailableInputError("unsupported_return_domain")
    # Native HTML has no public "these are returns" switch. Its internal helpers
    # also inspect sliced data, so a large positive observation is ambiguous even
    # when the entire input contains negative returns.
    if daily.ge(1).any():
        raise _UnavailableInputError("ambiguous_return_input")
    return daily


def _native_html(daily: pd.Series, periods: int | float, rate: float) -> str:
    # Keep optional plotting imports out of unavailable-data paths. QuantStats
    # lazily imports its plotting module, which changes rcParams on first import.
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    with _RENDER_LOCK, mpl.rc_context(), plt.ioff():
        # Resolving an unset backend can select a GUI, which cannot create
        # figures in MCP worker threads. Matplotlib >=3.10 preserves existing
        # figures when switching backends; rc_context does not restore backend.
        backend = mpl.get_backend(auto_select=False)
        existing_figures = set(plt.get_fignums())
        try:
            plt.switch_backend("agg")
            import quantstats.plots  # noqa: F401
            from quantstats import reports

            # Reset inside the context so the caller's styling and first-import
            # effects cannot change the artifact. rc_context restores styling.
            mpl.rcdefaults()
            mpl.rcParams.update({"svg.hashsalt": "tradingdev-quantstats-v1"})
            with TemporaryDirectory(prefix="tradingdev-quantstats-") as directory:
                output = Path(directory) / "report.html"
                native_html = cast("Callable[..., Any]", reports.html)
                native_html(
                    daily.copy(),
                    benchmark=None,
                    rf=rate,
                    output=str(output),
                    title="Strategy Performance",
                    compounded=True,
                    periods_per_year=periods,
                    figfmt="svg",
                    match_dates=False,
                )
                rendered = output.read_text(encoding="utf-8")
        finally:
            try:
                for number in set(plt.get_fignums()) - existing_figures:
                    plt.close(number)
            finally:
                # Leave Agg selected when the caller had no backend yet.
                if backend is not None:
                    plt.switch_backend(backend)
    # Matplotlib adds wall-clock SVG metadata; it is not result evidence.
    rendered = re.sub(r"<dc:date>[^<]*</dc:date>", "", rendered)
    # The native template otherwise consists of embedded styles/SVGs. Its favicon
    # is the only external resource and is irrelevant to a portable saved report.
    return re.sub(
        r'<link\b[^>]*\brel=["\'](?:shortcut )?icon["\'][^>]*>',
        "",
        rendered,
        flags=re.IGNORECASE,
    )


class _NativeTablesParser(HTMLParser):
    """Locate complete native tables without reserializing their contents."""

    def __init__(self, rendered: str) -> None:
        super().__init__(convert_charrefs=False)
        self._rendered = rendered
        self._line_offsets = [
            0,
            *(match.end() for match in re.finditer("\n", rendered)),
        ]
        self._depth = 0
        self._start = 0
        self.tables: list[str] = []

    def _offset(self) -> int:
        line, column = self.getpos()
        return self._line_offsets[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            if self._depth == 0:
                self._start = self._offset()
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._depth:
            self._depth -= 1
            if self._depth == 0:
                end = self._rendered.index(">", self._offset()) + 1
                self.tables.append(self._rendered[self._start : end])

    def complete_tables(self) -> str:
        self.feed(self._rendered)
        self.close()
        if self._depth:
            raise ValueError("Native QuantStats report contains an incomplete table")
        # Native CSS supplies some percent signs through pseudo-elements. Text
        # readers do not see those styles, so keep units in actual header text.
        return re.sub(
            r"(<th\b[^>]*>\s*)(Return|Cumulative|Drawdown)(\s*</th>)",
            r"\1\2 (%)\3",
            "\n".join(self.tables),
            flags=re.IGNORECASE,
        )


def _with_provenance(rendered: str, metadata: dict[str, Any]) -> str:
    """Keep a downloaded native page understandable without its parent report."""
    identity = []
    source = _object(metadata.get("source"))
    for label, value in (
        ("執行編號", source.get("run_id")),
        ("結果範圍", metadata.get("scope_id")),
        ("輸入資料 SHA-256", metadata["input_sha256"]),
    ):
        if value is not None:
            identity.append("<p>" + label + "：" + escape(value) + "</p>")
    settings = metadata["settings"]
    rate = escape(f"{settings['risk_free_rate']:.2%}")
    required_return = settings["saved_required_return"]
    target_note = (
        "保存設定的最低要求報酬率為 "
        + escape(f"{required_return:.2%}")
        + "；原生 Sortino 使用上述無風險利率作為目標。"
        if _finite_number(required_return)
        else "原生 Sortino 使用上述無風險利率作為目標。"
    )
    evidence = (
        json_text(metadata)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    banner = (
        '<aside id="tradingdev-report-source" lang="zh-Hant" '
        'style="max-width:960px;margin:20px auto;padding:16px;'
        'border:1px solid #dce5ef;background:#f4f7fb;overflow-wrap:anywhere">'
        "<h2>報告來源與計算口徑</h2>"
        + "".join(identity)
        + "<p>此原生報告分析期間報酬；勝率、獲利因子等統計依報酬期間計算，"
        "不能當作逐筆成交統計。</p><p>績效表無風險利率："
        + rate
        + "。"
        + target_note
        + "滾動 Sharpe／Sortino 圖的無風險利率固定為 0.00%。"
        "CAGR 使用未扣無風險利率的複利報酬。</p>"
        "<p>EOY 表的 Return 是當年度日報酬率加總；Cumulative 是當年度複利報酬率，"
        "不是跨年累積報酬率。兩欄數值單位均為百分比。</p>"
        '<script id="quantstats-report-evidence" type="application/json">'
        + evidence
        + "</script></aside>"
    )
    result, count = re.subn(
        r"<body\b[^>]*>",
        lambda match: match.group(0) + banner,
        rendered,
        count=1,
        flags=re.IGNORECASE,
    )
    if count != 1:
        raise ValueError("Native QuantStats report has no body element")
    return result


def build_quantstats_report(scope: dict[str, Any]) -> dict[str, Any]:
    """Generate native HTML without replacing the run's saved metric values.

    Known unsupported observations yield an explicit unavailable result. Unknown
    provider errors become QuantStatsReportError after resources are cleaned up.
    """
    settings = _object(_object(scope.get("metadata")).get("settings"))
    metadata: dict[str, Any] = {
        "renderer_version": _RENDER_VERSION,
        "providers": {
            package: version(package)
            for package in (
                "quantstats",
                "numpy",
                "pandas",
                "matplotlib",
                "seaborn",
                "scipy",
                "tabulate",
            )
        },
        "python_version": platform.python_version(),
        "saved_settings": settings.copy(),
        "settings": {
            "periods_per_year": settings.get("periods_per_year"),
            "risk_free_rate": settings.get("risk_free_rate"),
            "sortino_target": "risk_free_rate",
            "saved_required_return": settings.get("required_return"),
            "native_rolling_risk_free_rate": 0.0,
            "native_cagr_basis": "gross_compounded_returns",
            "native_eoy_return_basis": "yearly_sum_of_daily_returns",
            "native_eoy_cumulative_basis": "yearly_compounded_returns",
            "returns_sampling": "observed_utc_daily_compounded",
            "missing_dates": "not_filled",
            "first_return": "preserved_from_saved_observations",
            "benchmark": None,
            "compounded": True,
            "match_dates": False,
            "svg_date": "removed",
            "svg_hashsalt": "tradingdev-quantstats-v1",
        },
    }
    if source := _object(scope.get("report_source")):
        metadata["source"] = deepcopy(source)
    if isinstance(scope.get("scope_id"), str):
        metadata["scope_id"] = scope["scope_id"]
    try:
        daily = _daily_returns(scope, settings)
        periods = settings.get("periods_per_year")
        if not _finite_number(periods) or periods <= 0:
            raise _UnavailableInputError("missing_annualization")
        rate = settings.get("risk_free_rate")
        if not _finite_number(rate) or rate <= -1:
            raise _UnavailableInputError("invalid_risk_free_rate")
        if rate < 0:
            # QuantStats 0.0.86's scalar _rf_is_nonzero returns rf > 0.
            raise _UnavailableInputError("unsupported_negative_risk_free_rate")
    except _UnavailableInputError as error:
        reason = str(error)
        return {
            "available": False,
            "reason": reason,
            "message": _MESSAGES[reason],
            "html": None,
            "tables_html": None,
            "metadata": metadata,
        }
    observations = scope["observations"]
    metadata["input_sha256"] = _digest(
        {
            "kind": scope["kind"],
            "mode": scope["mode"],
            "settings": settings,
            "timestamps": observations["timestamps"],
            "returns": observations["returns"],
            "equity_curve": observations["equity_curve"],
            "init_cash": observations["init_cash"],
        }
    )
    metadata["daily_returns_sha256"] = _digest(
        {
            "dates": pd.DatetimeIndex(daily.index).strftime("%Y-%m-%d").tolist(),
            "returns": daily.tolist(),
        }
    )
    metadata["daily_observation_count"] = len(daily)
    metadata["start_date"] = daily.index[0].isoformat()
    metadata["end_date"] = daily.index[-1].isoformat()
    # Keep integer annualization values legible in the native subtitle.
    native_periods = int(periods) if float(periods).is_integer() else float(periods)
    try:
        native = _native_html(daily, native_periods, float(rate))
    except Exception as error:
        raise QuantStatsReportError(f"QuantStats rendering failed: {error}") from error
    tables_html = _NativeTablesParser(native).complete_tables()
    rendered = _with_provenance(native, metadata)
    return {
        "available": True,
        "reason": None,
        "message": None,
        "html": rendered,
        "tables_html": tables_html,
        "metadata": metadata,
    }
