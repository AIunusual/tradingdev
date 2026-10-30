"""Streamlit dashboard for backtest result visualization (viewer-only).

Launch::

    streamlit run src/tradingdev/adapters/dashboard/app.py \
        -- --run-id <run_id>

Requires a completed MCP backtest or walk-forward run with a pipeline_result
artifact under workspace/runs/<run_id>/.

"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

if TYPE_CHECKING:
    from tradingdev.domain.backtest.pipeline_result import (
        PipelineResult,
    )
    from tradingdev.domain.backtest.result import BacktestResult

from tradingdev.adapters.dashboard.analysis import (
    available_months,
    build_equity_series,
    build_trades_df,
    consecutive_loss_counts,
    cumulative_pnl,
    cumulative_pnl_pct,
    filter_by_month,
    metric_cards,
    monthly_volume,
    rolling_mdd_absolute,
)
from tradingdev.app.artifact_service import ArtifactService
from tradingdev.app.report_service import ReportService
from tradingdev.app.run_service import RunService

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Backtest Dashboard",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_PIPELINE_KEY = "_pipeline_result"
_RUN_ID_KEY = "_run_id"
_REPORT_KEY = "_report_download"


def _parse_args() -> str | None:
    """Parse optional ``--run-id`` from sys.argv."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    known, _ = parser.parse_known_args()
    return str(known.run_id) if known.run_id else None


# ---------------------------------------------------------------------------
# Data loading from run/artifact services
# ---------------------------------------------------------------------------
def _resolve_run(
    run_service: RunService, requested_run_id: str | None
) -> dict[str, Any]:
    """Resolve the run selected by CLI argument or Streamlit sidebar."""
    if requested_run_id:
        response = run_service.get_run(requested_run_id)
        if not response.get("success"):
            st.error(str(response.get("error", "Run not found.")))
            st.stop()
            return {}
        return cast("dict[str, Any]", response["run"])

    runs = run_service.list_runs()
    if not runs:
        st.error("No completed runs found in workspace/tradingdev.sqlite.")
        st.stop()
        return {}

    with st.sidebar:
        labels = [
            (f"{run['run_id']} | {run['strategy_id']} | {run.get('created_at', '')}")
            for run in runs
        ]
        selected = st.selectbox("Run", labels)
    return runs[labels.index(selected)]


def _load_pipeline(
    artifact_service: ArtifactService,
    run_id: str,
) -> PipelineResult:
    """Load PipelineResult from the run artifact store. Stops app if missing."""
    response = artifact_service.load_pipeline_result(run_id)
    if not response.get("success"):
        st.error(str(response.get("error", "Pipeline artifact not found.")))
        st.stop()
        return None  # unreachable; st.stop() raises
    st.toast("Loaded from run artifacts")
    return cast("PipelineResult", response["pipeline"])


# ---------------------------------------------------------------------------
# Shared chart rendering
# ---------------------------------------------------------------------------
def _render_charts(  # noqa: PLR0915
    result: BacktestResult,
    unit: str,
    month: str,
    raw_config: dict[str, Any],
) -> None:
    """Render KPI row and 5 chart tabs for a single BacktestResult."""
    is_volume_mode = result.mode == "volume"

    # --- KPI row ----------------------------------------------------------
    cards = metric_cards(result.metrics, mode=result.mode)
    for column, (label, value) in zip(st.columns(len(cards)), cards, strict=True):
        column.metric(label, value)
    st.caption("Monetary values and traded volume use the market's quote currency.")

    # --- Prepare data -----------------------------------------------------
    equity = build_equity_series(result.equity_curve, result.timestamps)
    trades_df = build_trades_df(result.trades, result.timestamps)
    eq_view, tr_view = filter_by_month(equity, trades_df, month)
    closed_trades = tr_view.loc[tr_view["status"] == "closed"]

    # --- Monthly target ---------------------------------------------------
    monthly_target: float | None = None
    if is_volume_mode:
        strat_section = raw_config.get("strategy", {})
        params_section = strat_section.get("parameters", {})
        target_val = params_section.get("monthly_volume_target")
        if isinstance(target_val, (int, float)):
            monthly_target = float(target_val)

    # --- Tabs -------------------------------------------------------------
    tab_pnl, tab_hist, tab_streak, tab_monthly, tab_mdd = st.tabs(
        [
            "Cumulative PnL",
            "Trade PnL Distribution",
            "Consecutive Losses",
            "Monthly Volume",
            "Rolling 30d MDD",
        ]
    )

    # -- Tab 1: Cumulative PnL
    with tab_pnl:
        if unit == "Absolute" or is_volume_mode:
            pnl_series = cumulative_pnl(eq_view, result.init_cash)
            y_label = "Cumulative PnL"
        else:
            pnl_series = cumulative_pnl_pct(eq_view, result.init_cash)
            y_label = "Cumulative PnL (%)"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pnl_series.index,
                y=pnl_series.values,
                mode="lines",
                name="PnL",
                line={"color": "#1f77b4"},
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        fig.update_layout(
            yaxis_title=y_label,
            xaxis_title="Time",
            height=420,
            margin={"t": 30},
        )
        st.plotly_chart(fig, use_container_width=True)

    # -- Tab 2: Trade PnL histogram
    with tab_hist:
        if closed_trades.empty:
            st.info("No closed trades in selected period.")
        else:
            pnl_col = closed_trades["net_pnl"]
            if unit == "%" and closed_trades["size_quote"].sum() > 0:
                notionals = closed_trades["size_quote"].where(
                    closed_trades["size_quote"] > 0
                )
                pnl_col = closed_trades["net_pnl"] / notionals * 100
                x_label = "Per-trade PnL (%)"
            else:
                x_label = "Per-trade PnL"

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=pnl_col,
                    nbinsx=60,
                    marker_color="#636EFA",
                )
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(
                xaxis_title=x_label,
                yaxis_title="Frequency",
                height=420,
                margin={"t": 30},
            )
            st.plotly_chart(fig, use_container_width=True)

    # -- Tab 3: Consecutive loss streaks
    with tab_streak:
        counts = consecutive_loss_counts(tr_view)
        if counts.empty:
            st.info("No losing streaks in selected period.")
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=counts.index.astype(str),
                    y=counts.values,
                    marker_color="#EF553B",
                )
            )
            fig.update_layout(
                xaxis_title="Consecutive Loss Count",
                yaxis_title="Frequency",
                height=420,
                margin={"t": 30},
            )
            st.plotly_chart(fig, use_container_width=True)

    # -- Tab 4: Monthly volume achievement
    with tab_monthly:
        vol_df = monthly_volume(trades_df)
        if vol_df.empty:
            st.info("No trade data with timestamps.")
        else:
            colors = []
            if monthly_target is not None:
                for v in vol_df["volume_quote"]:
                    colors.append("#00CC96" if v >= monthly_target else "#EF553B")
            else:
                colors = ["#636EFA"] * len(vol_df)

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=vol_df["month"],
                    y=vol_df["volume_quote"],
                    marker_color=colors,
                    name="Volume",
                )
            )
            if monthly_target is not None:
                fig.add_hline(
                    y=monthly_target,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=(f"Target: {monthly_target:,.0f}"),
                )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Volume",
                height=420,
                margin={"t": 30},
            )
            st.plotly_chart(fig, use_container_width=True)

    # -- Tab 5: Rolling 30-day MDD (absolute)
    with tab_mdd:
        st.caption(
            "Trailing 30 calendar days (t − 30 days, t]; observed equity only. "
            "Unavailable until 30 days of history; gaps are not interpolated."
        )
        try:
            mdd_series = rolling_mdd_absolute(eq_view, pd.Timedelta(days=30))
        except ValueError as error:
            st.info(f"Rolling drawdown unavailable: {error}")
            return

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=mdd_series.index,
                y=mdd_series.values,
                mode="lines",
                name="Rolling 30d MDD",
                line={"color": "#EF553B"},
                fill="tozeroy",
                fillcolor="rgba(239,85,59,0.15)",
            )
        )
        fig.update_layout(
            yaxis_title="Max Drawdown",
            xaxis_title="Time",
            height=420,
            margin={"t": 30},
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Mode-specific renderers
# ---------------------------------------------------------------------------
def _render_simple(
    pipeline: PipelineResult,
    run: dict[str, Any],
) -> None:
    """Render dashboard for a simple (non-walk-forward) backtest."""
    raw_config = pipeline.config_snapshot
    result = pipeline.backtest_result
    if result is None:
        st.error("No backtest result in pipeline.")
        st.stop()
        return

    strategy_name = raw_config["strategy"].get("id") or run["strategy_id"]
    is_volume_mode = result.mode == "volume"

    # --- Header
    st.title(f"Backtest Dashboard — {strategy_name}")
    st.caption(f"Mode: **simple** | Run: `{run['run_id']}`")

    # --- Sidebar controls
    with st.sidebar:
        st.header("Controls")
        default_unit = "Absolute" if is_volume_mode else "%"
        unit = st.radio(
            "Display unit",
            ["Absolute", "%"],
            index=0 if default_unit == "Absolute" else 1,
            horizontal=True,
        )
        months = ["全期間"] + available_months(result.timestamps)
        selected_month = st.selectbox("Period", months)

    _render_charts(
        result,
        unit,
        selected_month,
        raw_config,
    )


def _render_walk_forward(
    pipeline: PipelineResult,
    run: dict[str, Any],
) -> None:
    """Render dashboard for walk-forward validation results."""
    raw_config = pipeline.config_snapshot
    folds = pipeline.fold_results
    strategy_name = raw_config["strategy"].get("id") or run["strategy_id"]

    # --- Header
    st.title(f"Backtest Dashboard — {strategy_name}")
    st.caption(f"Mode: **walk-forward** | Run: `{run['run_id']}`")

    # --- Sidebar controls
    with st.sidebar:
        st.header("Controls")

        fold_labels = [f"Fold {f.fold_index}" for f in folds]
        selected_fold_label = st.selectbox("Fold", fold_labels)
        fold_idx = fold_labels.index(selected_fold_label)
        fold = folds[fold_idx]

        split = st.radio(
            "Split",
            ["Test (OOS)", "Train (IS)"],
            index=0,
            horizontal=True,
        )
        is_test = split == "Test (OOS)"

        result: BacktestResult | None = (
            fold.test_backtest if is_test else fold.train_backtest
        )
        if result is None:
            st.error("BacktestResult not available for this split.")
            st.stop()
            return

        is_volume_mode = result.mode == "volume"
        default_unit = "Absolute" if is_volume_mode else "%"
        unit = st.radio(
            "Display unit",
            ["Absolute", "%"],
            index=0 if default_unit == "Absolute" else 1,
            horizontal=True,
        )
        months = ["全期間"] + available_months(result.timestamps)
        selected_month = st.selectbox("Period", months)

    # --- Fold info caption
    split_label = "Test (OOS)" if is_test else "Train (IS)"
    st.caption(
        f"**{selected_fold_label}** | {split_label} | "
        f"Train: {fold.train_start} ~ {fold.train_end} | "
        f"Test: {fold.test_start} ~ {fold.test_end}"
    )

    _render_charts(
        result,
        unit,
        selected_month,
        raw_config,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _render_report_download(run_id: str, service: ReportService) -> None:
    """Keep report identity in the session and let the service verify every read."""
    if st.button("Generate full report with trade history"):
        st.session_state.pop(_REPORT_KEY, None)
        report = service.generate_report([run_id])
        if not report["success"]:
            st.error(str(report["error"]))
            return
        st.session_state[_REPORT_KEY] = {
            "run_id": run_id,
            "report_id": report["report_id"],
            "sha256": report["sha256"],
        }
    saved_report = st.session_state.get(_REPORT_KEY)
    if not saved_report or saved_report["run_id"] != run_id:
        return
    download = service.get_report_download(
        saved_report["report_id"], expected_sha256=saved_report["sha256"]
    )
    if not download["success"]:
        st.session_state.pop(_REPORT_KEY, None)
        st.error(str(download["error"]))
        return
    st.download_button(
        "Download offline HTML report",
        download["content"],
        file_name=f"tradingdev-{run_id}.html",
        mime="text/html",
    )


def main() -> None:
    requested_run_id = _parse_args()
    run_service = RunService()
    artifact_service = ArtifactService()
    run = _resolve_run(run_service, requested_run_id)
    run_id = str(run["run_id"])

    with st.sidebar:
        _render_report_download(run_id, ReportService())

    if (
        _PIPELINE_KEY not in st.session_state
        or st.session_state.get(_RUN_ID_KEY) != run_id
    ):
        st.session_state[_RUN_ID_KEY] = run_id
        st.session_state[_PIPELINE_KEY] = _load_pipeline(artifact_service, run_id)

    pipeline: PipelineResult = st.session_state[_PIPELINE_KEY]

    if pipeline.mode == "walk_forward":
        _render_walk_forward(pipeline, run)
    else:
        _render_simple(pipeline, run)


if __name__ == "__main__":
    main()
