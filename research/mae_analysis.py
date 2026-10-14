"""MAE (Max Adverse Excursion) 分析 — 評估 stop-loss 水位是否合理.

Usage:
    uv run streamlit run research/mae_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/processed/btcusdt_1m_2024_2025.parquet")
DEFAULT_HOLDING_BARS = 30
DEFAULT_SL = 0.015


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load OHLCV parquet and return a clean DataFrame."""
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Core computation (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def compute_mae(
    df: pd.DataFrame,
    holding_bars: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Compute MAE for every bar as a hypothetical entry.

    For each entry at close[t], look forward `holding_bars` bars and compute:
      - mae_long:  (entry - rolling_min(low))  / entry
      - mae_short: (rolling_max(high) - entry) / entry
      - mae_worst: max(mae_long, mae_short)
    """
    mask = (df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")) & (
        df["timestamp"] <= pd.Timestamp(end_date, tz="UTC")
    )
    data = df.loc[mask].copy().reset_index(drop=True)

    # Forward-looking rolling min/max (shift by -holding_bars to align)
    # We want min(low[t+1 : t+1+holding_bars]) for entry at t.
    # Strategy: reverse the series, apply rolling, reverse back, then shift.
    low_rev = data["low"].iloc[::-1]
    high_rev = data["high"].iloc[::-1]

    rolling_min_low = (
        low_rev.rolling(window=holding_bars, min_periods=1).min().iloc[::-1]
    )
    rolling_max_high = (
        high_rev.rolling(window=holding_bars, min_periods=1).max().iloc[::-1]
    )

    # Shift by -1 so we exclude the entry bar itself (look at t+1 onward).
    rolling_min_low = rolling_min_low.shift(-1).reset_index(drop=True)
    rolling_max_high = rolling_max_high.shift(-1).reset_index(drop=True)

    entry = data["close"]

    mae_long = (entry - rolling_min_low) / entry
    mae_short = (rolling_max_high - entry) / entry
    mae_worst = np.maximum(mae_long, mae_short)

    result = pd.DataFrame(
        {
            "timestamp": data["timestamp"],
            "mae_long": mae_long,
            "mae_short": mae_short,
            "mae_worst": mae_worst,
        }
    )
    # Drop tail rows that don't have a full forward window.
    return result.iloc[: len(result) - holding_bars].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_mae_histogram(
    mae_df: pd.DataFrame, sl_level: float, holding_bars: int
) -> go.Figure:
    """Histogram of MAE distributions with SL vertical line."""
    fig = go.Figure()

    for col, name, color in [
        ("mae_long", "Long MAE", "#EF553B"),
        ("mae_short", "Short MAE", "#636EFA"),
        ("mae_worst", "Worst MAE", "#AB63FA"),
    ]:
        fig.add_trace(
            go.Histogram(
                x=mae_df[col] * 100,
                name=name,
                opacity=0.55,
                nbinsx=200,
                marker_color=color,
            )
        )

    fig.add_vline(
        x=sl_level * 100,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"SL = {sl_level:.1%}",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"MAE Distribution (holding = {holding_bars} min)",
        xaxis_title="MAE (%)",
        yaxis_title="Count",
        barmode="overlay",
        height=450,
    )
    return fig


def plot_sl_trigger_curve(mae_worst: pd.Series, current_sl: float) -> go.Figure:
    """Trigger rate vs SL level sweep."""
    sl_levels = np.arange(0.002, 0.031, 0.001)
    trigger_rates = [(mae_worst >= sl).mean() * 100 for sl in sl_levels]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sl_levels * 100,
            y=trigger_rates,
            mode="lines+markers",
            name="Trigger Rate",
            line={"color": "#636EFA", "width": 2},
            marker={"size": 5},
        )
    )

    current_rate = float((mae_worst >= current_sl).mean() * 100)
    fig.add_trace(
        go.Scatter(
            x=[current_sl * 100],
            y=[current_rate],
            mode="markers+text",
            name=f"Current SL ({current_sl:.1%})",
            marker={"color": "red", "size": 12, "symbol": "diamond"},
            text=[f"{current_rate:.1f}%"],
            textposition="top center",
        )
    )

    fig.update_layout(
        title="SL Trigger Rate vs SL Level",
        xaxis_title="Stop-Loss Level (%)",
        yaxis_title="Trigger Rate (%)",
        height=400,
    )
    return fig


def plot_monthly_mae(mae_df: pd.DataFrame) -> go.Figure:
    """Monthly 95th percentile MAE time series."""
    monthly = mae_df.set_index("timestamp").resample("ME")

    fig = go.Figure()
    for col, name, _color in [
        ("mae_long", "Long P95", "#EF553B"),
        ("mae_short", "Short P95", "#636EFA"),
        ("mae_worst", "Worst P95", "#AB63FA"),
    ]:
        p95 = monthly[col].quantile(0.95) * 100
        fig.add_trace(
            go.Scatter(
                x=p95.index,
                y=p95.values,
                mode="lines+markers",
                name=name,
                line={"width": 2},
                marker={"size": 5},
            )
        )

    fig.update_layout(
        title="Monthly MAE 95th Percentile",
        xaxis_title="Month",
        yaxis_title="MAE (%)",
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="MAE Analysis", layout="wide")
    st.title("Max Adverse Excursion (MAE) Analysis")
    st.caption(
        "評估 BTC/USDT 在固定持倉時間內的最大逆向移動，判斷 stop-loss 設定是否合理"
    )

    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        return

    df = load_data(str(DATA_PATH))

    # --- Sidebar controls ---
    st.sidebar.header("Parameters")

    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if not isinstance(date_range, tuple) or len(date_range) != 2:
        st.sidebar.warning("Please select both start and end dates.")
        return

    holding_bars = st.sidebar.slider(
        "Holding bars (minutes)",
        min_value=5,
        max_value=60,
        value=DEFAULT_HOLDING_BARS,
        step=1,
    )
    sl_level = st.sidebar.number_input(
        "Stop-loss level",
        min_value=0.001,
        max_value=0.05,
        value=DEFAULT_SL,
        step=0.001,
        format="%.3f",
    )

    # --- Compute ---
    start_str = str(date_range[0])
    end_str = str(date_range[1])

    with st.spinner("Computing MAE..."):
        mae_df = compute_mae(df, holding_bars, start_str, end_str)

    st.success(f"Analyzed **{len(mae_df):,}** hypothetical entries")

    # --- 1. Histogram ---
    st.plotly_chart(
        plot_mae_histogram(mae_df, sl_level, holding_bars), use_container_width=True
    )

    # --- 2. Percentile table ---
    st.subheader("Percentile Table")
    percentiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ptable = pd.DataFrame(
        {
            "Percentile": [f"{p:.0%}" for p in percentiles],
            "Long MAE (%)": [
                f"{mae_df['mae_long'].quantile(p) * 100:.3f}" for p in percentiles
            ],
            "Short MAE (%)": [
                f"{mae_df['mae_short'].quantile(p) * 100:.3f}" for p in percentiles
            ],
            "Worst MAE (%)": [
                f"{mae_df['mae_worst'].quantile(p) * 100:.3f}" for p in percentiles
            ],
        }
    )
    st.dataframe(ptable, use_container_width=True, hide_index=True)

    trigger_rate = (mae_df["mae_worst"] >= sl_level).mean() * 100
    st.metric(
        label=f"SL Trigger Rate @ {sl_level:.1%}",
        value=f"{trigger_rate:.2f}%",
        help=(
            f"Percentage of entries where worst MAE"
            f" >= {sl_level:.1%} within {holding_bars} min"
        ),
    )

    # --- 3. SL trigger curve ---
    st.plotly_chart(
        plot_sl_trigger_curve(mae_df["mae_worst"], sl_level),
        use_container_width=True,
    )

    # --- 4. Monthly MAE ---
    st.plotly_chart(plot_monthly_mae(mae_df), use_container_width=True)


if __name__ == "__main__":
    main()
