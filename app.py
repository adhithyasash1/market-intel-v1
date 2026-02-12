"""
Market Intelligence & Sector Rotation Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A decision-ready Streamlit dashboard for sector rotation analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import logging
import html as html_mod
import textwrap

# ── Path setup ──
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import (
    WEIGHT_PRESETS, DEFAULT_PRESET, SIGNAL_COLORS,
    SECTOR_ETFS, BENCHMARK_ETF, DEFAULT_TRANSACTION_COST,
    DEFAULT_REBALANCE_FREQ, REPORTS_DIR,
)
from src.data_engine import load_or_fetch_snapshot, fetch_sector_etf_history
from src.features import compute_stock_features, compute_sector_aggregates
from src.scorer import score_pipeline, FEATURE_MAP
from src.backtest import run_backtest, bootstrap_test, sensitivity_analysis
from src.utils import format_pct, format_large_number

logger = logging.getLogger(__name__)

# Configure root logger so all modules' log messages are emitted
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def render_metric_card(label: str, value: str, delta: str = "", delta_type: str = "") -> str:
    """
    Generate HTML for a premium metric card.
    Consolidates repeated card markup into a single helper (DRY).
    HTML-escapes inputs to prevent XSS from untrusted data.
    """
    # Escape all dynamic content to prevent XSS
    label = html_mod.escape(str(label))
    value = html_mod.escape(str(value))
    delta = html_mod.escape(str(delta)) if delta else ""

    delta_html = ""
    if delta:
        delta_class = f"delta-{delta_type}" if delta_type else "delta"
        delta_html = f'<div class="delta {delta_class}">{delta}</div>'

    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>
    """


def _get_plotly_dark_layout(**overrides) -> dict:
    """Return common Plotly layout kwargs for dark theme. Reduces duplication."""
    base = dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc', family='Inter'),
        margin=dict(l=10, r=10, t=50, b=30),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
    )
    base.update(overrides)
    return base


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG & THEME
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Market Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium dark-theme CSS
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Header ── */
    .dashboard-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .dashboard-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #00f5d4, #00bbf9, #9b5de5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .dashboard-header p {
        color: rgba(255,255,255,0.6);
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
        font-weight: 400;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.35);
    }
    .metric-card .label {
        color: rgba(255,255,255,0.5);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-card .delta {
        font-size: 0.85rem;
        margin-top: 0.2rem;
        font-weight: 500;
    }
    .delta-positive { color: #00c853; }
    .delta-negative { color: #ef5350; }

    /* ── Signal Badges ── */
    .signal-overweight {
        background: linear-gradient(135deg, #00c853, #00e676);
        color: #000;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .signal-neutral {
        background: linear-gradient(135deg, #ff9800, #ffc107);
        color: #000;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .signal-avoid {
        background: linear-gradient(135deg, #ef5350, #ff1744);
        color: #fff;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
        display: inline-block;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e0e0e0;
        padding: 0.8rem 0 0.5rem 0;
        border-bottom: 2px solid rgba(155, 93, 229, 0.3);
        margin-bottom: 1rem;
        letter-spacing: -0.3px;
    }

    /* ── Table Styling ── */
    .dataframe-container {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Sidebar ── */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    st.markdown("---")

    # Weight Preset
    preset_name = st.selectbox(
        "📐 Scoring Preset",
        list(WEIGHT_PRESETS.keys()),
        index=list(WEIGHT_PRESETS.keys()).index(DEFAULT_PRESET),
        help="Choose how features are weighted in the composite score"
    )

    # Custom weights toggle
    use_custom = st.toggle("🎛️ Custom Weights", value=False)
    if use_custom:
        st.markdown("**Adjust Weights** (must sum to 1.0)")
        preset_w = WEIGHT_PRESETS[preset_name]
        w_mom = st.slider("Momentum",     0.0, 1.0, preset_w["momentum"],     0.05)
        w_brd = st.slider("Breadth",      0.0, 1.0, preset_w["breadth"],      0.05)
        w_vol = st.slider("Volatility",   0.0, 1.0, preset_w["volatility"],   0.05)
        w_liq = st.slider("Liquidity",    0.0, 1.0, preset_w["liquidity"],    0.05)
        w_acc = st.slider("Acceleration", 0.0, 1.0, preset_w["acceleration"], 0.05)
        total = w_mom + w_brd + w_vol + w_liq + w_acc
        if total < 1e-6:
            st.error("Weights cannot all be zero.")
            total = 1.0  # prevent division by zero
        elif abs(total - 1.0) > 0.01:
            st.warning(f"Weights sum to {total:.2f} — will be normalized")
        custom_weights = {
            "momentum": w_mom / total, "breadth": w_brd / total,
            "volatility": w_vol / total, "liquidity": w_liq / total,
            "acceleration": w_acc / total,
        }
    else:
        custom_weights = None

    st.markdown("---")

    # Backtest controls
    st.markdown("### 📈 Backtest Settings")
    rebalance_freq = st.selectbox(
        "Rebalance Frequency",
        ["Monthly", "Quarterly"],
        index=0,
    )
    transaction_cost = st.slider(
        "Transaction Cost (bps)",
        min_value=0, max_value=50, value=DEFAULT_TRANSACTION_COST, step=5,
        help="Cost per trade in basis points"
    )

    st.markdown("---")
    st.markdown("### 📋 Quick Actions")

    refresh_data = st.button("🔄 Refresh Screener Data", use_container_width=True)

    st.markdown("---")
    st.caption("Built with tvscreener + yfinance")
    st.caption("⚠️ Not financial advice")


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def load_data():
    """Load screener snapshot and compute features + scores."""
    snapshot = load_or_fetch_snapshot()
    stock_features = compute_stock_features(snapshot)
    sector_aggs = compute_sector_aggregates(stock_features)
    return snapshot, stock_features, sector_aggs


@st.cache_data(ttl=7200, show_spinner="Loading ETF history...")
def load_etf_data():
    """Load sector ETF historical prices for backtest."""
    return fetch_sector_etf_history()


# Handle refresh
if refresh_data:
    st.cache_data.clear()

try:
    snapshot, stock_features, sector_aggs = load_data()
except Exception as e:
    st.error(f"⚠️ Failed to load screener data: {str(e)}")
    st.info("This may be due to network issues or API rate limiting. Try again in a few moments.")
    st.stop()


# Score with selected weights
weights = custom_weights if custom_weights else None
scored_sectors = score_pipeline(sector_aggs, weights=weights, preset=preset_name)

if scored_sectors.empty:
    st.warning("No sectors scored — data may be insufficient.")
    st.stop()


# Load ETF data once, shared across Drilldown and Backtest tabs
etf_prices_cached = None
try:
    etf_prices_cached = load_etf_data()
except Exception as e:
    logger.warning("ETF data not available: %s", e)


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="dashboard-header">
    <h1>📊 Market Intelligence & Sector Rotation</h1>
    <p>Decision-ready composite scoring • Explainable signals • Backtested performance</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════
n_overweight = len(scored_sectors[scored_sectors['signal'] == 'Overweight'])
n_avoid = len(scored_sectors[scored_sectors['signal'] == 'Avoid'])
top_sector = textwrap.shorten(scored_sectors.index[0], width=18, placeholder="…")
top_score = scored_sectors['composite_score'].iloc[0]

cols = st.columns(5)
kpi_data = [
    ("SECTORS ANALYZED", str(len(scored_sectors)), "", ""),
    ("OVERWEIGHT", str(n_overweight), "", "positive"),
    ("AVOID", str(n_avoid), "", "negative"),
    ("TOP SECTOR", top_sector, "", ""),
    ("TOP SCORE", f"{top_score:.3f}", "", "positive" if top_score > 0 else "negative"),
]

for col, (label, value, delta, delta_type) in zip(cols, kpi_data):
    col.markdown(render_metric_card(label, value, delta, delta_type), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab_ranking, tab_drilldown, tab_backtest, tab_scenario, tab_export = st.tabs([
    "🏆 Sector Rankings", "🔍 Sector Drilldown", "📈 Backtest",
    "🎯 Scenario Analysis", "📥 Export & Reports"
])


# ══════════════════════════════════════════════════════════════════
# TAB 1: SECTOR RANKINGS
# ══════════════════════════════════════════════════════════════════
with tab_ranking:
    st.markdown('<div class="section-header">Composite Sector Scores & Rotation Signals</div>', unsafe_allow_html=True)

    # Prepare display table
    display_df = scored_sectors.copy()
    display_df.index.name = 'Sector'
    display_df = display_df.reset_index()

    display_cols = ['Sector', 'composite_score', 'signal', 'n_stocks',
                    'median_momentum', 'breadth', 'avg_volatility', 'liquidity_score',
                    'momentum_acceleration']
    available_cols = [c for c in display_cols if c in display_df.columns]
    table_df = display_df[available_cols].copy()

    rename_map = {
        'composite_score': 'Score',
        'signal': 'Signal',
        'n_stocks': '#Stocks',
        'median_momentum': 'Med. 1M Return %',
        'breadth': 'Breadth',
        'avg_volatility': 'Volatility',
        'liquidity_score': 'Liquidity',
        'momentum_acceleration': 'Momentum Accel.',
    }
    table_df = table_df.rename(columns=rename_map)

    # ── Bar chart: Composite Scores ──
    col_chart, col_table = st.columns([1, 1.2])

    with col_chart:
        colors = [SIGNAL_COLORS.get(s, "#888") for s in display_df['signal']]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=display_df['Sector'],
            x=display_df['composite_score'],
            orientation='h',
            marker_color=colors,
            text=display_df['signal'],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Score: %{x:.3f}<br>Signal: %{text}<extra></extra>',
        ))
        fig.update_layout(**_get_plotly_dark_layout(
            title=dict(text="Composite Score by Sector", font=dict(size=16, color='#e0e0e0')),
            xaxis_title="Composite Score",
            yaxis=dict(autorange="reversed"),
            height=max(400, len(display_df) * 35),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        def style_signal(val):
            if val == "Overweight":
                return 'background: linear-gradient(135deg, #00c853, #00e676); color: #000; font-weight: 700; border-radius: 12px; text-align: center;'
            elif val == "Avoid":
                return 'background: linear-gradient(135deg, #ef5350, #ff1744); color: #fff; font-weight: 700; border-radius: 12px; text-align: center;'
            return 'background: linear-gradient(135deg, #ff9800, #ffc107); color: #000; font-weight: 700; border-radius: 12px; text-align: center;'

        styled = table_df.style.format({
            'Score': '{:.3f}',
            'Med. 1M Return %': '{:.2f}',
            'Breadth': '{:.1%}',
            'Volatility': '{:.2f}',
            'Momentum Accel.': '{:.2f}',
        }, na_rep="—").map(style_signal, subset=['Signal'])

        st.dataframe(styled, use_container_width=True, height=max(400, len(table_df) * 35 + 40))

    # ── Signal Explanations ──
    st.markdown('<div class="section-header">🔎 Signal Explanations</div>', unsafe_allow_html=True)

    for idx, row in scored_sectors.iterrows():
        score_val = row['composite_score']

        with st.expander(f"**{idx}** — Score: {score_val:.3f}"):
            explanations = row.get('explanation', [])
            if isinstance(explanations, list) and explanations:
                for exp in explanations:
                    st.markdown(f"• {exp}")
            else:
                st.write("No explanation available")

            # Z-score breakdown
            z_cols = {
                'z_momentum': 'Momentum',
                'z_breadth': 'Breadth',
                'z_volatility': 'Volatility',
                'z_liquidity': 'Liquidity',
                'z_acceleration': 'Acceleration',
            }
            z_data = {v: row.get(k, 0) for k, v in z_cols.items()}
            fig_z = go.Figure(go.Bar(
                x=list(z_data.values()),
                y=list(z_data.keys()),
                orientation='h',
                marker_color=['#00c853' if v > 0 else '#ef5350' for v in z_data.values()],
            ))
            fig_z.update_layout(**_get_plotly_dark_layout(
                title="Z-Score Components",
                height=200,
                margin=dict(l=10, r=10, t=35, b=10),
                font=dict(color='#ccc', size=11),
            ))
            st.plotly_chart(fig_z, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2: SECTOR DRILLDOWN
# ══════════════════════════════════════════════════════════════════
with tab_drilldown:
    st.markdown('<div class="section-header">Sector Deep Dive</div>', unsafe_allow_html=True)

    sectors_list = scored_sectors.index.tolist()
    selected_sector = st.selectbox("Select a sector to drill down", sectors_list)

    if selected_sector and 'sector' in stock_features.columns:
        sector_stocks = stock_features[stock_features['sector'] == selected_sector].copy()

        if len(sector_stocks) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Constituents", len(sector_stocks))
            with col2:
                med_mom = sector_stocks['perf_1m'].median() if 'perf_1m' in sector_stocks.columns else 0
                st.metric("Median 1M Return", f"{med_mom:.1f}%")
            with col3:
                breadth_val = scored_sectors.loc[selected_sector, 'breadth'] if selected_sector in scored_sectors.index else 0
                st.metric("Breadth", f"{breadth_val:.0%}")
            with col4:
                signal = scored_sectors.loc[selected_sector, 'signal'] if selected_sector in scored_sectors.index else "N/A"
                st.metric("Signal", signal)

            # ── Top Movers ──
            top_col, bottom_col = st.columns(2)

            movers_rename = {
                'name': 'Name', 'price': 'Price',
                'perf_1m': '1M Return %', 'market_cap': 'Market Cap'
            }
            movers_cols = [c for c in ['name', 'price', 'perf_1m', 'market_cap'] if c in sector_stocks.columns]

            with top_col:
                st.markdown("**🟢 Top 5 Gainers**")
                if 'perf_1m' in sector_stocks.columns and movers_cols:
                    top5 = sector_stocks.nlargest(5, 'perf_1m')
                    st.dataframe(
                        top5[movers_cols].rename(columns=movers_rename),
                        use_container_width=True, hide_index=True,
                    )

            with bottom_col:
                st.markdown("**🔴 Bottom 5 Losers**")
                if 'perf_1m' in sector_stocks.columns and movers_cols:
                    bottom5 = sector_stocks.nsmallest(5, 'perf_1m')
                    st.dataframe(
                        bottom5[movers_cols].rename(columns=movers_rename),
                        use_container_width=True, hide_index=True,
                    )

            # ── Constituents Table ──
            st.markdown("**📋 All Constituents**")
            const_cols = [c for c in ['name', 'price', 'change_pct', 'perf_1m', 'perf_3m',
                         'rsi_14', 'market_cap', 'volume'] if c in sector_stocks.columns]
            if const_cols:
                sort_col = 'market_cap' if 'market_cap' in sector_stocks.columns else const_cols[0]
                st.dataframe(
                    sector_stocks[const_cols].sort_values(sort_col, ascending=False)
                    .rename(columns={
                        'name': 'Name', 'price': 'Price', 'change_pct': 'Chg %',
                        'perf_1m': '1M %', 'perf_3m': '3M %',
                        'rsi_14': 'RSI', 'market_cap': 'Mkt Cap', 'volume': 'Volume',
                    }),
                    use_container_width=True, hide_index=True, height=400,
                )

            # ── RSI Distribution ──
            if 'rsi_14' in sector_stocks.columns:
                rsi_data = sector_stocks.dropna(subset=['rsi_14'])
                if len(rsi_data) > 0:
                    fig_rsi = px.histogram(
                        rsi_data, x='rsi_14', nbins=20,
                        color_discrete_sequence=['#9b5de5'],
                        title=f"RSI Distribution — {selected_sector}",
                    )
                    fig_rsi.add_vline(x=30, line_dash="dash", line_color="#ef5350", annotation_text="Oversold")
                    fig_rsi.add_vline(x=70, line_dash="dash", line_color="#00c853", annotation_text="Overbought")
                    fig_rsi.update_layout(**_get_plotly_dark_layout(height=300))
                    st.plotly_chart(fig_rsi, use_container_width=True)

    # ── Cross-Sector Correlation Heatmap ──
    st.markdown('<div class="section-header">🔥 Cross-Sector Correlation</div>', unsafe_allow_html=True)
    st.caption("Based on sector ETF daily returns (last 1 year)")

    if etf_prices_cached is not None:
        try:
            etf_returns = etf_prices_cached.pct_change().dropna()
            recent_rets = etf_returns.tail(252)
            etf_cols = [c for c in SECTOR_ETFS if c in recent_rets.columns]

            if len(etf_cols) > 2:
                corr_matrix = recent_rets[etf_cols].corr()

                fig_corr = px.imshow(
                    corr_matrix.values,
                    x=etf_cols, y=etf_cols,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    text_auto='.2f',
                    aspect='auto',
                )
                fig_corr.update_layout(**_get_plotly_dark_layout(
                    title="Sector ETF Return Correlation (1Y)",
                    height=500,
                    margin=dict(l=10, r=10, t=50, b=10),
                ))
                st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render correlation heatmap: {e}")
    else:
        st.info("ETF data unavailable — cannot render correlation heatmap.")


# ══════════════════════════════════════════════════════════════════
# TAB 3: BACKTEST
# ══════════════════════════════════════════════════════════════════
with tab_backtest:
    st.markdown('<div class="section-header">📈 Strategy Backtest — Sector Tilt Simulation</div>', unsafe_allow_html=True)
    st.caption("Overweights top-2 scored sectors, underweights bottom-2, uses sector ETF daily returns")

    if etf_prices_cached is None:
        st.error("ETF price data unavailable. Check your network connection.")
    else:
        try:
            bt_weights = custom_weights if custom_weights else WEIGHT_PRESETS[preset_name]
            result = run_backtest(
                etf_prices_cached,
                weights=bt_weights,
                rebalance_freq=rebalance_freq,
                transaction_cost_bps=transaction_cost,
            )

            if result.metrics:
                m = result.metrics

                # ── Metrics Cards ──
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                alpha_val = m.get('alpha', 0)
                cards = [
                    (mc1, "ANN. RETURN", f"{m.get('annualized_return', 0):.1f}%",
                     f"α: {alpha_val:+.1f}% vs bench", "positive" if alpha_val > 0 else "negative"),
                    (mc2, "SHARPE RATIO", f"{m.get('sharpe_ratio', 0):.2f}",
                     f"Bench: {m.get('sharpe_ratio_bench', 0):.2f}", ""),
                    (mc3, "MAX DRAWDOWN", f"{m.get('max_drawdown', 0):.1f}%", "", ""),
                    (mc4, "HIT RATE", f"{m.get('hit_rate', 0):.0f}%", "of months beat bench", ""),
                    (mc5, "INFO RATIO", f"{m.get('information_ratio', 0):.3f}", "", ""),
                ]
                for col, label, value, delta, dt in cards:
                    col.markdown(render_metric_card(label, value, delta, dt), unsafe_allow_html=True)

                st.markdown("")

                # ── Cumulative Returns Chart ──
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(
                    x=result.cumulative_portfolio.index,
                    y=result.cumulative_portfolio.values,
                    name='Sector Tilt Strategy',
                    line=dict(color='#00bbf9', width=2.5),
                    fill='tozeroy',
                    fillcolor='rgba(0,187,249,0.08)',
                ))
                fig_bt.add_trace(go.Scatter(
                    x=result.cumulative_benchmark.index,
                    y=result.cumulative_benchmark.values,
                    name=f'Benchmark ({BENCHMARK_ETF})',
                    line=dict(color='#888', width=1.5, dash='dot'),
                ))
                fig_bt.update_layout(**_get_plotly_dark_layout(
                    title="Cumulative Returns: Strategy vs Benchmark",
                    yaxis_title="Growth of $1",
                    height=420,
                    legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.3)'),
                ))
                st.plotly_chart(fig_bt, use_container_width=True)

                # ── Bootstrap Test ──
                with st.expander("📊 Bootstrap Significance Test (1000 samples)"):
                    try:
                        bootstrap = bootstrap_test(
                            result.portfolio_returns,
                            result.benchmark_returns,
                            n_samples=1000,
                        )
                        if bootstrap:
                            for key, (mean_val, p5, p95) in bootstrap.items():
                                col_a, col_b = st.columns([1, 2])
                                with col_a:
                                    st.metric(key.title(), f"{mean_val:.2f}%")
                                with col_b:
                                    st.write(f"90% CI: [{p5:.2f}%, {p95:.2f}%]")
                                    significant = (p5 > 0) if key == 'alpha' else True
                                    if significant:
                                        st.success("✅ Statistically significant at 90% level")
                                    else:
                                        st.warning("⚠️ Not significant — CI includes zero")
                    except Exception as e:
                        st.warning(f"Bootstrap test failed: {e}")

                # ── Sensitivity Analysis ──
                with st.expander("🔬 Sensitivity Analysis (Presets × Rebalance Freq)"):
                    try:
                        sens_df = sensitivity_analysis(etf_prices_cached)
                        if len(sens_df) > 0:
                            st.dataframe(
                                sens_df.style.format({
                                    'annualized_return': '{:.1f}%',
                                    'sharpe_ratio': '{:.3f}',
                                    'max_drawdown': '{:.1f}%',
                                    'hit_rate': '{:.0f}%',
                                    'alpha': '{:.2f}%',
                                    'information_ratio': '{:.3f}',
                                }),
                                use_container_width=True, hide_index=True,
                            )
                    except Exception as e:
                        st.warning(f"Sensitivity analysis failed: {e}")

            else:
                st.warning("Insufficient data for backtest. Need at least 60 trading days of ETF history.")

        except Exception as e:
            st.error(f"Backtest error: {e}")
            st.info("Ensure yfinance can download ETF data. Check your network connection.")


# ══════════════════════════════════════════════════════════════════
# TAB 4: SCENARIO ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab_scenario:
    st.markdown('<div class="section-header">🎯 What-If Scenario Analysis</div>', unsafe_allow_html=True)
    st.caption("Simulate score changes by adjusting sector features")

    scenario_col1, scenario_col2 = st.columns([1, 2])

    with scenario_col1:
        st.markdown("**Adjust Sector Features**")

        scenario_sector = st.selectbox("Sector to stress", sectors_list, key="scenario_sector")

        vol_shock = st.slider(
            "Volatility Shock (%)",
            min_value=-50, max_value=200, value=0, step=10,
            help="Increase volatility by this percentage"
        )
        mom_shock = st.slider(
            "Momentum Shift (pp)",
            min_value=-20, max_value=20, value=0, step=1,
            help="Add/subtract percentage points to momentum"
        )
        breadth_shock = st.slider(
            "Breadth Shift (pp)",
            min_value=-30, max_value=30, value=0, step=5,
            help="Add/subtract percentage points to breadth"
        )

    with scenario_col2:
        # Apply scenario
        scenario_aggs = sector_aggs.copy()
        if scenario_sector in scenario_aggs.index:
            if vol_shock != 0 and 'avg_volatility' in scenario_aggs.columns:
                scenario_aggs.loc[scenario_sector, 'avg_volatility'] *= (1 + vol_shock / 100)
            if mom_shock != 0 and 'median_momentum' in scenario_aggs.columns:
                scenario_aggs.loc[scenario_sector, 'median_momentum'] += mom_shock
            if breadth_shock != 0 and 'breadth' in scenario_aggs.columns:
                scenario_aggs.loc[scenario_sector, 'breadth'] = max(0, min(1,
                    scenario_aggs.loc[scenario_sector, 'breadth'] + breadth_shock / 100))

        scenario_scored = score_pipeline(scenario_aggs, weights=weights, preset=preset_name)

        # Compare original vs scenario
        compare_df = pd.DataFrame({
            'Original Score': scored_sectors['composite_score'],
            'Scenario Score': scenario_scored['composite_score'],
            'Original Signal': scored_sectors['signal'],
            'Scenario Signal': scenario_scored['signal'],
        })
        compare_df['Score Change'] = compare_df['Scenario Score'] - compare_df['Original Score']
        compare_df = compare_df.sort_values('Score Change', ascending=False)

        # Highlight changes
        changed = compare_df[compare_df['Original Signal'] != compare_df['Scenario Signal']]

        if len(changed) > 0:
            st.markdown("**⚡ Signal Changes:**")
            for sector, row in changed.iterrows():
                st.markdown(
                    f"- **{sector}**: {row['Original Signal']} → {row['Scenario Signal']} "
                    f"(Δ score: {row['Score Change']:+.3f})"
                )
        else:
            st.info("No signal changes under this scenario.")

        st.markdown("")

        # Score change chart
        fig_scenario = go.Figure()
        fig_scenario.add_trace(go.Bar(
            y=compare_df.index,
            x=compare_df['Score Change'],
            orientation='h',
            marker_color=['#00c853' if v > 0 else '#ef5350' for v in compare_df['Score Change']],
        ))
        fig_scenario.update_layout(**_get_plotly_dark_layout(
            title="Score Change Under Scenario",
            xaxis_title="Δ Composite Score",
            yaxis=dict(autorange="reversed"),
            height=max(300, len(compare_df) * 30),
        ))
        st.plotly_chart(fig_scenario, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 5: EXPORT & REPORTS
# ══════════════════════════════════════════════════════════════════
with tab_export:
    st.markdown('<div class="section-header">📥 Export Data & Generate Reports</div>', unsafe_allow_html=True)

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        st.markdown("### 📊 Download Rankings CSV")
        export_df = scored_sectors.drop(columns=['explanation'], errors='ignore').reset_index()
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Sector Rankings",
            data=csv_data,
            file_name="sector_rankings.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("### 📋 Download Full Snapshot CSV")
        snap_csv = stock_features.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Stock Snapshot",
            data=snap_csv,
            file_name="stock_snapshot.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with export_col2:
        st.markdown("### 📝 Auto-Generated Recommendation Memo")

        # Generate memo text
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        overweight_sectors = scored_sectors[scored_sectors['signal'] == 'Overweight']
        avoid_sectors = scored_sectors[scored_sectors['signal'] == 'Avoid']

        memo = f"""# Market Intelligence — Sector Rotation Memo
**Date:** {today}  |  **Preset:** {preset_name}

---

## Recommended Actions

### ✅ OVERWEIGHT (Rotate Into)
"""
        for sector in overweight_sectors.index:
            score = overweight_sectors.loc[sector, 'composite_score']
            explanations = overweight_sectors.loc[sector].get('explanation', [])
            memo += f"\n**{sector}** (Score: {score:.3f})\n"
            if isinstance(explanations, list):
                for e in explanations[:2]:
                    memo += f"  - {e}\n"

        memo += "\n### ❌ AVOID (Reduce Exposure)\n"
        for sector in avoid_sectors.index:
            score = avoid_sectors.loc[sector, 'composite_score']
            explanations = avoid_sectors.loc[sector].get('explanation', [])
            memo += f"\n**{sector}** (Score: {score:.3f})\n"
            if isinstance(explanations, list):
                for e in explanations[:2]:
                    memo += f"  - {e}\n"

        memo += f"""
---

## Scoring Methodology
- **Preset:** {preset_name}
- **Features:** Momentum, Breadth, Volatility (inv.), Liquidity, Acceleration
- **Thresholds:** Top 20% → Overweight, Bottom 20% → Avoid

⚠️ *This is a quantitative signal and should be combined with fundamental analysis.
Not financial advice.*
"""
        st.text_area("Memo Preview", memo, height=400)
        st.download_button(
            label="⬇️ Download Memo (.md)",
            data=memo,
            file_name=f"sector_memo_{today}.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: rgba(255,255,255,0.3); font-size: 0.8rem;'>"
    "Market Intelligence Dashboard • Data: TradingView Screener + Yahoo Finance • "
    "⚠️ Quantitative signals only — not financial advice"
    "</div>",
    unsafe_allow_html=True,
)
