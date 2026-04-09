"""
Stock Market Dashboard
Streamlit app with live chart analysis, multi-stock comparison,
merged Kaggle datasets, technical indicators.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys, os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockPeers Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

:root {
    --bg:      #080c14;
    --card:    #0d1320;
    --border:  #1a2235;
    --accent:  #00d4aa;
    --red:     #ff4560;
    --yellow:  #f5a623;
    --blue:    #4facfe;
    --text:    #e2e8f0;
    --muted:   #64748b;
}

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: var(--bg); }
.block-container { padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1600px !important; }

/* Sidebar */
div[data-testid="stSidebar"] {
    background: #060a10 !important;
    border-right: 1px solid var(--border) !important;
}
div[data-testid="stSidebar"] * { color: var(--text) !important; }
div[data-testid="stSidebar"] .stSelectbox label,
div[data-testid="stSidebar"] .stMultiSelect label { color: var(--muted) !important; font-size: 11px !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }

/* Cards */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent);
}
.metric-card.red::before  { background: var(--red); }
.metric-card.blue::before { background: var(--blue); }
.metric-card.yellow::before { background: var(--yellow); }

.metric-label { font-size: 10px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 500; color: var(--text); }
.metric-sub   { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: var(--muted); margin-top: 4px; }
.metric-up    { color: var(--accent) !important; }
.metric-down  { color: var(--red) !important; }

/* Section titles */
.section-hdr {
    font-size: 11px; font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px; margin-bottom: 16px;
}

/* Ticker pill */
.ticker-pill {
    display: inline-block;
    background: rgba(0,212,170,0.1);
    border: 1px solid rgba(0,212,170,0.3);
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--accent);
    margin: 2px;
}
.ticker-pill.red { background: rgba(255,69,96,0.1); border-color: rgba(255,69,96,0.3); color: var(--red); }

/* Table */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 10px !important; }

/* Plotly chart border */
.js-plotly-plot { border-radius: 12px !important; }

/* Multiselect tags */
.stMultiSelect span[data-baseweb="tag"] {
    background: rgba(0,212,170,0.15) !important;
    border: 1px solid rgba(0,212,170,0.4) !important;
    color: var(--accent) !important;
}

h1, h2, h3 { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# ── Load / prepare data ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    csv = Path("data/merged_stocks.csv")
    if not csv.exists():
        # Run prepare_data inline
        import subprocess
        subprocess.run([sys.executable, "prepare_data.py"], check=True)
    df = pd.read_csv(csv, parse_dates=["Date"])
    return df

with st.spinner("Loading market data…"):
    df = load_data()

ALL_TICKERS = sorted(df["Ticker"].unique())
SECTORS     = df.groupby("Ticker")["Sector"].first().to_dict()
COMPANIES   = df.groupby("Ticker")["Company"].first().to_dict()
SOURCES     = df.groupby("Ticker")["Source"].first().to_dict()

COLORS = {
    "AAPL":"#00d4aa","MSFT":"#4facfe","GOOGL":"#f5a623","AMZN":"#ff6b6b",
    "NVDA":"#a78bfa","META":"#fb923c","JPM":"#34d399","BAC":"#60a5fa",
    "GS":"#f472b6","XOM":"#fbbf24","CVX":"#86efac","TSLA":"#ff4560",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 StockPeers")
    st.markdown("---")

    selected = st.multiselect(
        "Select Stocks",
        ALL_TICKERS,
        default=["AAPL","MSFT","TSLA","NVDA"],
    )
    if not selected:
        selected = ["AAPL"]

    st.markdown("### Time Range")
    date_min = df["Date"].min().date()
    date_max = df["Date"].max().date()
    date_from, date_to = st.date_input(
        "Period",
        value=[pd.Timestamp("2021-01-01").date(), date_max],
        min_value=date_min, max_value=date_max,
    )

    st.markdown("### Chart Type")
    chart_type = st.selectbox("Price Chart", ["Candlestick", "Line", "Area"])

    st.markdown("### Indicators")
    show_ma20  = st.checkbox("MA 20",         value=True)
    show_ma50  = st.checkbox("MA 50",         value=True)
    show_vol   = st.checkbox("Volume",        value=True)
    show_ret   = st.checkbox("Cumulative Return", value=False)

    st.markdown("---")
    st.markdown("### Data Sources")
    for src in df["Source"].unique():
        n = df[df["Source"]==src]["Ticker"].nunique()
        st.markdown(f"📦 `{src.replace('Kaggle_','').replace('_Dataset','')}`  \n*{n} stocks*")

# ── Filter data ───────────────────────────────────────────────────────────────
mask = (
    df["Ticker"].isin(selected) &
    (df["Date"] >= pd.Timestamp(date_from)) &
    (df["Date"] <= pd.Timestamp(date_to))
)
fdf = df[mask].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;">
    <div>
        <div style="font-size:11px;letter-spacing:3px;color:#64748b;text-transform:uppercase;margin-bottom:4px;">Real-Time Stock Analysis</div>
        <div style="font-size:2.2rem;font-weight:800;color:#e2e8f0;letter-spacing:-1px;">Market Dashboard</div>
    </div>
    <div style="text-align:right;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#64748b;">MERGED DATASET</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#00d4aa;">12 Stocks · 2 Sources</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
# FIXED VERSION (only changed prev calculation)

# Replace ONLY this section in your code:

# ── KPI Row ───────────────────────────────────────────────────────────────────
latest = fdf.groupby("Ticker").last().reset_index()

# ✅ FIXED LINE (no KeyError now)
prev = fdf.sort_values("Date").groupby("Ticker").nth(-2).reset_index()

kpi_cols = st.columns(min(len(selected), 6))
for i, ticker in enumerate(selected[:6]):
    row = latest[latest["Ticker"]==ticker]
    if row.empty: continue

    close = row["Close"].values[0]

    # safer handling if prev row missing
    prev_row = prev[prev["Ticker"]==ticker]
    prev_close = prev_row["Close"].values[0] if not prev_row.empty else close

    chg = close - prev_close
    pct = chg / prev_close * 100 if prev_close != 0 else 0
    up  = chg >= 0

    with kpi_cols[i]:
        st.markdown(f"""
        <div class="metric-card {'green' if up else 'red'}">
            <div class="metric-label">{COMPANIES.get(ticker, ticker)[:18]}</div>
            <div class="metric-value">${close:.2f}</div>
            <div class="metric-sub {'metric-up' if up else 'metric-down'}">
                {'▲' if up else '▼'} {abs(chg):.2f} ({abs(pct):.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

# END FIX


# ── Main price chart ──────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">📊 Price Analysis</div>', unsafe_allow_html=True)

primary = selected[0]
pdf = fdf[fdf["Ticker"]==primary].sort_values("Date")

rows_needed = 2 if show_vol else 1
row_heights = [0.7, 0.3] if show_vol else [1.0]
fig = make_subplots(rows=rows_needed, cols=1, shared_xaxes=True,
                    vertical_spacing=0.04, row_heights=row_heights)

BG   = "rgba(0,0,0,0)"
GRID = "rgba(26,34,53,0.8)"

if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=pdf["Date"], open=pdf["Open"], high=pdf["High"],
        low=pdf["Low"], close=pdf["Close"], name=primary,
        increasing_line_color="#00d4aa", decreasing_line_color="#ff4560",
        increasing_fillcolor="#00d4aa", decreasing_fillcolor="#ff4560",
    ), row=1, col=1)
elif chart_type == "Line":
    fig.add_trace(go.Scatter(
        x=pdf["Date"], y=pdf["Close"], name=primary,
        line=dict(color=COLORS.get(primary,"#00d4aa"), width=2),
        mode="lines",
    ), row=1, col=1)
else:  # Area
    fig.add_trace(go.Scatter(
        x=pdf["Date"], y=pdf["Close"], name=primary,
        line=dict(color=COLORS.get(primary,"#00d4aa"), width=2),
        fill="tozeroy", fillcolor=f"rgba(0,212,170,0.08)",
        mode="lines",
    ), row=1, col=1)

if show_ma20 and "MA_20" in pdf.columns:
    fig.add_trace(go.Scatter(x=pdf["Date"], y=pdf["MA_20"], name="MA20",
        line=dict(color="#f5a623", width=1.2, dash="dot"), opacity=0.8), row=1, col=1)

if show_ma50 and "MA_50" in pdf.columns:
    fig.add_trace(go.Scatter(x=pdf["Date"], y=pdf["MA_50"], name="MA50",
        line=dict(color="#a78bfa", width=1.2, dash="dash"), opacity=0.8), row=1, col=1)

if show_vol:
    colors_vol = ["#00d4aa" if c >= o else "#ff4560"
                  for c, o in zip(pdf["Close"], pdf["Open"])]
    fig.add_trace(go.Bar(x=pdf["Date"], y=pdf["Volume"], name="Volume",
        marker_color=colors_vol, opacity=0.6, showlegend=False), row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1,
                     title_font=dict(size=10, color="#64748b"))

fig.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(color="#e2e8f0", family="Syne"),
    height=520,
    margin=dict(l=10, r=10, t=20, b=10),
    xaxis_rangeslider_visible=False,
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hovermode="x unified",
)
fig.update_xaxes(gridcolor=GRID, showgrid=True, zeroline=False)
fig.update_yaxes(gridcolor=GRID, showgrid=True, zeroline=False)
st.plotly_chart(fig, use_container_width=True)

# ── Multi-stock comparison ────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">🔀 Multi-Stock Comparison</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📈 Normalized Performance", "📊 Correlation", "🏆 Returns Heatmap", "📋 Data Table"])

with tab1:
    fig2 = go.Figure()
    for ticker in selected:
        tdf = fdf[fdf["Ticker"]==ticker].sort_values("Date")
        if tdf.empty: continue
        base  = tdf["Close"].iloc[0]
        norm  = (tdf["Close"] / base - 1) * 100
        fig2.add_trace(go.Scatter(
            x=tdf["Date"], y=norm, name=ticker,
            line=dict(color=COLORS.get(ticker,"#888"), width=2),
            mode="lines",
            hovertemplate=f"<b>{ticker}</b><br>Return: %{{y:.2f}}%<extra></extra>",
        ))
    fig2.add_hline(y=0, line_dash="dot", line_color="#64748b", opacity=0.5)
    fig2.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color="#e2e8f0", family="Syne"),
        height=420, margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="Return %", hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig2.update_xaxes(gridcolor=GRID)
    fig2.update_yaxes(gridcolor=GRID)
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    pivot = fdf.pivot_table(index="Date", columns="Ticker", values="Close")
    corr  = pivot.corr().round(3)
    fig3  = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0,"#ff4560"],[0.5,"#1a2235"],[1,"#00d4aa"]],
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=11, family="IBM Plex Mono"),
        hovertemplate="<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))
    fig3.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color="#e2e8f0", family="Syne"),
        height=420, margin=dict(l=10, r=10, t=20, b=10),
        title=dict(text="Price Correlation Matrix", font=dict(size=13), x=0.01),
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    # Monthly returns heatmap for primary stock
    hdf = fdf[fdf["Ticker"]==primary].copy()
    hdf["Month"] = hdf["Date"].dt.to_period("M").astype(str)
    monthly = hdf.groupby("Month")["Close"].last().pct_change() * 100
    monthly = monthly.dropna().reset_index()
    monthly.columns = ["Month","Return"]
    monthly["Year"]  = monthly["Month"].str[:4]
    monthly["Mon"]   = monthly["Month"].str[5:]

    pivot_m = monthly.pivot_table(index="Year", columns="Mon", values="Return")
    fig4 = go.Figure(go.Heatmap(
        z=pivot_m.values, x=pivot_m.columns, y=pivot_m.index,
        colorscale=[[0,"#ff4560"],[0.5,"#1a2235"],[1,"#00d4aa"]],
        text=pivot_m.values.round(1),
        texttemplate="%{text}%",
        textfont=dict(size=10, family="IBM Plex Mono"),
        hovertemplate="<b>%{y}-%{x}</b><br>Return: %{z:.2f}%<extra></extra>",
    ))
    fig4.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color="#e2e8f0", family="Syne"),
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text=f"Monthly Returns — {primary}", font=dict(size=13), x=0.01),
    )
    st.plotly_chart(fig4, use_container_width=True)

with tab4:
    display_cols = ["Date","Ticker","Company","Sector","Source","Open","High","Low","Close","Volume","Daily_Return"]
    show_df = fdf[display_cols].copy()
    show_df["Date"] = show_df["Date"].dt.strftime("%Y-%m-%d")
    show_df["Daily_Return"] = (show_df["Daily_Return"] * 100).round(3).astype(str) + "%"
    st.dataframe(show_df.tail(200), use_container_width=True, height=380)

# ── Sector breakdown + Volume ─────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="section-hdr">🏭 Sector Performance</div>', unsafe_allow_html=True)
    sector_perf = []
    for ticker in ALL_TICKERS:
        tdf = df[df["Ticker"]==ticker].sort_values("Date")
        if len(tdf) < 2: continue
        ret = (tdf["Close"].iloc[-1] / tdf["Close"].iloc[0] - 1) * 100
        sector_perf.append({"Ticker": ticker, "Sector": SECTORS[ticker], "Return": round(ret,2)})
    spdf = pd.DataFrame(sector_perf)
    fig5 = px.bar(spdf, x="Ticker", y="Return", color="Sector",
                  color_discrete_sequence=["#00d4aa","#4facfe","#f5a623","#a78bfa"],
                  template="none")
    fig5.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color="#e2e8f0", family="Syne"),
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor=GRID),
        yaxis=dict(gridcolor=GRID, title="Total Return %"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
    )
    fig5.add_hline(y=0, line_color="#64748b", line_dash="dot", opacity=0.5)
    st.plotly_chart(fig5, use_container_width=True)

with col_b:
    st.markdown('<div class="section-hdr">📊 Average Daily Volume</div>', unsafe_allow_html=True)
    vol_df = fdf.groupby("Ticker")["Volume"].mean().reset_index()
    vol_df.columns = ["Ticker","Avg_Volume"]
    vol_df = vol_df.sort_values("Avg_Volume", ascending=True)
    fig6 = go.Figure(go.Bar(
        x=vol_df["Avg_Volume"], y=vol_df["Ticker"],
        orientation="h",
        marker=dict(
            color=[COLORS.get(t,"#888") for t in vol_df["Ticker"]],
            opacity=0.85,
        ),
        hovertemplate="<b>%{y}</b><br>Avg Volume: %{x:,.0f}<extra></extra>",
    ))
    fig6.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color="#e2e8f0", family="Syne"),
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor=GRID, title="Avg Daily Volume"),
        yaxis=dict(gridcolor=GRID),
    )
    st.plotly_chart(fig6, use_container_width=True)

# ── Dataset info ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-hdr">📦 Dataset Information</div>', unsafe_allow_html=True)

dc1, dc2, dc3, dc4 = st.columns(4)
with dc1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Total Records</div>
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-sub">Merged dataset</div>
    </div>""", unsafe_allow_html=True)
with dc2:
    st.markdown(f"""<div class="metric-card blue">
        <div class="metric-label">Stocks Tracked</div>
        <div class="metric-value">{df['Ticker'].nunique()}</div>
        <div class="metric-sub">Across 2 sources</div>
    </div>""", unsafe_allow_html=True)
with dc3:
    st.markdown(f"""<div class="metric-card yellow">
        <div class="metric-label">Date Range</div>
        <div class="metric-value" style="font-size:1rem;">2020–2023</div>
        <div class="metric-sub">{df['Date'].nunique()} trading days</div>
    </div>""", unsafe_allow_html=True)
with dc4:
    st.markdown(f"""<div class="metric-card red">
        <div class="metric-label">Data Sources</div>
        <div class="metric-value">2</div>
        <div class="metric-sub">Kaggle datasets merged</div>
    </div>""", unsafe_allow_html=True)

# Source breakdown table
src_summary = df.groupby("Source").agg(
    Stocks=("Ticker","nunique"),
    Records=("Date","count"),
    Tickers=("Ticker", lambda x: ", ".join(sorted(x.unique())))
).reset_index()
src_summary.columns = ["Source","Stocks","Records","Tickers"]
st.dataframe(src_summary, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:32px 0 8px;color:#334155;font-size:11px;letter-spacing:2px;">
    STOCK MARKET DASHBOARD · MERGED KAGGLE DATASETS · BUILT WITH STREAMLIT + PLOTLY
</div>
""", unsafe_allow_html=True)
