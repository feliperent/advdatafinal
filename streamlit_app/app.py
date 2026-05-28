"""advdatafinal demo app: 5 blocks per the design.

Run: streamlit run streamlit_app/app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make repo root importable so the app can use ingest/rag modules
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingest.common import pg_conn  # noqa: E402
from rag.answer import answer  # noqa: E402

st.set_page_config(page_title="advdatafinal | IN014", layout="wide")

# ----------------------------- Data loaders (cached) ---------------------------

@st.cache_data(ttl=60)
def load_companies() -> pd.DataFrame:
    with pg_conn() as conn:
        return pd.read_sql(
            "SELECT symbol, name, sector_name FROM gold.dim_company ORDER BY symbol", conn
        )

@st.cache_data(ttl=60)
def load_dates_for(symbol: str) -> list[str]:
    with pg_conn() as conn:
        df = pd.read_sql(
            """
            SELECT DISTINCT d.full_date
            FROM gold.fct_predictions p
            JOIN gold.dim_date d ON d.date_key = p.date_key
            JOIN gold.dim_company c ON c.company_key = p.company_key
            WHERE c.symbol = %s
            ORDER BY d.full_date DESC
            """,
            conn, params=(symbol,),
        )
    return [d.isoformat() for d in df["full_date"]]

@st.cache_data(ttl=60)
def load_predictions(symbol: str, date_iso: str) -> pd.DataFrame:
    with pg_conn() as conn:
        return pd.read_sql(
            """
            SELECT p.model_rung, p.prob_up::float AS prob_up, p.predicted_class, p.shap_json
            FROM gold.fct_predictions p
            JOIN gold.dim_date d ON d.date_key = p.date_key
            JOIN gold.dim_company c ON c.company_key = p.company_key
            WHERE c.symbol = %s AND d.full_date = %s
            ORDER BY p.model_rung
            """,
            conn, params=(symbol, date_iso),
        )

@st.cache_data(ttl=60)
def load_pnl() -> pd.DataFrame:
    with pg_conn() as conn:
        return pd.read_sql(
            """
            SELECT trade_date, model_rung, cum_net_ret::float AS cum_net_ret,
                   benchmark_cum_ret_eqw::float AS benchmark
            FROM gold.fct_backtest_pnl_daily
            ORDER BY trade_date
            """,
            conn,
        )


st.title("Final Project Advanced Data Processing by FR, DP, MA and NA ")
st.caption("Direction prediction + RAG over SEC filings on 20 US stocks.")

companies = load_companies()
if companies.empty:
    st.warning("No predictions yet; run `make train` first.")
    st.stop()

# Block 1: header (stock + date pickers)
col_h1, col_h2 = st.columns([2, 2])
with col_h1:
    symbol = st.selectbox("Stock", companies["symbol"].tolist(), index=0)
with col_h2:
    dates = load_dates_for(symbol)
    if not dates:
        st.warning(f"No predictions for {symbol}.")
        st.stop()
    date_iso = st.selectbox("Date", dates, index=0)

stock_row = companies[companies["symbol"] == symbol].iloc[0]
st.markdown(f"**{symbol}**  -  {stock_row['name']} *({stock_row['sector_name']})*")

# Block 2: predictions side by side
preds = load_predictions(symbol, date_iso)
st.subheader("Block 2 | Predictions per rung")
if preds.empty:
    st.info(f"No predictions for {symbol} on {date_iso}.")
else:
    cols = st.columns(3)
    rung_labels = {0: "Rung 0 (ARIMA)", 1: "Rung 1 (XGB structured)", 2: "Rung 2 (XGB + text)"}
    for r in range(3):
        with cols[r]:
            sub = preds[preds["model_rung"] == r]
            if sub.empty:
                st.metric(rung_labels.get(r, f"Rung {r}"), " - ", "no prediction")
                continue
            p = float(sub.iloc[0]["prob_up"])
            cls = "UP" if int(sub.iloc[0]["predicted_class"]) == 1 else "DOWN"
            st.metric(rung_labels.get(r, f"Rung {r}"), f"P(up) = {p:.2f}", cls)

# Block 3: SHAP attribution for Rung 2
st.subheader("Block 3 | SHAP attributions (Rung 2)")
rung2 = preds[preds["model_rung"] == 2]
if rung2.empty or rung2.iloc[0]["shap_json"] is None:
    st.info("No SHAP attributions stored for this (date, stock). SHAP is logged for the first 200 test rows per fold.")
else:
    raw = rung2.iloc[0]["shap_json"]
    shap_dict = raw if isinstance(raw, dict) else json.loads(raw)
    items = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    items_df = pd.DataFrame(items, columns=["feature", "shap_value"])
    items_df["abs"] = items_df["shap_value"].abs()
    items_df = items_df.sort_values("shap_value")
    fig = go.Figure(
        go.Bar(
            x=items_df["shap_value"],
            y=items_df["feature"],
            orientation="h",
            marker_color=["#d62728" if v < 0 else "#2ca02c" for v in items_df["shap_value"]],
        )
    )
    fig.update_layout(template="plotly_white", height=400, xaxis_title="SHAP value", yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

# Block 4: RAG ask
st.subheader("Block 4 | Ask the model 'why?'")
default_q = f"What does {symbol} disclose that supports the prediction on {date_iso}?"
question = st.text_input("Question", value=default_q)
if st.button("Ask"):
    with st.spinner("Retrieving + generating (Claude Haiku or template fallback)..."):
        rung_for_answer = 2 if not rung2.empty else (1 if not preds[preds["model_rung"] == 1].empty else 0)
        paragraph, citations, usage = answer(
            question, symbol=symbol, as_of_date=date_iso, rung=rung_for_answer
        )
    st.markdown(paragraph)
    st.caption(f"Citations: {', '.join(citations[:6])}")
    st.caption(
        f"Source: {usage.get('source', 'unknown')} | "
        f"tokens in/out: {usage.get('tokens_in', 0)}/{usage.get('tokens_out', 0)} | "
        f"latency: {usage.get('latency_ms', 0)} ms"
    )

# Block 5: cumulative P&L
st.subheader("Block 5 | Walk-forward backtest")
pnl = load_pnl()
if pnl.empty:
    st.info("Backtest not yet computed; run `make backtest` first.")
else:
    fig = go.Figure()
    for r in sorted(pnl["model_rung"].unique()):
        sub = pnl[pnl["model_rung"] == r].sort_values("trade_date")
        fig.add_scatter(x=sub["trade_date"], y=sub["cum_net_ret"], name=f"Rung {r}", mode="lines")
    bench = (
        pnl.drop_duplicates(subset=["trade_date"])
        .dropna(subset=["benchmark"])
        .sort_values("trade_date")
    )
    if not bench.empty:
        fig.add_scatter(x=bench["trade_date"], y=bench["benchmark"], name="Equal-weight 20-stock", mode="lines", line=dict(dash="dot"))
    fig.update_layout(template="plotly_white", height=500, xaxis_title="Date", yaxis_title="Cumulative net return")
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Data: 20 US stocks, 5 sectors, 2021-01 → 2025-12. "
    "Schemas: raw, datos_masked, silver, gold. "
    "Vector store: numpy cosine over bytea-stored 384-dim MiniLM embeddings."
)
