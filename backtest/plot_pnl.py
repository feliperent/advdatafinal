"""Plot cumulative net return per rung vs equal-weight benchmark."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from ingest.common import pg_conn

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    with pg_conn() as conn:
        df = pd.read_sql(
            """
            SELECT trade_date, model_rung, cum_net_ret::float AS cum_net_ret,
                   benchmark_cum_ret_eqw::float AS benchmark
            FROM gold.fct_backtest_pnl_daily
            ORDER BY trade_date
            """,
            conn,
        )
    if df.empty:
        print("No PNL data; run backtest first.")
        return

    fig = go.Figure()
    for rung in sorted(df["model_rung"].unique()):
        sub = df[df["model_rung"] == rung].sort_values("trade_date")
        fig.add_scatter(
            x=sub["trade_date"], y=sub["cum_net_ret"],
            name=f"Rung {rung}", mode="lines",
        )
    bench = df.drop_duplicates(subset=["trade_date"]).dropna(subset=["benchmark"]).sort_values("trade_date")
    if not bench.empty:
        fig.add_scatter(
            x=bench["trade_date"], y=bench["benchmark"],
            name="Equal-weight 20-stock", mode="lines",
            line=dict(dash="dot"),
        )
    fig.update_layout(
        title="Cumulative net return per rung (walk-forward, top-5 long, 5bp cost)",
        xaxis_title="Date", yaxis_title="Cumulative net return",
        template="plotly_white", height=520,
    )
    out = FIG_DIR / "backtest_pnl.png"
    try:
        fig.write_image(str(out), width=900, height=520)
        print(f"Wrote {out}")
    except Exception as e:
        # Fallback: write HTML if kaleido isn't available
        out_html = FIG_DIR / "backtest_pnl.html"
        fig.write_html(str(out_html))
        print(f"PNG export failed ({e}); wrote HTML instead at {out_html}")


if __name__ == "__main__":
    main()
