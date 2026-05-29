"""Generate BI charts for the final reporting pipeline.

Saves PNG figures into backtest/figures/ using matplotlib so the report can
embed them as static images. The Streamlit app produces its own interactive
Plotly versions of the same data.

Five charts:
  1. cumulative compound return per rung vs equal-weighted benchmark
  2. latest top-5 long picks per rung (bar chart of prob_up by symbol)
  3. per-sector average prob_up for the latest trade date
  4. distribution of prob_up across the universe for the latest date
  5. per-fold AUC line chart, Rung 1 vs Rung 2
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

from ingest.common import pg_conn

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "backtest" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {0: "#7f7f7f", 1: "#1f77b4", 2: "#d62728", "bench": "#2ca02c"}
RUNG_LABEL = {0: "Rung 0 (ARIMA)", 1: "Rung 1 (structured)", 2: "Rung 2 (structured + text + sector)"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 130,
})


def _save(name: str) -> Path:
    out = FIG_DIR / f"{name}.png"
    plt.savefig(out, bbox_inches="tight", dpi=160)
    plt.close()
    print(f"  wrote {out.name}")
    return out


# 1. cumulative compound return per rung -----------------------------------

def chart_cumulative_return() -> None:
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
        print("  [skip] cumulative_return: no backtest data")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for rung in sorted(df["model_rung"].unique()):
        sub = df[df["model_rung"] == rung].sort_values("trade_date")
        ax.plot(sub["trade_date"], sub["cum_net_ret"] * 100,
                label=RUNG_LABEL.get(int(rung), f"Rung {rung}"),
                color=PALETTE.get(int(rung), "#000"), linewidth=2)
    bench = df.drop_duplicates(subset=["trade_date"]).dropna(subset=["benchmark"]).sort_values("trade_date")
    if not bench.empty:
        ax.plot(bench["trade_date"], bench["benchmark"] * 100,
                label="Equal-weight 20-stock benchmark",
                color=PALETTE["bench"], linestyle="--", linewidth=1.5)
    ax.set_title("Cumulative compound return per rung (walk-forward, top-5 long, 5 bp turnover cost)")
    ax.set_xlabel("Trade date")
    ax.set_ylabel("Cumulative compound return (%)")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    _save("backtest_cumulative_return")


# 2. latest top-5 picks per rung -------------------------------------------

def chart_latest_top5() -> None:
    with pg_conn() as conn:
        df = pd.read_sql(
            """
            WITH latest AS (
              SELECT model_rung, MAX(d.full_date) AS latest_date
              FROM gold.fct_predictions p
              JOIN gold.dim_date d ON d.date_key = p.date_key
              GROUP BY model_rung
            )
            SELECT p.model_rung, d.full_date AS trade_date,
                   c.symbol, c.sector_name, p.prob_up::float AS prob_up
            FROM gold.fct_predictions p
            JOIN gold.dim_date d ON d.date_key = p.date_key
            JOIN gold.dim_company c ON c.company_key = p.company_key
            JOIN latest l ON l.model_rung = p.model_rung AND l.latest_date = d.full_date
            ORDER BY p.model_rung, p.prob_up DESC
            """,
            conn,
        )
    if df.empty:
        print("  [skip] latest_top5: no predictions")
        return
    rungs = sorted(df["model_rung"].unique())
    fig, axes = plt.subplots(1, len(rungs), figsize=(4 + 3 * len(rungs), 5), sharey=False)
    if len(rungs) == 1:
        axes = [axes]
    for ax, rung in zip(axes, rungs):
        sub = df[df["model_rung"] == rung].nlargest(5, "prob_up").sort_values("prob_up")
        ax.barh(sub["symbol"], sub["prob_up"], color=PALETTE.get(int(rung), "#000"))
        for y, p in zip(sub["symbol"], sub["prob_up"]):
            ax.text(p + 0.005, y, f"{p:.3f}", va="center", fontsize=9)
        ax.set_title(RUNG_LABEL.get(int(rung), f"Rung {rung}"))
        ax.set_xlim(0, max(0.7, sub["prob_up"].max() + 0.1))
        ax.set_xlabel("prob_up")
    fig.suptitle("Top-5 long picks per rung (latest trade date)", y=1.02)
    _save("latest_top5_picks")


# 3. per-sector forecast ----------------------------------------------------

def chart_sector_forecast() -> None:
    with pg_conn() as conn:
        df = pd.read_sql(
            """
            WITH latest AS (
              SELECT MAX(d.full_date) AS latest_date
              FROM gold.fct_predictions p
              JOIN gold.dim_date d ON d.date_key = p.date_key
              WHERE p.model_rung = 2
            )
            SELECT c.sector_name,
                   AVG(p.prob_up::float) AS avg_prob,
                   COUNT(*)::int AS n
            FROM gold.fct_predictions p
            JOIN gold.dim_date d   ON d.date_key   = p.date_key
            JOIN gold.dim_company c ON c.company_key = p.company_key
            JOIN latest l          ON l.latest_date = d.full_date
            WHERE p.model_rung = 2
            GROUP BY c.sector_name
            ORDER BY avg_prob DESC
            """,
            conn,
        )
    if df.empty:
        print("  [skip] sector_forecast: no predictions")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["sector_name"], df["avg_prob"], color=PALETTE[2])
    for x, p in zip(df["sector_name"], df["avg_prob"]):
        ax.text(x, p + 0.01, f"{p:.3f}", ha="center", fontsize=9)
    ax.set_title("Mean Rung 2 probability of 5-day up move by sector (latest date)")
    ax.set_xlabel("Sector")
    ax.set_ylabel("Mean predicted prob_up")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="#aaa", linestyle=":")
    _save("sector_forecast")


# 4. probability distribution across universe ------------------------------

def chart_probability_distribution() -> None:
    with pg_conn() as conn:
        df = pd.read_sql(
            """
            WITH latest AS (
              SELECT MAX(d.full_date) AS latest_date
              FROM gold.fct_predictions p
              JOIN gold.dim_date d ON d.date_key = p.date_key
              WHERE p.model_rung = 2
            )
            SELECT c.symbol, p.prob_up::float AS prob_up
            FROM gold.fct_predictions p
            JOIN gold.dim_date d   ON d.date_key   = p.date_key
            JOIN gold.dim_company c ON c.company_key = p.company_key
            JOIN latest l          ON l.latest_date = d.full_date
            WHERE p.model_rung = 2
            ORDER BY p.prob_up DESC
            """,
            conn,
        )
    if df.empty:
        print("  [skip] probability_distribution: no predictions")
        return
    df = df.reset_index(drop=True)
    cutoff = df["prob_up"].iloc[min(4, len(df) - 1)] if len(df) >= 5 else df["prob_up"].min()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = [PALETTE[2] if p >= cutoff else "#cccccc" for p in df["prob_up"]]
    ax.bar(df["symbol"], df["prob_up"], color=colors)
    ax.axhline(float(cutoff), color=PALETTE[1], linestyle="--", label=f"Top-5 cutoff = {cutoff:.3f}")
    for x, p in zip(df["symbol"], df["prob_up"]):
        ax.text(x, p + 0.015, f"{p:.2f}", ha="center", fontsize=8)
    ax.set_title("Rung 2 probability of 5-day up move per symbol (latest date, sorted)")
    ax.set_xlabel("Symbol")
    ax.set_ylabel("prob_up")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    _save("probability_distribution")


# 5. per-fold AUC comparison -----------------------------------------------

def chart_fold_auc() -> None:
    with pg_conn() as conn:
        df = pd.read_sql(
            """
            SELECT p.model_rung, p.fold_id, p.prob_up::float AS prob_up,
                   fp.y_5d_up
            FROM gold.fct_predictions p
            JOIN gold.fct_feature_panel_daily fp
                 ON fp.company_key = p.company_key AND fp.date_key = p.date_key
            WHERE p.model_rung IN (1, 2)
            """,
            conn,
        )
    if df.empty:
        print("  [skip] fold_auc: no predictions")
        return
    rows = []
    for (rung, fold_id), g in df.groupby(["model_rung", "fold_id"]):
        if g["y_5d_up"].nunique() == 2:
            rows.append((int(rung), fold_id, float(roc_auc_score(g["y_5d_up"], g["prob_up"]))))
    auc = pd.DataFrame(rows, columns=["rung", "fold_id", "auc"]).sort_values(["rung", "fold_id"])
    if auc.empty:
        print("  [skip] fold_auc: no folds with both classes")
        return
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for rung in sorted(auc["rung"].unique()):
        sub = auc[auc["rung"] == rung].sort_values("fold_id")
        ax.plot(sub["fold_id"], sub["auc"], marker="o",
                label=RUNG_LABEL.get(int(rung), f"Rung {rung}"),
                color=PALETTE.get(int(rung), "#000"), linewidth=1.8)
    ax.axhline(0.5, color="#aaa", linestyle=":", label="AUC = 0.5 (random)")
    ax.set_title("Per-fold AUC: Rung 1 vs Rung 2")
    ax.set_xlabel("Walk-forward fold (test-window start)")
    ax.set_ylabel("ROC-AUC on fold test set")
    ax.set_ylim(0.40, 0.65)
    ax.legend(loc="lower left")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    _save("fold_auc_per_rung")


def main() -> None:
    print(f"[plot_pnl] writing figures to {FIG_DIR}")
    chart_cumulative_return()
    chart_latest_top5()
    chart_sector_forecast()
    chart_probability_distribution()
    chart_fold_auc()


if __name__ == "__main__":
    main()
