"""Walk-forward backtest: long top-K by prob_up per rung, hold 5 days, 5bp transaction cost.

Reads gold.fct_predictions + gold.dim_date + silver.silver_prices_cleaned.
Writes gold.fct_backtest_pnl_daily."""
from __future__ import annotations

import pandas as pd

from ingest.common import pg_conn

TOP_K = 5
TX_COST_BPS = 5  # 5 basis points one-way

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    with pg_conn() as conn:
        preds = pd.read_sql(
            """
            SELECT p.date_key, p.company_key, p.model_rung, p.prob_up::float AS prob_up,
                   d.full_date AS trade_date
            FROM gold.fct_predictions p
            JOIN gold.dim_date d ON d.date_key = p.date_key
            ORDER BY model_rung, trade_date, prob_up DESC
            """,
            conn,
        )
        ret = pd.read_sql(
            """
            SELECT
              company_key, trade_date, close_px::float AS close_px,
              LEAD(close_px::float, 5) OVER (PARTITION BY company_key ORDER BY trade_date) AS close_t5
            FROM silver.silver_prices_cleaned
            """,
            conn,
        )
    ret["fwd_ret_5d"] = (ret["close_t5"] - ret["close_px"]) / ret["close_px"]
    return preds, ret

REBALANCE_EVERY = 5  # trading days

def compute_pnl(preds: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    """Pick top-K every 5 trading days (no daily overlap); 5-day forward return per rebalance.

    Critical: rebalancing every 5 days while measuring 5-day forward return avoids the daily-overlap
    bug that would multiply realised returns ~5x (because each 5-day return would be counted 5 times)."""
    out_rows: list[tuple] = []
    for rung, rung_group in preds.groupby("model_rung"):
        # Sort unique trade dates within this rung and pick every 5th as a rebalance day
        unique_dates = sorted(rung_group["trade_date"].unique())
        rebalance_dates = unique_dates[::REBALANCE_EVERY]
        for trade_date in rebalance_dates:
            day_group = rung_group[rung_group["trade_date"] == trade_date]
            top = day_group.nlargest(TOP_K, "prob_up")
            merged = top.merge(ret, on=["company_key", "trade_date"], how="left")
            if merged["fwd_ret_5d"].isna().all():
                continue
            gross = float(merged["fwd_ret_5d"].dropna().mean())
            n_long = int(len(merged["fwd_ret_5d"].dropna()))
            cost = TX_COST_BPS / 10000.0
            net = gross - cost
            out_rows.append((trade_date, int(rung), n_long, gross, cost, net))

    df = pd.DataFrame(out_rows, columns=["trade_date", "model_rung", "n_long", "gross_ret", "tx_cost", "net_ret"])
    df = df.sort_values(["model_rung", "trade_date"]).reset_index(drop=True)
    df["cum_net_ret"] = df.groupby("model_rung")["net_ret"].cumsum()
    return df

def benchmark_spy(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Equal-weight buy-and-hold across the 20-stock universe as a proxy for the cross-section."""
    with pg_conn() as conn:
        per_stock = pd.read_sql(
            """
            SELECT symbol, trade_date, close_px::float AS close_px,
                   LAG(close_px::float) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev_close
            FROM silver.silver_prices_cleaned
            """,
            conn,
        )
    per_stock["ret"] = (per_stock["close_px"] - per_stock["prev_close"]) / per_stock["prev_close"]
    bench = per_stock.dropna(subset=["ret"]).groupby("trade_date")["ret"].mean().reset_index()
    bench = bench.rename(columns={"ret": "eq_wt_ret"})
    bench = bench[(bench["trade_date"] >= start) & (bench["trade_date"] <= end)].sort_values("trade_date")
    bench["benchmark_cum"] = bench["eq_wt_ret"].cumsum()
    return bench[["trade_date", "benchmark_cum"]]

def write_to_db(pnl: pd.DataFrame, bench: pd.DataFrame) -> None:
    pnl = pnl.merge(bench, on="trade_date", how="left")
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS gold.fct_backtest_pnl_daily CASCADE")
        cur.execute(
            """
            CREATE TABLE gold.fct_backtest_pnl_daily (
              fact_pnl_key  bigserial PRIMARY KEY,
              trade_date    date NOT NULL,
              date_key      text NOT NULL,
              model_rung    smallint NOT NULL,
              n_long        smallint,
              gross_ret     numeric(18,8),
              tx_cost       numeric(18,8),
              net_ret       numeric(18,8),
              cum_net_ret   numeric(18,8),
              benchmark_cum_ret_eqw numeric(18,8)
            )
            """
        )
        rows = []
        for _, r in pnl.iterrows():
            rows.append(
                (
                    pd.to_datetime(r["trade_date"]).date(),
                    pd.util.hash_pandas_object(pd.Series([str(r["trade_date"])])).iloc[0],  # placeholder; replaced below
                    int(r["model_rung"]),
                    int(r["n_long"]),
                    float(r["gross_ret"]),
                    float(r["tx_cost"]),
                    float(r["net_ret"]),
                    float(r["cum_net_ret"]),
                    float(r["benchmark_cum"]) if pd.notna(r.get("benchmark_cum")) else None,
                )
            )
        # Use md5(date) for date_key
        import hashlib
        rows = [(d, hashlib.md5(str(d).encode()).hexdigest(), *rest) for d, _, *rest in rows]
        cur.executemany(
            """
            INSERT INTO gold.fct_backtest_pnl_daily
              (trade_date, date_key, model_rung, n_long, gross_ret, tx_cost, net_ret, cum_net_ret, benchmark_cum_ret_eqw)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
    print(f"Wrote {len(rows)} rows to gold.fct_backtest_pnl_daily")

def main() -> None:
    preds, ret = load_inputs()
    print(f"Loaded {len(preds)} predictions, {len(ret)} price rows")
    pnl = compute_pnl(preds, ret)
    print(f"Computed PNL: {len(pnl)} (date, rung) rows")
    if pnl.empty:
        print("No PNL rows; nothing to write.")
        return
    bench = benchmark_spy(pnl["trade_date"].min(), pnl["trade_date"].max())
    write_to_db(pnl, bench)
    # Summary per rung. NOTE: each P&L row covers 5 trading days, so to annualise:
    #   periods_per_year = 252 / 5 ≈ 50.4  (weekly rebalance)
    PERIODS_PER_YEAR = 252.0 / REBALANCE_EVERY
    summary = pnl.groupby("model_rung").agg(
        n_periods=("trade_date", "nunique"),
        final_cum=("cum_net_ret", "last"),
        ann_ret_pct=("net_ret", lambda x: float(x.mean()) * PERIODS_PER_YEAR * 100),
        sharpe=("net_ret", lambda x: (float(x.mean()) / max(float(x.std()), 1e-9)) * (PERIODS_PER_YEAR ** 0.5)),
    )
    print(summary)

if __name__ == "__main__":
    main()
