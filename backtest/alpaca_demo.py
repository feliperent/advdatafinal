"""Alpaca paper-trade demo: submit 5 sample orders for the latest Rung 2 top picks.

This is a demonstration, not the backtest. The backtest is a historical simulation
in `run_walkforward.py`. This script proves the same predictions can flow to a real
broker API end-to-end.

Run during US market hours (09:30 to 16:00 ET) to avoid the extended-hours code path.
Outside market hours the orders will reject; saving the screenshot of the Alpaca dashboard
is the alternative deliverable. Costs $0 (paper account, fictitious $100k portfolio)."""
from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def latest_picks(top_k: int = 5, rung: int = 2) -> list[str]:
    """Return the top-K symbols by prob_up on the most recent prediction date for the chosen rung."""
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST", "127.0.0.1"),
        port=os.getenv("PG_PORT", "5432"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        database=os.getenv("PG_DATABASE", "advdatafinal"),
    )
    df = pd.read_sql(
        """
        SELECT c.symbol, p.prob_up::float AS prob_up, d.full_date
        FROM gold.fct_predictions p
        JOIN gold.dim_company c USING (company_key)
        JOIN gold.dim_date    d ON d.date_key = p.date_key
        WHERE p.model_rung = %s
        ORDER BY d.full_date DESC, p.prob_up DESC
        LIMIT %s
        """,
        conn, params=(rung, top_k),
    )
    conn.close()
    if df.empty:
        raise RuntimeError(f"No predictions for rung {rung}; run models first.")
    return df["symbol"].tolist()

def main() -> None:
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
    except ImportError:
        print("alpaca-py is not installed; pip install alpaca-py")
        return

    key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET")
    if not key or not secret:
        print("ALPACA_API_KEY / ALPACA_SECRET_KEY missing from .env; cannot submit")
        return

    picks = latest_picks(top_k=5, rung=2)
    print(f"Picks (latest Rung 2 top 5): {picks}")

    client = TradingClient(key, secret, paper=True)
    acct = client.get_account()
    print(f"Starting equity: ${float(acct.equity):,.2f}")

    submitted = []
    for symbol in picks:
        try:
            req = MarketOrderRequest(
                symbol=symbol, qty=1, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = client.submit_order(req)
            submitted.append((symbol, order.id))
            print(f"  Submitted {symbol} qty=1  order_id={order.id}")
        except Exception as e:
            # outside market hours, "potential wash trade", etc.
            print(f"  {symbol} failed: {type(e).__name__}: {str(e)[:120]}")

    acct = client.get_account()
    print(f"Account equity (post-submission): ${float(acct.equity):,.2f}")
    print(f"Submitted {len(submitted)}/{len(picks)} orders.")
    return submitted

if __name__ == "__main__":
    main()
