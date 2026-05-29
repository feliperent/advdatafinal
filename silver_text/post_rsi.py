# Compute RSI-14 via Wilder smoothing in Pandas and write back into silver.
from __future__ import annotations

import pandas as pd

from ingest.common import pg_conn

def wilder_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1.0 / n, adjust=False).mean()
    avg_down = down.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, pd.NA)
    return 100 - 100 / (1 + rs)

def main() -> None:
    with pg_conn() as conn:
        df = pd.read_sql(
            "SELECT price_key, symbol, trade_date, close_px FROM silver.silver_prices_cleaned ORDER BY symbol, trade_date",
            conn,
        )
        df["rsi_14"] = df.groupby("symbol")["close_px"].transform(lambda s: wilder_rsi(s))

        with conn.cursor() as cur:
            # Use UNNEST for a fast bulk update instead of one UPDATE per row
            valid = df.dropna(subset=["rsi_14"])
            keys = valid["price_key"].tolist()
            vals = [float(v) for v in valid["rsi_14"].tolist()]
            cur.execute(
                """
                UPDATE silver.silver_prices_cleaned p
                SET rsi_14 = u.rsi
                FROM (SELECT UNNEST(%s::text[]) AS price_key, UNNEST(%s::numeric[]) AS rsi) u
                WHERE p.price_key = u.price_key
                """,
                (keys, vals),
            )
            print(f"Updated rsi_14 for {len(valid)} rows")

if __name__ == "__main__":
    main()
