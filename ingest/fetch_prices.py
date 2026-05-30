# Pull daily OHLCV for the 20-ticker universe via yfinance.
from __future__ import annotations

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from ingest.common import BRONZE_ROOT, all_tickers, log_ingest, pg_conn

def fetch_one(symbol: str, start: str = "2021-01-01", end: str = "2026-05-01") -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    df["symbol"] = symbol
    return df

def land_to_bronze(symbol: str, df: pd.DataFrame):
    out = BRONZE_ROOT / "prices" / f"{symbol}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log_ingest("yfinance_prices", symbol, "2021-2026-04", out, len(df))

def land_to_raw(symbol: str, df: pd.DataFrame):
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS raw.prices_raw (
              symbol     text,
              trade_date text,
              open       text,
              high       text,
              low        text,
              close      text,
              adj_close  text,
              volume     text,
              ingest_ts  timestamptz NOT NULL DEFAULT now(),
              PRIMARY KEY (symbol, trade_date)
            )
            """
        )
        rows = []
        for _, r in df.iterrows():
            d = r["date"]
            rows.append(
                (
                    symbol,
                    d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
                    str(r.get("open", "")),
                    str(r.get("high", "")),
                    str(r.get("low", "")),
                    str(r.get("close", "")),
                    str(r["adj_close"]) if "adj_close" in r and pd.notna(r["adj_close"]) else "",
                    str(int(r["volume"])) if pd.notna(r.get("volume")) else "",
                )
            )
        cur.executemany(
            """
            INSERT INTO raw.prices_raw (symbol, trade_date, open, high, low, close, adj_close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, trade_date) DO UPDATE SET
              open      = EXCLUDED.open,
              high      = EXCLUDED.high,
              low       = EXCLUDED.low,
              close     = EXCLUDED.close,
              adj_close = EXCLUDED.adj_close,
              volume    = EXCLUDED.volume,
              ingest_ts = now()
            """,
            rows,
        )

def main() -> None:
    for symbol in tqdm(all_tickers(), desc="prices"):
        df = fetch_one(symbol)
        if df.empty:
            print(f"  {symbol}: yfinance returned 0 rows")
            continue
        land_to_bronze(symbol, df)
        land_to_raw(symbol, df)

if __name__ == "__main__":
    main()
