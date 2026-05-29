# Pull income statement, balance sheet, cash flow from FMP into 3 raw tables.
from __future__ import annotations

import json
import os
from time import sleep

import requests
from tqdm import tqdm

from ingest.common import BRONZE_ROOT, all_tickers, log_ingest, pg_conn

FMP_BASE = "https://financialmodelingprep.com/stable"
FMP_KEY = os.getenv("FMP_API_KEY")

ENDPOINT_TABLE = {
    "income-statement": "raw.income_statement_raw",
    "balance-sheet-statement": "raw.balance_sheet_raw",
    "cash-flow-statement": "raw.cash_flow_raw",
}

def fetch_one(symbol: str, endpoint: str) -> list[dict]:
    """FMP /stable endpoint: symbol passed as query param, not path segment."""
    url = f"{FMP_BASE}/{endpoint}"
    r = requests.get(
        url,
        params={"apikey": FMP_KEY, "symbol": symbol, "period": "quarter", "limit": 20},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def _sanitize(k: str) -> str:
    """Postgres column names: lowercase, replace non-alphanumerics with underscore."""
    out = []
    for ch in k:
        if ch.isalnum() or ch == "_":
            out.append(ch.lower())
        else:
            out.append("_")
    s = "".join(out)
    # avoid starting with a digit
    if s and s[0].isdigit():
        s = "_" + s
    return s

def land_one(symbol: str, endpoint: str, payload: list[dict]) -> None:
    table = ENDPOINT_TABLE[endpoint]
    out = BRONZE_ROOT / "fund" / endpoint / f"{symbol}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    if not payload:
        log_ingest(f"fmp_{endpoint}", symbol, "2021-2025", out, 0)
        return

    raw_keys = list(payload[0].keys())
    safe_keys = [_sanitize(k) for k in raw_keys]
    cols_def = ", ".join(f'"{k}" text' for k in safe_keys)
    has_symbol = "symbol" in safe_keys
    has_date = "date" in safe_keys
    if has_symbol and has_date:
        pk = '"symbol", "date"'
    elif has_symbol:
        pk = '"symbol"'
    else:
        pk = '"date"'

    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
              {cols_def},
              ingest_ts timestamptz NOT NULL DEFAULT now(),
              PRIMARY KEY ({pk})
            )
            """
        )
        for row in payload:
            placeholders = ", ".join(["%s"] * len(raw_keys))
            update_clause = ", ".join(
                f'"{sk}" = EXCLUDED."{sk}"'
                for sk in safe_keys
                if sk not in {"symbol", "date"}
            )
            cur.execute(
                f"""
                INSERT INTO {table} ({", ".join(f'"{sk}"' for sk in safe_keys)})
                VALUES ({placeholders})
                ON CONFLICT ({pk}) DO UPDATE SET {update_clause}, ingest_ts = now()
                """,
                tuple(str(row.get(rk)) if row.get(rk) is not None else None for rk in raw_keys),
            )
    log_ingest(f"fmp_{endpoint}", symbol, "2021-2025", out, len(payload))

def main() -> None:
    for symbol in tqdm(all_tickers(), desc="fundamentals"):
        for endpoint in ENDPOINT_TABLE:
            try:
                payload = fetch_one(symbol, endpoint)
                land_one(symbol, endpoint, payload)
            except Exception as e:
                print(f"  {symbol} {endpoint} failed: {e}")
            sleep(0.25)

if __name__ == "__main__":
    main()
