# Pull press releases for the universe from FMP /press-releases (v4).
from __future__ import annotations

import json
import os
from time import sleep

import requests
from tqdm import tqdm

from ingest.common import BRONZE_ROOT, all_tickers, log_ingest, pg_conn

FMP_BASE = "https://financialmodelingprep.com/stable"
FMP_KEY = os.getenv("FMP_API_KEY")

def fetch_one(symbol: str, limit: int = 50) -> list[dict]:
    # FMP /stable/news/press-releases: symbols (plural) as query param.
    url = f"{FMP_BASE}/news/press-releases"
    r = requests.get(url, params={"symbols": symbol, "limit": limit, "apikey": FMP_KEY}, timeout=30)
    r.raise_for_status()
    return r.json()

def land(symbol: str, releases: list[dict]) -> None:
    out = BRONZE_ROOT / "press" / f"{symbol}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(releases, indent=2))

    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS raw.press_raw (
              press_id     text PRIMARY KEY,
              symbol       text,
              published_at text,
              title        text,
              body         text,
              ingest_ts    timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        import hashlib as _hl
        for p in releases:
            pub = p.get("publishedDate") or p.get("date") or ""
            body_hash = _hl.md5(((p.get("title") or "") + "|" + (p.get("text") or "")).encode()).hexdigest()[:12]
            pid = f"{symbol}::{pub}::{body_hash}"
            cur.execute(
                """
                INSERT INTO raw.press_raw (press_id, symbol, published_at, title, body)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (press_id) DO UPDATE SET
                  published_at = EXCLUDED.published_at,
                  title        = EXCLUDED.title,
                  body         = EXCLUDED.body,
                  ingest_ts    = now()
                """,
                (pid, symbol, pub or None, p.get("title"), p.get("text")),
            )
    log_ingest("fmp_press", symbol, "last_pull", out, len(releases))

def main() -> None:
    for symbol in tqdm(all_tickers(), desc="press"):
        try:
            releases = fetch_one(symbol)
            land(symbol, releases)
        except Exception as e:
            print(f"  {symbol} press failed: {e}")
        sleep(0.25)

if __name__ == "__main__":
    main()
