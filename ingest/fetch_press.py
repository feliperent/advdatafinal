# Pull press releases for the universe from FMP /stable/news/press-releases.
from __future__ import annotations

import json
import os
from datetime import date, timedelta
from time import sleep

import requests
from tqdm import tqdm

from ingest.common import BRONZE_ROOT, all_tickers, log_ingest, pg_conn

FMP_BASE = "https://financialmodelingprep.com/stable"
FMP_KEY = os.getenv("FMP_API_KEY")


def fetch_one(symbol: str, limit: int = 100, frm: str | None = None, to: str | None = None) -> list[dict]:
    # FMP /stable/news/press-releases: symbols (plural) as query param.
    # `from` and `to` supported on Premium plan for historical backfill.
    url = f"{FMP_BASE}/news/press-releases"
    params: dict = {"symbols": symbol, "limit": limit, "apikey": FMP_KEY}
    if frm:
        params["from"] = frm
    if to:
        params["to"] = to
    r = requests.get(url, params=params, timeout=30)
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


def backfill_range(symbol: str, start: date, end: date, step_days: int = 60) -> int:
    total = 0
    cursor = start
    while cursor < end:
        nxt = min(cursor + timedelta(days=step_days - 1), end)
        try:
            releases = fetch_one(symbol, limit=500, frm=cursor.isoformat(), to=nxt.isoformat())
            land(symbol, releases)
            total += len(releases)
        except Exception as e:
            print(f"  {symbol} {cursor}->{nxt} failed: {e}")
        sleep(0.15)
        cursor = nxt + timedelta(days=1)
    return total


def main() -> None:
    backfill = os.getenv("BACKFILL", "").lower() in ("1", "true", "yes")
    if backfill:
        start = date.fromisoformat(os.getenv("BACKFILL_START", "2021-01-01"))
        end = date.fromisoformat(os.getenv("BACKFILL_END", date.today().isoformat()))
        print(f"=== press backfill {start} -> {end} (per-symbol 60-day chunks) ===")
        for symbol in tqdm(all_tickers(), desc="press backfill"):
            n = backfill_range(symbol, start, end)
            print(f"  {symbol}: {n} releases across the window")
    else:
        for symbol in tqdm(all_tickers(), desc="press"):
            try:
                releases = fetch_one(symbol)
                land(symbol, releases)
            except Exception as e:
                print(f"  {symbol} press failed: {e}")
            sleep(0.25)


if __name__ == "__main__":
    main()
