# Pull news articles for the universe from FMP /stable/news/stock.
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


def fetch_one(symbol: str, limit: int = 500, frm: str | None = None, to: str | None = None) -> list[dict]:
    # FMP /stable/news/stock: symbols (plural) as query param. Supports `from` / `to`
    # date filters on Premium plan tiers ($49/mo and above) for historical backfill.
    url = f"{FMP_BASE}/news/stock"
    params: dict = {"symbols": symbol, "limit": limit, "apikey": FMP_KEY}
    if frm:
        params["from"] = frm
    if to:
        params["to"] = to
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def land(symbol: str, articles: list[dict]) -> None:
    out = BRONZE_ROOT / "news" / f"{symbol}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(articles, indent=2))

    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS raw.news_raw (
              article_id   text PRIMARY KEY,
              symbol       text,
              published_at text,
              title        text,
              site         text,
              url          text,
              body         text,
              ingest_ts    timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        import hashlib as _hl
        for a in articles:
            url_hash = _hl.md5((a.get("url") or "").encode()).hexdigest()[:12]
            aid = f"{symbol}::{a.get('publishedDate', '')}::{url_hash}"
            cur.execute(
                """
                INSERT INTO raw.news_raw (article_id, symbol, published_at, title, site, url, body)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (article_id) DO UPDATE SET
                  title     = EXCLUDED.title,
                  site      = EXCLUDED.site,
                  body      = EXCLUDED.body,
                  ingest_ts = now()
                """,
                (
                    aid,
                    symbol,
                    a.get("publishedDate"),
                    a.get("title"),
                    a.get("site"),
                    a.get("url"),
                    a.get("text"),
                ),
            )
    log_ingest("fmp_news", symbol, "last_pull", out, len(articles))


def backfill_range(symbol: str, start: date, end: date, step_days: int = 30) -> int:
    # Walk the window in `step_days`-wide chunks, pull each chunk with from/to,
    # land into raw.news_raw via the UPSERT. Returns total articles ingested for this symbol.
    total = 0
    cursor = start
    while cursor < end:
        nxt = min(cursor + timedelta(days=step_days - 1), end)
        try:
            articles = fetch_one(symbol, limit=1000, frm=cursor.isoformat(), to=nxt.isoformat())
            land(symbol, articles)
            total += len(articles)
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
        print(f"=== news backfill {start} -> {end} (per-symbol 30-day chunks) ===")
        for symbol in tqdm(all_tickers(), desc="news backfill"):
            n = backfill_range(symbol, start, end)
            print(f"  {symbol}: {n} articles across the window")
    else:
        for symbol in tqdm(all_tickers(), desc="news"):
            try:
                articles = fetch_one(symbol)
                land(symbol, articles)
            except Exception as e:
                print(f"  {symbol} news failed: {e}")
            sleep(0.25)


if __name__ == "__main__":
    main()
