"""Pull 10-K Item 1A + 8-K material events from SEC EDGAR via edgartools."""
from __future__ import annotations

import os
from datetime import date, timedelta
from time import sleep

from edgar import Company, set_identity
from tqdm import tqdm

from ingest.common import BRONZE_ROOT, all_tickers, log_ingest, pg_conn


def init_sec() -> None:
    ua = os.getenv("SEC_USER_AGENT", "Felipe Renteria <felipe@bookline.ai>")
    set_identity(ua)


def fetch_10k_item1a(symbol: str, years: int = 5) -> list[dict]:
    """Pull the last `years` 10-Ks and extract the Item 1A Risk Factors text via tenk.risk_factors."""
    c = Company(symbol)
    filings = list(c.get_filings(form="10-K").head(years))
    out = []
    for f in filings:
        try:
            tenk = f.obj()
            body = str(tenk.risk_factors) if tenk and tenk.risk_factors else ""
            if not body:
                print(f"  10-K {symbol} {f.filing_date}: risk_factors empty, skipping")
                continue
            out.append(
                dict(
                    accession=str(f.accession_no),
                    filing_date=str(f.filing_date),
                    fiscal_year=str(getattr(tenk, "period_of_report", "") or ""),
                    body=body,
                )
            )
        except Exception as e:
            print(f"  10-K parse failed for {symbol} ({f.accession_no}): {type(e).__name__}: {e}")
    return out


def fetch_8k(symbol: str, months: int = 12) -> list[dict]:
    cutoff = date.today() - timedelta(days=30 * months)
    c = Company(symbol)
    filings = c.get_filings(form="8-K")
    out = []
    for f in filings:
        fd = f.filing_date if isinstance(f.filing_date, date) else None
        if fd and fd < cutoff:
            break
        try:
            eightk = f.obj()
            # eightk.text is a METHOD (not attribute) that returns the full filing body.
            # str(eightk) only returns a one-line summary, which is useless for RAG.
            body = ""
            if eightk and hasattr(eightk, "text"):
                t = eightk.text
                body = t() if callable(t) else str(t)
            if not body or len(body) < 200:
                # Fallback: try filing.markdown() then filing.text()
                for accessor in (getattr(f, "markdown", None), getattr(f, "text", None)):
                    if callable(accessor):
                        body = accessor()
                        if body and len(body) >= 200:
                            break
            out.append(
                dict(
                    accession=str(f.accession_no),
                    filing_date=str(f.filing_date),
                    body=body or "",
                )
            )
        except Exception as e:
            print(f"  8-K parse failed for {symbol}: {type(e).__name__}: {e}")
    return out


def land_10k(symbol: str, payloads: list[dict]) -> None:
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS raw.sec_10k_raw (
              accession    text PRIMARY KEY,
              symbol       text,
              filing_date  text,
              fiscal_year  text,
              body         text,
              ingest_ts    timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        for p in payloads:
            out = BRONZE_ROOT / "filings" / "10K" / symbol / f"{p['filing_date']}.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(p["body"])
            cur.execute(
                """
                INSERT INTO raw.sec_10k_raw (accession, symbol, filing_date, fiscal_year, body)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (accession) DO UPDATE SET body = EXCLUDED.body, ingest_ts = now()
                """,
                (p["accession"], symbol, p["filing_date"], p["fiscal_year"], p["body"]),
            )
    base = BRONZE_ROOT / "filings" / "10K" / symbol
    if base.exists():
        log_ingest("sec_10k", symbol, "last_5y", base, len(payloads))


def land_8k(symbol: str, payloads: list[dict]) -> None:
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS raw.sec_8k_raw (
              accession    text PRIMARY KEY,
              symbol       text,
              filing_date  text,
              body         text,
              ingest_ts    timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        for p in payloads:
            safe = p["accession"].replace("/", "_")
            out = BRONZE_ROOT / "filings" / "8K" / symbol / f"{p['filing_date']}_{safe}.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(p["body"])
            cur.execute(
                """
                INSERT INTO raw.sec_8k_raw (accession, symbol, filing_date, body)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (accession) DO UPDATE SET body = EXCLUDED.body, ingest_ts = now()
                """,
                (p["accession"], symbol, p["filing_date"], p["body"]),
            )
    base = BRONZE_ROOT / "filings" / "8K" / symbol
    if base.exists():
        log_ingest("sec_8k", symbol, "last_12m", base, len(payloads))


def main() -> None:
    init_sec()
    for symbol in tqdm(all_tickers(), desc="sec"):
        try:
            land_10k(symbol, fetch_10k_item1a(symbol))
        except Exception as e:
            print(f"  10-K all failed for {symbol}: {e}")
        try:
            land_8k(symbol, fetch_8k(symbol))
        except Exception as e:
            print(f"  8-K all failed for {symbol}: {e}")
        sleep(0.5)


if __name__ == "__main__":
    main()
