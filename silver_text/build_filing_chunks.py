# Chunk + embed every 10-K and 8-K filing, write to silver chunked tables + gold embedding store.
from __future__ import annotations

import hashlib
import struct

import numpy as np
import pandas as pd
from tqdm import tqdm

from ingest.common import pg_conn
from silver_text.chunk import chunk_text
from silver_text.embed import embed

EMBED_DIM = 384

def _to_bytes(vec: np.ndarray) -> bytes:
    """Pack a float32 vector as little-endian bytes."""
    v = vec.astype(np.float32, copy=False)
    return struct.pack(f"<{len(v)}f", *v.tolist())

def build_for(source_view: str, target_table: str, source_type_code: str) -> None:
    with pg_conn() as conn:
        df = pd.read_sql(
            f"SELECT accession, symbol, filing_date, body_masked AS body FROM {source_view}",
            conn,
        )

    if df.empty:
        print(f"No rows in {source_view}; skipping")
        return

    rows = []
    for _, r in df.iterrows():
        chunks = chunk_text(r["body"] or "")
        # Some companies file multiple 8-Ks on the same date (different accessions).
        # Disambiguate chunk_key by including the last 6 digits of the accession.
        acc_tail = (str(r["accession"]).replace("-", ""))[-6:] if r.get("accession") else "000000"
        for idx, chunk in enumerate(chunks):
            rows.append(
                dict(
                    chunk_key=f"{source_type_code}-{r['symbol']}-{r['filing_date']}-{acc_tail}-c{idx:02d}",
                    accession=r["accession"],
                    symbol=r["symbol"],
                    filing_date=r["filing_date"],
                    chunk_index=idx,
                    n_tokens=len(chunk.split()),
                    body_chunk=chunk,
                )
            )

    if not rows:
        print(f"No chunks generated for {target_table}")
        return

    df_chunks = pd.DataFrame(rows)
    print(f"Embedding {len(df_chunks)} chunks for {target_table}...")
    embeddings = embed(df_chunks["body_chunk"].tolist(), show_progress=True)

    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {target_table} CASCADE")
        cur.execute(
            f"""
            CREATE TABLE {target_table} (
              chunk_key      text PRIMARY KEY,
              accession      text NOT NULL,
              symbol         text NOT NULL,
              company_key    text NOT NULL,
              filing_date    date NOT NULL,
              chunk_index    smallint NOT NULL,
              n_tokens       smallint NOT NULL,
              body_chunk     text NOT NULL,
              embedding      bytea NOT NULL,
              as_of_date     date NOT NULL
            )
            """
        )
        for vec, row in zip(embeddings, df_chunks.itertuples()):
            ck = hashlib.md5(str(row.symbol).strip().lower().encode()).hexdigest()
            fd = pd.to_datetime(row.filing_date).date()
            cur.execute(
                f"""
                INSERT INTO {target_table}
                  (chunk_key, accession, symbol, company_key, filing_date, chunk_index, n_tokens, body_chunk, embedding, as_of_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    row.chunk_key,
                    row.accession,
                    row.symbol,
                    ck,
                    fd,
                    int(row.chunk_index),
                    int(row.n_tokens),
                    row.body_chunk,
                    _to_bytes(vec),
                    fd,
                ),
            )
        cur.execute(f"CREATE INDEX ON {target_table} (company_key, filing_date)")
    print(f"Wrote {len(df_chunks)} chunks + embeddings to {target_table}")

def main() -> None:
    build_for(
        source_view="datos_masked.filings_10k_redacted",
        target_table="silver.silver_filings_10k_chunked",
        source_type_code="10K",
    )
    build_for(
        source_view="datos_masked.filings_8k_redacted",
        target_table="silver.silver_filings_8k_chunked",
        source_type_code="8K",
    )

if __name__ == "__main__":
    main()
