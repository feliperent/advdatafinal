"""Numpy-based cosine retrieval over silver_filings_*_chunked embeddings (bytea-stored).

No pgvector. At ~13k chunks x 384 dims, in-memory dot product takes <30ms.
Embeddings are L2-normalised so cosine similarity == dot product."""
from __future__ import annotations

import struct
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from ingest.common import pg_conn
from silver_text.embed import embed

EMBED_DIM = 384


def _from_bytes(b: bytes) -> np.ndarray:
    return np.array(struct.unpack(f"<{EMBED_DIM}f", b), dtype=np.float32)


@lru_cache(maxsize=1)
def load_corpus() -> tuple[pd.DataFrame, np.ndarray]:
    """Load all chunks (10-K + 8-K) plus their embeddings as one (n, 384) numpy matrix.
    Cached for the lifetime of the process; ~10MB."""
    with pg_conn() as conn:
        df = pd.read_sql(
            """
            SELECT chunk_key, accession, symbol, company_key, filing_date,
                   chunk_index, n_tokens, body_chunk, embedding,
                   '10K' AS source_type, as_of_date
            FROM silver.silver_filings_10k_chunked
            UNION ALL
            SELECT chunk_key, accession, symbol, company_key, filing_date,
                   chunk_index, n_tokens, body_chunk, embedding,
                   '8K' AS source_type, as_of_date
            FROM silver.silver_filings_8k_chunked
            """,
            conn,
        )
    mat = np.vstack([_from_bytes(bytes(b)) for b in df["embedding"]])
    df = df.drop(columns=["embedding"]).reset_index(drop=True)
    return df, mat


def retrieve(
    question: str,
    symbol: str | None = None,
    as_of_date_max: date | str | None = None,
    top_k: int = 8,
    source_types: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return top-K chunks ranked by cosine similarity, filtered by symbol / as_of_date / source_type."""
    df, mat = load_corpus()

    mask = np.ones(len(df), dtype=bool)
    if symbol is not None:
        mask &= (df["symbol"].values == symbol)
    if as_of_date_max is not None:
        cutoff = pd.to_datetime(as_of_date_max).date()
        mask &= (pd.to_datetime(df["as_of_date"]).dt.date.values <= cutoff)
    if source_types is not None:
        mask &= df["source_type"].isin(source_types).values

    if not mask.any():
        return df.iloc[0:0].assign(similarity=[])

    q = embed([question], show_progress=False)[0]
    sims = mat[mask] @ q  # cosine sim since both are normalised
    sub = df[mask].reset_index(drop=True).copy()
    sub["similarity"] = sims
    sub = sub.nlargest(top_k, "similarity")
    return sub


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "supply chain risk concentration"
    sym = sys.argv[2] if len(sys.argv) > 2 else "AAPL"
    res = retrieve(q, symbol=sym, as_of_date_max="2025-12-01", top_k=5)
    for _, r in res.iterrows():
        print(f"  {r.chunk_key}  sim={r.similarity:.4f}  {(r.body_chunk or '')[:120]!r}")
