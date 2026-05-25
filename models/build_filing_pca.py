"""Average MiniLM embeddings per company from latest 10-K, fit PCA to top-5, write to gold.

Uses bytea-stored embeddings (the local pragmatic deviation from pgvector). Each chunk's
embedding is a packed float32 vector that we unpack with struct.unpack."""
from __future__ import annotations

import hashlib
import struct

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ingest.common import pg_conn

EMBED_DIM = 384

def _from_bytes(b: bytes) -> np.ndarray:
    if b is None:
        return np.zeros(EMBED_DIM, dtype=np.float32)
    return np.array(struct.unpack(f"<{EMBED_DIM}f", b), dtype=np.float32)

def main() -> None:
    with pg_conn() as conn:
        # ALL 10-K chunks across all (company, filing_date) pairs in the historical window
        df = pd.read_sql(
            """
            SELECT chunk_key, company_key, filing_date, embedding
            FROM silver.silver_filings_10k_chunked
            """,
            conn,
        )

    if df.empty:
        print("No 10-K chunks found; skipping PCA")
        return

    print(f"Loaded {len(df)} 10-K chunks across {df['company_key'].nunique()} companies, {df['filing_date'].nunique()} unique filing dates")
    df["vec"] = df["embedding"].apply(lambda b: _from_bytes(bytes(b)))

    # Mean-pool chunks per (company_key, filing_date) -> one embedding per 10-K filing event
    filing_means = (
        df.groupby(["company_key", "filing_date"])["vec"]
        .apply(lambda s: np.mean(np.vstack(s.values), axis=0))
        .reset_index()
        .rename(columns={"vec": "mean_vec"})
    )
    mat = np.vstack(filing_means["mean_vec"].values)
    print(f"PCA on matrix {mat.shape} (n_filings x 384)")

    pca = PCA(n_components=5)
    reduced = pca.fit_transform(mat)
    explained = pca.explained_variance_ratio_
    print(f"PCA explained-variance ratio (top 5): {[round(float(v), 4) for v in explained]}")
    print(f"Total variance captured: {float(explained.sum()):.4f}")

    rows = []
    for i, r in filing_means.iterrows():
        rows.append(
            (
                1000 + i,
                r["company_key"],
                r["filing_date"],
                float(reduced[i, 0]),
                float(reduced[i, 1]),
                float(reduced[i, 2]),
                float(reduced[i, 3]),
                float(reduced[i, 4]),
            )
        )

    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS gold.fct_embedding_per_company CASCADE")
        cur.execute(
            """
            CREATE TABLE gold.fct_embedding_per_company (
              fact_embedding_key bigint PRIMARY KEY,
              company_key        text   NOT NULL,
              as_of_date         date   NOT NULL,
              filing_pc1         numeric(18,8),
              filing_pc2         numeric(18,8),
              filing_pc3         numeric(18,8),
              filing_pc4         numeric(18,8),
              filing_pc5         numeric(18,8)
            )
            """
        )
        cur.executemany(
            """
            INSERT INTO gold.fct_embedding_per_company
              (fact_embedding_key, company_key, as_of_date,
               filing_pc1, filing_pc2, filing_pc3, filing_pc4, filing_pc5)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
        cur.execute("CREATE INDEX ON gold.fct_embedding_per_company (company_key)")
    print(f"Wrote {len(rows)} rows to gold.fct_embedding_per_company")

if __name__ == "__main__":
    main()
