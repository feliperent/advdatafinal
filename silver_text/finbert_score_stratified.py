"""Stratified FinBERT scoring for backfilled articles.

Memory-constrained dev machines cannot score all 148k articles in one pass.
This script picks N articles per (symbol, year-quarter) stratum and scores
them on CPU with small batches, UPSERTing in chunks so progress is visible
and resumable. Subsequent runs skip already-scored rows via Path A.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ingest.common import pg_conn

MODEL_NAME = "yiyanghkust/finbert-tone"
SAMPLE_PER_STRATUM = int(os.getenv("SAMPLE_PER_STRATUM", "30"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
FLUSH_EVERY = int(os.getenv("FLUSH_EVERY", "200"))
DEVICE = torch.device("cpu")  # MPS leaked memory on this host; CPU is stable


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(DEVICE)
    return tok, model


@torch.no_grad()
def score_batch(texts: list[str], tok, model) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * len(texts)
    payload_idx, payload_text = [], []
    for i, t in enumerate(texts):
        if not t or not str(t).strip():
            out[i] = (1.0, 0.0, 0.0)
        else:
            payload_idx.append(i)
            payload_text.append(str(t))
    if not payload_text:
        return out
    enc = tok(payload_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    probs = model(**enc).logits.softmax(dim=-1).tolist()
    for i, p in zip(payload_idx, probs):
        out[i] = (float(p[0]), float(p[1]), float(p[2]))
    return out


def ensure_silver(cur, target_table: str, id_col: str) -> None:
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {target_table} (
          {id_col}       text PRIMARY KEY,
          symbol         text NOT NULL,
          company_key    text NOT NULL,
          published_at   text,
          trade_date     date,
          pos_prob       numeric(8,5),
          neu_prob       numeric(8,5),
          neg_prob       numeric(8,5),
          finbert_score  numeric(8,5)
        )
        """
    )


def stage_stratified_queue(
    source_view: str,
    body_col: str,
    id_col: str,
    target_table: str,
) -> int:
    queue = f"silver._strat_queue_{id_col}"
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {queue}")
        # Stratify by (symbol, year-quarter). Random row pick within each stratum.
        cur.execute(
            f"""
            CREATE TABLE {queue} AS
            SELECT work_id, symbol, published_at, body FROM (
                SELECT
                    r.{id_col}         AS work_id,
                    r.symbol           AS symbol,
                    r.published_at     AS published_at,
                    r.{body_col}       AS body,
                    ROW_NUMBER() OVER (
                        PARTITION BY r.symbol, DATE_TRUNC('quarter', r.published_at::timestamp)
                        ORDER BY md5(r.{id_col})
                    ) AS rnk
                FROM {source_view} r
                LEFT JOIN {target_table} s ON s.{id_col} = r.work_id_dummy_placeholder
                WHERE r.published_at IS NOT NULL
            ) ranked
            WHERE ranked.rnk <= {SAMPLE_PER_STRATUM}
            """
        )
        cur.execute(f"SELECT COUNT(*) FROM {queue}")
        n_sample = cur.fetchone()[0]
        # Strip already-scored
        cur.execute(
            f"DELETE FROM {queue} WHERE work_id IN (SELECT {id_col} FROM {target_table})"
        )
        cur.execute(f"CREATE INDEX ON {queue} (work_id)")
        cur.execute(f"SELECT COUNT(*) FROM {queue}")
        n_remaining = cur.fetchone()[0]
        print(f"  stratified sample: {n_sample} picked, {n_remaining} unscored to process", flush=True)
    return n_remaining


def stage_simple_queue(
    source_view: str,
    body_col: str,
    id_col: str,
    target_table: str,
) -> int:
    """Stratification is broken when source_view has no work_id column. Use this
    for the press table since the placeholder JOIN won't work there either; we
    stratify in plain Python on the fetched rows instead.
    """
    queue = f"silver._strat_queue_{id_col}"
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {queue}")
        cur.execute(
            f"""
            CREATE TABLE {queue} AS
            SELECT
                r.{id_col}     AS work_id,
                r.symbol       AS symbol,
                r.published_at AS published_at,
                r.{body_col}   AS body,
                ROW_NUMBER() OVER (
                    PARTITION BY r.symbol, DATE_TRUNC('quarter', r.published_at::timestamp)
                    ORDER BY md5(r.{id_col})
                ) AS rnk
            FROM {source_view} r
            LEFT JOIN {target_table} s ON s.{id_col} = r.{id_col}
            WHERE s.{id_col} IS NULL AND r.published_at IS NOT NULL
            """
        )
        cur.execute(f"DELETE FROM {queue} WHERE rnk > {SAMPLE_PER_STRATUM}")
        cur.execute(f"CREATE INDEX ON {queue} (work_id)")
        cur.execute(f"SELECT COUNT(*) FROM {queue}")
        n = cur.fetchone()[0]
        print(f"  stratified sample for {target_table}: {n} unscored to process (per-stratum cap = {SAMPLE_PER_STRATUM})", flush=True)
    return n


def score_streaming(target_table: str, id_col: str, tok, model) -> int:
    queue = f"silver._strat_queue_{id_col}"
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {queue}")
        total = cur.fetchone()[0]
    if total == 0:
        return 0
    bar = tqdm(total=total, desc=target_table.split(".")[-1], unit="art", mininterval=2.0)
    scored = 0
    while True:
        with pg_conn() as conn:
            df = pd.read_sql(
                f"SELECT work_id, symbol, published_at, body FROM {queue} LIMIT {FLUSH_EVERY}",
                conn,
            )
        if df.empty:
            break
        n = len(df)
        bodies = df["body"].tolist()
        probs: list[tuple[float, float, float]] = []
        for start in range(0, n, BATCH_SIZE):
            probs.extend(score_batch(bodies[start:start + BATCH_SIZE], tok, model))
        df["neu_prob"] = [p[0] for p in probs]
        df["pos_prob"] = [p[1] for p in probs]
        df["neg_prob"] = [p[2] for p in probs]
        df["finbert_score"] = df["pos_prob"] - df["neg_prob"]

        rows, work_ids = [], []
        for _, r in df.iterrows():
            try:
                td = pd.to_datetime(r["published_at"]).date() if pd.notna(r["published_at"]) else None
            except Exception:
                td = None
            ck = hashlib.md5(str(r["symbol"]).strip().lower().encode()).hexdigest()
            rows.append((
                r["work_id"], r["symbol"], ck, r["published_at"], td,
                float(r["pos_prob"]), float(r["neu_prob"]), float(r["neg_prob"]), float(r["finbert_score"]),
            ))
            work_ids.append(r["work_id"])
        with pg_conn() as conn, conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {target_table}
                  ({id_col}, symbol, company_key, published_at, trade_date,
                   pos_prob, neu_prob, neg_prob, finbert_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT ({id_col}) DO UPDATE SET
                  pos_prob = EXCLUDED.pos_prob,
                  neu_prob = EXCLUDED.neu_prob,
                  neg_prob = EXCLUDED.neg_prob,
                  finbert_score = EXCLUDED.finbert_score
                """,
                rows,
            )
            cur.execute(f"DELETE FROM {queue} WHERE work_id = ANY(%s)", (work_ids,))
        scored += n
        bar.update(n)
    bar.close()
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {queue}")
    return scored


def main() -> None:
    t0 = time.time()
    print(f"=== Stratified FinBERT scorer: device={DEVICE} batch={BATCH_SIZE} flush={FLUSH_EVERY} per_stratum={SAMPLE_PER_STRATUM} ===", flush=True)
    tok, model = load_model()
    print(f"  model ready in {time.time()-t0:.1f}s", flush=True)

    # Ensure silver tables exist before staging the work-queue (LEFT JOIN needs them).
    with pg_conn() as conn, conn.cursor() as cur:
        ensure_silver(cur, "silver.silver_news_scored", "article_id")
        ensure_silver(cur, "silver.silver_press_scored", "press_id")

    # News
    print("=== News ===", flush=True)
    stage_simple_queue("datos_masked.news_redacted", "body_masked", "article_id", "silver.silver_news_scored")
    n_news = score_streaming("silver.silver_news_scored", "article_id", tok, model)
    print(f"  news upserted: {n_news} in {time.time()-t0:.1f}s elapsed", flush=True)

    # Press (smaller table, run also stratified for consistency)
    print("=== Press ===", flush=True)
    stage_simple_queue("datos_masked.press_redacted", "body_masked", "press_id", "silver.silver_press_scored")
    n_press = score_streaming("silver.silver_press_scored", "press_id", tok, model)
    print(f"  press upserted: {n_press} in {time.time()-t0:.1f}s elapsed", flush=True)

    print(f"=== Done in {time.time()-t0:.1f}s, news+{n_news} press+{n_press} ===", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
