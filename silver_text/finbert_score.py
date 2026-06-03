# Score news + press articles with FinBERT, write to silver. Append-only / upsert
# so a re-ingest never destroys previously scored rows. Uses MPS (Apple Silicon)
# or CUDA when available and batches inference for throughput.
from __future__ import annotations

import hashlib

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ingest.common import pg_conn

MODEL_NAME = "yiyanghkust/finbert-tone"
BATCH_SIZE = 64


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(_device())
    return tok, model


@torch.no_grad()
def score_text(text: str, tok, model) -> tuple[float, float, float]:
    # Return the three FinBERT-tone class probabilities (neu, pos, neg). Single-text path
    # kept for unit tests; bulk path uses score_batch below.
    if not text or not str(text).strip():
        return 1.0, 0.0, 0.0
    dev = _device()
    enc = tok(str(text), return_tensors="pt", truncation=True, max_length=512).to(dev)
    out = model(**enc).logits.softmax(dim=-1).squeeze().cpu().tolist()
    return float(out[0]), float(out[1]), float(out[2])


@torch.no_grad()
def score_batch(texts: list[str], tok, model) -> list[tuple[float, float, float]]:
    # Batched inference. Empty / whitespace texts get (1.0, 0.0, 0.0) without a forward pass.
    out: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * len(texts)
    payload_idx: list[int] = []
    payload_text: list[str] = []
    for i, t in enumerate(texts):
        if not t or not str(t).strip():
            out[i] = (1.0, 0.0, 0.0)
        else:
            payload_idx.append(i)
            payload_text.append(str(t))
    if not payload_text:
        return out
    dev = _device()
    enc = tok(
        payload_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(dev)
    probs = model(**enc).logits.softmax(dim=-1).cpu().tolist()
    for idx, p in zip(payload_idx, probs):
        out[idx] = (float(p[0]), float(p[1]), float(p[2]))
    return out


def _ensure_table(cur, target_table: str, id_col: str) -> None:
    # Create the silver table on first run. Idempotent via PRIMARY KEY + UPSERT.
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
    cur.execute(f"CREATE INDEX IF NOT EXISTS ix_{id_col.replace('_id','')}_ck_td ON {target_table} (company_key, trade_date)")


FLUSH_EVERY = 2000  # rows per UPSERT commit so progress is visible and resumable


def _flush(rows: list, cur, target_table: str, id_col: str) -> None:
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


def score_table(source_view: str, target_table: str, body_col: str, id_col: str) -> None:
    # Path A incremental: stage the work-queue once into a temp table to avoid
    # re-evaluating the upstream regex view on every chunk read. Then stream
    # through it in FLUSH_EVERY-row windows: score on device, UPSERT to silver,
    # delete from the work-queue, loop until empty. A kill is resumable because
    # the source view + LEFT ANTI JOIN against silver is deterministic.
    tok, model = load_model()
    with pg_conn() as conn, conn.cursor() as cur:
        _ensure_table(cur, target_table, id_col)
    print(f"  Staging work-queue for {target_table}...", flush=True)
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS silver._finbert_queue_{id_col}")
        cur.execute(
            f"""
            CREATE TABLE silver._finbert_queue_{id_col} AS
            SELECT r.{id_col} AS work_id, r.symbol, r.published_at, r.{body_col} AS body
            FROM {source_view} r
            LEFT JOIN {target_table} s ON s.{id_col} = r.{id_col}
            WHERE s.{id_col} IS NULL
            """
        )
        cur.execute(f"CREATE INDEX ON silver._finbert_queue_{id_col} (work_id)")
        cur.execute(f"SELECT COUNT(*) FROM silver._finbert_queue_{id_col}")
        total = cur.fetchone()[0]
    if total == 0:
        print(f"  {target_table}: nothing new to score (incremental).", flush=True)
        with pg_conn() as conn, conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS silver._finbert_queue_{id_col}")
        return
    print(f"  Scoring {total} new rows for {target_table} on device={_device()} batch={BATCH_SIZE} flush={FLUSH_EVERY}", flush=True)
    bar = tqdm(total=total, desc=target_table.split(".")[-1], unit="art", mininterval=2.0)
    scored_so_far = 0
    while True:
        with pg_conn() as conn:
            df = pd.read_sql(
                f"SELECT work_id, symbol, published_at, body FROM silver._finbert_queue_{id_col} LIMIT {FLUSH_EVERY}",
                conn,
            )
        if df.empty:
            break
        n = len(df)
        bodies = df["body"].tolist()
        probs: list[tuple[float, float, float]] = []
        for start in range(0, n, BATCH_SIZE):
            chunk = bodies[start:start + BATCH_SIZE]
            probs.extend(score_batch(chunk, tok, model))
        df["neu_prob"] = [p[0] for p in probs]
        df["pos_prob"] = [p[1] for p in probs]
        df["neg_prob"] = [p[2] for p in probs]
        df["finbert_score"] = df["pos_prob"] - df["neg_prob"]

        rows = []
        work_ids = []
        for _, r in df.iterrows():
            try:
                td = pd.to_datetime(r["published_at"]).date() if pd.notna(r["published_at"]) else None
            except Exception:
                td = None
            ck = hashlib.md5(str(r["symbol"]).strip().lower().encode()).hexdigest()
            rows.append((
                r["work_id"],
                r["symbol"],
                ck,
                r["published_at"],
                td,
                float(r["pos_prob"]),
                float(r["neu_prob"]),
                float(r["neg_prob"]),
                float(r["finbert_score"]),
            ))
            work_ids.append(r["work_id"])
        with pg_conn() as conn, conn.cursor() as cur:
            _flush(rows, cur, target_table, id_col)
            cur.execute(
                f"DELETE FROM silver._finbert_queue_{id_col} WHERE work_id = ANY(%s)",
                (work_ids,),
            )
        scored_so_far += n
        bar.update(n)
    bar.close()
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS silver._finbert_queue_{id_col}")
    print(f"Upserted {scored_so_far} rows into {target_table}", flush=True)


def main() -> None:
    score_table(
        source_view="datos_masked.news_redacted",
        target_table="silver.silver_news_scored",
        body_col="body_masked",
        id_col="article_id",
    )
    score_table(
        source_view="datos_masked.press_redacted",
        target_table="silver.silver_press_scored",
        body_col="body_masked",
        id_col="press_id",
    )

if __name__ == "__main__":
    main()
