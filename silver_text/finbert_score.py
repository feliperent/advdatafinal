"""Score news + press articles with FinBERT, write to silver.silver_news_scored / silver_press_scored.

FinBERT (yiyanghkust/finbert-tone) returns 3-class probabilities. Scalar score = P(pos) - P(neg) in [-1, +1]."""
from __future__ import annotations

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ingest.common import pg_conn

MODEL_NAME = "yiyanghkust/finbert-tone"

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tok, model

@torch.no_grad()
def score_text(text: str, tok, model) -> float:
    if not text or not str(text).strip():
        return 0.0
    # Let the tokenizer truncate to 512 TOKENS (do NOT pre-slice to 512 chars; that loses ~90% of long articles).
    enc = tok(str(text), return_tensors="pt", truncation=True, max_length=512)
    out = model(**enc).logits.softmax(dim=-1).squeeze().tolist()
    # FinBERT-tone label order (verified empirically against m.config.id2label):
    # {0: Neutral, 1: Positive, 2: Negative}. Scalar score = P(Positive) - P(Negative) in [-1, +1].
    return float(out[1] - out[2])

def score_table(source_view: str, target_table: str, body_col: str, id_col: str) -> None:
    tok, model = load_model()
    with pg_conn() as conn:
        df = pd.read_sql(
            f"SELECT {id_col}, symbol, published_at, {body_col} AS body FROM {source_view}",
            conn,
        )
    print(f"  Scoring {len(df)} rows for {target_table}...")
    df["finbert_score"] = [
        score_text(t, tok, model) for t in tqdm(df["body"].tolist(), desc=target_table.split(".")[-1])
    ]

    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {target_table} CASCADE")
        cur.execute(
            f"""
            CREATE TABLE {target_table} (
              {id_col}       text PRIMARY KEY,
              symbol         text NOT NULL,
              company_key    text NOT NULL,
              published_at   text,
              trade_date     date,
              finbert_score  numeric(8,5)
            )
            """
        )
        import hashlib
        rows = []
        for _, r in df.iterrows():
            try:
                td = pd.to_datetime(r["published_at"]).date() if pd.notna(r["published_at"]) else None
            except Exception:
                td = None
            ck = hashlib.md5(str(r["symbol"]).strip().lower().encode()).hexdigest()
            rows.append((r[id_col], r["symbol"], ck, r["published_at"], td, float(r["finbert_score"])))
        cur.executemany(
            f"INSERT INTO {target_table} ({id_col}, symbol, company_key, published_at, trade_date, finbert_score) VALUES (%s, %s, %s, %s, %s, %s)",
            rows,
        )
        cur.execute(f"CREATE INDEX ON {target_table} (company_key, trade_date)")
    print(f"Wrote {len(df)} rows to {target_table}")

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
