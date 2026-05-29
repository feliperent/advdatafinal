"""Score news + press articles with FinBERT, write to silver.silver_news_scored / silver_press_scored.

FinBERT (yiyanghkust/finbert-tone) returns 3-class probabilities. We persist all three
(pos_prob, neu_prob, neg_prob) so downstream models can use them directly. The legacy scalar
score = P(pos) - P(neg) is kept as well for backwards compatibility with prior dashboards."""
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
def score_text(text: str, tok, model) -> tuple[float, float, float]:
    """Return the three FinBERT-tone class probabilities (neu, pos, neg).

    Empirical label order from m.config.id2label is {0: Neutral, 1: Positive, 2: Negative}.
    Returning a triple lets downstream code keep the scalar via pos - neg if it wants it,
    while also exposing the raw probabilities for richer features."""
    if not text or not str(text).strip():
        return 1.0, 0.0, 0.0  # treat empty text as fully neutral
    enc = tok(str(text), return_tensors="pt", truncation=True, max_length=512)
    out = model(**enc).logits.softmax(dim=-1).squeeze().tolist()
    return float(out[0]), float(out[1]), float(out[2])

def score_table(source_view: str, target_table: str, body_col: str, id_col: str) -> None:
    tok, model = load_model()
    with pg_conn() as conn:
        df = pd.read_sql(
            f"SELECT {id_col}, symbol, published_at, {body_col} AS body FROM {source_view}",
            conn,
        )
    print(f"  Scoring {len(df)} rows for {target_table}...")
    probs = [
        score_text(t, tok, model) for t in tqdm(df["body"].tolist(), desc=target_table.split(".")[-1])
    ]
    df["neu_prob"] = [p[0] for p in probs]
    df["pos_prob"] = [p[1] for p in probs]
    df["neg_prob"] = [p[2] for p in probs]
    df["finbert_score"] = df["pos_prob"] - df["neg_prob"]

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
              pos_prob       numeric(8,5),
              neu_prob       numeric(8,5),
              neg_prob       numeric(8,5),
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
            rows.append((
                r[id_col],
                r["symbol"],
                ck,
                r["published_at"],
                td,
                float(r["pos_prob"]),
                float(r["neu_prob"]),
                float(r["neg_prob"]),
                float(r["finbert_score"]),
            ))
        cur.executemany(
            f"INSERT INTO {target_table} ({id_col}, symbol, company_key, published_at, trade_date, pos_prob, neu_prob, neg_prob, finbert_score) "
            f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
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
