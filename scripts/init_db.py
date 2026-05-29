# Idempotent DB bootstrap: create advdatafinal database + schemas + pgvector extension.
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure .env from the repo root is loaded regardless of cwd
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import psycopg2

PG = dict(
    host=os.getenv("PG_HOST", "127.0.0.1"),
    port=os.getenv("PG_PORT", "5432"),
    user=os.getenv("PG_USER", "postgres"),
    password=os.getenv("PG_PASSWORD"),
)

def create_db() -> None:
    conn = psycopg2.connect(database="postgres", **PG)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = 'advdatafinal'")
    if not cur.fetchone():
        cur.execute("CREATE DATABASE advdatafinal")
        print("Created database advdatafinal")
    else:
        print("Database advdatafinal already exists")
    conn.close()

def init_schemas() -> None:
    conn = psycopg2.connect(database="advdatafinal", **PG)
    conn.autocommit = True
    cur = conn.cursor()
    for schema in ("raw", "datos_masked", "silver", "gold"):
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    # pgvector intentionally NOT used: this Postgres install (EDB 18) needs a
    # build-from-source + sudo for the vector extension. Embeddings are stored
    # as bytea and retrieved via numpy cosine in Python instead.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS raw.ingest_log (
          ingest_id    bigserial PRIMARY KEY,
          source       text NOT NULL,
          symbol       text,
          window_key   text NOT NULL,
          file_path    text NOT NULL,
          row_count    integer,
          sha256       char(64) NOT NULL,
          ingested_at  timestamptz NOT NULL DEFAULT now(),
          UNIQUE (source, symbol, window_key)
        )
        """
    )
    conn.close()
    print("Schemas (raw, datos_masked, silver, gold) + raw.ingest_log ready")
    print("(pgvector skipped; numpy cosine retrieval used instead)")

if __name__ == "__main__":
    if not PG.get("password"):
        sys.exit("PG_PASSWORD missing from .env; aborting")
    create_db()
    init_schemas()
