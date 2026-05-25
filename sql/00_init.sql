
-- 00_init.sql  -- Database bootstrap

-- Idempotent: safe to re-run. Same shape as the IN014 midterm's first cells.

CREATE DATABASE advdatafinal;
\c advdatafinal

-- Four-tier medallion schemas (lowercase, matching midterm convention)
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS datos_masked;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

-- pgvector intentionally NOT used. Embeddings are stored as bytea + numpy cosine
-- in Python (the local EDB Postgres 18 install would need build-from-source + sudo
-- for the vector extension). See docs/IMPLEMENTATION_PLAN.md for the rationale.

-- Governance audit table (every raw write logs sha256 here).
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
);
