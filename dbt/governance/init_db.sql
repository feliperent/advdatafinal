-- Idempotent: safe to re-run.
-- Run as Postgres superuser the first time:
--   psql -h 127.0.0.1 -U postgres -f dbt/governance/init_db.sql

CREATE DATABASE advdatafinal;
\c advdatafinal

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS datos_masked;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS raw.ingest_log (
  ingest_id    bigserial PRIMARY KEY,
  source       text   NOT NULL,
  symbol       text,
  window_key   text   NOT NULL,
  file_path    text   NOT NULL,
  row_count    integer,
  sha256       char(64) NOT NULL,
  ingested_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (source, symbol, window_key)
);
