
-- 01_raw_schema.sql  -- Raw tables (8 source mirrors + ingest_log)

-- All raw tables are text-typed mirrors of the source API. Casting + cleaning
-- happen in silver. Idempotency via ON CONFLICT DO UPDATE (the Postgres
-- equivalent of the midterm's APPLY CHANGES INTO ... STORED AS SCD TYPE 1).

-- -- Prices (yfinance) ------------------------------------------------
CREATE TABLE IF NOT EXISTS raw.prices_raw (
  symbol     text,
  trade_date text,
  open       text,
  high       text,
  low        text,
  close      text,
  adj_close  text,
  volume     text,
  ingest_ts  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (symbol, trade_date)
);

-- Ingest pattern (used by ingest/fetch_prices.py):
-- INSERT INTO raw.prices_raw (symbol, trade_date, open, high, low, close, adj_close, volume)
-- VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
-- ON CONFLICT (symbol, trade_date) DO UPDATE SET
--   open = EXCLUDED.open, ..., ingest_ts = now();

-- -- Fundamentals (FMP /stable/* x 3 endpoints) ----------------------
-- Created dynamically by ingest/fetch_fundamentals.py from the API response shape.
-- Both PK and columns are derived from the FMP keys (camelCase -> lowercase).
-- Three tables result, one per endpoint:
--   raw.income_statement_raw
--   raw.balance_sheet_raw
--   raw.cash_flow_raw

-- Shape of raw.income_statement_raw (full schema with all FMP columns; ~40 columns):
CREATE TABLE IF NOT EXISTS raw.income_statement_raw (
  date              text,
  symbol            text,
  reportedcurrency  text,
  cik               text,
  filingdate        text,
  accepteddate      text,
  fiscalyear        text,
  period            text,
  revenue           text,
  costofrevenue     text,
  grossprofit       text,
  operatingincome   text,
  netincome         text,
  ebitda            text,
  -- (remaining ~30 columns omitted for brevity; full set in pg_dump)
  ingest_ts         timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (symbol, date)
);
-- balance_sheet_raw and cash_flow_raw follow the same shape (different columns).

-- -- News (FMP /stable/news/stock) -----------------------------------
CREATE TABLE IF NOT EXISTS raw.news_raw (
  article_id   text PRIMARY KEY,   -- composite: symbol::publishedDate::md5(url[:12])
  symbol       text,
  published_at text,
  title        text,
  site         text,
  url          text,
  body         text,
  ingest_ts    timestamptz NOT NULL DEFAULT now()
);

-- -- Press releases (FMP /stable/news/press-releases) ----------------
CREATE TABLE IF NOT EXISTS raw.press_raw (
  press_id     text PRIMARY KEY,   -- symbol::publishedDate::md5(title|text)
  symbol       text,
  published_at text,
  title        text,
  body         text,
  ingest_ts    timestamptz NOT NULL DEFAULT now()
);

-- -- SEC 10-K (edgartools, Item 1A Risk Factors) ----------------------
CREATE TABLE IF NOT EXISTS raw.sec_10k_raw (
  accession    text PRIMARY KEY,
  symbol       text,
  filing_date  text,
  fiscal_year  text,
  body         text,
  ingest_ts    timestamptz NOT NULL DEFAULT now()
);

-- -- SEC 8-K (edgartools, full body via eightk.text()) ---------------
CREATE TABLE IF NOT EXISTS raw.sec_8k_raw (
  accession    text PRIMARY KEY,
  symbol       text,
  filing_date  text,
  body         text,
  ingest_ts    timestamptz NOT NULL DEFAULT now()
);

-- -- Governance audit log -------------------------------------------
-- Defined in 00_init.sql. Repeated here for visibility.
-- CREATE TABLE raw.ingest_log (
--   ingest_id    bigserial PRIMARY KEY,
--   source       text NOT NULL,
--   symbol       text,
--   window_key   text NOT NULL,
--   file_path    text NOT NULL,
--   row_count    integer,
--   sha256       char(64) NOT NULL,
--   ingested_at  timestamptz NOT NULL DEFAULT now(),
--   UNIQUE (source, symbol, window_key)
-- );

-- -- Final counts after a complete ingest run ----------------------
-- raw.prices_raw            : 25,100 rows × 20 symbols
-- raw.income_statement_raw  :    400 rows × 20 symbols
-- raw.balance_sheet_raw     :    400 rows × 20 symbols
-- raw.cash_flow_raw         :    400 rows × 20 symbols
-- raw.news_raw              :  3,853 rows × 20 symbols
-- raw.press_raw             :    563 rows × 20 symbols
-- raw.sec_10k_raw           :     95 rows × 20 symbols (~5 per stock × 5 years)
-- raw.sec_8k_raw            :    255 rows × 20 symbols (last 12 months)
-- raw.ingest_log            :    160 audit rows
