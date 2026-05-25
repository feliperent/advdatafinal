-- ============================================================================
-- dlt_equivalents.sql  -- Local Postgres <-> Databricks DLT mapping reference
-- ============================================================================
-- For every model in the local Postgres pipeline, this file shows the Databricks
-- DLT equivalent and a one-line comment explaining the materialisation choice
-- (streaming table vs materialised view) plus which window function (if any)
-- forced the choice.

-- ----------------------------------------------------------------------------
-- RAW LAYER (8 streaming tables)
-- ----------------------------------------------------------------------------

-- Local:  CREATE TABLE raw.prices_raw (...)
--         + ingest/fetch_prices.py with ON CONFLICT DO UPDATE
-- DLT:    CREATE STREAMING TABLE raw.prices_raw AS SELECT ... FROM STREAM read_files('/Volumes/...')
-- Why:    Streaming table = incremental file ingest. Same SCD-1 idempotency as the
--         Postgres ON CONFLICT pattern. No window functions, so streaming is fine.

-- ----------------------------------------------------------------------------
-- DATOS_MASKED (4 views)
-- ----------------------------------------------------------------------------

-- Local:  CREATE VIEW datos_masked.news_redacted AS SELECT regexp_replace(...) FROM raw.news_raw;
-- DLT:    CREATE OR REFRESH STREAMING TABLE datos_masked.news_redacted AS
--           SELECT regexp_replace(...) FROM STREAM(raw.news_raw);
-- Why:    Streaming table because the view body is stateless per row (no window). Pure regex.

-- ----------------------------------------------------------------------------
-- SILVER LAYER (6 tables; some streaming, some materialised)
-- ----------------------------------------------------------------------------

-- Local:  silver.silver_prices_cleaned  (CREATE TABLE AS with window functions inline)
-- DLT:    SCD-1 stream into silver.silver_prices_cleaned + a downstream materialised
--         view silver.silver_prices_features for the window-function side.
-- Why:    Streaming for the SCD-1 typed copy; materialised view for the 10 features
--         (LAG, AVG OVER, STDDEV_POP OVER) which streaming cannot evaluate
--         incrementally. Same constraint the midterm hit with ROW_NUMBER.

-- Local:  silver.silver_fundamentals_cleaned  (joins 3 raw tables + 4-quarter TTM SUM OVER)
-- DLT:    CREATE MATERIALIZED VIEW silver.silver_fundamentals_cleaned
-- Why:    SUM OVER ROWS BETWEEN 3 PRECEDING AND CURRENT ROW (the TTM rollup) needs
--         to look across 4 quarters. Window function -> materialised view.

-- Local:  silver.silver_news_scored, silver.silver_press_scored (Python: FinBERT)
-- DLT:    Python-step inside the pipeline OR materialised view that joins to a
--         precomputed Delta table of scored articles.
-- Why:    FinBERT scoring is a torch model call, not pure SQL. The DLT pipeline can
--         either call out to a Python cell that writes a Delta table, then UNION to it,
--         or score on the fly via pandas-on-spark UDF. The midterm did not face this
--         since its only ML step was Streamlit-side.

-- Local:  silver.silver_filings_{10k,8k}_chunked (Python: chunker + MiniLM embeddings)
-- DLT:    Similar to news_scored. Python cell writes Delta with embeddings as ARRAY<FLOAT>
--         (Databricks native; not bytea as in Postgres).
-- Why:    Same as above. Embedding bytes are a Postgres-specific implementation detail
--         (because we can't use pgvector locally). Databricks supports ARRAY<FLOAT> natively.

-- ----------------------------------------------------------------------------
-- GOLD DIMENSIONS (5 streaming tables)
-- ----------------------------------------------------------------------------

-- Local:  gold.dim_date  (calendar via generate_series)
-- DLT:    CREATE OR REFRESH STREAMING TABLE gold.dim_date AS SELECT ... GROUP BY date;
-- Why:    Pure aggregation. Streaming-safe.

-- Local:  gold.dim_company, dim_sector, dim_filing_type
-- DLT:    Same pattern: streaming tables with md5(...) hash key + GROUP BY.
--         Mirrors midterm gold.dim_area / dim_element / dim_detail exactly.

-- Local:  gold.dim_chunk  (UNION ALL of 10K + 8K chunks)
-- DLT:    CREATE OR REFRESH STREAMING TABLE gold.dim_chunk AS
--           SELECT ... FROM STREAM(silver.silver_filings_10k_chunked)
--           UNION ALL SELECT ... FROM STREAM(silver.silver_filings_8k_chunked);
-- Why:    UNION over two streams; both are streaming-safe.

-- ----------------------------------------------------------------------------
-- GOLD FACTS (5 facts: most are materialised views)
-- ----------------------------------------------------------------------------

-- Local:  gold.fct_sentiment_per_day (window-function 3-day and 30-day rolling means)
-- DLT:    CREATE MATERIALIZED VIEW gold.fct_sentiment_per_day
-- Why:    ROWS BETWEEN N PRECEDING. Window function -> materialised view.

-- Local:  gold.fct_embedding_per_company (PCA computed by Python; table is a flat store)
-- DLT:    Python-cell materialises a Delta table; downstream gold facts join to it.
-- Why:    PCA is sklearn, not SQL. Same as silver chunked tables.

-- Local:  gold.fct_feature_panel_daily (the ML training table)
-- DLT:    CREATE MATERIALIZED VIEW gold.fct_feature_panel_daily
-- Why:    ROW_NUMBER() OVER (...) + LAG / LEAD for the target. Same constraint as
--         the midterm's gold.fct_complaints_daily.

-- Local:  gold.fct_predictions (written by Python model trainers, not pure SQL)
-- DLT:    Standard Delta table written by Python notebook cells calling
--         spark.write... or pandas-to-Delta. Not a DLT-managed table.

-- Local:  gold.fct_backtest_pnl_daily, gold.fct_rag_queries
-- DLT:    Same as fct_predictions -- Python-managed Delta tables.

-- ============================================================================
-- COST IMPLICATION (the midterm lesson, restated)
-- ============================================================================
-- Streaming tables are cheap: each new file in the landing volume is appended
-- incrementally. Materialised views recompute on every refresh, which at this
-- project's scale (25k rows, refresh manually) is fine but at production scale
-- (>1M rows) would warrant moving to Delta Live Tables ENHANCED or a watermark-
-- aware streaming aggregation. The midterm hit this wall with gold.fct_complaints_daily;
-- our project hits it with gold.fct_feature_panel_daily for the same reason
-- (window functions across time-series).
