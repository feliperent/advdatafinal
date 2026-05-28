-- Local Postgres to Databricks DLT mapping reference.
-- One block per object showing what we built locally and how it maps to DLT.

-- raw.prices_raw, raw.income_statement_raw, raw.balance_sheet_raw, raw.cash_flow_raw,
-- raw.news_raw, raw.press_raw, raw.sec_10k_raw, raw.sec_8k_raw (8 streaming tables)
-- Local:  CREATE TABLE + Python loader with ON CONFLICT DO UPDATE.
-- DLT:    CREATE OR REFRESH STREAMING TABLE ... FROM STREAM read_files('/Volumes/...').
-- Auto Loader gives the same idempotency the Postgres upsert pattern gives.

-- datos_masked.news_redacted, press_redacted, filings_10k_redacted, filings_8k_redacted
-- Local:  CREATE VIEW with regexp_replace email + phone.
-- DLT:    Streaming table with the same regex; views are stateless per row.

-- silver.silver_prices_cleaned (SCD-1 typed prices)
-- Local:  Plain CTAS.
-- DLT:    Empty STREAMING TABLE + APPLY CHANGES INTO ... STORED AS SCD TYPE 1.
-- silver.silver_prices_features (10 technical features)
-- Local:  Window functions in the same CTAS.
-- DLT:    Separate MATERIALIZED VIEW. Streaming cannot evaluate LAG / AVG OVER incrementally.
-- silver.silver_fundamentals_cleaned (joined statements + 4-quarter TTM)
-- DLT:    MATERIALIZED VIEW. SUM OVER ROWS BETWEEN 3 PRECEDING needs a window pass.

-- silver.silver_news_scored, silver.silver_press_scored (FinBERT)
-- silver.silver_filings_10k_chunked, silver.silver_filings_8k_chunked (chunker + MiniLM)
-- These need torch + sentence-transformers and live in mlpipeline.py, not in pipelinedatos.sql.

-- gold.dim_date, dim_sector, dim_company, dim_filing_type
-- DLT:    MATERIALIZED VIEW with md5(...) keys. No streaming source for these dimensions.

-- gold.fct_sentiment_per_day, gold.fct_embedding_per_company, gold.fct_feature_panel_daily
-- gold.fct_predictions, gold.fct_backtest_pnl_daily, gold.fct_rag_queries
-- These depend on Python ML (FinBERT, PCA, XGBoost, backtest) and live in mlpipeline.py.

-- Cost note: streaming tables are append-only and cheap; materialised views recompute
-- on every refresh. At 25k rows that is fine. At >1M rows the window-function MVs
-- (silver_prices_features, fct_feature_panel_daily) would warrant a watermark-aware
-- streaming aggregation or DLT Enhanced.
