# Databricks notebook source
# MAGIC %md
# MAGIC # 01_data_pipeline | advdatafinal medallion DLT pipeline
# MAGIC
# MAGIC Builds the four-schema medallion (raw / datos_masked / silver / gold) on Delta tables.
# MAGIC Direct port of the local Postgres pipeline. Same DDL shape, same column names, same
# MAGIC SCD-1 idempotency. Outputs feed `02_ml_pipeline.py` which trains the 3 model rungs.
# MAGIC
# MAGIC **Prerequisites**: bronze/ uploaded to `/Volumes/advdatafinal/raw/landing/` (see `docs/DATABRICKS_UPLOAD.md`).
# MAGIC
# MAGIC **Materialisation choices**: streaming tables for everything except window-function-heavy
# MAGIC facts. The midterm encountered this same wall with `gold.fct_complaints_daily` (forced
# MAGIC into a materialised view by `ROW_NUMBER`).

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Catalog + schemas + landing volume

# COMMAND ----------
# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS advdatafinal;
# MAGIC CREATE SCHEMA  IF NOT EXISTS advdatafinal.raw;
# MAGIC CREATE SCHEMA  IF NOT EXISTS advdatafinal.datos_masked;
# MAGIC CREATE SCHEMA  IF NOT EXISTS advdatafinal.silver;
# MAGIC CREATE SCHEMA  IF NOT EXISTS advdatafinal.gold;
# MAGIC CREATE VOLUME  IF NOT EXISTS advdatafinal.raw.landing;

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Raw layer | 8 streaming tables via Auto Loader

# COMMAND ----------
# MAGIC %sql
# MAGIC -- raw.prices_raw (yfinance OHLCV)
# MAGIC CREATE STREAMING TABLE advdatafinal.raw.prices_raw
# MAGIC AS SELECT
# MAGIC   cast(symbol     as STRING) as symbol,
# MAGIC   cast(trade_date as STRING) as trade_date,
# MAGIC   cast(open       as STRING) as open,
# MAGIC   cast(high       as STRING) as high,
# MAGIC   cast(low        as STRING) as low,
# MAGIC   cast(close      as STRING) as close,
# MAGIC   cast(adj_close  as STRING) as adj_close,
# MAGIC   cast(volume     as STRING) as volume,
# MAGIC   _metadata.file_path as _source_path,
# MAGIC   current_timestamp() as ingest_ts
# MAGIC FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/prices/', format => 'parquet');

# COMMAND ----------
# MAGIC %sql
# MAGIC -- raw.income_statement_raw, raw.balance_sheet_raw, raw.cash_flow_raw (FMP)
# MAGIC CREATE STREAMING TABLE advdatafinal.raw.income_statement_raw
# MAGIC AS SELECT *, current_timestamp() as ingest_ts
# MAGIC FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/income_statement/', format => 'json');
# MAGIC
# MAGIC CREATE STREAMING TABLE advdatafinal.raw.balance_sheet_raw
# MAGIC AS SELECT *, current_timestamp() as ingest_ts
# MAGIC FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/balance_sheet/', format => 'json');
# MAGIC
# MAGIC CREATE STREAMING TABLE advdatafinal.raw.cash_flow_raw
# MAGIC AS SELECT *, current_timestamp() as ingest_ts
# MAGIC FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/cash_flow/', format => 'json');

# COMMAND ----------
# MAGIC %sql
# MAGIC -- raw.news_raw, raw.press_raw (FMP, JSON)
# MAGIC CREATE STREAMING TABLE advdatafinal.raw.news_raw
# MAGIC AS SELECT *, current_timestamp() as ingest_ts
# MAGIC FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/news/', format => 'json');
# MAGIC
# MAGIC CREATE STREAMING TABLE advdatafinal.raw.press_raw
# MAGIC AS SELECT *, current_timestamp() as ingest_ts
# MAGIC FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/press/', format => 'json');

# COMMAND ----------
# MAGIC %sql
# MAGIC -- raw.sec_10k_raw, raw.sec_8k_raw (text files)
# MAGIC CREATE STREAMING TABLE advdatafinal.raw.sec_10k_raw
# MAGIC AS SELECT
# MAGIC   regexp_extract(_metadata.file_path, '/10K/([^/]+)/', 1) as symbol,
# MAGIC   regexp_extract(_metadata.file_path, '/([0-9]{4}-[0-9]{2}-[0-9]{2})\\.txt$', 1) as filing_date,
# MAGIC   value as body,
# MAGIC   _metadata.file_path as accession,
# MAGIC   current_timestamp() as ingest_ts
# MAGIC FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/sec_10k/', format => 'text', wholeText => true);
# MAGIC
# MAGIC CREATE STREAMING TABLE advdatafinal.raw.sec_8k_raw
# MAGIC AS SELECT
# MAGIC   regexp_extract(_metadata.file_path, '/8K/([^/]+)/', 1) as symbol,
# MAGIC   regexp_extract(_metadata.file_path, '/([0-9]{4}-[0-9]{2}-[0-9]{2})_', 1) as filing_date,
# MAGIC   value as body,
# MAGIC   _metadata.file_path as accession,
# MAGIC   current_timestamp() as ingest_ts
# MAGIC FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/sec_8k/', format => 'text', wholeText => true);

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. datos_masked layer | 4 PII redaction views (regex email + phone)

# COMMAND ----------
# MAGIC %sql
# MAGIC CREATE OR REFRESH STREAMING TABLE advdatafinal.datos_masked.news_redacted AS
# MAGIC SELECT
# MAGIC   symbol, publishedDate as published_at, title, site, url,
# MAGIC   regexp_replace(
# MAGIC     regexp_replace(coalesce(text, ''),
# MAGIC       '[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}', '[EMAIL]'),
# MAGIC     '\\+?1?[\\s.\\-]?\\(?\\d{3}\\)?[\\s.\\-]?\\d{3}[\\s.\\-]?\\d{4}', '[PHONE]'
# MAGIC   ) AS body_masked,
# MAGIC   ingest_ts
# MAGIC FROM STREAM(advdatafinal.raw.news_raw);
# MAGIC
# MAGIC -- 3 analogous streaming tables follow the same pattern:
# MAGIC --   advdatafinal.datos_masked.press_redacted    (from raw.press_raw)
# MAGIC --   advdatafinal.datos_masked.filings_10k_redacted  (from raw.sec_10k_raw)
# MAGIC --   advdatafinal.datos_masked.filings_8k_redacted   (from raw.sec_8k_raw)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Silver layer | SCD-1 idempotency via APPLY CHANGES INTO

# COMMAND ----------
# MAGIC %sql
# MAGIC -- silver.silver_prices_cleaned -- typed prices (the 10 technical features land in a
# MAGIC -- downstream materialised view because LAG/AVG OVER cannot be streamed incrementally).
# MAGIC CREATE TEMPORARY STREAMING LIVE VIEW prices_typed AS
# MAGIC SELECT
# MAGIC   symbol,
# MAGIC   cast(trade_date as DATE)         as trade_date,
# MAGIC   cast(open as DECIMAL(18,6))      as open_px,
# MAGIC   cast(high as DECIMAL(18,6))      as high_px,
# MAGIC   cast(low as DECIMAL(18,6))       as low_px,
# MAGIC   cast(close as DECIMAL(18,6))     as close_px,
# MAGIC   cast(adj_close as DECIMAL(18,6)) as adj_close_px,
# MAGIC   cast(volume as BIGINT)           as volume
# MAGIC FROM STREAM(advdatafinal.raw.prices_raw)
# MAGIC WHERE close IS NOT NULL AND close <> '';
# MAGIC
# MAGIC CREATE OR REFRESH STREAMING TABLE advdatafinal.silver.silver_prices_cleaned;
# MAGIC APPLY CHANGES INTO advdatafinal.silver.silver_prices_cleaned
# MAGIC   FROM STREAM(prices_typed)
# MAGIC   KEYS (symbol, trade_date)
# MAGIC   SEQUENCE BY trade_date
# MAGIC   STORED AS SCD TYPE 1;
# MAGIC -- The 10 technical features (LAG/AVG OVER lookback) live in a downstream MV.

# COMMAND ----------
# MAGIC %sql
# MAGIC -- silver.silver_prices_features (the 10 technical features; materialised view)
# MAGIC CREATE MATERIALIZED VIEW advdatafinal.silver.silver_prices_features AS
# MAGIC SELECT *,
# MAGIC   LN(close_px / NULLIF(LAG(close_px, 1) OVER (PARTITION BY symbol ORDER BY trade_date), 0)) AS log_ret_1d,
# MAGIC   AVG(close_px) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN  4 PRECEDING AND CURRENT ROW) AS sma_5,
# MAGIC   AVG(close_px) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sma_20,
# MAGIC   AVG(close_px) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS sma_50,
# MAGIC   AVG(close_px) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS ema_12,
# MAGIC   AVG(close_px) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 25 PRECEDING AND CURRENT ROW) AS ema_26,
# MAGIC   STDDEV_POP(close_px) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sd_close_20
# MAGIC FROM advdatafinal.silver.silver_prices_cleaned;

# COMMAND ----------
# MAGIC %sql
# MAGIC -- silver.silver_fundamentals_cleaned (joins 3 raw statements + 4-quarter TTM rollups)
# MAGIC CREATE MATERIALIZED VIEW advdatafinal.silver.silver_fundamentals_cleaned AS
# MAGIC WITH inc AS (...), bs AS (...), cf AS (...), joined AS (...), ttm AS (...)
# MAGIC SELECT
# MAGIC   md5(symbol || '|' || filing_date::text) as fund_key,
# MAGIC   md5(LOWER(TRIM(symbol)))                 as company_key,
# MAGIC   symbol, filing_date,
# MAGIC   revenue_ttm, gross_profit_ttm, ...
# MAGIC FROM ttm WHERE revenue_ttm IS NOT NULL;
# MAGIC -- Full SQL in the local sql/03_silver_schema.sql (identical here).

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Python-built silver tables (FinBERT scoring + MiniLM embedding)

# COMMAND ----------
# Run the FinBERT scorer + chunker as Python cells. These are NOT DLT; they write
# directly to Delta tables that downstream gold tables read from.
import sys
sys.path.insert(0, '/Workspace/Repos/<your-repo>/advdatafinal')
# from silver_text.finbert_score import main as score_main
# from silver_text.build_filing_chunks import main as chunk_main
# score_main()  # writes silver.silver_news_scored + silver.silver_press_scored
# chunk_main()  # writes silver.silver_filings_{10k,8k}_chunked

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Gold layer | dimensions (streaming) + facts (materialised views)

# COMMAND ----------
# MAGIC %sql
# MAGIC -- gold.dim_company (streaming, md5 hash key)
# MAGIC CREATE OR REFRESH STREAMING TABLE advdatafinal.gold.dim_company AS
# MAGIC SELECT md5(LOWER(TRIM(symbol))) as company_key, symbol, name, sector
# MAGIC FROM STREAM(advdatafinal.silver.silver_prices_cleaned)
# MAGIC GROUP BY symbol, name, sector;

# COMMAND ----------
# MAGIC %sql
# MAGIC -- gold.fct_feature_panel_daily (materialised view; window functions force this)
# MAGIC CREATE MATERIALIZED VIEW advdatafinal.gold.fct_feature_panel_daily AS
# MAGIC SELECT
# MAGIC   ROW_NUMBER() OVER (ORDER BY trade_date, company_key) + 1000 AS fact_panel_key,
# MAGIC   /* 30 features + LEAD(close,5) target */
# MAGIC FROM advdatafinal.silver.silver_prices_features p
# MAGIC JOIN advdatafinal.silver.silver_fundamentals_cleaned f ON ...
# MAGIC JOIN advdatafinal.gold.fct_sentiment_per_day s ON ...
# MAGIC JOIN advdatafinal.gold.fct_embedding_per_company e ON ...;
# MAGIC -- Identical to sql/05_gold_facts.sql.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Hand-off to 02_ml_pipeline.py
# MAGIC
# MAGIC At this point `advdatafinal.gold.fct_feature_panel_daily` is populated with
# MAGIC 25,080 rows and 30 features. The ML notebook reads from here.
# MAGIC
# MAGIC If running via a Databricks Job (workflow.yaml), the Job's downstream task is
# MAGIC `02_ml_pipeline.py` which triggers automatically on this notebook's completion.
