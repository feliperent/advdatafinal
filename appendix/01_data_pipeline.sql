-- ============================================================================
-- 01_data_pipeline.sql  | advdatafinal medallion DLT pipeline (pure SQL)
-- ============================================================================
-- Builds the four-schema medallion (raw / datos_masked / silver / gold) on Delta
-- tables. Pure SQL, mirroring the IN014 midterm shape exactly. Python steps
-- (FinBERT scoring, MiniLM embedding, PCA) live in 02_ml_pipeline.py.
--
-- Materialisation choices: streaming tables for everything except window-function
-- facts. The midterm encountered this same wall with gold.fct_complaints_daily
-- (ROW_NUMBER forced a materialised view).
--
-- Prerequisites: bronze/ uploaded to /Volumes/advdatafinal/raw/landing/ (see
-- docs/DATABRICKS_UPLOAD.md).

-- COMMAND ----------
-- ## 1. Catalog + schemas + landing volume

CREATE CATALOG IF NOT EXISTS advdatafinal;
CREATE SCHEMA  IF NOT EXISTS advdatafinal.raw;
CREATE SCHEMA  IF NOT EXISTS advdatafinal.datos_masked;
CREATE SCHEMA  IF NOT EXISTS advdatafinal.silver;
CREATE SCHEMA  IF NOT EXISTS advdatafinal.gold;
CREATE VOLUME  IF NOT EXISTS advdatafinal.raw.landing;

-- COMMAND ----------
-- ## 2. Raw layer | 8 streaming tables via Auto Loader

-- raw.prices_raw (yfinance OHLCV)
CREATE STREAMING TABLE advdatafinal.raw.prices_raw
AS SELECT
    cast(symbol     as STRING) as symbol,
    cast(trade_date as STRING) as trade_date,
    cast(open       as STRING) as open,
    cast(high       as STRING) as high,
    cast(low        as STRING) as low,
    cast(close      as STRING) as close,
    cast(adj_close  as STRING) as adj_close,
    cast(volume     as STRING) as volume,
    _metadata.file_path as _source_path,
    current_timestamp() as ingest_ts
FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/prices/', format => 'parquet');

-- COMMAND ----------

-- raw.income_statement_raw, balance_sheet_raw, cash_flow_raw (FMP, JSON)
CREATE STREAMING TABLE advdatafinal.raw.income_statement_raw
AS SELECT *, current_timestamp() as ingest_ts
FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/income_statement/', format => 'json');

CREATE STREAMING TABLE advdatafinal.raw.balance_sheet_raw
AS SELECT *, current_timestamp() as ingest_ts
FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/balance_sheet/', format => 'json');

CREATE STREAMING TABLE advdatafinal.raw.cash_flow_raw
AS SELECT *, current_timestamp() as ingest_ts
FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/cash_flow/', format => 'json');

-- COMMAND ----------

-- raw.news_raw, raw.press_raw (FMP, JSON)
CREATE STREAMING TABLE advdatafinal.raw.news_raw
AS SELECT *, current_timestamp() as ingest_ts
FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/news/', format => 'json');

CREATE STREAMING TABLE advdatafinal.raw.press_raw
AS SELECT *, current_timestamp() as ingest_ts
FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/press/', format => 'json');

-- COMMAND ----------

-- raw.sec_10k_raw, raw.sec_8k_raw (plain text)
-- Extract symbol + filing_date from the file path
CREATE STREAMING TABLE advdatafinal.raw.sec_10k_raw
AS SELECT
    regexp_extract(_metadata.file_path, '/sec_10k/([^/]+)/', 1) as symbol,
    regexp_extract(_metadata.file_path, '/([0-9]{4}-[0-9]{2}-[0-9]{2})\\.txt$', 1) as filing_date,
    value as body,
    _metadata.file_path as accession,
    current_timestamp() as ingest_ts
FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/sec_10k/', format => 'text', wholeText => true);

CREATE STREAMING TABLE advdatafinal.raw.sec_8k_raw
AS SELECT
    regexp_extract(_metadata.file_path, '/sec_8k/([^/]+)/', 1) as symbol,
    regexp_extract(_metadata.file_path, '/([0-9]{4}-[0-9]{2}-[0-9]{2})_', 1) as filing_date,
    value as body,
    _metadata.file_path as accession,
    current_timestamp() as ingest_ts
FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/sec_8k/', format => 'text', wholeText => true);

-- COMMAND ----------
-- ## 3. datos_masked layer | 4 PII redaction streaming tables (regex email + phone)

CREATE OR REFRESH STREAMING TABLE advdatafinal.datos_masked.news_redacted AS
SELECT
    cast(symbol as STRING) as symbol,
    cast(publishedDate as STRING) as published_at,
    cast(title as STRING) as title,
    cast(site as STRING) as site,
    cast(url as STRING) as url,
    regexp_replace(
        regexp_replace(
            coalesce(cast(text as STRING), ''),
            '[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}', '[EMAIL]'
        ),
        '\\+?1?[\\s.\\-]?\\(?\\d{3}\\)?[\\s.\\-]?\\d{3}[\\s.\\-]?\\d{4}', '[PHONE]'
    ) AS body_masked,
    ingest_ts
FROM STREAM(advdatafinal.raw.news_raw);

CREATE OR REFRESH STREAMING TABLE advdatafinal.datos_masked.press_redacted AS
SELECT
    cast(symbol as STRING) as symbol,
    cast(publishedDate as STRING) as published_at,
    cast(title as STRING) as title,
    regexp_replace(
        regexp_replace(
            coalesce(cast(text as STRING), ''),
            '[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}', '[EMAIL]'
        ),
        '\\+?1?[\\s.\\-]?\\(?\\d{3}\\)?[\\s.\\-]?\\d{3}[\\s.\\-]?\\d{4}', '[PHONE]'
    ) AS body_masked,
    ingest_ts
FROM STREAM(advdatafinal.raw.press_raw);

CREATE OR REFRESH STREAMING TABLE advdatafinal.datos_masked.filings_10k_redacted AS
SELECT
    symbol, filing_date, accession,
    regexp_replace(
        regexp_replace(coalesce(body, ''),
            '[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}', '[EMAIL]'),
        '\\+?1?[\\s.\\-]?\\(?\\d{3}\\)?[\\s.\\-]?\\d{3}[\\s.\\-]?\\d{4}', '[PHONE]'
    ) AS body_masked,
    ingest_ts
FROM STREAM(advdatafinal.raw.sec_10k_raw);

CREATE OR REFRESH STREAMING TABLE advdatafinal.datos_masked.filings_8k_redacted AS
SELECT
    symbol, filing_date, accession,
    regexp_replace(
        regexp_replace(coalesce(body, ''),
            '[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}', '[EMAIL]'),
        '\\+?1?[\\s.\\-]?\\(?\\d{3}\\)?[\\s.\\-]?\\d{3}[\\s.\\-]?\\d{4}', '[PHONE]'
    ) AS body_masked,
    ingest_ts
FROM STREAM(advdatafinal.raw.sec_8k_raw);

-- COMMAND ----------
-- ## 4. Silver layer | SCD-1 idempotency via APPLY CHANGES INTO + window-function materialised views

-- silver.silver_prices_cleaned -- typed prices (technical features in downstream MV)
CREATE TEMPORARY STREAMING LIVE VIEW prices_typed AS
SELECT
    symbol,
    cast(trade_date as DATE)         as trade_date,
    cast(open as DECIMAL(18,6))      as open_px,
    cast(high as DECIMAL(18,6))      as high_px,
    cast(low  as DECIMAL(18,6))      as low_px,
    cast(close as DECIMAL(18,6))     as close_px,
    cast(adj_close as DECIMAL(18,6)) as adj_close_px,
    cast(volume as BIGINT)           as volume
FROM STREAM(advdatafinal.raw.prices_raw)
WHERE close IS NOT NULL AND close <> '';

CREATE OR REFRESH STREAMING TABLE advdatafinal.silver.silver_prices_cleaned;

APPLY CHANGES INTO advdatafinal.silver.silver_prices_cleaned
  FROM STREAM(prices_typed)
  KEYS (symbol, trade_date)
  SEQUENCE BY trade_date
  STORED AS SCD TYPE 1;

-- COMMAND ----------

-- silver.silver_prices_features -- the 10 technical features (materialised view; window functions)
CREATE MATERIALIZED VIEW advdatafinal.silver.silver_prices_features AS
WITH base AS (
    SELECT *,
        LN(close_px / NULLIF(LAG(close_px, 1) OVER (PARTITION BY symbol ORDER BY trade_date), 0))
            AS log_ret_1d
    FROM advdatafinal.silver.silver_prices_cleaned
),
features AS (
    SELECT *,
        AVG(close_px) OVER w5  AS sma_5,
        AVG(close_px) OVER w20 AS sma_20,
        AVG(close_px) OVER w50 AS sma_50,
        AVG(close_px) OVER w12 AS ema_12,
        AVG(close_px) OVER w26 AS ema_26,
        STDDEV_POP(close_px)   OVER w20 AS sd_close_20,
        STDDEV_POP(log_ret_1d) OVER w20 AS sd_logret_20
    FROM base
    WINDOW
        w5  AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN  4 PRECEDING AND CURRENT ROW),
        w20 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
        w50 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW),
        w12 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW),
        w26 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 25 PRECEDING AND CURRENT ROW)
)
SELECT
    md5(symbol || '|' || trade_date::STRING) AS price_key,
    md5(trade_date::STRING)                  AS date_key,
    md5(LOWER(TRIM(symbol)))                 AS company_key,
    symbol, trade_date,
    open_px, high_px, low_px, close_px, adj_close_px, volume,
    log_ret_1d, sma_5, sma_20, sma_50, ema_12, ema_26,
    (ema_12 - ema_26) AS macd_hist,
    CASE WHEN sd_close_20 > 0 THEN (close_px - sma_20) / sd_close_20 ELSE 0 END AS bb_z,
    sd_logret_20 * SQRT(252) AS vol_20d,
    trade_date AS as_of_date
FROM features;

-- COMMAND ----------

-- silver.silver_fundamentals_cleaned -- joins 3 raw statement tables + 4-quarter TTM rollups
CREATE MATERIALIZED VIEW advdatafinal.silver.silver_fundamentals_cleaned AS
WITH inc AS (
    SELECT
        symbol, cast(date as DATE) AS filing_date,
        cast(revenue as DECIMAL(20,2))         AS revenue,
        cast(grossProfit as DECIMAL(20,2))     AS gross_profit,
        cast(operatingIncome as DECIMAL(20,2)) AS operating_income,
        cast(netIncome as DECIMAL(20,2))       AS net_income,
        cast(ebitda as DECIMAL(20,2))          AS ebitda
    FROM advdatafinal.raw.income_statement_raw
),
bs AS (
    SELECT
        symbol, cast(date as DATE) AS filing_date,
        cast(totalAssets as DECIMAL(20,2))              AS total_assets,
        cast(totalLiabilities as DECIMAL(20,2))         AS total_liabilities,
        cast(totalStockholdersEquity as DECIMAL(20,2))  AS total_equity,
        cast(totalDebt as DECIMAL(20,2))                AS total_debt,
        cast(cashAndCashEquivalents as DECIMAL(20,2))   AS cash
    FROM advdatafinal.raw.balance_sheet_raw
),
cf AS (
    SELECT
        symbol, cast(date as DATE) AS filing_date,
        cast(freeCashFlow as DECIMAL(20,2))      AS free_cash_flow,
        cast(operatingCashFlow as DECIMAL(20,2)) AS operating_cash_flow
    FROM advdatafinal.raw.cash_flow_raw
),
joined AS (
    SELECT inc.symbol, inc.filing_date,
           inc.revenue, inc.gross_profit, inc.operating_income, inc.net_income, inc.ebitda,
           bs.total_assets, bs.total_liabilities, bs.total_equity, bs.total_debt, bs.cash,
           cf.free_cash_flow, cf.operating_cash_flow
    FROM inc
    LEFT JOIN bs USING (symbol, filing_date)
    LEFT JOIN cf USING (symbol, filing_date)
),
ttm AS (
    SELECT symbol, filing_date,
        SUM(revenue)          OVER w AS revenue_ttm,
        SUM(gross_profit)     OVER w AS gross_profit_ttm,
        SUM(operating_income) OVER w AS operating_income_ttm,
        SUM(net_income)       OVER w AS net_income_ttm,
        SUM(ebitda)           OVER w AS ebitda_ttm,
        SUM(free_cash_flow)   OVER w AS free_cash_flow_ttm,
        AVG(total_assets)     OVER w AS avg_assets_ttm,
        AVG(total_equity)     OVER w AS avg_equity_ttm,
        total_assets, total_liabilities, total_equity, total_debt, cash
    FROM joined
    WINDOW w AS (PARTITION BY symbol ORDER BY filing_date ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
)
SELECT
    md5(symbol || '|' || filing_date::STRING) AS fund_key,
    md5(LOWER(TRIM(symbol)))                  AS company_key,
    symbol, filing_date,
    revenue_ttm, gross_profit_ttm, operating_income_ttm, net_income_ttm, ebitda_ttm,
    free_cash_flow_ttm, avg_assets_ttm, avg_equity_ttm,
    total_assets, total_liabilities, total_equity, total_debt, cash,
    CASE WHEN avg_equity_ttm > 0 THEN net_income_ttm / avg_equity_ttm END AS roe,
    CASE WHEN avg_assets_ttm > 0 THEN net_income_ttm / avg_assets_ttm END AS roa,
    CASE WHEN total_equity > 0   THEN total_debt / total_equity END        AS debt_eq,
    CASE WHEN revenue_ttm > 0    THEN gross_profit_ttm / revenue_ttm END     AS gross_margin,
    CASE WHEN revenue_ttm > 0    THEN operating_income_ttm / revenue_ttm END AS op_margin,
    CASE WHEN avg_assets_ttm > 0 THEN revenue_ttm / avg_assets_ttm END       AS asset_turnover,
    filing_date AS as_of_date
FROM ttm
WHERE revenue_ttm IS NOT NULL;

-- COMMAND ----------
-- ## 5. Gold layer | dimensions (streaming) + intermediate facts (materialised)

-- gold.dim_date -- calendar covering the project window
CREATE OR REFRESH STREAMING TABLE advdatafinal.gold.dim_date AS
SELECT
    md5(cast(full_date as STRING)) AS date_key,
    full_date,
    YEAR(full_date)       AS year,
    MONTH(full_date)      AS month,
    DAY(full_date)        AS day,
    WEEKOFYEAR(full_date) AS week,
    DAYOFWEEK(full_date)  AS day_of_week,
    (DAYOFWEEK(full_date) BETWEEN 2 AND 6) AS is_trading_day  -- Spark DOW: 1=Sun,7=Sat
FROM (
    SELECT explode(sequence(DATE '2021-01-01', DATE '2025-12-31', INTERVAL 1 DAY)) AS full_date
);

-- gold.dim_sector
CREATE OR REFRESH STREAMING TABLE advdatafinal.gold.dim_sector AS
SELECT md5(LOWER(TRIM(sector_name))) AS sector_key, sector_name
FROM (VALUES
    ('Technology'), ('Financials'), ('Healthcare'), ('Industrials'), ('Consumer')
) AS s(sector_name);

-- gold.dim_company -- 20-stock universe (same pattern as midterm gold.dim_area)
CREATE OR REFRESH STREAMING TABLE advdatafinal.gold.dim_company AS
WITH symbols AS (
    SELECT DISTINCT symbol FROM STREAM(advdatafinal.silver.silver_prices_cleaned)
),
mapping AS (
    SELECT * FROM (VALUES
        ('AAPL','Apple','Technology'),  ('MSFT','Microsoft','Technology'),
        ('GOOGL','Alphabet','Technology'),('NVDA','NVIDIA','Technology'),
        ('JPM','JPMorgan Chase','Financials'),('BAC','Bank of America','Financials'),
        ('GS','Goldman Sachs','Financials'),('AXP','American Express','Financials'),
        ('JNJ','Johnson & Johnson','Healthcare'),('UNH','UnitedHealth','Healthcare'),
        ('PFE','Pfizer','Healthcare'),('LLY','Eli Lilly','Healthcare'),
        ('BA','Boeing','Industrials'),('CAT','Caterpillar','Industrials'),
        ('HON','Honeywell','Industrials'),('GE','GE Aerospace','Industrials'),
        ('AMZN','Amazon','Consumer'),('WMT','Walmart','Consumer'),
        ('KO','Coca-Cola','Consumer'),('NKE','Nike','Consumer')
    ) AS m(symbol, name, sector_name)
)
SELECT
    md5(LOWER(TRIM(m.symbol)))      AS company_key,
    m.symbol, m.name,
    md5(LOWER(TRIM(m.sector_name))) AS sector_key,
    m.sector_name
FROM mapping m
WHERE m.symbol IN (SELECT symbol FROM symbols);

-- gold.dim_filing_type
CREATE OR REFRESH STREAMING TABLE advdatafinal.gold.dim_filing_type AS
SELECT md5(LOWER(TRIM(code))) AS filing_type_key, code, label
FROM (VALUES
    ('10K',   '10-K Annual Report'),
    ('8K',    '8-K Material Event'),
    ('NEWS',  'News Article'),
    ('PRESS', 'Press Release')
) AS t(code, label);

-- COMMAND ----------
-- ## 6. Hand-off note
--
-- At this point the DLT pipeline has built:
--   advdatafinal.raw.*                 (8 streaming tables)
--   advdatafinal.datos_masked.*        (4 streaming tables with PII masking)
--   advdatafinal.silver.silver_prices_cleaned + silver_prices_features
--   advdatafinal.silver.silver_fundamentals_cleaned
--   advdatafinal.gold.dim_date, dim_sector, dim_company, dim_filing_type
--
-- The remaining objects depend on Python ML steps and live in 02_ml_pipeline.py:
--   silver.silver_news_scored                 (FinBERT)
--   silver.silver_press_scored                (FinBERT)
--   silver.silver_filings_10k_chunked         (chunk + MiniLM embeddings)
--   silver.silver_filings_8k_chunked          (same)
--   gold.dim_chunk                            (union of 10K + 8K chunks)
--   gold.fct_sentiment_per_day                (aggregation; Spark SQL)
--   gold.fct_embedding_per_company            (sklearn PCA)
--   gold.fct_feature_panel_daily              (the 30-feature ML training table)
--   gold.fct_predictions                      (3 rungs, written by training loop)
--   gold.fct_backtest_pnl_daily               (weekly walk-forward)
--   gold.fct_rag_queries                      (audit; appended by rag/answer.py)
