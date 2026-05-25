-- Databricks notebook source

-- raw.prices_raw (yfinance OHLCV)
CREATE OR REFRESH STREAMING TABLE advdatafinal.raw.prices_raw AS
SELECT
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

-- raw.income_statement_raw (FMP /stable/income-statement)
CREATE OR REFRESH STREAMING TABLE advdatafinal.raw.income_statement_raw AS
SELECT *, current_timestamp() as ingest_ts
FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/income_statement/', format => 'json', multiLine => 'true');

-- raw.balance_sheet_raw (FMP /stable/balance-sheet-statement)
CREATE OR REFRESH STREAMING TABLE advdatafinal.raw.balance_sheet_raw AS
SELECT *, current_timestamp() as ingest_ts
FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/balance_sheet/', format => 'json', multiLine => 'true');

-- raw.cash_flow_raw (FMP /stable/cash-flow-statement)
CREATE OR REFRESH STREAMING TABLE advdatafinal.raw.cash_flow_raw AS
SELECT *, current_timestamp() as ingest_ts
FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/cash_flow/', format => 'json', multiLine => 'true');

-- raw.news_raw (FMP /stable/news/stock)
CREATE OR REFRESH STREAMING TABLE advdatafinal.raw.news_raw AS
SELECT *, current_timestamp() as ingest_ts
FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/news/', format => 'json', multiLine => 'true');

-- raw.press_raw (FMP /stable/news/press-releases)
CREATE OR REFRESH STREAMING TABLE advdatafinal.raw.press_raw AS
SELECT *, current_timestamp() as ingest_ts
FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/press/', format => 'json', multiLine => 'true');

-- raw.sec_10k_raw (SEC EDGAR 10-K Item 1A)
CREATE OR REFRESH STREAMING TABLE advdatafinal.raw.sec_10k_raw AS
SELECT
    regexp_extract(_metadata.file_path, '/sec_10k/([^/]+)/', 1) as symbol,
    regexp_extract(_metadata.file_path, '/([0-9]{4}-[0-9]{2}-[0-9]{2})\\.txt$', 1) as filing_date,
    value as body,
    _metadata.file_path as accession,
    current_timestamp() as ingest_ts
FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/sec_10k/', format => 'text', wholetext => 'true');

-- raw.sec_8k_raw (SEC EDGAR 8-K events)
CREATE OR REFRESH STREAMING TABLE advdatafinal.raw.sec_8k_raw AS
SELECT
    regexp_extract(_metadata.file_path, '/sec_8k/([^/]+)/', 1) as symbol,
    regexp_extract(_metadata.file_path, '/([0-9]{4}-[0-9]{2}-[0-9]{2})_', 1) as filing_date,
    value as body,
    _metadata.file_path as accession,
    current_timestamp() as ingest_ts
FROM STREAM read_files('/Volumes/advdatafinal/raw/landing/sec_8k/', format => 'text', wholetext => 'true');

-- datos_masked.news_redacted (regex mask email + US phone in news bodies)
CREATE OR REFRESH STREAMING TABLE advdatafinal.datos_masked.news_redacted AS
SELECT
    cast(symbol as STRING) as symbol,
    cast(publishedDate as STRING) as published_at,
    cast(title as STRING) as title,
    cast(site as STRING) as site,
    cast(url as STRING) as url,
    regexp_replace(
        regexp_replace(coalesce(cast(text as STRING), ''),
            '[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}', '[EMAIL]'),
        '\\+?1?[\\s.\\-]?\\(?\\d{3}\\)?[\\s.\\-]?\\d{3}[\\s.\\-]?\\d{4}', '[PHONE]'
    ) as body_masked,
    ingest_ts
FROM STREAM(advdatafinal.raw.news_raw);

-- datos_masked.press_redacted (regex mask email + US phone in press bodies)
CREATE OR REFRESH STREAMING TABLE advdatafinal.datos_masked.press_redacted AS
SELECT
    cast(symbol as STRING) as symbol,
    cast(publishedDate as STRING) as published_at,
    cast(title as STRING) as title,
    regexp_replace(
        regexp_replace(coalesce(cast(text as STRING), ''),
            '[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}', '[EMAIL]'),
        '\\+?1?[\\s.\\-]?\\(?\\d{3}\\)?[\\s.\\-]?\\d{3}[\\s.\\-]?\\d{4}', '[PHONE]'
    ) as body_masked,
    ingest_ts
FROM STREAM(advdatafinal.raw.press_raw);

-- datos_masked.filings_10k_redacted (regex mask email + US phone in 10-K bodies)
CREATE OR REFRESH STREAMING TABLE advdatafinal.datos_masked.filings_10k_redacted AS
SELECT
    symbol, filing_date, accession,
    regexp_replace(
        regexp_replace(coalesce(body, ''),
            '[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}', '[EMAIL]'),
        '\\+?1?[\\s.\\-]?\\(?\\d{3}\\)?[\\s.\\-]?\\d{3}[\\s.\\-]?\\d{4}', '[PHONE]'
    ) as body_masked,
    ingest_ts
FROM STREAM(advdatafinal.raw.sec_10k_raw);

-- datos_masked.filings_8k_redacted (regex mask email + US phone in 8-K bodies)
CREATE OR REFRESH STREAMING TABLE advdatafinal.datos_masked.filings_8k_redacted AS
SELECT
    symbol, filing_date, accession,
    regexp_replace(
        regexp_replace(coalesce(body, ''),
            '[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}', '[EMAIL]'),
        '\\+?1?[\\s.\\-]?\\(?\\d{3}\\)?[\\s.\\-]?\\d{3}[\\s.\\-]?\\d{4}', '[PHONE]'
    ) as body_masked,
    ingest_ts
FROM STREAM(advdatafinal.raw.sec_8k_raw);

-- silver.prices_typed (intermediate view: cast prices to typed columns)
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

-- silver.silver_prices_cleaned (SCD-1 idempotent typed prices)
CREATE OR REFRESH STREAMING TABLE advdatafinal.silver.silver_prices_cleaned;

APPLY CHANGES INTO advdatafinal.silver.silver_prices_cleaned
  FROM STREAM(prices_typed)
  KEYS (symbol, trade_date)
  SEQUENCE BY trade_date
  STORED AS SCD TYPE 1;

-- silver.silver_prices_features (10 technical features via window functions)
CREATE OR REFRESH MATERIALIZED VIEW advdatafinal.silver.silver_prices_features AS
WITH base AS (
    SELECT *,
        LN(close_px / NULLIF(LAG(close_px, 1) OVER (PARTITION BY symbol ORDER BY trade_date), 0)) as log_ret_1d
    FROM advdatafinal.silver.silver_prices_cleaned
),
features AS (
    SELECT *,
        AVG(close_px) OVER w5  as sma_5,
        AVG(close_px) OVER w20 as sma_20,
        AVG(close_px) OVER w50 as sma_50,
        AVG(close_px) OVER w12 as ema_12,
        AVG(close_px) OVER w26 as ema_26,
        STDDEV_POP(close_px)   OVER w20 as sd_close_20,
        STDDEV_POP(log_ret_1d) OVER w20 as sd_logret_20
    FROM base
    WINDOW
        w5  AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN  4 PRECEDING AND CURRENT ROW),
        w20 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
        w50 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW),
        w12 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW),
        w26 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 25 PRECEDING AND CURRENT ROW)
)
SELECT
    md5(symbol || '|' || cast(trade_date as STRING)) as price_key,
    md5(cast(trade_date as STRING))                  as date_key,
    md5(LOWER(TRIM(symbol)))                         as company_key,
    symbol, trade_date,
    open_px, high_px, low_px, close_px, adj_close_px, volume,
    log_ret_1d, sma_5, sma_20, sma_50, ema_12, ema_26,
    (ema_12 - ema_26) as macd_hist,
    CASE WHEN sd_close_20 > 0 THEN (close_px - sma_20) / sd_close_20 ELSE 0 END as bb_z,
    sd_logret_20 * SQRT(252) as vol_20d,
    trade_date as as_of_date
FROM features;

-- silver.silver_fundamentals_cleaned (joins 3 raw statements + 4-quarter TTM rollups)
CREATE OR REFRESH MATERIALIZED VIEW advdatafinal.silver.silver_fundamentals_cleaned AS
WITH inc AS (
    SELECT
        symbol, cast(date as DATE) as filing_date,
        cast(revenue as DECIMAL(20,2))         as revenue,
        cast(grossProfit as DECIMAL(20,2))     as gross_profit,
        cast(operatingIncome as DECIMAL(20,2)) as operating_income,
        cast(netIncome as DECIMAL(20,2))       as net_income,
        cast(ebitda as DECIMAL(20,2))          as ebitda
    FROM advdatafinal.raw.income_statement_raw
),
bs AS (
    SELECT
        symbol, cast(date as DATE) as filing_date,
        cast(totalAssets as DECIMAL(20,2))              as total_assets,
        cast(totalLiabilities as DECIMAL(20,2))         as total_liabilities,
        cast(totalStockholdersEquity as DECIMAL(20,2))  as total_equity,
        cast(totalDebt as DECIMAL(20,2))                as total_debt,
        cast(cashAndCashEquivalents as DECIMAL(20,2))   as cash
    FROM advdatafinal.raw.balance_sheet_raw
),
cf AS (
    SELECT
        symbol, cast(date as DATE) as filing_date,
        cast(freeCashFlow as DECIMAL(20,2))      as free_cash_flow,
        cast(operatingCashFlow as DECIMAL(20,2)) as operating_cash_flow
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
        SUM(revenue)          OVER w as revenue_ttm,
        SUM(gross_profit)     OVER w as gross_profit_ttm,
        SUM(operating_income) OVER w as operating_income_ttm,
        SUM(net_income)       OVER w as net_income_ttm,
        SUM(ebitda)           OVER w as ebitda_ttm,
        SUM(free_cash_flow)   OVER w as free_cash_flow_ttm,
        AVG(total_assets)     OVER w as avg_assets_ttm,
        AVG(total_equity)     OVER w as avg_equity_ttm,
        total_assets, total_liabilities, total_equity, total_debt, cash
    FROM joined
    WINDOW w AS (PARTITION BY symbol ORDER BY filing_date ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
)
SELECT
    md5(symbol || '|' || cast(filing_date as STRING)) as fund_key,
    md5(LOWER(TRIM(symbol)))                          as company_key,
    symbol, filing_date,
    revenue_ttm, gross_profit_ttm, operating_income_ttm, net_income_ttm, ebitda_ttm,
    free_cash_flow_ttm, avg_assets_ttm, avg_equity_ttm,
    total_assets, total_liabilities, total_equity, total_debt, cash,
    CASE WHEN avg_equity_ttm > 0 THEN net_income_ttm / avg_equity_ttm END as roe,
    CASE WHEN avg_assets_ttm > 0 THEN net_income_ttm / avg_assets_ttm END as roa,
    CASE WHEN total_equity > 0   THEN total_debt / total_equity END        as debt_eq,
    CASE WHEN revenue_ttm > 0    THEN gross_profit_ttm / revenue_ttm END     as gross_margin,
    CASE WHEN revenue_ttm > 0    THEN operating_income_ttm / revenue_ttm END as op_margin,
    CASE WHEN avg_assets_ttm > 0 THEN revenue_ttm / avg_assets_ttm END       as asset_turnover,
    filing_date as as_of_date
FROM ttm
WHERE revenue_ttm IS NOT NULL;

-- gold.dim_date (calendar 2021-2025 with is_trading_day flag)
CREATE OR REFRESH MATERIALIZED VIEW advdatafinal.gold.dim_date AS
SELECT
    md5(cast(full_date as STRING)) as date_key,
    full_date,
    YEAR(full_date)                              as year,
    MONTH(full_date)                             as month,
    DAY(full_date)                               as day,
    WEEKOFYEAR(full_date)                        as week,
    DAYOFWEEK(full_date)                         as day_of_week,
    (DAYOFWEEK(full_date) BETWEEN 2 AND 6)       as is_trading_day
FROM (
    SELECT explode(sequence(DATE '2021-01-01', DATE '2025-12-31', INTERVAL 1 DAY)) as full_date
);

-- gold.dim_sector (5 sector codes hashed)
CREATE OR REFRESH MATERIALIZED VIEW advdatafinal.gold.dim_sector AS
SELECT md5(LOWER(TRIM(sector_name))) as sector_key, sector_name
FROM (VALUES
    ('Technology'), ('Financials'), ('Healthcare'), ('Industrials'), ('Consumer')
) AS s(sector_name);

-- gold.dim_company (20 stocks across 5 sectors)
CREATE OR REFRESH MATERIALIZED VIEW advdatafinal.gold.dim_company AS
WITH symbols AS (
    SELECT DISTINCT symbol FROM advdatafinal.silver.silver_prices_cleaned
),
mapping AS (
    SELECT * FROM (VALUES
        ('AAPL','Apple','Technology'),('MSFT','Microsoft','Technology'),
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
    md5(LOWER(TRIM(m.symbol)))      as company_key,
    m.symbol, m.name,
    md5(LOWER(TRIM(m.sector_name))) as sector_key,
    m.sector_name
FROM mapping m
WHERE m.symbol IN (SELECT symbol FROM symbols);

-- gold.dim_filing_type (4 source codes: 10K, 8K, NEWS, PRESS)
CREATE OR REFRESH MATERIALIZED VIEW advdatafinal.gold.dim_filing_type AS
SELECT md5(LOWER(TRIM(code))) as filing_type_key, code, label
FROM (VALUES
    ('10K',   '10-K Annual Report'),
    ('8K',    '8-K Material Event'),
    ('NEWS',  'News Article'),
    ('PRESS', 'Press Release')
) AS t(code, label);
