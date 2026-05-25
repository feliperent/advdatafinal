
-- 03_silver_schema.sql  -- Silver layer (6 tables, one-tier midterm pattern)

-- Naming: silver.silver_<source>_<state> (matches midterm silver.silver_quejas_cleaned).
-- Each silver table is denormalised: every row carries both the hashed dim keys
-- (md5 stored as text) AND the dim attribute names, so analysts can query silver
-- directly without joining to gold.

-- -- 1) silver.silver_prices_cleaned --------------------------------
-- 25,100 rows. OHLCV + log return + 10 technical features (LAG/AVG OVER lookback).
-- RSI-14 is filled by a Python post-hook (Wilder smoothing is awkward in pure SQL).

CREATE TABLE silver.silver_prices_cleaned AS
WITH typed AS (
    SELECT
        symbol,
        trade_date::date                     AS trade_date,
        NULLIF(open, '')::numeric(18,6)      AS open_px,
        NULLIF(high, '')::numeric(18,6)      AS high_px,
        NULLIF(low, '')::numeric(18,6)       AS low_px,
        NULLIF(close, '')::numeric(18,6)     AS close_px,
        NULLIF(adj_close, '')::numeric(18,6) AS adj_close_px,
        NULLIF(volume, '')::bigint           AS volume
    FROM raw.prices_raw
    WHERE close IS NOT NULL AND close <> ''
),
with_returns AS (
    SELECT *,
           LN(close_px / NULLIF(LAG(close_px, 1) OVER (PARTITION BY symbol ORDER BY trade_date), 0))
             AS log_ret_1d
    FROM typed
),
with_features AS (
    SELECT *,
           AVG(close_px) OVER w5   AS sma_5,
           AVG(close_px) OVER w20  AS sma_20,
           AVG(close_px) OVER w50  AS sma_50,
           AVG(close_px) OVER w12  AS ema_12_approx,
           AVG(close_px) OVER w26  AS ema_26_approx,
           STDDEV_POP(close_px)    OVER w20 AS sd_close_20,
           STDDEV_POP(log_ret_1d)  OVER w20 AS sd_logret_20
    FROM with_returns
    WINDOW
        w5  AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN  4 PRECEDING AND CURRENT ROW),
        w20 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
        w50 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW),
        w12 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW),
        w26 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 25 PRECEDING AND CURRENT ROW)
)
SELECT
    MD5(symbol || '|' || trade_date::text)             AS price_key,
    MD5(trade_date::text)                              AS date_key,
    MD5(LOWER(TRIM(symbol)))                           AS company_key,
    symbol, trade_date,
    open_px, high_px, low_px, close_px, adj_close_px, volume,
    log_ret_1d, sma_5, sma_20, sma_50,
    ema_12_approx AS ema_12,
    ema_26_approx AS ema_26,
    (ema_12_approx - ema_26_approx) AS macd_hist,
    CASE WHEN sd_close_20 > 0 THEN (close_px - sma_20) / sd_close_20 ELSE 0 END AS bb_z,
    NULL::numeric(18,4) AS rsi_14,  -- post-hook: silver_text/post_rsi.py
    sd_logret_20 * SQRT(252) AS vol_20d,
    trade_date AS as_of_date
FROM with_features;

CREATE INDEX ON silver.silver_prices_cleaned (date_key, company_key);
CREATE INDEX ON silver.silver_prices_cleaned (company_key);
CREATE INDEX ON silver.silver_prices_cleaned (trade_date);

-- -- 2) silver.silver_fundamentals_cleaned -------------------------
-- 400 rows. Joins 3 FMP statement tables + 4-quarter TTM rollups + ratios.

CREATE TABLE silver.silver_fundamentals_cleaned AS
WITH inc AS (
    SELECT symbol, date::date AS filing_date,
           NULLIF(revenue, '')::numeric(20,2)         AS revenue,
           NULLIF(grossprofit, '')::numeric(20,2)     AS gross_profit,
           NULLIF(operatingincome, '')::numeric(20,2) AS operating_income,
           NULLIF(netincome, '')::numeric(20,2)       AS net_income,
           NULLIF(ebitda, '')::numeric(20,2)          AS ebitda
    FROM raw.income_statement_raw
),
bs AS (
    SELECT symbol, date::date AS filing_date,
           NULLIF(totalassets, '')::numeric(20,2)              AS total_assets,
           NULLIF(totalliabilities, '')::numeric(20,2)         AS total_liabilities,
           NULLIF(totalstockholdersequity, '')::numeric(20,2)  AS total_equity,
           NULLIF(totaldebt, '')::numeric(20,2)                AS total_debt,
           NULLIF(cashandcashequivalents, '')::numeric(20,2)   AS cash
    FROM raw.balance_sheet_raw
),
cf AS (
    SELECT symbol, date::date AS filing_date,
           NULLIF(freecashflow, '')::numeric(20,2)     AS free_cash_flow,
           NULLIF(operatingcashflow, '')::numeric(20,2) AS operating_cash_flow
    FROM raw.cash_flow_raw
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
           SUM(revenue)             OVER w AS revenue_ttm,
           SUM(gross_profit)        OVER w AS gross_profit_ttm,
           SUM(operating_income)    OVER w AS operating_income_ttm,
           SUM(net_income)          OVER w AS net_income_ttm,
           SUM(ebitda)              OVER w AS ebitda_ttm,
           SUM(free_cash_flow)      OVER w AS free_cash_flow_ttm,
           SUM(operating_cash_flow) OVER w AS operating_cash_flow_ttm,
           AVG(total_assets)        OVER w AS avg_assets_ttm,
           AVG(total_equity)        OVER w AS avg_equity_ttm,
           total_assets, total_liabilities, total_equity, total_debt, cash
    FROM joined
    WINDOW w AS (PARTITION BY symbol ORDER BY filing_date ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
)
SELECT
    MD5(symbol || '|' || filing_date::text) AS fund_key,
    MD5(LOWER(TRIM(symbol)))                AS company_key,
    symbol, filing_date,
    revenue_ttm, gross_profit_ttm, operating_income_ttm, net_income_ttm, ebitda_ttm,
    free_cash_flow_ttm, operating_cash_flow_ttm, avg_assets_ttm, avg_equity_ttm,
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

-- -- 3) silver.silver_news_scored (Python-built via silver_text/finbert_score.py)
-- 3,853 rows. FinBERT-tone scalar = P(Positive) - P(Negative) in [-1, +1].
-- IMPORTANT: id2label is {0:Neutral, 1:Positive, 2:Negative}, NOT the more
-- intuitive {0:Pos, 1:Neg, 2:Neu}. An earlier version inverted the formula
-- and was caught by the Phase 3a code review.

CREATE TABLE silver.silver_news_scored (
  article_id     text PRIMARY KEY,
  symbol         text NOT NULL,
  company_key    text NOT NULL,
  published_at   text,
  trade_date     date,
  finbert_score  numeric(8,5)
);

-- -- 4) silver.silver_press_scored (same shape, 563 rows) -----------
CREATE TABLE silver.silver_press_scored (
  press_id       text PRIMARY KEY,
  symbol         text NOT NULL,
  company_key    text NOT NULL,
  published_at   text,
  trade_date     date,
  finbert_score  numeric(8,5)
);

-- -- 5) silver.silver_filings_10k_chunked (Python-built via silver_text/build_filing_chunks.py)
-- 3,040 rows = 95 filings x ~32 chunks each. 500-token chunks, 50-token overlap.
-- Embeddings stored as bytea (384 float32 = 1,536 bytes). NumPy cosine in Python at retrieve time.

CREATE TABLE silver.silver_filings_10k_chunked (
  chunk_key    text PRIMARY KEY,    -- 10K-{symbol}-{filing_date}-c{idx:02d}
  accession    text NOT NULL,
  symbol       text NOT NULL,
  company_key  text NOT NULL,
  filing_date  date NOT NULL,
  chunk_index  smallint NOT NULL,
  n_tokens     smallint NOT NULL,
  body_chunk   text NOT NULL,
  embedding    bytea NOT NULL,
  as_of_date   date NOT NULL
);
CREATE INDEX ON silver.silver_filings_10k_chunked (company_key, filing_date);

-- -- 6) silver.silver_filings_8k_chunked (same shape, 10,807 rows) -
-- chunk_key includes the last 6 digits of accession to disambiguate companies
-- that file multiple 8-Ks on the same date (e.g. JPM, AXP, WMT).

CREATE TABLE silver.silver_filings_8k_chunked (
  chunk_key    text PRIMARY KEY,    -- 8K-{symbol}-{filing_date}-{acc_tail}-c{idx:02d}
  accession    text NOT NULL,
  symbol       text NOT NULL,
  company_key  text NOT NULL,
  filing_date  date NOT NULL,
  chunk_index  smallint NOT NULL,
  n_tokens     smallint NOT NULL,
  body_chunk   text NOT NULL,
  embedding    bytea NOT NULL,
  as_of_date   date NOT NULL
);
CREATE INDEX ON silver.silver_filings_8k_chunked (company_key, filing_date);
