-- ============================================================================
-- 05_gold_facts.sql  -- Gold fact tables (5 facts: 2 intermediate + 3 main + audit)
-- ============================================================================
-- Same shape as the midterm's gold.fct_complaints_daily:
--   - bigint surrogate PK via ROW_NUMBER() OVER (...) + 1000
--   - real FK constraints to every referenced dimension
--   - btree indexes on (date_key, company_key)
-- Window-function-heavy facts must be CREATE TABLE AS (materialised) because
-- streaming-table semantics would force a full recompute (the cost lesson from
-- the team's IN014 midterm with ROW_NUMBER in gold.fct_complaints_daily).

-- ============================================================================
-- INTERMEDIATE FACT 1: sentiment per (date, company)
-- ============================================================================
-- 26,080 rows = 20 stocks x 1,304 trading days. 5 sentiment features per row.
-- finbert_*_30d uses a trailing 30-day window over silver_news/press_scored.
-- n_8k_30d counts unique 8-K filings in the trailing 30 days from raw.sec_8k_raw.

CREATE TABLE gold.fct_sentiment_per_day AS
WITH date_grid AS (
    SELECT d.date_key, d.full_date, c.company_key, c.symbol
    FROM gold.dim_date d
    CROSS JOIN gold.dim_company c
    WHERE d.is_trading_day = true
),
news_per_day AS (
    SELECT company_key, trade_date,
           AVG(finbert_score) AS news_mean_day,
           COUNT(*)::int      AS n_news_day
    FROM silver.silver_news_scored
    WHERE trade_date IS NOT NULL
    GROUP BY company_key, trade_date
),
press_per_day AS (
    SELECT company_key, trade_date,
           AVG(finbert_score) AS press_mean_day
    FROM silver.silver_press_scored
    WHERE trade_date IS NOT NULL
    GROUP BY company_key, trade_date
),
filings_per_day AS (
    SELECT md5(LOWER(TRIM(symbol))) AS company_key,
           filing_date::date        AS filing_date,
           COUNT(DISTINCT accession)::int AS n_8k_on_day
    FROM raw.sec_8k_raw
    WHERE filing_date IS NOT NULL AND filing_date <> ''
    GROUP BY symbol, filing_date::date
),
joined AS (
    SELECT g.date_key, g.full_date, g.company_key, g.symbol,
           COALESCE(n.news_mean_day, 0::numeric)  AS news_mean_day,
           COALESCE(n.n_news_day, 0)              AS n_news_day,
           COALESCE(p.press_mean_day, 0::numeric) AS press_mean_day,
           COALESCE(fk.n_8k_on_day, 0)            AS n_8k_on_day
    FROM date_grid g
    LEFT JOIN news_per_day  n ON g.company_key = n.company_key AND g.full_date = n.trade_date
    LEFT JOIN press_per_day p ON g.company_key = p.company_key AND g.full_date = p.trade_date
    LEFT JOIN filings_per_day fk ON g.company_key = fk.company_key AND g.full_date = fk.filing_date
),
windowed AS (
    SELECT date_key, full_date, company_key, symbol,
           AVG(NULLIF(news_mean_day, 0))  OVER w3   AS finbert_news_mean_3d,
           AVG(NULLIF(news_mean_day, 0))  OVER w30  AS finbert_news_mean_30d,
           AVG(NULLIF(press_mean_day, 0)) OVER w30  AS finbert_press_30d,
           COALESCE(SUM(n_news_day) OVER w3, 0)     AS n_news_3d,
           COALESCE(SUM(n_8k_on_day) OVER w30, 0)   AS n_8k_30d
    FROM joined
    WINDOW
      w3  AS (PARTITION BY company_key ORDER BY full_date ROWS BETWEEN  2 PRECEDING AND CURRENT ROW),
      w30 AS (PARTITION BY company_key ORDER BY full_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
)
SELECT
    ROW_NUMBER() OVER (ORDER BY full_date, company_key) + 1000 AS fact_sentiment_key,
    date_key, company_key,
    COALESCE(finbert_news_mean_3d,  0::numeric) AS finbert_news_mean_3d,
    COALESCE(finbert_news_mean_30d, 0::numeric) AS finbert_news_mean_30d,
    COALESCE(finbert_press_30d,     0::numeric) AS finbert_press_30d,
    n_news_3d, n_8k_30d,
    full_date AS as_of_date
FROM windowed;

ALTER TABLE gold.fct_sentiment_per_day ADD PRIMARY KEY (fact_sentiment_key);
ALTER TABLE gold.fct_sentiment_per_day ADD CONSTRAINT fk_fct_sent_date
  FOREIGN KEY (date_key) REFERENCES gold.dim_date(date_key);
ALTER TABLE gold.fct_sentiment_per_day ADD CONSTRAINT fk_fct_sent_company
  FOREIGN KEY (company_key) REFERENCES gold.dim_company(company_key);
CREATE INDEX ON gold.fct_sentiment_per_day (date_key, company_key);

-- ============================================================================
-- INTERMEDIATE FACT 2: PCA-reduced 10-K embedding per (company, filing_date)
-- ============================================================================
-- 95 rows. Built by models/build_filing_pca.py (Python, not pure SQL).
-- Mean-pool MiniLM embeddings per filing -> fit PCA on (95 x 384) -> top 5 components.
-- Explained-variance ratio: [0.22, 0.14, 0.11, 0.09, 0.06] = 61.4% cumulative.

CREATE TABLE gold.fct_embedding_per_company (
  fact_embedding_key bigint PRIMARY KEY,
  company_key        text   NOT NULL REFERENCES gold.dim_company(company_key),
  as_of_date         date   NOT NULL,
  filing_pc1         numeric(18,8),
  filing_pc2         numeric(18,8),
  filing_pc3         numeric(18,8),
  filing_pc4         numeric(18,8),
  filing_pc5         numeric(18,8)
);
CREATE INDEX ON gold.fct_embedding_per_company (company_key);

-- ============================================================================
-- MAIN FACT 1: the ML feature panel
-- ============================================================================
-- 25,080 rows = 20 stocks x ~1,254 trading days (after dropping rows with
-- NULL sma_50, i.e. the first 49 days per stock). 30 numerical features + target.
-- The model rungs (Phase 5) all train against this single table.

CREATE TABLE gold.fct_feature_panel_daily AS
WITH price AS (
    SELECT * FROM silver.silver_prices_cleaned
),
fund_latest AS (
    SELECT p.symbol, p.trade_date,
           (SELECT f.fund_key FROM silver.silver_fundamentals_cleaned f
            WHERE f.symbol = p.symbol AND f.filing_date <= p.trade_date
            ORDER BY f.filing_date DESC LIMIT 1) AS fund_key
    FROM price p
),
fund_full AS (
    SELECT p.symbol AS symbol, p.trade_date AS trade_date,
           f.roe, f.roa, f.debt_eq, f.gross_margin, f.op_margin, f.asset_turnover,
           f.net_income_ttm, f.total_equity, f.ebitda_ttm, f.total_assets,
           f.cash, f.total_debt, f.free_cash_flow_ttm,
           f.as_of_date AS as_of_date
    FROM price p
    LEFT JOIN fund_latest fl ON fl.symbol = p.symbol AND fl.trade_date = p.trade_date
    LEFT JOIN silver.silver_fundamentals_cleaned f ON f.fund_key = fl.fund_key
),
emb_latest AS (
    SELECT p.company_key, p.trade_date,
           (SELECT e.fact_embedding_key FROM gold.fct_embedding_per_company e
            WHERE e.company_key = p.company_key AND e.as_of_date <= p.trade_date
            ORDER BY e.as_of_date DESC LIMIT 1) AS emb_key
    FROM price p
),
embedded AS (
    SELECT el.company_key, el.trade_date, e.as_of_date,
           e.filing_pc1, e.filing_pc2, e.filing_pc3, e.filing_pc4, e.filing_pc5
    FROM emb_latest el
    LEFT JOIN gold.fct_embedding_per_company e ON e.fact_embedding_key = el.emb_key
),
joined AS (
    SELECT
        p.price_key, p.date_key, p.company_key, c.sector_key, p.symbol, p.trade_date, p.close_px,
        -- 10 PRICE features
        p.log_ret_1d, p.sma_5, p.sma_20, p.sma_50, p.ema_12, p.ema_26,
        p.rsi_14, p.macd_hist, p.bb_z, p.vol_20d,
        -- 10 FUNDAMENTALS features
        ff.roe, ff.roa, ff.debt_eq, ff.gross_margin, ff.op_margin, ff.asset_turnover,
        CASE WHEN ff.net_income_ttm > 0 AND p.close_px > 0 THEN p.close_px / ff.net_income_ttm * 1e9 END AS pe_ttm_proxy,
        CASE WHEN ff.total_equity > 0    AND p.close_px > 0 THEN p.close_px / (ff.total_equity / 1e9) END   AS pb_proxy,
        CASE WHEN ff.ebitda_ttm > 0      AND ff.total_assets > 0
             THEN ff.ebitda_ttm / NULLIF(ff.total_assets - ff.cash + ff.total_debt, 0) END AS ebitda_to_ev_proxy,
        CASE WHEN ff.free_cash_flow_ttm IS NOT NULL AND ff.total_assets > 0
             THEN ff.free_cash_flow_ttm / ff.total_assets END                              AS fcf_to_assets,
        -- 5 SENTIMENT features
        COALESCE(s.finbert_news_mean_3d,  0::numeric) AS finbert_news_mean_3d,
        COALESCE(s.finbert_news_mean_30d, 0::numeric) AS finbert_news_mean_30d,
        COALESCE(s.finbert_press_30d,     0::numeric) AS finbert_press_30d,
        COALESCE(s.n_news_3d, 0) AS n_news_3d,
        COALESCE(s.n_8k_30d, 0)  AS n_8k_30d,
        -- 5 PCA-EMBEDDING features
        COALESCE(e.filing_pc1, 0::numeric) AS filing_pc1,
        COALESCE(e.filing_pc2, 0::numeric) AS filing_pc2,
        COALESCE(e.filing_pc3, 0::numeric) AS filing_pc3,
        COALESCE(e.filing_pc4, 0::numeric) AS filing_pc4,
        COALESCE(e.filing_pc5, 0::numeric) AS filing_pc5,
        -- TARGET: 5-day forward direction
        CASE
          WHEN LEAD(p.close_px, 5) OVER (PARTITION BY p.company_key ORDER BY p.trade_date) > p.close_px
          THEN 1 ELSE 0
        END AS y_5d_up,
        -- governance
        GREATEST(p.as_of_date,
                 COALESCE(ff.as_of_date, p.as_of_date),
                 COALESCE(s.as_of_date,  p.as_of_date),
                 COALESCE(e.as_of_date,  p.as_of_date)) AS as_of_date
    FROM price p
    LEFT JOIN gold.dim_company c ON c.company_key = p.company_key
    LEFT JOIN fund_full ff ON ff.symbol = p.symbol AND ff.trade_date = p.trade_date
    LEFT JOIN gold.fct_sentiment_per_day s
           ON s.date_key = p.date_key AND s.company_key = p.company_key
    LEFT JOIN embedded e ON e.company_key = p.company_key AND e.trade_date = p.trade_date
)
SELECT
    ROW_NUMBER() OVER (ORDER BY trade_date, company_key) + 1000 AS fact_panel_key,
    *
FROM joined
WHERE log_ret_1d IS NOT NULL AND sma_50 IS NOT NULL;

ALTER TABLE gold.fct_feature_panel_daily ADD PRIMARY KEY (fact_panel_key);
ALTER TABLE gold.fct_feature_panel_daily ADD CONSTRAINT fk_panel_date
  FOREIGN KEY (date_key) REFERENCES gold.dim_date(date_key);
ALTER TABLE gold.fct_feature_panel_daily ADD CONSTRAINT fk_panel_company
  FOREIGN KEY (company_key) REFERENCES gold.dim_company(company_key);
CREATE INDEX ON gold.fct_feature_panel_daily (date_key, company_key);
CREATE INDEX ON gold.fct_feature_panel_daily (company_key, trade_date);

-- -- Leakage-guard test (governance evidence) ---------------------
-- SELECT COUNT(*) FROM gold.fct_feature_panel_daily WHERE as_of_date > trade_date;
-- Expected: 0 rows. Verified Phase 4.

-- ============================================================================
-- MAIN FACT 2: model predictions (one row per (date, company, rung))
-- ============================================================================
-- 19,480 rows so far (Rung 1 + Rung 2 fully done across 11 walk-forward folds;
-- Rung 0 ARIMA in progress).

CREATE TABLE gold.fct_predictions (
  prediction_key  bigserial PRIMARY KEY,
  date_key        text     NOT NULL REFERENCES gold.dim_date(date_key),
  company_key     text     NOT NULL REFERENCES gold.dim_company(company_key),
  model_rung      smallint NOT NULL,       -- 0=ARIMA, 1=XGB-structured, 2=XGB-full
  fold_id         text     NOT NULL,       -- e.g. '2024-01-01' (test_start)
  prob_up         numeric(8,5) NOT NULL,
  predicted_class smallint NOT NULL,
  shap_json       jsonb                    -- top-10 SHAP per row for Rungs 1+2 (sampled)
);
CREATE INDEX ix_pred_dcr ON gold.fct_predictions (date_key, company_key, model_rung);

-- ============================================================================
-- MAIN FACT 3: backtest P&L per (date, rung)
-- ============================================================================
-- 196 rows = ~96 weekly rebalances x 2 rungs (Rung 1+2 complete; Rung 0 partial).
-- Rebalance every 5 trading days; top-5 long, equal weight; 5 bp tx cost.

CREATE TABLE gold.fct_backtest_pnl_daily (
  fact_pnl_key            bigserial PRIMARY KEY,
  trade_date              date NOT NULL,
  date_key                text NOT NULL,
  model_rung              smallint NOT NULL,
  n_long                  smallint,
  gross_ret               numeric(18,8),
  tx_cost                 numeric(18,8),
  net_ret                 numeric(18,8),
  cum_net_ret             numeric(18,8),
  benchmark_cum_ret_eqw   numeric(18,8)
);
ALTER TABLE gold.fct_backtest_pnl_daily ADD CONSTRAINT fk_pnl_date
  FOREIGN KEY (date_key) REFERENCES gold.dim_date(date_key);

-- ============================================================================
-- AUDIT FACT: RAG query log (append-only)
-- ============================================================================
-- Every call to rag.answer() writes one row. Lets a regulator replay any
-- historical RAG response. SEC Rule 17a-4 spirit.

CREATE TABLE gold.fct_rag_queries (
  query_id        bigserial PRIMARY KEY,
  ts              timestamptz NOT NULL DEFAULT now(),
  question        text NOT NULL,
  symbol          text NOT NULL,
  as_of_date      date NOT NULL,
  rung            smallint,
  top_k_chunk_ids text[] NOT NULL,
  llm_model       text NOT NULL,           -- 'claude-haiku-...' or 'template-fallback'
  tokens_in       integer,
  tokens_out      integer,
  response        text NOT NULL,
  latency_ms      integer
);
