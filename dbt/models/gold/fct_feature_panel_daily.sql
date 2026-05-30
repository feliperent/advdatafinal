{{ config(
    materialized='table',
    indexes=[
        {'columns': ['date_key', 'company_key'], 'type': 'btree'},
        {'columns': ['company_key', 'trade_date'], 'type': 'btree'}
    ]
) }}

-- gold.fct_feature_panel_daily: ONE row per (trading_date, company) with 30 features + target.
-- Joins silver_prices_cleaned (10 price feats) + silver_fundamentals_cleaned (10 fundamentals)
-- + gold.fct_sentiment_per_day (5 sentiment feats) + gold.fct_embedding_per_company (5 PCA feats, may not exist yet).
--
-- COALESCE on PCA columns so the table builds even before Phase 3b (filings chunked + embedded).
-- as_of_date is the GREATEST of all source as_of_dates -- enforced by the leakage-guard test.

WITH price AS (
    SELECT * FROM {{ ref('silver_prices_cleaned') }}
),
-- For each (symbol, trade_date) pick the latest fundamentals row with filing_date <= trade_date
fund_latest AS (
    SELECT
        p.symbol, p.trade_date,
        (SELECT f.fund_key FROM {{ ref('silver_fundamentals_cleaned') }} f
         WHERE f.symbol = p.symbol AND f.filing_date <= p.trade_date
         ORDER BY f.filing_date DESC LIMIT 1) AS fund_key
    FROM price p
),
fund_full AS (
    -- Pull only the fundamentals fields we actually use; avoid name collisions with `price`.
    SELECT
        p.symbol                    AS symbol,
        p.trade_date                AS trade_date,
        f.roe, f.roa, f.debt_eq, f.gross_margin, f.op_margin, f.asset_turnover,
        f.net_income_ttm, f.total_equity, f.ebitda_ttm, f.total_assets,
        f.cash, f.total_debt, f.free_cash_flow_ttm,
        f.as_of_date                AS as_of_date
    FROM price p
    LEFT JOIN fund_latest fl ON fl.symbol = p.symbol AND fl.trade_date = p.trade_date
    LEFT JOIN {{ ref('silver_fundamentals_cleaned') }} f ON f.fund_key = fl.fund_key
),
-- For each (company, trade_date) pick the MOST RECENT 10-K PCA snapshot whose as_of_date <= trade_date.
-- Without this restriction the LEFT JOIN multiplies rows by the number of historical filings.
emb_latest AS (
    SELECT
        p.company_key,
        p.trade_date,
        (SELECT e.fact_embedding_key
         FROM gold.fct_embedding_per_company e
         WHERE e.company_key = p.company_key AND e.as_of_date <= p.trade_date
         ORDER BY e.as_of_date DESC LIMIT 1) AS emb_key
    FROM price p
),
embedded AS (
    SELECT
        el.company_key,
        el.trade_date,
        e.as_of_date,
        e.filing_pc1, e.filing_pc2, e.filing_pc3, e.filing_pc4, e.filing_pc5
    FROM emb_latest el
    LEFT JOIN gold.fct_embedding_per_company e ON e.fact_embedding_key = el.emb_key
),
joined AS (
    SELECT
        p.price_key, p.date_key, p.company_key, c.sector_key,
        p.symbol, p.trade_date, p.close_px,
        -- 10 PRICE features
        p.log_ret_1d, p.sma_5, p.sma_20, p.sma_50, p.ema_12, p.ema_26,
        p.rsi_14, p.macd_hist, p.bb_z, p.vol_20d,
        -- 10 FUNDAMENTALS features (TTM ratios + 4 market-cap-dependent ones computed on the fly)
        ff.roe, ff.roa, ff.debt_eq, ff.gross_margin, ff.op_margin, ff.asset_turnover,
        CASE WHEN ff.net_income_ttm > 0 AND p.close_px > 0 THEN p.close_px / ff.net_income_ttm * 1e9 END AS pe_ttm_proxy,
        CASE WHEN ff.total_equity > 0    AND p.close_px > 0 THEN p.close_px / (ff.total_equity / 1e9) END   AS pb_proxy,
        CASE WHEN ff.ebitda_ttm > 0      AND ff.total_assets > 0
             THEN ff.ebitda_ttm / NULLIF(ff.total_assets - ff.cash + ff.total_debt, 0) END                  AS ebitda_to_ev_proxy,
        CASE WHEN ff.free_cash_flow_ttm IS NOT NULL AND ff.total_assets > 0
             THEN ff.free_cash_flow_ttm / ff.total_assets END                                               AS fcf_to_assets,
        -- 5 LEGACY sentiment features (kept so prior Rung 2 still resolves).
        COALESCE(s.finbert_news_mean_3d,  0::numeric) AS finbert_news_mean_3d,
        COALESCE(s.finbert_news_mean_30d, 0::numeric) AS finbert_news_mean_30d,
        COALESCE(s.finbert_press_30d,     0::numeric) AS finbert_press_30d,
        COALESCE(s.n_news_3d, 0)                       AS n_news_3d,
        COALESCE(s.n_8k_30d, 0)                        AS n_8k_30d,
        -- 6 NEW per-class probability + derivative features for the upgraded Rung 2.
        COALESCE(s.finbert_news_pos_3d,    0::numeric) AS finbert_news_pos_3d,
        COALESCE(s.finbert_news_neg_3d,    0::numeric) AS finbert_news_neg_3d,
        COALESCE(s.finbert_news_pos_30d,   0::numeric) AS finbert_news_pos_30d,
        COALESCE(s.finbert_news_neg_30d,   0::numeric) AS finbert_news_neg_30d,
        COALESCE(s.news_mean_change_5d,    0::numeric) AS news_mean_change_5d,
        COALESCE(s.news_disp_3d,           0::numeric) AS news_disp_3d,
        -- Sector one-hot dummies (5 binary columns the tree model can split on).
        -- Replaces the earlier sector_int hash approach (XGBoost range-splits on a hash
        -- buy-bucket would never recover the sector grouping).
        CASE WHEN c.sector_name = 'Technology'   THEN 1 ELSE 0 END AS is_tech,
        CASE WHEN c.sector_name = 'Financials'   THEN 1 ELSE 0 END AS is_financials,
        CASE WHEN c.sector_name = 'Healthcare'   THEN 1 ELSE 0 END AS is_healthcare,
        CASE WHEN c.sector_name = 'Industrials'  THEN 1 ELSE 0 END AS is_industrials,
        CASE WHEN c.sector_name = 'Consumer'     THEN 1 ELSE 0 END AS is_consumer,
        -- Sentiment-x-price cross features (interactions XGBoost might otherwise miss at this data size).
        (p.vol_20d * COALESCE(s.finbert_news_mean_3d, 0::numeric))   AS senti_x_vol,
        (ABS(p.log_ret_1d) * COALESCE(s.finbert_news_mean_3d, 0::numeric)) AS senti_x_absret,
        (COALESCE(s.n_news_3d, 0) * p.vol_20d)                       AS newsvol_x_vol,
        -- 5 PCA-EMBEDDING features (may be NULL until Phase 3b runs)
        COALESCE(e.filing_pc1, 0::numeric) AS filing_pc1,
        COALESCE(e.filing_pc2, 0::numeric) AS filing_pc2,
        COALESCE(e.filing_pc3, 0::numeric) AS filing_pc3,
        COALESCE(e.filing_pc4, 0::numeric) AS filing_pc4,
        COALESCE(e.filing_pc5, 0::numeric) AS filing_pc5,
        -- target: y_5d_up
        CASE
          WHEN LEAD(p.close_px, 5) OVER (PARTITION BY p.company_key ORDER BY p.trade_date) > p.close_px
          THEN 1 ELSE 0
        END AS y_5d_up,
        -- governance: latest as_of_date across all sources used
        GREATEST(
            p.as_of_date,
            COALESCE(ff.as_of_date, p.as_of_date),
            COALESCE(s.as_of_date, p.as_of_date),
            COALESCE(e.as_of_date, p.as_of_date)
        ) AS as_of_date
    FROM price p
    INNER JOIN {{ ref('dim_date') }}    d ON d.full_date  = p.trade_date
    INNER JOIN {{ ref('dim_company') }} c ON c.company_key = p.company_key
    INNER JOIN {{ ref('dim_sector') }}  sd ON sd.sector_key = c.sector_key
    LEFT  JOIN fund_full ff ON ff.symbol = p.symbol AND ff.trade_date = p.trade_date
    LEFT  JOIN {{ ref('fct_sentiment_per_day') }} s
           ON s.date_key = p.date_key AND s.company_key = p.company_key
    LEFT  JOIN embedded e ON e.company_key = p.company_key AND e.trade_date = p.trade_date
)
SELECT
    ROW_NUMBER() OVER (ORDER BY trade_date, company_key) + 1000 AS fact_panel_key,
    *
FROM joined
WHERE log_ret_1d IS NOT NULL AND sma_50 IS NOT NULL
