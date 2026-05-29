{{ config(materialized='table') }}
-- gold.fct_sentiment_per_day: per-company per-trading-day sentiment features.
--   Rolling means of the three FinBERT class probabilities over 3- and 30-day windows.
--   Rolling means of the scalar (P(pos) - P(neg)) over the same windows.
--   News-volume counts and 8-K event counts.
--   Day-over-day sentiment deltas and within-window dispersion.

WITH date_grid AS (
    SELECT d.date_key, d.full_date, c.company_key, c.symbol
    FROM {{ ref('dim_date') }} d
    CROSS JOIN {{ ref('dim_company') }} c
    WHERE d.is_trading_day = true
),
news_per_day AS (
    SELECT company_key, trade_date,
           AVG(pos_prob)      AS news_pos_day,
           AVG(neu_prob)      AS news_neu_day,
           AVG(neg_prob)      AS news_neg_day,
           AVG(finbert_score) AS news_mean_day,
           STDDEV_SAMP(finbert_score) AS news_disp_day,
           COUNT(*)::int      AS n_news_day
    FROM {{ source('silver_python', 'silver_news_scored') }}
    WHERE trade_date IS NOT NULL
    GROUP BY company_key, trade_date
),
press_per_day AS (
    SELECT company_key, trade_date,
           AVG(pos_prob)      AS press_pos_day,
           AVG(neu_prob)      AS press_neu_day,
           AVG(neg_prob)      AS press_neg_day,
           AVG(finbert_score) AS press_mean_day
    FROM {{ source('silver_python', 'silver_press_scored') }}
    WHERE trade_date IS NOT NULL
    GROUP BY company_key, trade_date
),
filings_per_day AS (
    SELECT
        md5(LOWER(TRIM(symbol))) AS company_key,
        filing_date::date        AS filing_date,
        COUNT(DISTINCT accession)::int AS n_8k_on_day
    FROM {{ source('raw', 'sec_8k_raw') }}
    WHERE filing_date IS NOT NULL AND filing_date <> ''
    GROUP BY symbol, filing_date::date
),
joined AS (
    SELECT
        g.date_key,
        g.full_date,
        g.company_key,
        g.symbol,
        COALESCE(n.news_pos_day,   0::numeric) AS news_pos_day,
        COALESCE(n.news_neu_day,   1::numeric) AS news_neu_day,
        COALESCE(n.news_neg_day,   0::numeric) AS news_neg_day,
        COALESCE(n.news_mean_day,  0::numeric) AS news_mean_day,
        COALESCE(n.news_disp_day,  0::numeric) AS news_disp_day,
        COALESCE(n.n_news_day, 0)              AS n_news_day,
        COALESCE(p.press_pos_day,  0::numeric) AS press_pos_day,
        COALESCE(p.press_neu_day,  1::numeric) AS press_neu_day,
        COALESCE(p.press_neg_day,  0::numeric) AS press_neg_day,
        COALESCE(p.press_mean_day, 0::numeric) AS press_mean_day,
        COALESCE(fk.n_8k_on_day, 0)            AS n_8k_on_day
    FROM date_grid g
    LEFT JOIN news_per_day  n  ON g.company_key = n.company_key  AND g.full_date = n.trade_date
    LEFT JOIN press_per_day p  ON g.company_key = p.company_key  AND g.full_date = p.trade_date
    LEFT JOIN filings_per_day fk ON g.company_key = fk.company_key AND g.full_date = fk.filing_date
),
windowed AS (
    SELECT
        date_key, full_date, company_key, symbol,
        -- Three-day rolling probabilities (per class).
        AVG(NULLIF(news_pos_day, 0))   OVER w3 AS news_pos_3d,
        AVG(NULLIF(news_neu_day, 1))   OVER w3 AS news_neu_3d,
        AVG(NULLIF(news_neg_day, 0))   OVER w3 AS news_neg_3d,
        AVG(NULLIF(news_mean_day, 0))  OVER w3 AS finbert_news_mean_3d,
        -- 30-day rolling.
        AVG(NULLIF(news_pos_day, 0))   OVER w30 AS news_pos_30d,
        AVG(NULLIF(news_neg_day, 0))   OVER w30 AS news_neg_30d,
        AVG(NULLIF(news_mean_day, 0))  OVER w30 AS finbert_news_mean_30d,
        AVG(NULLIF(press_mean_day, 0)) OVER w30 AS finbert_press_30d,
        -- Volume and dispersion.
        COALESCE(SUM(n_news_day) OVER w3, 0)   AS n_news_3d,
        AVG(NULLIF(news_disp_day, 0)) OVER w3  AS news_disp_3d,
        COALESCE(SUM(n_8k_on_day) OVER w30, 0) AS n_8k_30d,
        -- Day-over-day sentiment change.
        news_mean_day - LAG(news_mean_day, 5) OVER (PARTITION BY company_key ORDER BY full_date) AS news_mean_change_5d
    FROM joined
    WINDOW
      w3  AS (PARTITION BY company_key ORDER BY full_date ROWS BETWEEN 2  PRECEDING AND CURRENT ROW),
      w30 AS (PARTITION BY company_key ORDER BY full_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
)
SELECT
    ROW_NUMBER() OVER (ORDER BY full_date, company_key) + 1000 AS fact_sentiment_key,
    date_key,
    company_key,
    -- Original five features (kept so prior Rung-2 panel still resolves).
    COALESCE(finbert_news_mean_3d,  0::numeric) AS finbert_news_mean_3d,
    COALESCE(finbert_news_mean_30d, 0::numeric) AS finbert_news_mean_30d,
    COALESCE(finbert_press_30d,     0::numeric) AS finbert_press_30d,
    n_news_3d,
    n_8k_30d,
    -- New per-class probability features for Rung 2.
    COALESCE(news_pos_3d,   0::numeric)         AS finbert_news_pos_3d,
    COALESCE(news_neg_3d,   0::numeric)         AS finbert_news_neg_3d,
    COALESCE(news_pos_30d,  0::numeric)         AS finbert_news_pos_30d,
    COALESCE(news_neg_30d,  0::numeric)         AS finbert_news_neg_30d,
    -- New derivatives.
    COALESCE(news_mean_change_5d, 0::numeric)   AS news_mean_change_5d,
    COALESCE(news_disp_3d, 0::numeric)          AS news_disp_3d,
    full_date AS as_of_date
FROM windowed
