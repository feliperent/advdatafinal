{{ config(materialized='table') }}
-- gold.fct_sentiment_per_day: 5 sentiment features per (date, company), trading-days only.
-- Aggregates silver.silver_news_scored and silver.silver_press_scored over rolling 3 and 30 trading-day windows.
-- n_8k_30d counts unique 8-K filings (from raw.sec_8k_raw) in the trailing 30 days.

WITH date_grid AS (
    -- (trading_date, company) cross-product
    SELECT d.date_key, d.full_date, c.company_key, c.symbol
    FROM {{ ref('dim_date') }} d
    CROSS JOIN {{ ref('dim_company') }} c
    WHERE d.is_trading_day = true
),
news_per_day AS (
    SELECT company_key, trade_date,
           AVG(finbert_score) AS news_mean_day,
           COUNT(*)::int      AS n_news_day
    FROM {{ source('silver_python', 'silver_news_scored') }}
    WHERE trade_date IS NOT NULL
    GROUP BY company_key, trade_date
),
press_per_day AS (
    SELECT company_key, trade_date,
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
        COALESCE(n.news_mean_day, 0::numeric)  AS news_mean_day,
        COALESCE(n.n_news_day, 0)              AS n_news_day,
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
        AVG(NULLIF(news_mean_day, 0))  OVER w3   AS finbert_news_mean_3d,
        AVG(NULLIF(news_mean_day, 0))  OVER w30  AS finbert_news_mean_30d,
        AVG(NULLIF(press_mean_day, 0)) OVER w30  AS finbert_press_30d,
        COALESCE(SUM(n_news_day) OVER w3, 0)     AS n_news_3d,
        COALESCE(SUM(n_8k_on_day) OVER w30, 0)   AS n_8k_30d
    FROM joined
    WINDOW
      w3  AS (PARTITION BY company_key ORDER BY full_date ROWS BETWEEN 2  PRECEDING AND CURRENT ROW),
      w30 AS (PARTITION BY company_key ORDER BY full_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
)
SELECT
    ROW_NUMBER() OVER (ORDER BY full_date, company_key) + 1000 AS fact_sentiment_key,
    date_key,
    company_key,
    COALESCE(finbert_news_mean_3d,  0::numeric) AS finbert_news_mean_3d,
    COALESCE(finbert_news_mean_30d, 0::numeric) AS finbert_news_mean_30d,
    COALESCE(finbert_press_30d,     0::numeric) AS finbert_press_30d,
    n_news_3d,
    n_8k_30d,
    full_date AS as_of_date
FROM windowed
