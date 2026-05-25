{{ config(
    materialized='table',
    indexes=[
        {'columns': ['company_key', 'filing_date'], 'type': 'btree'},
        {'columns': ['filing_date'], 'type': 'btree'}
    ]
) }}

-- silver.silver_fundamentals_cleaned
-- One-tier silver: joins the 3 raw fundamentals tables (income, balance, cash flow)
-- and computes TTM rollups + classic ratios in a single denormalised table.
-- Postgres column names are all-lowercase (sanitiser collapsed camelCase at ingest).

WITH inc AS (
    SELECT
        symbol,
        date::date                                AS filing_date,
        NULLIF(revenue, '')::numeric(20,2)        AS revenue,
        NULLIF(grossprofit, '')::numeric(20,2)    AS gross_profit,
        NULLIF(operatingincome, '')::numeric(20,2) AS operating_income,
        NULLIF(netincome, '')::numeric(20,2)      AS net_income,
        NULLIF(ebitda, '')::numeric(20,2)         AS ebitda
    FROM {{ source('raw', 'income_statement_raw') }}
),
bs AS (
    SELECT
        symbol,
        date::date                                       AS filing_date,
        NULLIF(totalassets, '')::numeric(20,2)           AS total_assets,
        NULLIF(totalliabilities, '')::numeric(20,2)      AS total_liabilities,
        NULLIF(totalstockholdersequity, '')::numeric(20,2) AS total_equity,
        NULLIF(totaldebt, '')::numeric(20,2)             AS total_debt,
        NULLIF(cashandcashequivalents, '')::numeric(20,2) AS cash
    FROM {{ source('raw', 'balance_sheet_raw') }}
),
cf AS (
    SELECT
        symbol,
        date::date                                   AS filing_date,
        NULLIF(freecashflow, '')::numeric(20,2)      AS free_cash_flow,
        NULLIF(operatingcashflow, '')::numeric(20,2) AS operating_cash_flow
    FROM {{ source('raw', 'cash_flow_raw') }}
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
    -- 4-quarter trailing window per (symbol, filing_date). Postgres requires ORDER BY for window frames.
    SELECT
        symbol, filing_date,
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
    symbol,
    filing_date,
    revenue_ttm,
    gross_profit_ttm,
    operating_income_ttm,
    net_income_ttm,
    ebitda_ttm,
    free_cash_flow_ttm,
    operating_cash_flow_ttm,
    avg_assets_ttm,
    avg_equity_ttm,
    total_assets, total_liabilities, total_equity, total_debt, cash,
    -- Ratios
    CASE WHEN avg_equity_ttm > 0 THEN net_income_ttm / avg_equity_ttm END AS roe,
    CASE WHEN avg_assets_ttm > 0 THEN net_income_ttm / avg_assets_ttm END AS roa,
    CASE WHEN total_equity > 0   THEN total_debt / total_equity END     AS debt_eq,
    CASE WHEN revenue_ttm > 0    THEN gross_profit_ttm / revenue_ttm END  AS gross_margin,
    CASE WHEN revenue_ttm > 0    THEN operating_income_ttm / revenue_ttm END AS op_margin,
    CASE WHEN avg_assets_ttm > 0 THEN revenue_ttm / avg_assets_ttm END    AS asset_turnover,
    -- PE TTM, PB, FCF yield, EV/EBITDA all depend on market_cap (close_px * shares_out).
    -- Those are computed JOIN-side in gold.fct_feature_panel_daily where prices and shares meet.
    filing_date AS as_of_date
FROM ttm
WHERE revenue_ttm IS NOT NULL
