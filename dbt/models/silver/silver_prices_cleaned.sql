{{ config(
    materialized='table',
    indexes=[
        {'columns': ['date_key', 'company_key'], 'type': 'btree'},
        {'columns': ['company_key'], 'type': 'btree'},
        {'columns': ['trade_date'], 'type': 'btree'}
    ]
) }}

-- silver.silver_prices_cleaned
-- One-tier silver model (matches IN014 midterm style: silver.silver_<source>_cleaned).
-- Carries cleaned prices PLUS the 10 technical features in one denormalised table.
-- Window functions are computed in SQL except RSI-14 (Wilder smoothing), which is
-- filled by silver_text/post_rsi.py because the recursive form is awkward in pure SQL.

WITH typed AS (
    SELECT
        symbol,
        trade_date::date                       AS trade_date,
        NULLIF(open, '')::numeric(18,6)        AS open_px,
        NULLIF(high, '')::numeric(18,6)        AS high_px,
        NULLIF(low, '')::numeric(18,6)         AS low_px,
        NULLIF(close, '')::numeric(18,6)       AS close_px,
        NULLIF(adj_close, '')::numeric(18,6)   AS adj_close_px,
        NULLIF(volume, '')::bigint             AS volume
    FROM {{ source('raw', 'prices_raw') }}
    WHERE close IS NOT NULL AND close <> ''
),
with_returns AS (
    SELECT
        *,
        LN(close_px / NULLIF(LAG(close_px, 1) OVER (PARTITION BY symbol ORDER BY trade_date), 0)) AS log_ret_1d
    FROM typed
),
with_features AS (
    SELECT
        *,
        AVG(close_px) OVER w5   AS sma_5,
        AVG(close_px) OVER w20  AS sma_20,
        AVG(close_px) OVER w50  AS sma_50,
        AVG(close_px) OVER w12  AS ema_12_approx,
        AVG(close_px) OVER w26  AS ema_26_approx,
        STDDEV_POP(close_px)    OVER w20 AS sd_close_20,
        STDDEV_POP(log_ret_1d)  OVER w20 AS sd_logret_20
    FROM with_returns
    WINDOW
        w5  AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 4  PRECEDING AND CURRENT ROW),
        w20 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
        w50 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW),
        w12 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW),
        w26 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 25 PRECEDING AND CURRENT ROW)
)
SELECT
    MD5(symbol || '|' || trade_date::text)              AS price_key,
    MD5(trade_date::text)                               AS date_key,
    MD5(LOWER(TRIM(symbol)))                            AS company_key,
    symbol,
    trade_date,
    open_px, high_px, low_px, close_px, adj_close_px, volume,
    log_ret_1d,
    sma_5,
    sma_20,
    sma_50,
    ema_12_approx                                       AS ema_12,
    ema_26_approx                                       AS ema_26,
    (ema_12_approx - ema_26_approx)                     AS macd_hist,
    CASE
        WHEN sd_close_20 IS NOT NULL AND sd_close_20 > 0
        THEN (close_px - sma_20) / sd_close_20
        ELSE 0
    END                                                 AS bb_z,
    NULL::numeric(18,4)                                 AS rsi_14,  -- post-hook: silver_text/post_rsi.py
    sd_logret_20 * SQRT(252)                            AS vol_20d,
    trade_date                                          AS as_of_date
FROM with_features
