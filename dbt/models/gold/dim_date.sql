{{ config(materialized='table') }}
-- gold.dim_date: daily calendar covering the project's date range.
-- Mirrors the midterm's gold.dim_date shape (md5 surrogate key, friendly attributes).

WITH dates AS (
    SELECT generate_series('2021-01-01'::date, '2025-12-31'::date, '1 day')::date AS full_date
)
SELECT
    md5(full_date::text)                          AS date_key,
    full_date,
    EXTRACT(YEAR  FROM full_date)::smallint       AS year,
    EXTRACT(MONTH FROM full_date)::smallint       AS month,
    EXTRACT(DAY   FROM full_date)::smallint       AS day,
    EXTRACT(WEEK  FROM full_date)::smallint       AS week,
    EXTRACT(DOW   FROM full_date)::smallint       AS day_of_week,
    (EXTRACT(DOW FROM full_date)::int BETWEEN 1 AND 5) AS is_trading_day
FROM dates
