{{ config(materialized='table') }}
-- gold.dim_sector: 5 sectors used by the 20-ticker universe.
SELECT
    md5(LOWER(TRIM(sector_name))) AS sector_key,
    sector_name
FROM (VALUES
    ('Technology'),
    ('Financials'),
    ('Healthcare'),
    ('Industrials'),
    ('Consumer')
) AS s(sector_name)
