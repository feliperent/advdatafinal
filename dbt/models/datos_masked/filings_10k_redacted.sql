{{ config(materialized='view') }}
SELECT
    accession,
    symbol,
    filing_date,
    fiscal_year,
    REGEXP_REPLACE(
        COALESCE(body, ''),
        '[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}',
        '[EMAIL]', 'g'
    ) AS body_masked,
    ingest_ts
FROM {{ source('raw', 'sec_10k_raw') }}
