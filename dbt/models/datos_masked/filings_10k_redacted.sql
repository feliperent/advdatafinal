{{ config(materialized='view') }}
SELECT
    accession,
    symbol,
    filing_date,
    fiscal_year,
    REGEXP_REPLACE(
        REGEXP_REPLACE(
            COALESCE(body, ''),
            '[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}',
            '[EMAIL]', 'g'
        ),
        '\+?1?[\s.\-]?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}',
        '[PHONE]', 'g'
    ) AS body_masked,
    ingest_ts
FROM {{ source('raw', 'sec_10k_raw') }}
