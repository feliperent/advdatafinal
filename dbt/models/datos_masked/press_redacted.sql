{{ config(materialized='view') }}
SELECT
    press_id,
    symbol,
    published_at,
    title,
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
FROM {{ source('raw', 'press_raw') }}
