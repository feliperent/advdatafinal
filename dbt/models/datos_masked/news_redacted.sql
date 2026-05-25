{{ config(materialized='view') }}
-- Redact email addresses in news body text, mirroring the midterm's complaint_text_masked pattern.
SELECT
    article_id,
    symbol,
    published_at,
    title,
    site,
    url,
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
FROM {{ source('raw', 'news_raw') }}
