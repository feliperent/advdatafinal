
-- 02_datos_masked_schema.sql  -- PII redaction views (between raw and silver)

-- Four views, one per text-bearing raw table. Regex-masks emails + US phones.
-- Silver text pipelines read FROM these views, NEVER from raw.* directly,
-- so silver never sees PII in plaintext.

-- The midterm-style pattern: complaint_text_masked / phone_masked on
-- silver.silver_quejas_cleaned. Here it's applied at the layer BEFORE silver
-- so multiple downstream silver tables can reuse it.

CREATE OR REPLACE VIEW datos_masked.news_redacted AS
SELECT
    article_id, symbol, published_at, title, site, url,
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
FROM raw.news_raw;

CREATE OR REPLACE VIEW datos_masked.press_redacted AS
SELECT
    press_id, symbol, published_at, title,
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
FROM raw.press_raw;

CREATE OR REPLACE VIEW datos_masked.filings_10k_redacted AS
SELECT
    accession, symbol, filing_date, fiscal_year,
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
FROM raw.sec_10k_raw;

CREATE OR REPLACE VIEW datos_masked.filings_8k_redacted AS
SELECT
    accession, symbol, filing_date,
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
FROM raw.sec_8k_raw;

-- -- Verification queries (Section 2 Governance evidence) -----------
-- SELECT 'news' v,
--        COUNT(*) FILTER (WHERE body_masked ~ '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}') emails,
--        COUNT(*) FILTER (WHERE body_masked ~ '\+?1?[\s.\-]?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}') phones
-- FROM datos_masked.news_redacted
-- UNION ALL ... (same for press, 10K, 8K)
-- Expected: all rows = 0 leaks. Verified after the FinBERT-inversion fix + phone-regex add.
