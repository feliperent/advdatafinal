{{ config(materialized='table') }}
-- gold.dim_filing_type: 4 filing-type codes used by the RAG corpus.
SELECT
    md5(LOWER(TRIM(code))) AS filing_type_key,
    code,
    label
FROM (VALUES
    ('10K',   '10-K Annual Report'),
    ('8K',    '8-K Material Event'),
    ('NEWS',  'News Article'),
    ('PRESS', 'Press Release')
) AS t(code, label)
