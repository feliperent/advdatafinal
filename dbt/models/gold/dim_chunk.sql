{{ config(materialized='table') }}
-- gold.dim_chunk: every 10-K and 8-K chunk used by the RAG layer.
-- JOINs dim_filing_type so the lineage edge filing_type -> chunk appears in dbt docs.

WITH chunks AS (
    SELECT chunk_key, accession, '10K' AS source_type, company_key, chunk_index, n_tokens, as_of_date
    FROM {{ source('silver_python', 'silver_filings_10k_chunked') }}
    UNION ALL
    SELECT chunk_key, accession, '8K', company_key, chunk_index, n_tokens, as_of_date
    FROM {{ source('silver_python', 'silver_filings_8k_chunked') }}
)
SELECT
    c.chunk_key,
    c.accession                                AS filing_accession,
    c.source_type,
    ft.filing_type_key,
    c.company_key,
    c.chunk_index,
    c.n_tokens,
    NULL::smallint                             AS page_approx,
    'https://www.sec.gov/Archives/edgar/data'  AS source_url_root,
    c.as_of_date
FROM chunks c
INNER JOIN {{ ref('dim_filing_type') }} ft ON ft.code = c.source_type
