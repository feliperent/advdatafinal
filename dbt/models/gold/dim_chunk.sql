{{ config(materialized='table') }}
-- gold.dim_chunk: union of all RAG-corpus chunks across 10-K and 8-K filings.
-- Provides the citation lookup the RAG layer uses (chunk_key -> source_url, page_approx, as_of_date).

SELECT
    chunk_key,
    accession                                  AS filing_accession,
    '10K'                                      AS source_type,
    md5(LOWER(TRIM('10K')))                    AS filing_type_key,
    company_key,
    chunk_index,
    n_tokens,
    NULL::smallint                             AS page_approx,
    'https://www.sec.gov/Archives/edgar/data'  AS source_url_root,
    as_of_date
FROM {{ source('silver_python', 'silver_filings_10k_chunked') }}

UNION ALL

SELECT
    chunk_key,
    accession,
    '8K',
    md5(LOWER(TRIM('8K'))),
    company_key,
    chunk_index,
    n_tokens,
    NULL::smallint,
    'https://www.sec.gov/Archives/edgar/data',
    as_of_date
FROM {{ source('silver_python', 'silver_filings_8k_chunked') }}
