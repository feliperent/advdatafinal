{{ config(materialized='view') }}
-- gold.fct_rag_chunk_citations: one row per (query, cited_chunk) by unnesting top_k_chunk_ids.
-- Bridges fct_rag_queries -> dim_chunk so the lineage edge appears in dbt docs.

SELECT
    q.query_id,
    q.ts,
    q.symbol,
    q.as_of_date,
    q.rung,
    cited.chunk_key,
    cited.rank_position,
    c.company_key,
    c.filing_type_key,
    c.filing_accession,
    c.chunk_index,
    c.n_tokens
FROM {{ source('gold_python', 'fct_rag_queries') }} q
CROSS JOIN LATERAL unnest(q.top_k_chunk_ids) WITH ORDINALITY AS cited(chunk_key, rank_position)
INNER JOIN {{ ref('dim_chunk') }} c ON c.chunk_key = cited.chunk_key
