
-- 04_gold_dims.sql  -- Gold dimensions (5 dim tables, star-schema vocabulary)

-- Same shape as the IN014 midterm's gold.dim_area / dim_element / dim_detail:
-- md5(LOWER(TRIM(col))) primary keys stored as text, attribute columns alongside.

-- -- gold.dim_date -------------------------------------------------
-- 1,826 calendar days from 2021-01-01 to 2025-12-31.
-- is_trading_day flags Mon-Fri (a coarse approximation; market holidays handled
-- downstream by inner-joining to silver.silver_prices_cleaned).

CREATE TABLE gold.dim_date AS
WITH dates AS (
    SELECT generate_series('2021-01-01'::date, '2025-12-31'::date, '1 day')::date AS full_date
)
SELECT
    md5(full_date::text)                              AS date_key,
    full_date,
    EXTRACT(YEAR  FROM full_date)::smallint           AS year,
    EXTRACT(MONTH FROM full_date)::smallint           AS month,
    EXTRACT(DAY   FROM full_date)::smallint           AS day,
    EXTRACT(WEEK  FROM full_date)::smallint           AS week,
    EXTRACT(DOW   FROM full_date)::smallint           AS day_of_week,
    (EXTRACT(DOW FROM full_date)::int BETWEEN 1 AND 5) AS is_trading_day
FROM dates;

ALTER TABLE gold.dim_date ADD PRIMARY KEY (date_key);

-- -- gold.dim_sector -----------------------------------------------
CREATE TABLE gold.dim_sector AS
SELECT
    md5(LOWER(TRIM(sector_name))) AS sector_key,
    sector_name
FROM (VALUES
    ('Technology'),
    ('Financials'),
    ('Healthcare'),
    ('Industrials'),
    ('Consumer')
) AS s(sector_name);

ALTER TABLE gold.dim_sector ADD PRIMARY KEY (sector_key);

-- -- gold.dim_company ----------------------------------------------
-- 20 stocks, locked at project start. company_key = md5(lower(trim(symbol))).
-- sector_key is a FK to gold.dim_sector.

CREATE TABLE gold.dim_company AS
WITH symbols AS (
    SELECT DISTINCT symbol FROM silver.silver_prices_cleaned
),
mapping AS (
    SELECT * FROM (VALUES
        ('AAPL', 'Apple',              'Technology'),
        ('MSFT', 'Microsoft',          'Technology'),
        ('GOOGL','Alphabet',           'Technology'),
        ('NVDA', 'NVIDIA',             'Technology'),
        ('JPM',  'JPMorgan Chase',     'Financials'),
        ('BAC',  'Bank of America',    'Financials'),
        ('GS',   'Goldman Sachs',      'Financials'),
        ('AXP',  'American Express',   'Financials'),
        ('JNJ',  'Johnson & Johnson',  'Healthcare'),
        ('UNH',  'UnitedHealth',       'Healthcare'),
        ('PFE',  'Pfizer',             'Healthcare'),
        ('LLY',  'Eli Lilly',          'Healthcare'),
        ('BA',   'Boeing',             'Industrials'),
        ('CAT',  'Caterpillar',        'Industrials'),
        ('HON',  'Honeywell',          'Industrials'),
        ('GE',   'GE Aerospace',       'Industrials'),
        ('AMZN', 'Amazon',             'Consumer'),
        ('WMT',  'Walmart',            'Consumer'),
        ('KO',   'Coca-Cola',          'Consumer'),
        ('NKE',  'Nike',               'Consumer')
    ) AS m(symbol, name, sector_name)
)
SELECT
    md5(LOWER(TRIM(m.symbol)))      AS company_key,
    m.symbol, m.name,
    md5(LOWER(TRIM(m.sector_name))) AS sector_key,
    m.sector_name
FROM mapping m
WHERE m.symbol IN (SELECT symbol FROM symbols);

ALTER TABLE gold.dim_company ADD PRIMARY KEY (company_key);
ALTER TABLE gold.dim_company ADD CONSTRAINT fk_dim_company_sector
  FOREIGN KEY (sector_key) REFERENCES gold.dim_sector(sector_key);

-- -- gold.dim_filing_type ----------------------------------------
CREATE TABLE gold.dim_filing_type AS
SELECT
    md5(LOWER(TRIM(code))) AS filing_type_key,
    code, label
FROM (VALUES
    ('10K',   '10-K Annual Report'),
    ('8K',    '8-K Material Event'),
    ('NEWS',  'News Article'),
    ('PRESS', 'Press Release')
) AS t(code, label);

ALTER TABLE gold.dim_filing_type ADD PRIMARY KEY (filing_type_key);

-- -- gold.dim_chunk -------------------------------------------------
-- 13,847 chunks = 3,040 10-K + 10,807 8-K. Citation lookup for the RAG layer.
-- as_of_date is the filing date; the RAG retriever enforces
-- WHERE dim_chunk.as_of_date <= query_date so future filings cannot leak.

CREATE TABLE gold.dim_chunk AS
SELECT
    chunk_key, accession AS filing_accession,
    '10K' AS source_type, md5(LOWER(TRIM('10K'))) AS filing_type_key,
    company_key, chunk_index, n_tokens,
    NULL::smallint AS page_approx,
    'https://www.sec.gov/Archives/edgar/data' AS source_url_root,
    as_of_date
FROM silver.silver_filings_10k_chunked
UNION ALL
SELECT
    chunk_key, accession, '8K', md5(LOWER(TRIM('8K'))),
    company_key, chunk_index, n_tokens, NULL::smallint,
    'https://www.sec.gov/Archives/edgar/data', as_of_date
FROM silver.silver_filings_8k_chunked;

ALTER TABLE gold.dim_chunk ADD PRIMARY KEY (chunk_key);
ALTER TABLE gold.dim_chunk ADD CONSTRAINT fk_dim_chunk_company
  FOREIGN KEY (company_key) REFERENCES gold.dim_company(company_key);
ALTER TABLE gold.dim_chunk ADD CONSTRAINT fk_dim_chunk_filing_type
  FOREIGN KEY (filing_type_key) REFERENCES gold.dim_filing_type(filing_type_key);
