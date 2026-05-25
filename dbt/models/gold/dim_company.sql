{{ config(materialized='table') }}
-- gold.dim_company: one row per stock in the universe with sector linkage.

WITH symbols AS (
    SELECT DISTINCT symbol FROM {{ ref('silver_prices_cleaned') }}
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
    m.symbol,
    m.name,
    md5(LOWER(TRIM(m.sector_name))) AS sector_key,
    m.sector_name
FROM mapping m
WHERE m.symbol IN (SELECT symbol FROM symbols)
