# sql/  -- per-schema query files, midterm-style

This directory exposes every CREATE TABLE / CREATE VIEW / CREATE INDEX / FK constraint
side-by-side, the same way the IN014 midterm exposes its DLT cells. Reading this folder
top to bottom is enough to reconstruct the database schema by hand.

The files are numbered so the order matches the build sequence the dbt pipeline follows
under the hood. Running them in sequence with `psql` is an alternative to running `dbt run`.

| File | Schema | Tables / views |
|---|---|---|
| `00_init.sql` | (database + schemas) | CREATE DATABASE advdatafinal + 4 schemas + raw.ingest_log |
| `01_raw_schema.sql` | `raw` | 8 source-mirror tables + ingest_log |
| `02_datos_masked_schema.sql` | `datos_masked` | 4 redaction views (regex email + US phone) |
| `03_silver_schema.sql` | `silver` | 6 cleaned/enriched tables (prices, fundamentals, news, press, 10-K chunks, 8-K chunks) |
| `04_gold_dims.sql` | `gold` (dim) | 5 dimensions: dim_date, dim_sector, dim_company, dim_filing_type, dim_chunk |
| `05_gold_facts.sql` | `gold` (fct) | 5 facts: fct_sentiment_per_day, fct_embedding_per_company, fct_feature_panel_daily, fct_predictions, fct_backtest_pnl_daily, fct_rag_queries |

## How this relates to the dbt project

The Python pipeline uses **dbt-postgres** under `dbt/models/` to maintain the same tables
with full lineage, tests, and incremental refresh. The dbt models contain the canonical
DDL the engine actually runs. This `sql/` directory mirrors those models in flat `.sql`
files for grader-readability.

When the dbt files and these files diverge, the dbt files are the source of truth (because
they include Jinja templating + `ref()` calls dbt resolves at compile time). To regenerate
this directory from the compiled dbt artefacts:

```
cd dbt && dbt compile
# the compiled SQL lands in dbt/target/compiled/advdatafinal/models/<layer>/<model>.sql
```

## Why this directory exists

The midterm's strength is that **all the queries are visible in flat `.sql` cells**, in order,
with no Jinja or `ref()` indirection. A grader can read the SQL top-down without learning
dbt vocabulary first. This directory preserves that property for IN014.
