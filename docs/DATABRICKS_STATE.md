# State of the project: Databricks workspace + local Postgres

A snapshot of what is currently deployed, where, and the exact commands to reproduce.

## Databricks (what is up right now)

### Catalog and schemas

`advdatafinal` catalog with four schemas mirroring the local Postgres database:

```
advdatafinal
  raw            -- 8 streaming tables (Auto Loader from /Volumes/advdatafinal/raw/landing/)
  datos_masked   -- 4 streaming tables with email + US-phone regex masking
  silver         -- silver_prices_cleaned (SCD-1) + 2 materialised views
  gold           -- 4 materialised view dimensions
```

A volume `advdatafinal.raw.landing` holds the bronze files, organised one folder per
source: `prices/`, `income_statement/`, `balance_sheet/`, `cash_flow/`, `news/`, `press/`,
`sec_10k/<symbol>/`, `sec_8k/<symbol>/`.

### Tables and row counts (verified 2026-05-26 01:38)

| Schema | Table | Type | Rows | Derived from |
|---|---|---|---|---|
| raw | prices_raw | streaming table | 25,100 | Volume parquet (Auto Loader) |
| raw | income_statement_raw | streaming table | 400 | Volume JSON (Auto Loader) |
| raw | balance_sheet_raw | streaming table | 400 | Volume JSON (Auto Loader) |
| raw | cash_flow_raw | streaming table | 400 | Volume JSON (Auto Loader) |
| raw | news_raw | streaming table | 3,856 | Volume JSON (Auto Loader) |
| raw | press_raw | streaming table | 564 | Volume JSON (Auto Loader) |
| raw | sec_10k_raw | streaming table | 95 | Volume text (Auto Loader) |
| raw | sec_8k_raw | streaming table | 262 | Volume text (Auto Loader) |
| datos_masked | news_redacted | streaming table | 3,856 | raw.news_raw |
| datos_masked | press_redacted | streaming table | 564 | raw.press_raw |
| datos_masked | filings_10k_redacted | streaming table | 95 | raw.sec_10k_raw |
| datos_masked | filings_8k_redacted | streaming table | 262 | raw.sec_8k_raw |
| silver | silver_prices_cleaned | SCD-1 streaming | 25,100 | raw.prices_raw via APPLY CHANGES |
| silver | silver_prices_features | materialised view | 25,100 | silver_prices_cleaned + window functions |
| silver | silver_fundamentals_cleaned | materialised view | 400 | raw.{inc,bs,cf}_statement_raw + TTM |
| gold | dim_date | materialised view | 1,255 | DISTINCT trade_date from silver_prices_cleaned |
| gold | dim_company | materialised view | 20 | DISTINCT symbol from silver_prices_cleaned + sector map |
| gold | dim_sector | materialised view | 5 | DISTINCT sector_* from dim_company |
| gold | dim_filing_type | materialised view | 4 | datos_masked.*_redacted COUNT > 0 |
| gold | fct_feature_panel_daily | materialised view | 25,100 | silver_prices_features + silver_fundamentals_cleaned (asof) |

Every dimension and the only fact materialised in the DLT DAG derive from a silver
table or another gold table - no static VALUES are emitted into gold any more.

### DAG lineage

```
raw.prices_raw           --> silver.silver_prices_cleaned (SCD-1)
                            --> silver.silver_prices_features (MV, window fn)
                                  --> gold.fct_feature_panel_daily (MV)
                            --> gold.dim_date (MV, DISTINCT trade_date)
                            --> gold.dim_company (MV, DISTINCT symbol + map)
                                  --> gold.dim_sector (MV, DISTINCT sector)
raw.{inc,bs,cf}_raw      --> silver.silver_fundamentals_cleaned (MV)
                                  --> gold.fct_feature_panel_daily (joined asof)
raw.news_raw             --> datos_masked.news_redacted        --> gold.dim_filing_type
raw.press_raw            --> datos_masked.press_redacted       --> gold.dim_filing_type
raw.sec_10k_raw          --> datos_masked.filings_10k_redacted --> gold.dim_filing_type
raw.sec_8k_raw           --> datos_masked.filings_8k_redacted  --> gold.dim_filing_type
```

The text-side gold tables (`fct_sentiment_per_day`, `fct_embedding_per_company`,
`fct_predictions`, `fct_backtest_pnl_daily`) live in `mlpipeline.py` because they
need PyTorch (FinBERT, MiniLM) and sklearn / xgboost (PCA, training, backtest).
The notebook runs as the second task in `databricksstuff/workflow.yaml`, reading
`gold.fct_feature_panel_daily` and writing back to gold.

### DLT pipeline

- Name: `advdatafinal_dlt`
- Source notebook: `/Workspace/Repos/lfrenteria33@gmail.com/advdatafinal/databricksstuff/pipelinedatos`
- Catalog / default schema: `advdatafinal` / `raw`
- Compute: Serverless
- Mode: Triggered (manual full refresh)
- Spark config: `spark.sql.parquet.enableNanosAsLong=true` (left in for safety even though prices were re-encoded)
- Last full refresh: COMPLETED on 2026-05-26 around 01:17

### Notebooks in the Databricks Repo

`/Workspace/Repos/lfrenteria33@gmail.com/advdatafinal/`

- `databricksstuff/pipelinedatos` (SQL notebook) - the DLT pipeline above
- `databricksstuff/mlpipeline` (Python notebook) - FinBERT + MiniLM + PCA + 3 rungs + backtest stub
- `databricksstuff/workflow.yaml` - Databricks Job chaining DLT and the ML notebook
- `databricksstuff/grants.sql` - one-time CATALOG and SCHEMA grants
- `databricksstuff/dlt_equivalents.sql` - local-Postgres-to-DLT mapping reference

### SQL warehouse

`Serverless Starter Warehouse` (id `f75f590c1d849a46`). Used for ad-hoc queries from
the SQL Editor and for the grants script.

## What was actually wrong before it worked

The DLT pipeline failed several times before COMPLETED. Each failure and its fix:

1. **No write permission on advdatafinal.raw.**
   The catalog was created in the UI but the user identity lacked CREATE TABLE / MODIFY.
   Fix: run `databricksstuff/grants.sql` once in a SQL Editor query (USE CATALOG, USE SCHEMA,
   CREATE TABLE, MODIFY, CREATE MATERIALIZED VIEW on every schema).

2. **Notebook path included the .sql extension.**
   `/Workspace/Repos/.../databricksstuff/pipelinedatos.sql` returned NOTEBOOK_NOT_FOUND_EXCEPTION.
   Databricks identifies SQL notebooks without extensions. Fix: drop the `.sql`.

3. **Parquet timestamps were nanoseconds.**
   yfinance writes `date` as `timestamp[ns]`. Spark rejects INT64 nanosecond parquet
   with `[PARQUET_TYPE_ILLEGAL]`. Fix: rewrite all 20 price parquet files with pyarrow,
   converting `date` to a YYYY-MM-DD string column also renamed to `trade_date`.

4. **JSON files were single-array, not NDJSON.**
   FMP files are `[{...}, {...}]`. Spark `read_files(format=>'json')` defaults to
   NDJSON and only emits one bad row per file. Fix: add `multiLine => 'true'` to all
   five FMP endpoints.

5. **CREATE CATALOG / SCHEMA / VOLUME inside a DLT pipeline.**
   DLT only owns tables and views. Those DDL statements at the top of the file made
   the parser fail. Fix: remove them from `pipelinedatos.sql`. The catalog and the
   volume already existed (created via the UI / CLI).

6. **Dimension tables declared as STREAMING TABLE with no streaming source.**
   `dim_date`, `dim_sector`, `dim_company`, `dim_filing_type` are computed from
   VALUES or batch SELECT DISTINCT - no streaming source available. Fix: change to
   `CREATE OR REFRESH MATERIALIZED VIEW`.

7. **dim_company tried to STREAM from a SCD-1 table.**
   `STREAM(advdatafinal.silver.silver_prices_cleaned)` after APPLY CHANGES INTO is
   restricted. Fix: read it as batch with `SELECT DISTINCT symbol`.

8. **Mid-merge conflict in the Databricks Repo blocked all pulls.**
   The user edited `pipelinedatos.sql` in the Databricks editor while we pushed to
   GitHub, hitting a merge state that `databricks repos update` refused to clear.
   Fix: delete the repo via CLI and re-clone fresh.

## How to run the full workflow locally (Postgres connection)

The DBClient saved queries you see are SELECT and DDL references. The actual data
INSERTs run from Python (`ingest/*.py`) because every source needs HTTP calls and
not just SQL. That is why there is no "injection query" file you can run from
DBClient.

The local workflow has three layers:

1. SQL DDL (creates schemas and empty tables)
2. Python ingest (calls APIs, runs INSERT...ON CONFLICT)
3. dbt + Python ML (transformations and models)

### One-time setup

```bash
cd /Users/renteeee/Desktop/advdatafinal
cp .env.example .env       # fill in DATABASE_URL + FMP_API_KEY + ANTHROPIC_API_KEY
uv sync                    # install Python deps
```

`.env` needs:
```
DATABASE_URL=postgresql://postgres:rente@127.0.0.1:5432/advdatafinal
FMP_API_KEY=...
ANTHROPIC_API_KEY=...
```

### Step 1: build the schemas and empty tables

```bash
make init
```

That runs `sql/00_init.sql` through `sql/05_gold_facts.sql` against your local Postgres,
creating the four schemas and all DDL.

### Step 2: ingest raw data

```bash
make ingest
```

That runs `ingest/run_all.py`, which calls every fetcher in order:

- `ingest/fetch_prices.py` - yfinance OHLCV, writes parquet to `bronze/prices/`, then INSERTs into `raw.prices_raw`
- `ingest/fetch_fundamentals.py` - FMP income / balance / cash flow, writes JSON, INSERTs into the three `raw.*_raw` tables
- `ingest/fetch_news.py` - FMP news, INSERTs into `raw.news_raw`
- `ingest/fetch_press.py` - FMP press releases, INSERTs into `raw.press_raw`
- `ingest/fetch_sec.py` - edgartools for 10-K Item 1A and 8-K bodies, INSERTs into `raw.sec_10k_raw` and `raw.sec_8k_raw`

All use `INSERT ... ON CONFLICT DO UPDATE` so they are idempotent. You can re-run safely.

### Step 3: dbt transformations

```bash
make build
```

That runs `dbt run` + `dbt test` against the four medallion schemas:

- `datos_masked.*_redacted` views (regex masking)
- `silver.silver_prices_cleaned` (typed, deduped)
- `silver.silver_fundamentals_cleaned` (joined + TTM)
- `gold.dim_date / dim_sector / dim_company / dim_filing_type / dim_chunk`
- `gold.fct_sentiment_per_day`
- `gold.fct_feature_panel_daily` (the 30-feature ML table)

### Step 4: text models

```bash
python silver_text/finbert_score.py    # FinBERT sentiment on news + press
python silver_text/build_filing_chunks.py   # chunker + MiniLM embeddings for 10-K and 8-K
python models/build_filing_pca.py      # 5 principal components per company
```

### Step 5: train models and backtest

```bash
make train       # walks forward all 3 rungs, writes gold.fct_predictions, logs to MLflow
make backtest    # weekly walk-forward, writes gold.fct_backtest_pnl_daily
```

### Step 6: RAG and demo

```bash
make rag         # build pgvector index, evaluate retrieval
make demo        # Streamlit on :8501
```

### Or do it all in one shot

```bash
make all
```

That is the order `make init -> ingest -> build -> train -> backtest -> rag -> demo`.

## Mapping local objects to Databricks objects

Same object names, same shapes. Every local Postgres table has a Databricks equivalent
in the same schema and with the same name. The Python text models (FinBERT, MiniLM, PCA)
run identically in both places: locally they write to Postgres, in Databricks they would
write to Delta tables (`mlpipeline.py` carries the same logic but reads from Spark and
logs to MLflow).

The only objects that exist in only one place:
- `raw.ingest_log` is a local-only audit table written by the Python fetchers
- The DLT pipeline's `pipelines.parquet.enableNanosAsLong` and the bronze parquet rewrite
  step are Databricks-only safety measures
