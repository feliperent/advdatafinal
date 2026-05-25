# Uploading bronze/ to Databricks Volumes

The local pipeline writes raw API responses to `bronze/` on disk. The DLT pipeline
(`appendix/pipelinedatos.sql`) reads them from a Databricks Volume, so the folder needs to
go up to the workspace once. About 51 MB across 477 files.

## What gets uploaded

| Local path | Files | Format |
|---|---|---|
| `bronze/prices/` | 20 .parquet | yfinance OHLCV |
| `bronze/fund/income-statement/` | 20 .json | FMP income statements |
| `bronze/fund/balance-sheet-statement/` | 20 .json | FMP balance sheets |
| `bronze/fund/cash-flow-statement/` | 20 .json | FMP cash flows |
| `bronze/news/` | 20 .json | FMP news bodies |
| `bronze/press/` | 20 .json | FMP press releases |
| `bronze/filings/10K/{symbol}/` | 95 .txt | 10-K Item 1A |
| `bronze/filings/8K/{symbol}/` | 262 .txt | 8-K bodies |

## Where it lands on Databricks

```
/Volumes/advdatafinal/raw/landing/prices/<symbol>.parquet
/Volumes/advdatafinal/raw/landing/income_statement/<symbol>.json
/Volumes/advdatafinal/raw/landing/balance_sheet/<symbol>.json
/Volumes/advdatafinal/raw/landing/cash_flow/<symbol>.json
/Volumes/advdatafinal/raw/landing/news/<symbol>.json
/Volumes/advdatafinal/raw/landing/press/<symbol>.json
/Volumes/advdatafinal/raw/landing/sec_10k/<symbol>/<filing_date>.txt
/Volumes/advdatafinal/raw/landing/sec_8k/<symbol>/<filing_date>_<accession>.txt
```

## Upload commands

One-time setup:

```bash
pip install databricks-cli
databricks configure --token
```

Upload (creates the Volume if missing):

```bash
cd /Users/renteeee/Desktop/advdatafinal
databricks fs mkdirs dbfs:/Volumes/advdatafinal/raw/landing/

databricks fs cp -r bronze/prices/                       dbfs:/Volumes/advdatafinal/raw/landing/prices/
databricks fs cp -r bronze/fund/income-statement/        dbfs:/Volumes/advdatafinal/raw/landing/income_statement/
databricks fs cp -r bronze/fund/balance-sheet-statement/ dbfs:/Volumes/advdatafinal/raw/landing/balance_sheet/
databricks fs cp -r bronze/fund/cash-flow-statement/     dbfs:/Volumes/advdatafinal/raw/landing/cash_flow/
databricks fs cp -r bronze/news/                         dbfs:/Volumes/advdatafinal/raw/landing/news/
databricks fs cp -r bronze/press/                        dbfs:/Volumes/advdatafinal/raw/landing/press/
databricks fs cp -r bronze/filings/10K/                  dbfs:/Volumes/advdatafinal/raw/landing/sec_10k/
databricks fs cp -r bronze/filings/8K/                   dbfs:/Volumes/advdatafinal/raw/landing/sec_8k/
```

Verify:

```bash
databricks fs ls dbfs:/Volumes/advdatafinal/raw/landing/
```

## Parquet prep note

`yfinance` writes prices with timestamp[ns] in the `date` column. Spark in Databricks
rejects INT64 nanosecond parquet timestamps by default. Before uploading, rewrite
the parquet files so `date` becomes a YYYY-MM-DD string column named `trade_date`.
A small pyarrow script at `scripts/fix_prices_parquet.py` does this in place.

## After uploading

1. Run the DLT pipeline that points at `appendix/pipelinedatos.sql` (raw, datos_masked,
   silver, gold dimensions).
2. Open `appendix/mlpipeline.py` and run it (FinBERT, MiniLM, PCA, 3 rungs, backtest).
3. The Job in `appendix/workflow.yaml` chains the two if you want them automated.
