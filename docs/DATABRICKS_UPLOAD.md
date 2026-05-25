# Uploading bronze/ to Databricks Volumes

The local pipeline writes raw API responses to `bronze/` on disk. To run the Databricks DLT
notebook (`appendix/01_data_pipeline.py`), this folder must be uploaded to a Volume in your
Databricks workspace. 51 MB across 477 files; takes ~2 minutes on a normal connection.

## Inventory of what gets uploaded

| Path | Files | Size | Format |
|---|---|---|---|
| `bronze/prices/` | 20 .parquet files | 1.3 MB | Parquet (yfinance OHLCV per stock) |
| `bronze/fund/income-statement/` | 20 .json files | ~700 KB | JSON (FMP income statements) |
| `bronze/fund/balance-sheet-statement/` | 20 .json files | ~700 KB | JSON (FMP balance sheets) |
| `bronze/fund/cash-flow-statement/` | 20 .json files | ~700 KB | JSON (FMP cash flows) |
| `bronze/news/` | 20 .json files | 2.7 MB | JSON (FMP news bodies) |
| `bronze/press/` | 20 .json files | 620 KB | JSON (FMP press releases) |
| `bronze/filings/10K/{symbol}/` | 95 .txt files | ~6 MB | Plain text (Item 1A Risk Factors) |
| `bronze/filings/8K/{symbol}/` | 262 .txt files | ~38 MB | Plain text (8-K bodies) |

Total: 477 files, 51 MB.

## Target structure on Databricks

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

## Option A: Databricks CLI (recommended for batch uploads)

Install once on your laptop:

```bash
pip install databricks-cli
databricks configure --token
# host:  https://<your-workspace>.cloud.databricks.com
# token: <personal access token from User Settings -> Developer -> Access tokens>
```

Then upload the entire bronze/ folder in one command (creates the Volume if missing):

```bash
cd /Users/renteeee/Desktop/advdatafinal

databricks fs mkdirs dbfs:/Volumes/advdatafinal/raw/landing/

# yfinance prices
databricks fs cp -r bronze/prices/                             dbfs:/Volumes/advdatafinal/raw/landing/prices/

# FMP fundamentals (3 endpoints in 3 sub-folders)
databricks fs cp -r bronze/fund/income-statement/              dbfs:/Volumes/advdatafinal/raw/landing/income_statement/
databricks fs cp -r bronze/fund/balance-sheet-statement/       dbfs:/Volumes/advdatafinal/raw/landing/balance_sheet/
databricks fs cp -r bronze/fund/cash-flow-statement/           dbfs:/Volumes/advdatafinal/raw/landing/cash_flow/

# FMP news + press
databricks fs cp -r bronze/news/                                dbfs:/Volumes/advdatafinal/raw/landing/news/
databricks fs cp -r bronze/press/                               dbfs:/Volumes/advdatafinal/raw/landing/press/

# SEC filings (preserves the {symbol}/ subdirectories)
databricks fs cp -r bronze/filings/10K/                         dbfs:/Volumes/advdatafinal/raw/landing/sec_10k/
databricks fs cp -r bronze/filings/8K/                          dbfs:/Volumes/advdatafinal/raw/landing/sec_8k/
```

Verify with:

```bash
databricks fs ls dbfs:/Volumes/advdatafinal/raw/landing/
databricks fs ls dbfs:/Volumes/advdatafinal/raw/landing/prices/ | head
```

## Option B: Databricks UI (drag-and-drop)

1. Open your workspace `Catalog` browser.
2. Create catalog `advdatafinal` (if not present).
3. Inside it, create schema `raw`.
4. Inside `raw`, create Volume `landing`.
5. Click `landing`, then `Upload`, and drag the `bronze/` folder. The UI preserves the directory structure.

This works for small batches but is slow for 477 files. CLI is faster.

## Option C: Compressed tarball (if upload is slow)

```bash
cd /Users/renteeee/Desktop/advdatafinal
tar czf bronze.tar.gz bronze/                                   # 51 MB -> ~15 MB compressed
databricks fs cp bronze.tar.gz dbfs:/Volumes/advdatafinal/raw/landing/bronze.tar.gz
```

Then in a Databricks notebook cell:

```python
%sh
cd /Volumes/advdatafinal/raw/landing/
tar xzf bronze.tar.gz
```

## After uploading

Run the pipeline notebook in this order:

1. `appendix/01_data_pipeline.py` — DLT pipeline; Auto Loader will pick up the new files in
   the Volume and build raw → datos_masked → silver → gold.
2. `appendix/02_ml_pipeline.py` — ML notebook; trains the 3 rungs against the gold tables
   and writes predictions + backtest into Delta. MLflow logs to the Databricks workspace.

If you set up `appendix/workflow.yaml` as a Databricks Job, the two notebooks chain
automatically (`01` triggers `02` on completion).
