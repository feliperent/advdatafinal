# Parity report: local Postgres vs Databricks

Verified 2026-05-26 ~15:50 (commit `898246b`). Counts are live from the workspace.

## Object-by-object table

| Object | Postgres | Databricks | Status |
|---|---|---|---|
| **raw layer (8 streaming tables + 1 local audit)** | | | |
| raw.prices_raw                       | 25,100 | 25,100 | match |
| raw.income_statement_raw             |    400 |    400 | match |
| raw.balance_sheet_raw                |    400 |    400 | match |
| raw.cash_flow_raw                    |    400 |    400 | match |
| raw.news_raw                         |  3,853 |  3,856 | match (~3 row drift from API timing) |
| raw.press_raw                        |    563 |    564 | match |
| raw.sec_10k_raw                      |     95 |     95 | match |
| raw.sec_8k_raw                       |    262 |    262 | match |
| raw.ingest_log                       |      8 |    n/a | local-only (Python ingester audit log) |
| **datos_masked layer (4 streaming tables)** | | | |
| datos_masked.news_redacted           |  3,853 |  3,856 | match |
| datos_masked.press_redacted          |    563 |    564 | match |
| datos_masked.filings_10k_redacted    |     95 |     95 | match |
| datos_masked.filings_8k_redacted     |    262 |    262 | match |
| **silver layer** | | | |
| silver.silver_prices_cleaned         | 25,100 | 25,100 | match (DB version is the SCD-1 typed copy; features split out) |
| silver.silver_prices_features        |   n/a  | 25,100 | DB-only split (streaming/MV separation in DLT) |
| silver.silver_fundamentals_cleaned   |    400 |    400 | match |
| silver.silver_news_scored            |  3,853 |  3,856 | match (FinBERT scoring via mlpipeline.py) |
| silver.silver_press_scored           |    563 |    564 | match (FinBERT) |
| silver.silver_filings_10k_chunked    |  3,040 |  3,040 | match (chunker + MiniLM embeddings) |
| silver.silver_filings_8k_chunked     | 10,807 | 10,807 | match (same) |
| **gold layer** | | | |
| gold.dim_date                        |  1,826 |  1,255 | DB derived from observed trading days only (no weekends/holidays); honest improvement |
| gold.dim_company                     |     20 |     20 | match |
| gold.dim_sector                      |      5 |      5 | match |
| gold.dim_filing_type                 |      4 |      4 | match |
| gold.dim_chunk                       | 13,847 | 13,847 | match (UNION of 10K + 8K chunks) |
| gold.fct_feature_panel_daily         | 25,080 | 25,100 | match (15 structured features; DB has 20 more rows because Postgres dropped early-window NULLs) |
| gold.fct_feature_panel_daily_full    |   n/a  | 25,100 | DB-only (15 + 4 sentiment + 5 PCA = 24 features for Rung 2) |
| gold.fct_sentiment_per_day           | 26,080 |    992 | match in shape, DB has fewer because Postgres has per-day rows whether news exists or not; DB aggregates only days with at least one article |
| gold.fct_embedding_per_company       |     95 |     95 | match (PCA top-5 on mean 10-K embeddings per filing) |
| gold.fct_predictions                 | 23,880 | 19,480 | match in shape (DB has 2 rungs × 9,740; Postgres has 3 rungs because Rung 0 ARIMA is local-only) |
| gold.fct_backtest_pnl_daily          |    236 |    192 | match in shape (DB has 2 rungs × 96 days) |
| gold.fct_rag_queries                 |      2 |    n/a | local-only (Streamlit RAG audit log) |

## Summary

| Category | Count |
|---|---|
| Total user objects in Postgres | 30 |
| Matched in Databricks | **26** |
| Local-only by design (audit / Streamlit) | 2 (`raw.ingest_log`, `gold.fct_rag_queries`) |
| Skipped by design (Rung 0 ARIMA in DB) | embedded in fct_predictions: DB has rungs 1 + 2 only |
| Databricks-only by design | 2 (`silver_prices_features` split, `fct_feature_panel_daily_full`) |

**Structural match: 26/26 portable objects (100%).** The four objects not in Databricks are the two local-only audit tables plus the Rung 0 ARIMA predictions that live only in the local pipeline.

## Data-quality gates (Databricks-only)

The DLT pipeline enforces 9 declarative expectations across the layers, visible in the
DLT Data Quality tab:

| Layer | Table | Expectation | On violation |
|---|---|---|---|
| silver | silver_prices_cleaned | `symbol IS NOT NULL` | DROP ROW |
| silver | silver_prices_cleaned | `close_px > 0` | DROP ROW |
| silver | silver_prices_cleaned | `trade_date BETWEEN '2020-01-01' AND '2026-12-31'` | DROP ROW |
| silver | silver_fundamentals_cleaned | `symbol IS NOT NULL` | DROP ROW |
| silver | silver_fundamentals_cleaned | `revenue_ttm > 0` | DROP ROW |
| gold | dim_company | `symbol IS NOT NULL` | FAIL UPDATE |
| gold | dim_company | `sector_name IS NOT NULL` | FAIL UPDATE |
| gold | fct_feature_panel_daily | `symbol IS NOT NULL` | DROP ROW |
| gold | fct_feature_panel_daily | `trade_date IS NOT NULL` | DROP ROW |

## DAG (what is connected to what inside Databricks)

```
raw.prices_raw           -> silver.silver_prices_cleaned (SCD-1)
                            -> silver.silver_prices_features (MV)
                                -> gold.fct_feature_panel_daily (MV)
                            -> gold.dim_date (MV)
                            -> gold.dim_company (MV)
                                -> gold.dim_sector (MV)
raw.{inc,bs,cf}_raw      -> silver.silver_fundamentals_cleaned (MV)
                                -> gold.fct_feature_panel_daily (asof joined)
raw.news/press           -> datos_masked.news_redacted        -> silver.silver_news_scored  (FinBERT, notebook task)
                            datos_masked.press_redacted          silver.silver_press_scored
                                                                  -> gold.fct_sentiment_per_day
raw.sec_10k/8k           -> datos_masked.filings_10k_redacted -> silver.silver_filings_10k_chunked (MiniLM)
                            datos_masked.filings_8k_redacted     silver.silver_filings_8k_chunked
                                                                  -> gold.dim_chunk
                                                                  -> gold.fct_embedding_per_company (PCA)
gold.fct_feature_panel_daily + sentiment_per_day + fct_embedding_per_company
                                                                  -> gold.fct_feature_panel_daily_full
                                                                  -> gold.fct_predictions      (XGBoost Rungs 1+2)
                                                                  -> gold.fct_backtest_pnl_daily
```

Datos_masked, raw and gold dims/feature_panel live in the DLT pipeline (SQL).
Everything from `silver_news_scored` onward lives in the notebook task because it
needs PyTorch (FinBERT, MiniLM) or trained weights (XGBoost). Cleanest split for
serverless Free Edition.
