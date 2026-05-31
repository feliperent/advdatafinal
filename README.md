# advdatafinal

Advanced Data Processing and Analysis final project.

A medallion data pipeline on 20 US stocks predicting 5-day directional returns, with a RAG layer over SEC filings for explainability.

## Quickstart

    cp .env.example .env       # fill in real values
    make init                  # create Postgres DB + 4 medallion schemas + ingest_log
    make ingest                # pull all six raw sources
    make build                 # dbt run + dbt test
    make train                 # walk-forward all three rungs
    make backtest              # P&L curves + BI figures
    make rag                   # build retrieval index, run RAG eval
    make demo                  # launch Streamlit on :8501
    make all                   # ingest -> build -> train -> backtest -> rag

## Repo layout

| Directory | Purpose |
|---|---|
| `ingest/` | Six API fetchers + audit log; raw schema landing |
| `silver_text/` | FinBERT scoring, MiniLM chunk-and-embed |
| `dbt/` | Medallion transformations (silver + gold) |
| `models/` | Three XGBoost rungs + walk-forward + PCA |
| `backtest/` | Walk-forward backtest + BI figures |
| `rag/` | Retrieval (cosine on bytea-packed embeddings) + Claude Haiku answer |
| `streamlit_app/` | Demo UI on :8501 |
| `databricksstuff/` | Databricks DLT pipeline + ML notebook mirror |
| `sql/` | (removed; superseded by dbt models) |
| `scripts/` | One-time bootstrap (`init_db.py`) |
| `tests/` | pytest suite (ingest helpers + leakage guard) |

## Storage choice for embeddings

`pgvector` is intentionally not used. The Postgres install on the dev machine (EDB 18) does not have the extension available without a sudo-level rebuild. Embeddings are stored as bytea-packed float32 vectors and cosine retrieval is done in Python with numpy. At 14,000 chunks the latency is ~50 milliseconds per query, invisible to the user. See `silver_text/embed.py` and `rag/retrieve.py`.
