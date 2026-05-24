# advdatafinal

IN014 Advanced Data Processing and Analysis final project. Felipe Rentería Zuleta, 2026.

A medallion data pipeline on 20 US stocks predicting 5-day directional returns, with a RAG layer over SEC filings for explainability.

See `../finprojectv1/docs/IN014_advdatafinal_full_description.md` for the full design.

## Quickstart

    cp .env.example .env       # fill in real values
    make init                  # create Postgres DB + schemas + pgvector
    make ingest                # pull all six raw sources
    make build                 # dbt run + dbt test
    make train                 # walk-forward all three rungs
    make backtest              # P&L curves
    make rag                   # build pgvector index, RAG eval
    make demo                  # launch Streamlit on :8501
    make all                   # everything end to end

## Repo layout

See `docs/repo_layout.md` for the directory tree and what each module is for.
