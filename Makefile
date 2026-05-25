.PHONY: help init ingest build train backtest rag demo all clean docs gate

PY := /opt/anaconda3/bin/python3

help:
	@echo "advdatafinal targets:"
	@echo "  init       create Postgres DB + schemas + pgvector"
	@echo "  ingest     pull all six raw sources into raw.*"
	@echo "  build      dbt run + dbt test (silver + gold)"
	@echo "  train      walk-forward training of all three rungs"
	@echo "  backtest   compute P&L per rung + benchmarks"
	@echo "  rag        build pgvector index + run retrieval eval"
	@echo "  demo       launch Streamlit on :8501"
	@echo "  docs       dbt docs generate + serve"
	@echo "  gate       lint + pytest + dbt test"
	@echo "  all        everything end to end"

init:
	$(PY) scripts/init_db.py

ingest:
	$(PY) -m ingest.run_all

build:
	cd dbt && dbt run && dbt test

train:
	$(PY) -m models.walkforward

backtest:
	$(PY) -m backtest.run_walkforward
	$(PY) -m backtest.plot_pnl

rag:
	$(PY) -m rag.eval_retrieval

demo:
	streamlit run streamlit_app/app.py

docs:
	cd dbt && dbt docs generate && dbt docs serve

gate:
	ruff check .
	pytest -q
	cd dbt && dbt test

clean:
	rm -rf bronze/ mlruns/ dbt/target/ dbt/logs/

all: ingest build train backtest rag
