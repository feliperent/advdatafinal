.PHONY: help init ingest build train backtest rag demo all clean docs gate rebuild-rung2

PY := /opt/anaconda3/bin/python3

help:
	@echo "advdatafinal targets:"
	@echo "  init       create Postgres DB + 4 medallion schemas + ingest_log"
	@echo "  ingest     pull all six raw sources into raw.*"
	@echo "  build      dbt run + dbt test (silver + gold)"
	@echo "  train      walk-forward training of all three rungs"
	@echo "  backtest   compute P&L per rung + BI figures"
	@echo "  rag        build retrieval index + run retrieval eval"
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
	$(PY) -m models.rung0_arima
	$(PY) -m models.rung1_xgb_structured
	$(PY) -m models.rung2_xgb_with_text

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

rebuild-rung2:
	$(PY) -m silver_text.finbert_score
	cd dbt && dbt run --select fct_sentiment_per_day fct_feature_panel_daily
	$(PY) -m models.rung2_xgb_with_text
	$(PY) -m backtest.run_walkforward
	$(PY) -m backtest.plot_pnl
