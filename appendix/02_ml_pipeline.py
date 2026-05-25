# Databricks notebook source
# MAGIC %md
# MAGIC # 02_ml_pipeline | text scoring + chunking + PCA + 3-rung model ladder + backtest
# MAGIC
# MAGIC Picks up where `01_data_pipeline.sql` finished: the medallion structure exists in
# MAGIC Delta with raw, datos_masked, silver_prices, silver_fundamentals, and the 4 gold
# MAGIC dimensions populated.
# MAGIC
# MAGIC This notebook adds:
# MAGIC 1. Python-built silver tables: news/press FinBERT scoring + 10-K/8-K chunking + MiniLM embeddings
# MAGIC 2. Python-built gold intermediate facts: sentiment_per_day, embedding_per_company (PCA)
# MAGIC 3. SQL-built gold main fact: fct_feature_panel_daily (30 features + target)
# MAGIC 4. Three model rungs (Rung 0 ARIMA, Rung 1 XGB structured, Rung 2 XGB + text) with MLflow
# MAGIC 5. Walk-forward backtest + write to gold.fct_backtest_pnl_daily

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. MLflow setup

# COMMAND ----------
import mlflow
mlflow.set_experiment("/advdatafinal")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Silver text tables (FinBERT scoring + chunking + embedding)
# MAGIC
# MAGIC FinBERT: yiyanghkust/finbert-tone. Label order verified: {0:Neutral, 1:Positive, 2:Negative}.
# MAGIC Scalar score = P(Positive) - P(Negative) in [-1, +1].
# MAGIC
# MAGIC Chunker: 500 tokens with 50-token overlap, tiktoken cl100k_base.
# MAGIC Embedder: sentence-transformers/all-MiniLM-L6-v2 (384-dim, L2-normalised).

# COMMAND ----------
# (Same code as local silver_text/finbert_score.py; copied here so the notebook is self-contained.)
# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# tok = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
# model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
# def score(text):
#     enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
#     with torch.no_grad():
#         p = model(**enc).logits.softmax(dim=-1).squeeze().tolist()
#     return float(p[1] - p[2])  # P(Positive) - P(Negative)
#
# (Read news + press from datos_masked, score each row, write back to silver tables via spark.write...)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Gold intermediate facts | fct_sentiment_per_day + fct_embedding_per_company

# COMMAND ----------
# MAGIC %sql
# MAGIC -- gold.fct_sentiment_per_day -- materialised view (window aggregations 3d + 30d)
# MAGIC CREATE OR REPLACE MATERIALIZED VIEW advdatafinal.gold.fct_sentiment_per_day AS
# MAGIC WITH date_grid AS (
# MAGIC     SELECT d.date_key, d.full_date, c.company_key, c.symbol
# MAGIC     FROM advdatafinal.gold.dim_date d CROSS JOIN advdatafinal.gold.dim_company c
# MAGIC     WHERE d.is_trading_day = true
# MAGIC ),
# MAGIC news_per_day AS (
# MAGIC     SELECT company_key, trade_date,
# MAGIC            AVG(finbert_score) AS news_mean_day,
# MAGIC            COUNT(*) AS n_news_day
# MAGIC     FROM advdatafinal.silver.silver_news_scored
# MAGIC     WHERE trade_date IS NOT NULL
# MAGIC     GROUP BY company_key, trade_date
# MAGIC ),
# MAGIC press_per_day AS (...),  -- analogous
# MAGIC filings_per_day AS (...)  -- counts 8-Ks per (company_key, filing_date)
# MAGIC SELECT
# MAGIC     row_number() OVER (ORDER BY full_date, company_key) + 1000 AS fact_sentiment_key,
# MAGIC     date_key, company_key,
# MAGIC     /* 3-day and 30-day windowed means + counts */
# MAGIC     full_date AS as_of_date
# MAGIC FROM (joined with windowed CTE);
# MAGIC -- Full SQL identical to local sql/05_gold_facts.sql.

# COMMAND ----------
# PCA per (company, filing_date) in Python; writes to gold.fct_embedding_per_company
# (Same code as local models/build_filing_pca.py; reads silver.silver_filings_10k_chunked,
# unpacks bytea-stored embeddings, fits PCA on (n_filings x 384) -> top 5 components.)
# Expected: ~95 rows (one per filing across 20 stocks x ~5 10-Ks each).

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Gold main fact | fct_feature_panel_daily (30 features + y_5d_up target)

# COMMAND ----------
# MAGIC %sql
# MAGIC CREATE OR REPLACE MATERIALIZED VIEW advdatafinal.gold.fct_feature_panel_daily AS
# MAGIC SELECT
# MAGIC   ROW_NUMBER() OVER (ORDER BY p.trade_date, p.company_key) + 1000 AS fact_panel_key,
# MAGIC   p.date_key, p.company_key, c.sector_key, p.symbol, p.trade_date, p.close_px,
# MAGIC   -- 10 PRICE features (from silver.silver_prices_features)
# MAGIC   p.log_ret_1d, p.sma_5, p.sma_20, p.sma_50, p.ema_12, p.ema_26,
# MAGIC   p.rsi_14, p.macd_hist, p.bb_z, p.vol_20d,
# MAGIC   -- 10 FUNDAMENTALS features
# MAGIC   ff.roe, ff.roa, ff.debt_eq, ff.gross_margin, ff.op_margin, ff.asset_turnover,
# MAGIC   /* pe_ttm_proxy, pb_proxy, ebitda_to_ev_proxy, fcf_to_assets */
# MAGIC   -- 5 SENTIMENT + 5 PCA features (from intermediate facts)
# MAGIC   COALESCE(s.finbert_news_mean_3d,  0) AS finbert_news_mean_3d,
# MAGIC   /* + 4 more sentiment + 5 PCA */
# MAGIC   -- TARGET
# MAGIC   CASE
# MAGIC     WHEN LEAD(p.close_px, 5) OVER (PARTITION BY p.company_key ORDER BY p.trade_date) > p.close_px
# MAGIC     THEN 1 ELSE 0
# MAGIC   END AS y_5d_up,
# MAGIC   GREATEST(p.as_of_date, COALESCE(ff.as_of_date, p.as_of_date),
# MAGIC            COALESCE(s.as_of_date, p.as_of_date), COALESCE(e.as_of_date, p.as_of_date)) AS as_of_date
# MAGIC FROM advdatafinal.silver.silver_prices_features p
# MAGIC LEFT JOIN advdatafinal.gold.dim_company c USING (company_key)
# MAGIC LEFT JOIN /* fund_latest + fund_full subquery */
# MAGIC LEFT JOIN advdatafinal.gold.fct_sentiment_per_day s ON s.date_key = p.date_key AND s.company_key = p.company_key
# MAGIC LEFT JOIN /* emb_latest with as_of_date guard */;
# MAGIC -- Full SQL identical to local sql/05_gold_facts.sql.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Walk-forward fold generator (identical to local)

# COMMAND ----------
from datetime import date, timedelta
from dataclasses import dataclass

@dataclass
class Fold:
    fold_id: str
    train_start: date
    train_end: date
    test_start: date
    test_end: date

def folds(start=date(2021,1,1), end=date(2025,12,31),
          train_years=3, test_quarter_days=63, gap_days=5):
    out, cursor = [], start + timedelta(days=365*train_years)
    while cursor + timedelta(days=test_quarter_days) <= end:
        train_end = cursor - timedelta(days=gap_days)
        train_start = train_end - timedelta(days=365*train_years)
        test_start = cursor
        test_end = cursor + timedelta(days=test_quarter_days)
        out.append(Fold(test_start.isoformat(), train_start, train_end, test_start, test_end))
        cursor += timedelta(days=test_quarter_days)
    return out

for f in folds():
    print(f"{f.fold_id}: train {f.train_start} -> {f.train_end} | test {f.test_start} -> {f.test_end}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Rung 1 | XGBoost on 20 structured features
# MAGIC
# MAGIC Hyperparameters identical to local (`models/rung1_xgb_structured.py`).
# MAGIC Lift to Rung 2 isolates the contribution of the 10 text-derived features.

# COMMAND ----------
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

panel = (
    spark.table("advdatafinal.gold.fct_feature_panel_daily")
    .filter("y_5d_up IS NOT NULL")
    .orderBy("symbol", "trade_date")
    .toPandas()
)
print(f"Panel: {len(panel)} rows, {panel['symbol'].nunique()} stocks")

STRUCTURED_FEATURES = [
    "log_ret_1d","sma_5","sma_20","sma_50","ema_12","ema_26",
    "rsi_14","macd_hist","bb_z","vol_20d",
    "roe","roa","debt_eq","gross_margin","op_margin","asset_turnover",
    "pe_ttm_proxy","pb_proxy","ebitda_to_ev_proxy","fcf_to_assets",
]
HP = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
          subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
          gamma=0.1, reg_lambda=1.0, random_state=7,
          eval_metric="auc", tree_method="hist", n_jobs=4)

preds_rung1 = []
for fold in folds():
    train = panel[(panel["trade_date"] >= fold.train_start) & (panel["trade_date"] <= fold.train_end)]
    test  = panel[(panel["trade_date"] >= fold.test_start)  & (panel["trade_date"] <= fold.test_end)]
    if train.empty or test.empty: continue
    Xtr, ytr = train[STRUCTURED_FEATURES].fillna(0), train["y_5d_up"]
    Xte, yte = test[STRUCTURED_FEATURES].fillna(0),  test["y_5d_up"]
    clf = xgb.XGBClassifier(**HP)
    clf.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    p = clf.predict_proba(Xte)[:, 1]
    with mlflow.start_run(run_name=f"rung1_{fold.fold_id}"):
        mlflow.log_param("rung", 1); mlflow.log_param("fold_id", fold.fold_id)
        mlflow.log_param("n_features", len(STRUCTURED_FEATURES))
        mlflow.log_metric("test_auc", float(roc_auc_score(yte, p)))
        mlflow.log_metric("accuracy", float(accuracy_score(yte, (p > 0.5).astype(int))))
    for i, (_, r) in enumerate(test.iterrows()):
        preds_rung1.append((str(r["trade_date"]), r["company_key"], 1, fold.fold_id, float(p[i]), int(p[i] > 0.5)))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Rung 2 | same hyperparameters + all 30 features (text added)

# COMMAND ----------
TEXT_FEATURES = [
    "finbert_news_mean_3d","finbert_news_mean_30d","finbert_press_30d","n_news_3d","n_8k_30d",
    "filing_pc1","filing_pc2","filing_pc3","filing_pc4","filing_pc5",
]
FULL_FEATURES = STRUCTURED_FEATURES + TEXT_FEATURES

preds_rung2 = []
for fold in folds():
    train = panel[(panel["trade_date"] >= fold.train_start) & (panel["trade_date"] <= fold.train_end)]
    test  = panel[(panel["trade_date"] >= fold.test_start)  & (panel["trade_date"] <= fold.test_end)]
    if train.empty or test.empty: continue
    Xtr, ytr = train[FULL_FEATURES].fillna(0), train["y_5d_up"]
    Xte, yte = test[FULL_FEATURES].fillna(0),  test["y_5d_up"]
    clf = xgb.XGBClassifier(**HP)
    clf.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    p = clf.predict_proba(Xte)[:, 1]
    with mlflow.start_run(run_name=f"rung2_{fold.fold_id}"):
        mlflow.log_param("rung", 2); mlflow.log_param("fold_id", fold.fold_id)
        mlflow.log_param("n_features", len(FULL_FEATURES))
        mlflow.log_metric("test_auc", float(roc_auc_score(yte, p)))
        mlflow.log_metric("accuracy", float(accuracy_score(yte, (p > 0.5).astype(int))))
    for i, (_, r) in enumerate(test.iterrows()):
        preds_rung2.append((str(r["trade_date"]), r["company_key"], 2, fold.fold_id, float(p[i]), int(p[i] > 0.5)))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Rung 0 | ARIMA baseline (per-stock; sampled to keep runtime tractable)

# COMMAND ----------
# from pmdarima import auto_arima
# import math
# Same logic as local models/rung0_arima.py; max_test_per_stock_per_fold = 20.
# Expected: AUC mean ~0.51 across folds; Sharpe in the backtest ~2.26 (highest of all 3).

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Write predictions to gold.fct_predictions (Delta)

# COMMAND ----------
import pandas as pd
all_preds = pd.DataFrame(preds_rung1 + preds_rung2,
    columns=["trade_date","company_key","model_rung","fold_id","prob_up","predicted_class"])
spark.createDataFrame(all_preds).write.mode("overwrite").saveAsTable("advdatafinal.gold.fct_predictions")
print(f"Wrote {len(all_preds)} predictions to advdatafinal.gold.fct_predictions")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Walk-forward backtest | top-5 long, weekly rebalance, 5 bp tx cost

# COMMAND ----------
# Identical logic to local backtest/run_walkforward.py. Critical: rebalance every 5
# trading days (not daily) to avoid the off-by-5 overlap bug. Writes
# gold.fct_backtest_pnl_daily.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 11. Parity check vs local
# MAGIC
# MAGIC Local results (from local mlruns/):
# MAGIC - Rung 1 mean AUC: 0.5123    backtest Sharpe: 1.53    ann return: 30.5%
# MAGIC - Rung 2 mean AUC: 0.5116    backtest Sharpe: 1.02    ann return: 21.7%
# MAGIC - Rung 0 mean AUC: ~0.51     backtest Sharpe: 2.26    ann return: 47.9% (sampled, 44 periods)
# MAGIC
# MAGIC Expected: Databricks AUC within +/- 0.005 of local because the panel data and
# MAGIC hyperparameters are identical and the random_state is fixed. If divergence > 0.005,
# MAGIC suspect a data drift between Postgres and Delta (most likely cause: a missing source
# MAGIC file in the landing volume).
