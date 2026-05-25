# Databricks notebook source

# COMMAND ----------

# This notebook runs AFTER the DLT pipeline finishes.
# DLT built: raw -> silver_prices_cleaned / silver_prices_features / silver_fundamentals_cleaned
#            -> gold.dim_* + gold.fct_feature_panel_daily (15 price + fundamental features).
# This notebook adds the text side (FinBERT sentiment + MiniLM embedding PCA),
# trains 2 XGBoost rungs, and writes predictions and backtest tables to gold.

# COMMAND ----------

# MAGIC %pip install -q transformers==4.43.0 sentence-transformers==3.0.1 xgboost==2.0.3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.set_experiment("/advdatafinal")

# COMMAND ----------

# silver.silver_news_scored | FinBERT (Positive - Negative) per article
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pyspark.sql.functions as F

tok = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
mdl = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").eval()

def finbert_score(texts):
    enc = tok(list(texts), padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(mdl(**enc).logits, dim=-1).numpy()
    # FinBERT-tone label order: 0=Neutral, 1=Positive, 2=Negative -> score = P(pos) - P(neg)
    return (probs[:, 1] - probs[:, 2]).tolist()

news_df = (
    spark.table("advdatafinal.datos_masked.news_redacted")
    .selectExpr("symbol", "to_timestamp(published_at) as published_at",
                "title", "body_masked as body")
    .toPandas()
)
news_df["finbert_score"] = finbert_score(news_df["body"].fillna("").tolist())

(spark.createDataFrame(news_df)
    .write.mode("overwrite")
    .saveAsTable("advdatafinal.silver.silver_news_scored"))
print(f"silver.silver_news_scored: {len(news_df)} rows")

# COMMAND ----------

# silver.silver_press_scored | same FinBERT pass on press releases
press_df = (
    spark.table("advdatafinal.datos_masked.press_redacted")
    .selectExpr("symbol", "to_timestamp(published_at) as published_at",
                "title", "body_masked as body")
    .toPandas()
)
press_df["finbert_score"] = finbert_score(press_df["body"].fillna("").tolist())
(spark.createDataFrame(press_df)
    .write.mode("overwrite")
    .saveAsTable("advdatafinal.silver.silver_press_scored"))
print(f"silver.silver_press_scored: {len(press_df)} rows")

# COMMAND ----------

# gold.fct_sentiment_per_day | 3-day and 30-day rolling mean per (symbol, trade_date)
spark.sql("""
CREATE OR REPLACE TABLE advdatafinal.gold.fct_sentiment_per_day AS
WITH news_daily AS (
    SELECT symbol, date(published_at) as trade_date, AVG(finbert_score) as news_score, COUNT(*) as n_news
    FROM advdatafinal.silver.silver_news_scored
    GROUP BY symbol, date(published_at)
),
press_daily AS (
    SELECT symbol, date(published_at) as trade_date, AVG(finbert_score) as press_score, COUNT(*) as n_press
    FROM advdatafinal.silver.silver_press_scored
    GROUP BY symbol, date(published_at)
)
SELECT
    coalesce(n.symbol, p.symbol) as symbol,
    coalesce(n.trade_date, p.trade_date) as trade_date,
    n.news_score, n.n_news,
    p.press_score, p.n_press,
    AVG(n.news_score)  OVER w3  as finbert_news_mean_3d,
    AVG(n.news_score)  OVER w30 as finbert_news_mean_30d,
    AVG(p.press_score) OVER w30 as finbert_press_30d,
    SUM(n.n_news)      OVER w3  as n_news_3d
FROM news_daily n
FULL OUTER JOIN press_daily p USING (symbol, trade_date)
WINDOW
    w3  AS (PARTITION BY coalesce(n.symbol, p.symbol) ORDER BY coalesce(n.trade_date, p.trade_date)
            ROWS BETWEEN 2  PRECEDING AND CURRENT ROW),
    w30 AS (PARTITION BY coalesce(n.symbol, p.symbol) ORDER BY coalesce(n.trade_date, p.trade_date)
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
""")
print("gold.fct_sentiment_per_day written")

# COMMAND ----------

# Augment gold.fct_feature_panel_daily with the 4 sentiment columns (left-join, NULLs allowed)
spark.sql("""
CREATE OR REPLACE TABLE advdatafinal.gold.fct_feature_panel_daily_full AS
SELECT
    p.*,
    s.finbert_news_mean_3d,
    s.finbert_news_mean_30d,
    s.finbert_press_30d,
    s.n_news_3d
FROM advdatafinal.gold.fct_feature_panel_daily p
LEFT JOIN advdatafinal.gold.fct_sentiment_per_day s
    USING (symbol, trade_date)
""")
panel = (spark.table("advdatafinal.gold.fct_feature_panel_daily_full")
         .filter("y_5d_up IS NOT NULL")
         .orderBy("symbol", "trade_date")
         .toPandas())
print(f"Panel: {len(panel)} rows, {panel['symbol'].nunique()} stocks, up_rate {panel['y_5d_up'].mean():.3f}")

# COMMAND ----------

# Walk-forward fold generator (3y train / 5d gap / 63d test, advancing weekly)
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

# COMMAND ----------

# Rung 1 | XGBoost on 14 price + fundamental features (no text)
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

STRUCTURED_FEATURES = [
    "log_ret_1d","sma_5","sma_20","sma_50","ema_12","ema_26",
    "macd_hist","bb_z","vol_20d",
    "roe","roa","debt_eq","gross_margin","op_margin","asset_turnover",
]
HP = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
          subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
          gamma=0.1, reg_lambda=1.0, random_state=7,
          eval_metric="auc", tree_method="hist", n_jobs=4)

import pandas as pd
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

# Rung 2 | same hyperparameters plus 4 sentiment features (no MiniLM PCA in this build)
TEXT_FEATURES = ["finbert_news_mean_3d","finbert_news_mean_30d","finbert_press_30d","n_news_3d"]
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

# Write predictions to gold.fct_predictions
all_preds = pd.DataFrame(
    preds_rung1 + preds_rung2,
    columns=["trade_date","company_key","model_rung","fold_id","prob_up","predicted_class"],
)
(spark.createDataFrame(all_preds)
    .write.mode("overwrite")
    .saveAsTable("advdatafinal.gold.fct_predictions"))
print(f"Wrote {len(all_preds)} rows to advdatafinal.gold.fct_predictions")

# COMMAND ----------

# Walk-forward backtest (top-5 long, weekly rebalance, 5 bp tx cost)
# Identical math to backtest/run_walkforward.py in the local repo.
def backtest(preds_df, panel_df, top_n=5, tc_bp=5):
    df = preds_df.merge(panel_df[["trade_date","company_key","symbol","y_5d_logret"]],
                        on=["trade_date","company_key"], how="left").dropna(subset=["y_5d_logret"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    rows = []
    for rung in sorted(df["model_rung"].unique()):
        r = df[df["model_rung"] == rung].sort_values("trade_date")
        rebal_dates = sorted(r["trade_date"].unique())[::5]
        prev_basket = set()
        for d in rebal_dates:
            day = r[r["trade_date"] == d]
            basket = set(day.nlargest(top_n, "prob_up")["symbol"])
            turnover = len(basket.symmetric_difference(prev_basket)) / max(1, top_n * 2)
            tc = turnover * tc_bp / 10000.0
            ret = day[day["symbol"].isin(basket)]["y_5d_logret"].mean()
            rows.append({"rung": rung, "trade_date": d, "ret": ret - tc})
            prev_basket = basket
    return pd.DataFrame(rows)

panel_for_bt = panel[["trade_date","company_key","symbol","y_5d_logret"]].copy()
bt = backtest(all_preds, panel_for_bt)
(spark.createDataFrame(bt)
    .write.mode("overwrite")
    .saveAsTable("advdatafinal.gold.fct_backtest_pnl_daily"))
print(f"Wrote {len(bt)} rows to advdatafinal.gold.fct_backtest_pnl_daily")

# COMMAND ----------

# Parity expectation against the local pipeline (run with the full 30-feature set)
# Local mean AUC per rung:
#   Rung 0 ARIMA: ~0.51 (Sharpe 2.26)
#   Rung 1 XGB structured: 0.5123 (Sharpe 1.53, ann ret 30.5%)
#   Rung 2 XGB + text: 0.5116 (Sharpe 1.02, ann ret 21.7%)
# Databricks build uses a 14-feature + 4-sentiment subset (no MiniLM PCA, no value ratios),
# so absolute AUC is expected to be a few bps lower but the rung ordering should hold.
