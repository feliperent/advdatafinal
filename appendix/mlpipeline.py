# Databricks notebook source

# COMMAND ----------

# This notebook runs as Job task 2, AFTER the DLT pipeline finishes.
# DLT (pipelinedatos.sql + mlpipeline_dlt.py) builds everything derivable from raw:
#   raw -> silver -> gold.dim_* + gold.fct_sentiment_per_day + gold.fct_feature_panel_daily_full
# This notebook does the only two things DLT cannot:
#   - XGBoost training (weights come from a fit, not from a deterministic read)
#   - Walk-forward backtest (depends on those trained predictions)
# It writes gold.fct_predictions and gold.fct_backtest_pnl_daily as plain Delta tables.

# COMMAND ----------

# MAGIC %pip install -q xgboost==2.0.3

# COMMAND ----------

import mlflow
mlflow.set_experiment("/advdatafinal")

# COMMAND ----------

import pandas as pd
panel = (
    spark.table("advdatafinal.gold.fct_feature_panel_daily_full")
    .filter("y_5d_up IS NOT NULL")
    .orderBy("symbol", "trade_date")
    .toPandas()
)
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

# Rung 1 | XGBoost on 15 price + fundamental features
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

# Rung 2 | same hyperparameters + 4 sentiment features
TEXT_FEATURES = ["news_score","n_news","press_score","n_press"]
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
            rows.append({"rung": rung, "trade_date": d, "net_ret": ret - tc})
            prev_basket = basket
    return pd.DataFrame(rows)

panel_for_bt = panel[["trade_date","company_key","symbol","y_5d_logret"]].copy()
bt = backtest(all_preds, panel_for_bt)
(spark.createDataFrame(bt)
    .write.mode("overwrite")
    .saveAsTable("advdatafinal.gold.fct_backtest_pnl_daily"))
print(f"Wrote {len(bt)} rows to advdatafinal.gold.fct_backtest_pnl_daily")
