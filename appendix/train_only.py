# Databricks notebook source
# MAGIC %pip install -q xgboost==2.0.3

# COMMAND ----------

# One-off training notebook | reads gold.fct_feature_panel_daily_full (built by mlpipeline.py)
# and writes gold.fct_predictions + gold.fct_backtest_pnl_daily.
# Skips the long FinBERT + MiniLM passes so it runs in ~5 minutes instead of ~50.

import pandas as pd
from datetime import date, timedelta
from dataclasses import dataclass
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

panel = (spark.table("advdatafinal.gold.fct_feature_panel_daily_full")
         .filter("y_5d_up IS NOT NULL")
         .orderBy("symbol", "trade_date")
         .toPandas())
# Spark DATE comes back as object; normalise to pandas datetime64 for clean masking
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
print(f"Panel: {len(panel)} rows, {panel['symbol'].nunique()} stocks, "
      f"date range {panel['trade_date'].min()} to {panel['trade_date'].max()}, "
      f"up_rate {panel['y_5d_up'].mean():.3f}")

# COMMAND ----------

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

STRUCTURED_FEATURES = [
    "log_ret_1d","sma_5","sma_20","sma_50","ema_12","ema_26",
    "macd_hist","bb_z","vol_20d",
    "roe","roa","debt_eq","gross_margin","op_margin","asset_turnover",
]
TEXT_FEATURES = ["news_score","n_news","press_score","n_press",
                 "filing_pc1","filing_pc2","filing_pc3","filing_pc4","filing_pc5"]
FULL_FEATURES = STRUCTURED_FEATURES + TEXT_FEATURES

HP = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
          subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
          gamma=0.1, reg_lambda=1.0, random_state=7,
          eval_metric="auc", tree_method="hist", n_jobs=4)

# COMMAND ----------

# Rung 1 | 15 structured features
preds_rung1 = []
for fold in folds():
    train = panel[(panel["trade_date"] >= fold.train_start) & (panel["trade_date"] <= fold.train_end)]
    test  = panel[(panel["trade_date"] >= fold.test_start)  & (panel["trade_date"] <= fold.test_end)]
    if train.empty or test.empty: continue
    Xtr = train[STRUCTURED_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    ytr = train["y_5d_up"].astype(int)
    Xte = test[STRUCTURED_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    yte = test["y_5d_up"].astype(int)
    clf = xgb.XGBClassifier(**HP).fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    p = clf.predict_proba(Xte)[:, 1]
    for i, (_, r) in enumerate(test.iterrows()):
        preds_rung1.append((str(r["trade_date"]), r["company_key"], 1, fold.fold_id, float(p[i]), int(p[i] > 0.5)))
print(f"Rung 1 done: {len(preds_rung1)} preds")

# Rung 2 | 24 features (15 + 4 sentiment + 5 PCA)
preds_rung2 = []
for fold in folds():
    train = panel[(panel["trade_date"] >= fold.train_start) & (panel["trade_date"] <= fold.train_end)]
    test  = panel[(panel["trade_date"] >= fold.test_start)  & (panel["trade_date"] <= fold.test_end)]
    if train.empty or test.empty: continue
    Xtr = train[FULL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    ytr = train["y_5d_up"].astype(int)
    Xte = test[FULL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    yte = test["y_5d_up"].astype(int)
    clf = xgb.XGBClassifier(**HP).fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    p = clf.predict_proba(Xte)[:, 1]
    for i, (_, r) in enumerate(test.iterrows()):
        preds_rung2.append((str(r["trade_date"]), r["company_key"], 2, fold.fold_id, float(p[i]), int(p[i] > 0.5)))
print(f"Rung 2 done: {len(preds_rung2)} preds")

# COMMAND ----------

# gold.fct_predictions
all_preds = pd.DataFrame(
    preds_rung1 + preds_rung2,
    columns=["trade_date","company_key","model_rung","fold_id","prob_up","predicted_class"],
)
(spark.createDataFrame(all_preds)
    .write.mode("overwrite")
    .saveAsTable("advdatafinal.gold.fct_predictions"))
print(f"gold.fct_predictions: {len(all_preds)} rows")

# COMMAND ----------

# gold.fct_backtest_pnl_daily (top-5 long, weekly rebalance, 5 bp tx cost)
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
print(f"gold.fct_backtest_pnl_daily: {len(bt)} rows")
