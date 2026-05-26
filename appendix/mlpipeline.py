# Databricks notebook source
# MAGIC %pip install -q transformers==4.43.0 torch==2.3.1 sentence-transformers==3.0.1 tiktoken==0.7.0 xgboost==2.0.3

# COMMAND ----------

# Runs as Job task 2, after the DLT pipeline finishes.
# DLT (pipelinedatos.sql) builds: raw -> silver -> gold dims + fct_feature_panel_daily.
# This notebook adds the ML-derived objects (10 more tables) and the trained outputs.
#
# Path A "incremental scoring": every FinBERT and MiniLM cell does a LEFT ANTI JOIN
# against its target table so only NEW rows are scored. On a fresh run the targets
# don't exist and everything is scored. On subsequent runs (after new news/press
# arrive), only the deltas are scored -> ~3 min instead of ~55 min.

import mlflow, pandas as pd, numpy as np, hashlib
import pyspark.sql.functions as F
try:
    mlflow.set_experiment("/Users/lfrenteria33@gmail.com/advdatafinal")
except Exception as e:
    print(f"MLflow experiment setup skipped: {e}")

def _new_rows(source_df, target_table_name, key_cols):
    """Return rows from source_df not yet present in target_table_name."""
    if not spark.catalog.tableExists(target_table_name):
        return source_df
    existing = spark.table(target_table_name).select(*key_cols).distinct()
    return source_df.join(existing, key_cols, "left_anti")

# COMMAND ----------

# FinBERT (Positive - Negative) | cached at module level so the model loads once per worker
_FB_CACHE = {}
def _finbert_score(texts):
    if "tok" not in _FB_CACHE:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _FB_CACHE["tok"] = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        _FB_CACHE["mdl"] = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").eval()
    import torch
    tok, mdl = _FB_CACHE["tok"], _FB_CACHE["mdl"]
    out, B = [], 16
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i+B], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(mdl(**enc).logits, dim=-1).numpy()
        # Label order: 0=Neutral, 1=Positive, 2=Negative -> P(pos) - P(neg)
        out.extend((probs[:, 1] - probs[:, 2]).tolist())
    return out

# silver.silver_news_scored | incremental
new_news = _new_rows(
    spark.table("advdatafinal.datos_masked.news_redacted"),
    "advdatafinal.silver.silver_news_scored",
    ["symbol", "published_at", "title"],
).toPandas()
if len(new_news):
    new_news["finbert_score"] = _finbert_score(new_news["body_masked"].fillna("").tolist())
    (spark.createDataFrame(new_news[["symbol","published_at","title","finbert_score"]])
        .write.mode("append").saveAsTable("advdatafinal.silver.silver_news_scored"))
print(f"silver.silver_news_scored: +{len(new_news)} new (target now "
      f"{spark.table('advdatafinal.silver.silver_news_scored').count()})")

# silver.silver_press_scored | incremental
new_press = _new_rows(
    spark.table("advdatafinal.datos_masked.press_redacted"),
    "advdatafinal.silver.silver_press_scored",
    ["symbol", "published_at", "title"],
).toPandas()
if len(new_press):
    new_press["finbert_score"] = _finbert_score(new_press["body_masked"].fillna("").tolist())
    (spark.createDataFrame(new_press[["symbol","published_at","title","finbert_score"]])
        .write.mode("append").saveAsTable("advdatafinal.silver.silver_press_scored"))
print(f"silver.silver_press_scored: +{len(new_press)} new (target now "
      f"{spark.table('advdatafinal.silver.silver_press_scored').count()})")

# COMMAND ----------

# MiniLM chunker (500 tokens with 50 overlap) + 384-dim L2-normalised embeddings
import tiktoken
from sentence_transformers import SentenceTransformer

_ENC = tiktoken.get_encoding("cl100k_base")
_MINILM = None
def _minilm():
    global _MINILM
    if _MINILM is None:
        _MINILM = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MINILM

def _chunk_text(text):
    if not text or not text.strip(): return []
    tokens = _ENC.encode(text)
    step = 500 - 50
    out = []
    for i in range(0, len(tokens), step):
        sl = tokens[i:i+500]
        if len(sl) < 50: break
        out.append(_ENC.decode(sl))
    return out

def _build_chunks(source_view, target_table, source_code):
    src = spark.table(source_view).toPandas()
    rows = []
    for _, r in src.iterrows():
        chunks = _chunk_text(r["body_masked"] or "")
        acc_tail = str(r["accession"]).replace("-", "")[-6:] if r.get("accession") else "000000"
        for idx, chunk in enumerate(chunks):
            rows.append({
                "chunk_key":   f"{source_code}-{r['symbol']}-{r['filing_date']}-{acc_tail}-c{idx:02d}",
                "accession":   r["accession"],
                "symbol":      r["symbol"],
                "company_key": hashlib.md5(r["symbol"].strip().lower().encode()).hexdigest(),
                "filing_date": str(r["filing_date"]),
                "chunk_index": idx,
                "n_tokens":    len(chunk.split()),
                "body_chunk":  chunk,
            })
    if not rows:
        return 0
    cdf = pd.DataFrame(rows)
    cdf_spark = spark.createDataFrame(cdf)
    new = _new_rows(cdf_spark, target_table, ["chunk_key"]).toPandas()
    if not len(new):
        return 0
    embs = _minilm().encode(new["body_chunk"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    new["embedding"] = [list(map(float, v)) for v in embs]
    (spark.createDataFrame(new).write.mode("append").saveAsTable(target_table))
    return len(new)

n10 = _build_chunks("advdatafinal.datos_masked.filings_10k_redacted",
                    "advdatafinal.silver.silver_filings_10k_chunked", "10K")
print(f"silver.silver_filings_10k_chunked: +{n10} new (target now "
      f"{spark.table('advdatafinal.silver.silver_filings_10k_chunked').count()})")

n8 = _build_chunks("advdatafinal.datos_masked.filings_8k_redacted",
                   "advdatafinal.silver.silver_filings_8k_chunked", "8K")
print(f"silver.silver_filings_8k_chunked: +{n8} new (target now "
      f"{spark.table('advdatafinal.silver.silver_filings_8k_chunked').count()})")

# COMMAND ----------

# gold.dim_chunk (UNION) + gold.fct_embedding_per_company (PCA top-5)
spark.sql("""
CREATE OR REPLACE TABLE advdatafinal.gold.dim_chunk AS
SELECT '10K' AS filing_type, chunk_key, accession, symbol, company_key, filing_date, chunk_index, n_tokens, body_chunk, embedding
FROM advdatafinal.silver.silver_filings_10k_chunked
UNION ALL
SELECT '8K' AS filing_type, chunk_key, accession, symbol, company_key, filing_date, chunk_index, n_tokens, body_chunk, embedding
FROM advdatafinal.silver.silver_filings_8k_chunked
""")
print("gold.dim_chunk written")

from sklearn.decomposition import PCA
k10 = spark.table("advdatafinal.silver.silver_filings_10k_chunked").toPandas()
k10["vec"] = k10["embedding"].apply(np.array)
filings = (k10.groupby(["company_key","filing_date"])["vec"]
            .apply(lambda s: np.mean(np.vstack(s.values), axis=0))
            .reset_index().rename(columns={"vec": "mean_vec"}))
mat = np.vstack(filings["mean_vec"].values)
pca = PCA(n_components=5, random_state=7).fit(mat)
pcs = pca.transform(mat)
for i in range(5):
    filings[f"filing_pc{i+1}"] = pcs[:, i].astype(float)
filings_out = filings.drop(columns=["mean_vec"])
(spark.createDataFrame(filings_out)
    .write.mode("overwrite").saveAsTable("advdatafinal.gold.fct_embedding_per_company"))
print(f"gold.fct_embedding_per_company: {len(filings_out)} rows "
      f"({pca.explained_variance_ratio_.sum():.1%} variance explained)")

# COMMAND ----------

# gold.fct_sentiment_per_day + gold.fct_feature_panel_daily_full
spark.sql("""
CREATE OR REPLACE TABLE advdatafinal.gold.fct_sentiment_per_day AS
WITH news_daily AS (
    SELECT symbol, date(published_at) as trade_date,
           AVG(finbert_score) as news_score, COUNT(*) as n_news
    FROM advdatafinal.silver.silver_news_scored
    GROUP BY symbol, date(published_at)
),
press_daily AS (
    SELECT symbol, date(published_at) as trade_date,
           AVG(finbert_score) as press_score, COUNT(*) as n_press
    FROM advdatafinal.silver.silver_press_scored
    GROUP BY symbol, date(published_at)
)
SELECT coalesce(n.symbol, p.symbol) as symbol,
       coalesce(n.trade_date, p.trade_date) as trade_date,
       n.news_score, n.n_news, p.press_score, p.n_press
FROM news_daily n FULL OUTER JOIN press_daily p USING (symbol, trade_date)
""")
print("gold.fct_sentiment_per_day written")

spark.sql("""
CREATE OR REPLACE TABLE advdatafinal.gold.fct_feature_panel_daily_full AS
WITH pca_asof AS (
    SELECT p.symbol, p.trade_date,
           e.filing_pc1, e.filing_pc2, e.filing_pc3, e.filing_pc4, e.filing_pc5,
           ROW_NUMBER() OVER (PARTITION BY p.symbol, p.trade_date
                              ORDER BY e.filing_date DESC NULLS LAST) AS rn
    FROM advdatafinal.gold.fct_feature_panel_daily p
    LEFT JOIN advdatafinal.gold.fct_embedding_per_company e
        ON p.company_key = e.company_key AND e.filing_date <= cast(p.trade_date as STRING)
)
SELECT p.*, s.news_score, s.n_news, s.press_score, s.n_press,
       pca.filing_pc1, pca.filing_pc2, pca.filing_pc3, pca.filing_pc4, pca.filing_pc5
FROM advdatafinal.gold.fct_feature_panel_daily p
LEFT JOIN advdatafinal.gold.fct_sentiment_per_day s USING (symbol, trade_date)
LEFT JOIN (SELECT * FROM pca_asof WHERE rn = 1) pca USING (symbol, trade_date)
""")
panel = (spark.table("advdatafinal.gold.fct_feature_panel_daily_full")
         .filter("y_5d_up IS NOT NULL")
         .orderBy("symbol","trade_date").toPandas())
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
print(f"Panel: {len(panel)} rows, {panel['symbol'].nunique()} stocks, "
      f"up_rate {panel['y_5d_up'].mean():.3f}")

# COMMAND ----------

# Walk-forward folds + 2 XGBoost rungs
from datetime import date, timedelta
from dataclasses import dataclass
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

@dataclass
class Fold:
    fold_id: str; train_start: date; train_end: date; test_start: date; test_end: date

def folds(start=date(2021,1,1), end=date(2025,12,31), train_years=3, test_quarter_days=63, gap_days=5):
    out, cursor = [], start + timedelta(days=365*train_years)
    while cursor + timedelta(days=test_quarter_days) <= end:
        train_end = cursor - timedelta(days=gap_days)
        train_start = train_end - timedelta(days=365*train_years)
        test_start = cursor; test_end = cursor + timedelta(days=test_quarter_days)
        out.append(Fold(test_start.isoformat(), train_start, train_end, test_start, test_end))
        cursor += timedelta(days=test_quarter_days)
    return out

STRUCTURED = ["log_ret_1d","sma_5","sma_20","sma_50","ema_12","ema_26","macd_hist","bb_z","vol_20d",
              "roe","roa","debt_eq","gross_margin","op_margin","asset_turnover"]
TEXT       = ["news_score","n_news","press_score","n_press",
              "filing_pc1","filing_pc2","filing_pc3","filing_pc4","filing_pc5"]
HP = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
          subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
          gamma=0.1, reg_lambda=1.0, random_state=7,
          eval_metric="auc", tree_method="hist", n_jobs=4)

def run_rung(features, rung_id):
    preds = []
    for fold in folds():
        train = panel[(panel["trade_date"] >= pd.Timestamp(fold.train_start)) & (panel["trade_date"] <= pd.Timestamp(fold.train_end))]
        test  = panel[(panel["trade_date"] >= pd.Timestamp(fold.test_start))  & (panel["trade_date"] <= pd.Timestamp(fold.test_end))]
        if train.empty or test.empty: continue
        Xtr = train[features].apply(pd.to_numeric, errors="coerce").fillna(0)
        ytr = train["y_5d_up"].astype(int)
        Xte = test[features].apply(pd.to_numeric, errors="coerce").fillna(0)
        yte = test["y_5d_up"].astype(int)
        clf = xgb.XGBClassifier(**HP).fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        p = clf.predict_proba(Xte)[:, 1]
        try:
            with mlflow.start_run(run_name=f"rung{rung_id}_{fold.fold_id}"):
                mlflow.log_param("rung", rung_id); mlflow.log_param("fold_id", fold.fold_id)
                mlflow.log_param("n_features", len(features))
                mlflow.log_metric("test_auc", float(roc_auc_score(yte, p)))
                mlflow.log_metric("accuracy", float(accuracy_score(yte, (p > 0.5).astype(int))))
        except Exception:
            pass
        for i, (_, r) in enumerate(test.iterrows()):
            preds.append((str(r["trade_date"]), r["company_key"], rung_id, fold.fold_id, float(p[i]), int(p[i] > 0.5)))
    return preds

preds_rung1 = run_rung(STRUCTURED, 1)
preds_rung2 = run_rung(STRUCTURED + TEXT, 2)
print(f"rung1={len(preds_rung1)}  rung2={len(preds_rung2)}")

all_preds = pd.DataFrame(preds_rung1 + preds_rung2,
    columns=["trade_date","company_key","model_rung","fold_id","prob_up","predicted_class"])
(spark.createDataFrame(all_preds)
    .write.mode("overwrite").saveAsTable("advdatafinal.gold.fct_predictions"))
print(f"gold.fct_predictions: {len(all_preds)} rows")

# COMMAND ----------

# Walk-forward backtest (top-5 long, weekly rebalance, 5 bp tx cost)
def backtest(preds_df, panel_df, top_n=5, tc_bp=5):
    preds_df = preds_df.copy()
    preds_df["trade_date"] = pd.to_datetime(preds_df["trade_date"])
    df = preds_df.merge(panel_df[["trade_date","company_key","symbol","y_5d_logret"]],
                        on=["trade_date","company_key"], how="left").dropna(subset=["y_5d_logret"])
    rows = []
    for rung in sorted(df["model_rung"].unique()):
        r = df[df["model_rung"] == rung].sort_values("trade_date")
        rebal = sorted(r["trade_date"].unique())[::5]
        prev = set()
        for d in rebal:
            day = r[r["trade_date"] == d]
            basket = set(day.nlargest(top_n, "prob_up")["symbol"])
            turnover = len(basket.symmetric_difference(prev)) / max(1, top_n * 2)
            tc = turnover * tc_bp / 10000.0
            ret = day[day["symbol"].isin(basket)]["y_5d_logret"].mean()
            rows.append({"rung": rung, "trade_date": d, "net_ret": ret - tc})
            prev = basket
    return pd.DataFrame(rows)

bt = backtest(all_preds, panel[["trade_date","company_key","symbol","y_5d_logret"]].copy())
(spark.createDataFrame(bt).write.mode("overwrite")
    .saveAsTable("advdatafinal.gold.fct_backtest_pnl_daily"))
print(f"gold.fct_backtest_pnl_daily: {len(bt)} rows")
