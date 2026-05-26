# Databricks notebook source

# Python DLT notebook | second source of advdatafinal_dlt.
# Builds the text-side silver tables and the gold facts that depend on them.
# Training (XGBoost) and backtest stay in appendix/mlpipeline.py (Job task 2)
# because trained weights cannot be derived from one read.

# COMMAND ----------

# MAGIC %pip install -q transformers==4.43.0 torch==2.3.1

# COMMAND ----------

import dlt
import pyspark.sql.functions as F
import pyspark.sql.types as T

# COMMAND ----------

def _finbert_score(texts):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    tok = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    mdl = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").eval()
    out = []
    B = 16
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i+B], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(mdl(**enc).logits, dim=-1).numpy()
        # FinBERT-tone label order: 0=Neutral, 1=Positive, 2=Negative -> P(pos) - P(neg)
        out.extend((probs[:, 1] - probs[:, 2]).tolist())
    return out

# COMMAND ----------

# silver.silver_news_scored | FinBERT (Pos - Neg) per article
@dlt.table(name="silver.silver_news_scored",
           comment="FinBERT-tone (Positive minus Negative) per news article")
def silver_news_scored():
    pdf = spark.read.table("advdatafinal.datos_masked.news_redacted").toPandas()
    pdf["finbert_score"] = _finbert_score(pdf["body_masked"].fillna("").tolist()) if len(pdf) else []
    return spark.createDataFrame(pdf[["symbol", "published_at", "title", "finbert_score"]])

# COMMAND ----------

# silver.silver_press_scored | FinBERT on press releases
@dlt.table(name="silver.silver_press_scored",
           comment="FinBERT-tone (Positive minus Negative) per press release")
def silver_press_scored():
    pdf = spark.read.table("advdatafinal.datos_masked.press_redacted").toPandas()
    pdf["finbert_score"] = _finbert_score(pdf["body_masked"].fillna("").tolist()) if len(pdf) else []
    return spark.createDataFrame(pdf[["symbol", "published_at", "title", "finbert_score"]])

# COMMAND ----------

# gold.fct_sentiment_per_day | 3d + 30d rolling FinBERT means per (symbol, trade_date)
@dlt.table(name="gold.fct_sentiment_per_day",
           comment="3d and 30d rolling FinBERT means per company-day")
def fct_sentiment_per_day():
    news = (
        spark.read.table("advdatafinal.silver.silver_news_scored")
        .withColumn("trade_date", F.to_date("published_at"))
        .groupBy("symbol", "trade_date")
        .agg(F.avg("finbert_score").alias("news_score"),
             F.count(F.lit(1)).alias("n_news"))
    )
    press = (
        spark.read.table("advdatafinal.silver.silver_press_scored")
        .withColumn("trade_date", F.to_date("published_at"))
        .groupBy("symbol", "trade_date")
        .agg(F.avg("finbert_score").alias("press_score"),
             F.count(F.lit(1)).alias("n_press"))
    )
    return (
        news.join(press, ["symbol", "trade_date"], "full_outer")
            .withColumn("date_key",    F.md5(F.col("trade_date").cast("string")))
            .withColumn("company_key", F.md5(F.lower(F.trim("symbol"))))
            .select("date_key", "company_key", "symbol", "trade_date",
                    "news_score", "n_news", "press_score", "n_press")
    )

# COMMAND ----------

# gold.fct_feature_panel_daily_full | panel (15 features) + sentiment (4 features)
@dlt.table(name="gold.fct_feature_panel_daily_full",
           comment="ML training table: 15 price+fundamental features + 4 sentiment features")
def fct_feature_panel_daily_full():
    panel = spark.read.table("advdatafinal.gold.fct_feature_panel_daily")
    sent  = (
        spark.read.table("advdatafinal.gold.fct_sentiment_per_day")
        .select("symbol", "trade_date", "news_score", "n_news", "press_score", "n_press")
    )
    return panel.join(sent, ["symbol", "trade_date"], "left")
