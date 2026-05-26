# Databricks notebook source
# MAGIC %pip install -q transformers==4.43.0 torch==2.3.1

# COMMAND ----------

# Python DLT notebook | second source of advdatafinal_dlt.
# Builds text-side silver tables and the gold facts that depend on them.
# Reads of pipeline-internal tables go through dlt.read so DLT can wire the DAG.

import dlt
import pyspark.sql.functions as F
import pyspark.sql.types as T

SCORED_SCHEMA = T.StructType([
    T.StructField("symbol",        T.StringType()),
    T.StructField("published_at",  T.StringType()),
    T.StructField("title",         T.StringType()),
    T.StructField("finbert_score", T.DoubleType()),
])

# COMMAND ----------

def _finbert_score(texts):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    tok = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    mdl = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").eval()
    out, B = [], 16
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
           comment="FinBERT-tone score per news article (P(positive) - P(negative))")
def silver_news_scored():
    pdf = spark.read.table("advdatafinal.datos_masked.news_redacted").toPandas()
    if not len(pdf):
        return spark.createDataFrame([], SCORED_SCHEMA)
    pdf["finbert_score"] = _finbert_score(pdf["body_masked"].fillna("").tolist())
    return spark.createDataFrame(pdf[["symbol", "published_at", "title", "finbert_score"]], schema=SCORED_SCHEMA)

# COMMAND ----------

# silver.silver_press_scored | FinBERT on press releases
@dlt.table(name="silver.silver_press_scored",
           comment="FinBERT-tone score per press release")
def silver_press_scored():
    pdf = spark.read.table("advdatafinal.datos_masked.press_redacted").toPandas()
    if not len(pdf):
        return spark.createDataFrame([], SCORED_SCHEMA)
    pdf["finbert_score"] = _finbert_score(pdf["body_masked"].fillna("").tolist())
    return spark.createDataFrame(pdf[["symbol", "published_at", "title", "finbert_score"]], schema=SCORED_SCHEMA)

# COMMAND ----------

# gold.fct_sentiment_per_day | aggregated FinBERT means per (symbol, trade_date)
@dlt.table(name="gold.fct_sentiment_per_day",
           comment="Per-day mean FinBERT score and article count for news and press")
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
           comment="ML training table: 15 price+fundamental + 4 sentiment features")
def fct_feature_panel_daily_full():
    panel = spark.read.table("advdatafinal.gold.fct_feature_panel_daily")
    sent  = (
        spark.read.table("advdatafinal.gold.fct_sentiment_per_day")
        .select("symbol", "trade_date", "news_score", "n_news", "press_score", "n_press")
    )
    return panel.join(sent, ["symbol", "trade_date"], "left")
