# Databricks notebook source
# MAGIC %pip install -q transformers==4.43.0 torch==2.3.1

# COMMAND ----------

# Python DLT notebook | second source of advdatafinal_dlt.
# Builds text-side silver tables and the gold facts that depend on them.
# Uses dlt.read_stream + pandas_udf so DLT registers the dependency on
# datos_masked.* and orders the flows correctly during a full refresh.

import dlt
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.functions import pandas_udf

# COMMAND ----------

# Load FinBERT on the driver once, broadcast to executors.
# This is the pattern Databricks recommends for HuggingFace inference in pipelines:
# https://www.databricks.com/blog/2023/02/06/getting-started-nlp-using-hugging-face-transformers-pipelines.html
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

_tok = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
_mdl = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").eval()
BC_TOK = spark.sparkContext.broadcast(_tok)
BC_MDL = spark.sparkContext.broadcast(_mdl)

@pandas_udf(T.DoubleType())
def finbert_udf(texts: pd.Series) -> pd.Series:
    tok = BC_TOK.value
    mdl = BC_MDL.value
    out, B = [], 16
    bodies = texts.fillna("").tolist()
    for i in range(0, len(bodies), B):
        enc = tok(bodies[i:i+B], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(mdl(**enc).logits, dim=-1).numpy()
        # FinBERT-tone label order: 0=Neutral, 1=Positive, 2=Negative -> P(pos) - P(neg)
        out.extend((probs[:, 1] - probs[:, 2]).tolist())
    return pd.Series(out)

# COMMAND ----------

# silver.silver_news_scored | FinBERT score per news article
@dlt.table(name="silver.silver_news_scored",
           comment="FinBERT-tone score per news article (P(positive) - P(negative))")
@dlt.expect_or_drop("valid_symbol_present", "symbol IS NOT NULL")
def silver_news_scored():
    return (
        dlt.read_stream("datos_masked.news_redacted")
        .withColumn("finbert_score", finbert_udf(F.col("body_masked")))
        .select("symbol", "published_at", "title", "finbert_score")
    )

# COMMAND ----------

# silver.silver_press_scored | FinBERT on press releases
@dlt.table(name="silver.silver_press_scored",
           comment="FinBERT-tone score per press release")
@dlt.expect_or_drop("valid_symbol_present", "symbol IS NOT NULL")
def silver_press_scored():
    return (
        dlt.read_stream("datos_masked.press_redacted")
        .withColumn("finbert_score", finbert_udf(F.col("body_masked")))
        .select("symbol", "published_at", "title", "finbert_score")
    )

# COMMAND ----------

# gold.fct_sentiment_per_day | aggregated FinBERT means per (symbol, trade_date)
@dlt.table(name="gold.fct_sentiment_per_day",
           comment="Per-day mean FinBERT score and article count for news and press")
def fct_sentiment_per_day():
    news = (
        dlt.read("silver.silver_news_scored")
        .withColumn("trade_date", F.to_date("published_at"))
        .groupBy("symbol", "trade_date")
        .agg(F.avg("finbert_score").alias("news_score"),
             F.count(F.lit(1)).alias("n_news"))
    )
    press = (
        dlt.read("silver.silver_press_scored")
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
    panel = dlt.read("gold.fct_feature_panel_daily")
    sent  = (
        dlt.read("gold.fct_sentiment_per_day")
        .select("symbol", "trade_date", "news_score", "n_news", "press_score", "n_press")
    )
    return panel.join(sent, ["symbol", "trade_date"], "left")
