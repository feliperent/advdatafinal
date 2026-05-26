# Databricks notebook source
# MAGIC %pip install -q transformers==4.43.0 torch==2.3.1 sentence-transformers==3.0.1 tiktoken==0.7.0 scikit-learn==1.5.0

# COMMAND ----------

# Python DLT source | second library of advdatafinal_dlt.
# Adds the text-side silver tables and the gold facts that depend on them.
# Uses dlt.read_stream so DLT registers the dependency and processes only new rows
# on subsequent refreshes (Path A "incremental scoring" comes for free).
# Models are cached at module level so they load once per Python worker process,
# without going through broadcast (which fails to deserialise on Free Edition serverless).

import dlt
import pandas as pd
import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.types as T
import hashlib

# COMMAND ----------

# FinBERT-tone (label order: 0=Neutral, 1=Positive, 2=Negative)
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
        out.extend((probs[:, 1] - probs[:, 2]).tolist())
    return out

SCORED_SCHEMA = T.StructType([
    T.StructField("symbol",        T.StringType()),
    T.StructField("published_at",  T.StringType()),
    T.StructField("title",         T.StringType()),
    T.StructField("finbert_score", T.DoubleType()),
])

# COMMAND ----------

# silver.silver_news_scored | streaming, incremental via dlt.read_stream
@dlt.table(name="silver.silver_news_scored",
           comment="FinBERT-tone (Positive - Negative) per news article")
@dlt.expect_or_drop("valid_symbol_present", "symbol IS NOT NULL")
def silver_news_scored():
    pdf = dlt.read_stream("datos_masked.news_redacted").toPandas()
    if not len(pdf):
        return spark.createDataFrame([], SCORED_SCHEMA)
    pdf["finbert_score"] = _finbert_score(pdf["body_masked"].fillna("").tolist())
    return spark.createDataFrame(pdf[["symbol", "published_at", "title", "finbert_score"]], schema=SCORED_SCHEMA)

# COMMAND ----------

# silver.silver_press_scored | streaming
@dlt.table(name="silver.silver_press_scored",
           comment="FinBERT-tone per press release")
@dlt.expect_or_drop("valid_symbol_present", "symbol IS NOT NULL")
def silver_press_scored():
    pdf = dlt.read_stream("datos_masked.press_redacted").toPandas()
    if not len(pdf):
        return spark.createDataFrame([], SCORED_SCHEMA)
    pdf["finbert_score"] = _finbert_score(pdf["body_masked"].fillna("").tolist())
    return spark.createDataFrame(pdf[["symbol", "published_at", "title", "finbert_score"]], schema=SCORED_SCHEMA)

# COMMAND ----------

# MiniLM chunker (500 tokens with 50 overlap) + 384-dim L2-normalised embeddings
_ML_CACHE = {}
def _chunk_text(text, enc):
    if not text or not text.strip(): return []
    tokens = enc.encode(text)
    step = 500 - 50
    out = []
    for i in range(0, len(tokens), step):
        sl = tokens[i:i+500]
        if len(sl) < 50: break
        out.append(enc.decode(sl))
    return out

CHUNKED_SCHEMA = T.StructType([
    T.StructField("chunk_key",   T.StringType()),
    T.StructField("accession",   T.StringType()),
    T.StructField("symbol",      T.StringType()),
    T.StructField("company_key", T.StringType()),
    T.StructField("filing_date", T.StringType()),
    T.StructField("chunk_index", T.IntegerType()),
    T.StructField("n_tokens",    T.IntegerType()),
    T.StructField("body_chunk",  T.StringType()),
    T.StructField("embedding",   T.ArrayType(T.FloatType())),
])

def _chunk_and_embed(pdf, source_code):
    if "enc" not in _ML_CACHE:
        import tiktoken
        from sentence_transformers import SentenceTransformer
        _ML_CACHE["enc"]    = tiktoken.get_encoding("cl100k_base")
        _ML_CACHE["minilm"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    enc, minilm = _ML_CACHE["enc"], _ML_CACHE["minilm"]
    rows = []
    for _, r in pdf.iterrows():
        chunks = _chunk_text(r["body_masked"] or "", enc)
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
        return None
    cdf = pd.DataFrame(rows)
    embeddings = minilm.encode(cdf["body_chunk"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    cdf["embedding"] = [list(map(float, v)) for v in embeddings]
    return cdf

@dlt.table(name="silver.silver_filings_10k_chunked",
           comment="Chunked 10-K bodies with MiniLM embeddings (384 dim, L2 normalised)")
def silver_filings_10k_chunked():
    pdf = dlt.read_stream("datos_masked.filings_10k_redacted").toPandas()
    if not len(pdf):
        return spark.createDataFrame([], CHUNKED_SCHEMA)
    cdf = _chunk_and_embed(pdf, "10K")
    if cdf is None or not len(cdf):
        return spark.createDataFrame([], CHUNKED_SCHEMA)
    return spark.createDataFrame(cdf, schema=CHUNKED_SCHEMA)

@dlt.table(name="silver.silver_filings_8k_chunked",
           comment="Chunked 8-K bodies with MiniLM embeddings")
def silver_filings_8k_chunked():
    pdf = dlt.read_stream("datos_masked.filings_8k_redacted").toPandas()
    if not len(pdf):
        return spark.createDataFrame([], CHUNKED_SCHEMA)
    cdf = _chunk_and_embed(pdf, "8K")
    if cdf is None or not len(cdf):
        return spark.createDataFrame([], CHUNKED_SCHEMA)
    return spark.createDataFrame(cdf, schema=CHUNKED_SCHEMA)

# COMMAND ----------

# gold.fct_embedding_per_company | mean-pool 10-K chunks per filing, PCA top-5
PCA_SCHEMA = T.StructType([
    T.StructField("company_key", T.StringType()),
    T.StructField("filing_date", T.StringType()),
    T.StructField("filing_pc1",  T.DoubleType()),
    T.StructField("filing_pc2",  T.DoubleType()),
    T.StructField("filing_pc3",  T.DoubleType()),
    T.StructField("filing_pc4",  T.DoubleType()),
    T.StructField("filing_pc5",  T.DoubleType()),
])

@dlt.table(name="gold.fct_embedding_per_company",
           comment="PCA top-5 components of mean 10-K embeddings per filing event")
def fct_embedding_per_company():
    pdf = dlt.read("silver.silver_filings_10k_chunked").toPandas()
    if not len(pdf):
        return spark.createDataFrame([], PCA_SCHEMA)
    pdf["vec"] = pdf["embedding"].apply(np.array)
    grouped = (pdf.groupby(["company_key", "filing_date"])["vec"]
               .apply(lambda s: np.mean(np.vstack(s.values), axis=0))
               .reset_index().rename(columns={"vec": "mean_vec"}))
    if len(grouped) < 5:
        return spark.createDataFrame([], PCA_SCHEMA)
    from sklearn.decomposition import PCA
    mat = np.vstack(grouped["mean_vec"].values)
    pca = PCA(n_components=5, random_state=7).fit(mat)
    pcs = pca.transform(mat)
    for i in range(5):
        grouped[f"filing_pc{i+1}"] = pcs[:, i].astype(float)
    return spark.createDataFrame(
        grouped[["company_key","filing_date","filing_pc1","filing_pc2","filing_pc3","filing_pc4","filing_pc5"]],
        schema=PCA_SCHEMA,
    )
