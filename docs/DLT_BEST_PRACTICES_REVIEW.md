# Best-practice review of the three open questions

Written after the second DLT pipeline COMPLETED but before any further edits. Pipeline state is
locked as-is overnight; this document is *advisory only*. Each section gives the question,
what the docs and the community actually recommend, and a verdict on whether we should apply
it to advdatafinal.

## 1. Loading FinBERT inside `@dlt.table` (re-downloaded every refresh)

### What we do today (in `databricksstuff/mlpipeline_dlt.py`)

```python
def _finbert_score(texts):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    mdl = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").eval()
    ...
```

The model is downloaded from HuggingFace on every pipeline refresh. With a `--full-refresh`
that runs the FinBERT cell twice (news + press), that is two ~440 MB downloads, ~30 s each.

### What Databricks actually recommends

Three escalating tiers in the official docs:

- **Tier 1 - "Pandas UDF with broadcast".** Load the model once on the driver, broadcast it
  to workers, apply the UDF on a Spark DataFrame. From the Databricks NLP blog:
  > "Spark uses broadcast to efficiently transmit any objects required by the pandas UDFs to
  > the worker nodes."

  Code shape:

  ```python
  bc_pipeline = sc.broadcast(transformers.pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone"))

  @pandas_udf("double")
  def finbert_udf(texts: pd.Series) -> pd.Series:
      pipe = bc_pipeline.value
      return pd.Series([r["score"] if r["label"] == "Positive" else -r["score"]
                        for r in pipe(texts.tolist(), batch_size=16)])
  ```

  The model still has to be downloaded the first time on the driver, but the broadcast
  variable is cached for the life of the SparkSession; subsequent stages reuse it.

- **Tier 2 - "Log to MLflow, load via `mlflow.pyfunc.spark_udf`".** Log the model once with
  `mlflow.transformers.log_model(task="text-classification")`, then in DLT use:

  ```python
  predict_udf = mlflow.pyfunc.spark_udf(
      spark, model_uri="models:/finbert-tone/Production", result_type="string"
  )
  df.withColumn("score", predict_udf("body_masked"))
  ```

  This is the path Databricks officially recommends for production: model lives in the Unity
  Catalog Model Registry, weights are cached in DBFS, versioning is explicit.

- **Tier 3 - "`ai_query()`".** The newest API for batch inference; Databricks now points
  users at this in the docs banner:
  > "Databricks recommends using `ai_query` for batch inference instead."

  For HuggingFace-style sentiment, you would serve the model via Mosaic AI Model Serving
  and call `SELECT ai_query('finbert-endpoint', body_masked) FROM news`. Requires a Pro
  workspace; not available on Free Edition.

### Verdict for advdatafinal

**Do not refactor.** Reasons:

- Tier 2 (MLflow registry) is the cleanest answer at production scale but adds a workflow
  step (log the model, register it, point the pipeline at the URI) that buys nothing at our
  4,420-article scale. The course rubric does not ask for Model Registry use.
- Tier 1 (broadcast) is the "more correct" pattern but requires running on a cluster with
  GPU or multi-worker CPU to actually pay off. On Free Edition serverless DLT we are on a
  single small node anyway; broadcast within one JVM is no faster than reloading.
- Tier 3 requires features Free Edition lacks.

**If we ship to production later**, the migration is: log FinBERT once via
`mlflow.transformers.log_model`, replace the `_finbert_score()` helper with a
`mlflow.pyfunc.spark_udf` call. That is a one-day change.

## 2. `.toPandas()` inside `@dlt.table` defeats Spark parallelism

### What we do today

```python
def silver_news_scored():
    pdf = dlt.read("datos_masked.news_redacted").toPandas()   # pulls all rows to driver
    pdf["finbert_score"] = _finbert_score(pdf["body_masked"].fillna("").tolist())
    return spark.createDataFrame(pdf[...], schema=SCORED_SCHEMA)
```

This pulls every row to the driver, scores in a Python for-loop, then re-creates a Spark
DataFrame. We get zero parallelism even if the cluster has 4 workers.

### What Databricks actually recommends

Same Pandas UDF pattern as above:

```python
@pandas_udf("double")
def finbert_udf(texts: pd.Series) -> pd.Series:
    ...

@dlt.table(name="silver.silver_news_scored")
def silver_news_scored():
    return (
        dlt.read("datos_masked.news_redacted")
        .repartition(8)                              # one partition per worker core
        .withColumn("finbert_score", finbert_udf(F.col("body_masked")))
        .select("symbol", "published_at", "title", "finbert_score")
    )
```

This keeps the DataFrame distributed end to end, Spark splits the rows across the cluster,
each executor runs FinBERT on its slice in parallel. Documented advice from the docs:
> "To make good utilization of the hardware in your cluster, you may need to repartition
> your Spark DataFrame. Generally some multiple of the number of GPUs on your workers (for
> GPU clusters) or number of cores across the workers in your cluster (for CPU clusters)
> works well in practice."

At our 4,420-row scale the difference is academic: even toPandas on a single small node
finishes in ~2 minutes. At 4 M rows the toPandas version would OOM the driver, while the
pandas UDF version would just take longer linearly.

### Verdict for advdatafinal

**Do not refactor.** Same reasoning as section 1: the gain only shows up on multi-worker
clusters and at scale. The toPandas pattern is acceptable in the academic context because:

- the dataset is small (under 5k articles total),
- the cluster is single-node serverless on Free Edition,
- the report can honestly note this as a known scale limitation,
- moving to pandas UDF is a 10-line change later.

**If we ship to production later**, swap `_finbert_score` for the pandas UDF above and add
`.repartition(n)` where n = number of cores. No other downstream change needed.

## 3. No `@dlt.expect_or_drop()` constraints on the streaming tables

### What we do today

Zero data-quality assertions. If `news_redacted.body_masked` is `NULL` for half the rows we
do not notice. If `silver_prices_cleaned.close_px <= 0` we do not notice.

### What Databricks actually recommends

Three severity tiers, applied per layer:

| Layer | Pattern | When to use |
|---|---|---|
| raw | `@dlt.expect("non_null_symbol", "symbol IS NOT NULL")` | Warning. Log bad rows, do not block. |
| datos_masked / silver | `@dlt.expect_or_drop("valid_dates", "trade_date BETWEEN '2021-01-01' AND '2025-12-31'")` | Drop bad rows silently. |
| gold | `@dlt.expect_or_fail("non_empty", "row_count > 0")` | Fail the pipeline if business invariants break. |

Direct quote from the patterns doc:
> "Start in Warning mode, learn the data, then tighten rules over time to promote to Drop
> or Fail where appropriate."

Naming convention from the docs: `valid_<col>_<constraint>`, e.g.
`valid_close_px_positive`, `valid_finbert_score_in_range`, `valid_symbol_in_universe`.

### Verdict for advdatafinal

**This is the one we SHOULD apply, lightly.** Reasons:

- Expectations cost almost nothing to write (~1 line per check) and are visible in the DLT
  UI as a quality dashboard, which is good for the academic deliverable.
- The course rubric rewards observability and data governance, and this is a CLS-adjacent
  control: it documents the invariants we expect from each layer.
- Failures are silent today; an expect_or_drop would surface NULL-bombs that would
  otherwise pass through and corrupt the feature panel.

**Concrete suggestion for tomorrow (one diff against `databricksstuff/pipelinedatos.sql`):**

```sql
CREATE OR REFRESH STREAMING TABLE advdatafinal.silver.silver_prices_cleaned (
    CONSTRAINT valid_close_px_positive  EXPECT (close_px > 0) ON VIOLATION DROP ROW,
    CONSTRAINT valid_symbol_present     EXPECT (symbol IS NOT NULL) ON VIOLATION DROP ROW,
    CONSTRAINT valid_date_in_range      EXPECT (trade_date BETWEEN '2021-01-01' AND '2025-12-31') ON VIOLATION DROP ROW
);
```

And similarly four to six expectations across silver and gold. Total work: under 30 minutes.
This is the only one of the three questions where I would actually change the pipeline.

## Pipeline architecture overall

Independent of the three questions, the overall structure is sound:

- raw -> silver -> gold lineage with one DLT pipeline that has two source notebooks (SQL +
  Python). Best practice for mixed SQL / ML pipelines.
- Trained outputs (`fct_predictions`, `fct_backtest_pnl_daily`) live OUTSIDE DLT as a
  follow-on notebook task. Correct architectural boundary because trained weights are not
  derivable from one read.
- Catalog + 4 schemas in Unity Catalog, materialised views for window-function stages,
  streaming tables for append-only flows. All idiomatic.
- The Python notebook uses `dlt.read()` for pipeline-internal references so the DAG
  connects properly (we just switched from `spark.read.table` to this in the last commit).

One *honest* concern that surfaced during testing: the most recent run completed in 26 s but
the FinBERT tables ended up with 0 rows. The likely cause is the cross-schema reference
`dlt.read("datos_masked.news_redacted")` not resolving in time during a full refresh on
serverless. The workaround tomorrow is to use `dlt.read_stream(...)` or fully qualify with
`spark.read.table("advdatafinal.datos_masked.news_redacted")` (the second loses DAG
connection but is reliable).

## Sources consulted

- [Model inference using Hugging Face Transformers for NLP - Databricks docs](https://docs.databricks.com/en/machine-learning/train-model/huggingface/model-inference-nlp.html)
- [Getting started with NLP using Hugging Face transformers pipelines - Databricks blog](https://www.databricks.com/blog/2023/02/06/getting-started-nlp-using-hugging-face-transformers-pipelines.html)
- [MLflow on Databricks](https://docs.databricks.com/aws/en/mlflow/)
- [mlflow.transformers API](https://mlflow.org/docs/latest/python_api/mlflow.transformers.html)
- [Manage data quality with pipeline expectations - Databricks docs](https://docs.databricks.com/aws/en/ldp/expectations)
- [Expectation recommendations and advanced patterns - Databricks docs](https://docs.databricks.com/aws/en/ldp/expectation-patterns)
- Course folder: `Advanced data processing/HuggingFacePipelinesText.ipynb`,
  `Advanced data processing/midterm_pipeline_notebook_LT.ipynb`,
  IN014 Sessions 14, 16, 17 PDFs (medallion + DLT material).
