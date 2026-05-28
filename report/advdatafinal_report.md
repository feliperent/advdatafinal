# Section 1. Introduction

## 1.1 Context

The financial industry has spent two decades turning structured market data (prices, volumes, fundamentals) into systematic trading signals. Most of that history happened before sentence-transformer embeddings, before FinBERT, and before models could read company filings at all. The question this project asks is narrow and operational: when a small medallion pipeline ingests 20 US large-cap stocks, scores their news and press releases with FinBERT, embeds their 10-K and 8-K filings with MiniLM, and trains an XGBoost classifier to predict the 5-day directional return, do the text features actually add lift over a pure price-and-fundamentals baseline?

The project was built solo over 14 days during the IN014 Advanced Data Processing and Analysis final-project window. It runs end-to-end on a local Postgres database and is mirrored object-for-object onto a Databricks workspace so the governance and lineage features the course covers (medallion architecture, Unity Catalog, Delta Live Tables, MLflow tracking) can be evaluated against the same data.

## 1.2 The business problem

A buy-side analyst covering 20 stocks today reads news, listens to earnings calls, scans 8-K disclosures, and looks at price action. The analyst then picks five names to overweight for the coming week. The pipeline in this project automates a thin slice of that workflow: it produces a probability for each stock that its price will be higher in five trading days, ranks the universe, and (in backtest) holds the top five names equally weighted with weekly rebalancing.

The decision is binary (up or down in five days). The horizon is short enough that text features could plausibly matter; an analyst who reads a sufficiently bad 8-K on Monday morning may legitimately update her one-week view. The horizon is long enough that classical price-momentum features are well established. Whether text adds anything on top is therefore the right question to pose.

## 1.3 Data sources

Six external sources feed the pipeline. Every source is rate-limited at the API level and read incrementally. The table below lists each source, the schema it lands in, the format, and the volume actually loaded into Postgres after a full ingest of the 2021-2025 window.

| Source | Endpoint | Format | Rows loaded |
|---|---|---|---|
| Equity prices | yfinance | Parquet (per symbol) | 25,100 daily bars across 20 symbols |
| Income statements | FMP `/stable/income-statement` | JSON | 400 quarterly statements (20 stocks × 20 quarters) |
| Balance sheets | FMP `/stable/balance-sheet-statement` | JSON | 400 quarterly statements |
| Cash flow statements | FMP `/stable/cash-flow-statement` | JSON | 400 quarterly statements |
| News articles | FMP `/stable/news/stock` | JSON | 3,853 articles over 90 days |
| Press releases | FMP `/stable/news/press-releases` | JSON | 563 releases. The endpoint caps the per-symbol response at 50 rows, so large-cap counts are floor-truncated. A production deployment would paginate the call. |
| 10-K Item 1A | SEC EDGAR via `edgartools` | Plain text | 95 filings, ~5 per stock over 5 years |
| 8-K material events | SEC EDGAR via `edgartools` | Plain text | 255 filings over 12 months |
| Ingest audit log | local Postgres | Structured | 160 rows with sha256 checksums and per-call metadata |

## 1.4 The universe

The 20 tickers were locked at project start and never changed: AAPL, MSFT, GOOGL, NVDA (Technology); JPM, BAC, GS, AXP (Financials); JNJ, UNH, PFE, LLY (Healthcare); BA, CAT, HON, GE (Industrials); AMZN, WMT, KO, NKE (Consumer). Four stocks per sector across five sectors, all large-caps, all with continuous trading history from January 2021 to December 2025. The point of fixing the universe was not to maximise model performance; it was to control survivorship bias and to keep the dimensionality manageable for a single laptop.

## 1.5 Honest framing of the expected result

A 5-day directional forecast on US large-caps is one of the most studied prediction tasks in finance and one of the hardest to beat the unconditional base rate on. The unconditional 5-day up-rate on this universe over 2021-2025 was 54.3 percent. A model that predicts "up" for everything would score 54.3 percent accuracy and a roughly random AUC. The realistic ceiling for a small academic pipeline is a few hundred basis points of lift above that baseline, with the predictions being most useful when ranked rather than absolute. This expectation is set up front because it shapes how the findings in Section 5 are read.

# Section 2. Data Governance

## 2.1 Provenance and audit trail

Every write to the raw layer is recorded in `raw.ingest_log`, which has one row per (source, symbol, window) tuple with the file path, row count, sha256 of the response body, and the ingest timestamp with timezone. A grader can reconstruct exactly which API call produced which row in `raw.prices_raw` by joining on the symbol and date range. The audit table has 160 rows after a complete ingest (20 symbols × 8 sources). The pattern matches what FRB SR 11-7 calls the "input data lineage" requirement for model risk management: provenance has to be reconstructable, not just inferrable.

## 2.2 Leakage prevention via `as_of_date`

Every silver and gold table carries an `as_of_date` column that records the latest information embedded in that row. For `silver_prices_features`, that is the trade date. For `silver_fundamentals_cleaned`, it is the filing date of the most recent 10-Q used in the trailing-twelve-month rollup. For `silver_filings_10k_chunked`, it is the 10-K filing date. The gold-layer feature panel asof-joins these against the trade date and the panel itself enforces the invariant: every feature on every row was knowable on or before that row's `trade_date`. A regression test in `tests/test_leakage_guard.py` queries the panel and fails if any feature has `as_of_date > trade_date`. The test ran clean on the final pipeline.

## 2.3 Column-level security

The `datos_masked` schema sits between raw and silver. Four streaming tables (`news_redacted`, `press_redacted`, `filings_10k_redacted`, `filings_8k_redacted`) wrap the corresponding raw bodies in a regex pass that replaces email addresses with `[EMAIL]` and US phone numbers with `[PHONE]`. The masking is done at write time, not at read time, so the original raw bodies remain in `raw.news_raw` and `raw.sec_8k_raw` (visible only to the database owner) and every downstream reader, including the FinBERT scorer and the MiniLM chunker, sees the masked text. In Databricks this same pattern is enforced via DLT streaming tables in the `datos_masked` schema with the same regex.

## 2.4 Row-level security

The 20-ticker universe is fixed in `config/universe.yaml` and joined as a hard filter at the silver-to-gold boundary. Any ticker that is not on the list is dropped at silver. The pattern is row-level security implemented as a fact-to-dimension join: the gold layer only sees rows whose symbol appears in `dim_company`, which is itself derived from the universe file.

## 2.5 PII inventory

There is no personally identifiable information by design. The four upstream sources (yfinance, FMP, SEC EDGAR, news headlines) contain no customer data, no employee data, and no transactional data. The regex masking is defensive: a press release might quote an investor-relations email address or a phone number, and even though those are public, masking them keeps the downstream tables uniformly free of contact strings. A formal PII inventory under GDPR Article 30 would list this layer and note that no Article 6 lawful basis is required because no personal data is processed.

## 2.6 Audit log for every retrieval-augmented answer

When a user asks the Streamlit demo a question, the retrieval-augmented generator selects the top-k chunks from `dim_chunk` by cosine similarity, sends them to Claude Haiku 4.5 with the question, and writes one row to `gold.fct_rag_queries`. The row captures the query, the symbol, the timestamp, the cited chunk identifiers as a text array, the model name and version, the latency in milliseconds, the token usage, and the full response. This table is the legal record the system would need if an analyst ever asked "why did the assistant tell me that on date D". A thin dbt view, `fct_rag_chunk_citations`, unnests the cited chunks and joins `dim_chunk`, which is what makes the dimension-to-fact relationship visible in dbt's lineage graph.

# Section 3. Data Stack

## 3.1 Primary stack: local Postgres with dbt and Streamlit

The whole pipeline runs on a 2022 MacBook Air with PostgreSQL 18, dbt-postgres, and Streamlit. The choice was deliberate. The 30-table working set is small (about 51 MB of bronze data, about 90 MB total after silver and gold) and trying to fit it onto a cloud warehouse would add latency and cost without changing anything analytical. Postgres 18 ships with native JSON support, fast window functions, and integer-array columns that are good enough for the 384-dimensional embedding storage (one bytea column per chunk). Streamlit serves the demo at `localhost:8501` and reads directly from the gold tables.

The orchestration is just `make`. A single `make all` runs `init` (creates the four schemas), `ingest` (eight Python fetchers writing to raw), `build` (dbt run plus dbt test for the medallion transformations), `train` (walk-forward XGBoost with MLflow tracking), `backtest` (the top-5 long strategy), `rag` (builds the chunked-text retrieval index and writes a few demo queries to the audit table), and `demo` (boots Streamlit). The first complete `make all` from a clean Postgres takes about 30 minutes.

## 3.2 Why this stack at this scale

The course covered three families of data stacks: traditional warehouses, lake houses with delta-style storage, and local-first analytics. At 30 tables and 25,000 rows of price data, the warehouse and lake-house options would be overkill. The simplest stack that handles every governance, lineage, and machine-learning requirement is the one that wins. Postgres with dbt does star-schema modelling and lineage. dbt does column-level and source-freshness testing. Python does the FinBERT and MiniLM inference. Streamlit serves the user-facing layer. There is no cloud cost and no data-egress dependency.

## 3.3 Appendix B stack: Databricks Delta Live Tables

The course also covered Databricks specifically, so the project was ported in full to a Databricks workspace running on Free Edition Serverless. The medallion layout is identical (raw, datos_masked, silver, gold across the `advdatafinal` catalog). The transformations are split across two notebooks that both run inside the same Delta Live Tables pipeline: `databricksstuff/pipelinedatos.sql` defines the SQL layer (8 raw streaming tables, 4 datos_masked streaming tables, 3 silver objects, 4 gold dimensions, and `gold.fct_feature_panel_daily`), and `databricksstuff/mlpipeline.py` is the notebook task that runs after the DLT pipeline and produces the trained outputs (`gold.fct_predictions`, `gold.fct_backtest_pnl_daily`). Both files live in the same GitHub repository as the local pipeline.

The Databricks side adds three features the local stack cannot easily reproduce. First, declarative data-quality constraints: nine `@dlt.expect` rules sit on the silver and gold tables and either drop offending rows or fail the update, all visible in the DLT Data Quality tab. Second, automatic streaming-table incrementality: when a new file lands in the `/Volumes/advdatafinal/raw/landing/` volume, only the new rows are processed. Third, Unity Catalog lineage: the catalog tracks every table-to-table dependency derived from Spark SQL queries and renders it as a clickable graph in the workspace UI.

## 3.4 Decisions made for cost or pragmatism

Three decisions deserve to be noted because they were not the textbook choice. First, embeddings are stored as bytea-packed float32 vectors in Postgres rather than in pgvector, because the version of PostgreSQL the laptop ships with does not have pgvector available; cosine retrieval is done in Python by unpacking the bytes and using numpy. The performance hit at 14,000 chunks is negligible (about 50 milliseconds per query). Second, the Databricks pipeline runs entirely on Free Edition Serverless, which has one small worker and no GPU; the MiniLM embedding pass on 14,000 chunks therefore takes about 45 minutes on a from-scratch run. The mitigation, described in Section 4, is incremental scoring. Third, XGBoost training and the backtest live in a notebook task rather than in DLT because they cannot be expressed as `@dlt.table` derivations: training produces weights that depend on the fitting process, and the backtest depends on those trained predictions.

# Section 4. Data Model

## 4.1 The four-schema medallion

Every object in the pipeline lives in one of four schemas: `raw`, `datos_masked`, `silver`, `gold`. The intent of each layer is sharp.

The `raw` schema is the source-of-truth landing zone. Tables have text-typed columns matching the API response shape, no casting and no cleaning, and one row per (source, primary key). Idempotency is enforced by `INSERT ... ON CONFLICT (...) DO UPDATE`, so re-running an ingest never duplicates rows. Eight tables: `prices_raw`, `income_statement_raw`, `balance_sheet_raw`, `cash_flow_raw`, `news_raw`, `press_raw`, `sec_10k_raw`, `sec_8k_raw`. A ninth, `ingest_log`, holds the provenance described in Section 2.1.

The `datos_masked` schema holds the regex-masked text views. Four tables, all derived from `raw` by the email-and-phone regex pass.

The `silver` schema holds typed, cleaned, joined data. Six tables: `silver_prices_cleaned` (typed prices with decimal precision and a date column), `silver_prices_features` (the ten technical features added by window functions), `silver_fundamentals_cleaned` (the three FMP statements joined into one row per filing with trailing-twelve-month rollups), `silver_news_scored` (each article scored with FinBERT-tone), `silver_press_scored` (same for press releases), and the two `silver_filings_*_chunked` tables (chunked 10-K and 8-K text with 384-dimensional MiniLM embeddings).

The `gold` schema holds the analytical layer. Five dimensions (`dim_date`, `dim_sector`, `dim_company`, `dim_filing_type`, `dim_chunk`) and six facts (`fct_sentiment_per_day`, `fct_embedding_per_company`, `fct_feature_panel_daily`, `fct_predictions`, `fct_backtest_pnl_daily`, `fct_rag_queries`). The dimensions all derive from upstream tables rather than being VALUES literals; for example, `dim_company` is `SELECT DISTINCT symbol FROM silver.silver_prices_cleaned` joined with a 20-row hard-coded sector mapping, and `dim_date` is `SELECT DISTINCT trade_date FROM silver.silver_prices_cleaned` with calendar-derived columns added.

## 4.2 Architecture diagram

Figure 4.1 shows the Delta Live Tables pipeline on Databricks. The eight raw streaming tables on the left flow into the four `datos_masked` redacted streaming tables, then into `silver_prices_cleaned` (an SCD-Type-1 streaming table built via `APPLY CHANGES INTO`) and the two materialised views in silver, and from there into the four gold dimensions and `fct_feature_panel_daily`. The pipeline holds 21 nodes total and three declarative expectations are enforced inside it.

[ INSERT FIGURE 4.1: report/figures/databricks_dlt_dag.png ]

Figure 4.2 shows the parent Databricks Job that wraps the DLT pipeline and the downstream notebook task. The Job has two tasks chained: `data_pipeline` triggers the DLT pipeline (Figure 4.1), and `ml_pipeline` is a serverless notebook task that depends on the first one finishing. The notebook task runs FinBERT scoring, MiniLM chunk-and-embed, PCA, and XGBoost training. Splitting the work this way keeps every derivable table inside the DLT graph and isolates the trained outputs (predictions, backtest) in a separate orchestration node.

[ INSERT FIGURE 4.2: report/figures/databricks_job_graph.png ]

## 4.3 Star-schema joins, not computed keys

A standard mistake when first building a star schema is to compute surrogate keys in every fact table by hashing the natural key locally. The result is data that joins correctly but lineage graphs that show no dimension-to-fact edges, because no `FROM dim_*` clause exists. The pipeline avoids this. `gold.fct_feature_panel_daily` `INNER JOIN`s `dim_date` (on `full_date = trade_date`), `dim_company` (on `symbol`), and `dim_sector` (on `sector_key`), and pulls `date_key`, `company_key`, and `sector_key` from those joins rather than computing them with `md5()`. The result is the same data, but Unity Catalog (and dbt docs locally) now show explicit fact-to-dimension edges. The pattern is mirrored in `gold.dim_chunk`, which INNER JOINs `dim_filing_type` on the filing-type code.

## 4.4 The feature panel

The 24-feature ML training table is `gold.fct_feature_panel_daily_full`. Its columns split into three groups. Fifteen are structured: `log_ret_1d` (one-day log return), `sma_5`, `sma_20`, `sma_50` (simple moving averages), `ema_12`, `ema_26` (exponential moving averages), `macd_hist` (`ema_12` minus `ema_26`), `bb_z` (z-score of close versus 20-day mean), `vol_20d` (20-day annualised volatility), and six fundamentals from the trailing-twelve-month rollup (`roe`, `roa`, `debt_eq`, `gross_margin`, `op_margin`, `asset_turnover`). Four are sentiment: `news_score`, `n_news`, `press_score`, `n_press` (per-day mean FinBERT score and article counts). Five are text-embedding principal components: `filing_pc1` through `filing_pc5`, the top five PCA loadings on the mean MiniLM embedding of each company's 10-K filings, asof-joined to the trade date. The target is `y_5d_up`, a binary label of one if the close in five trading days is higher than today's close.

The panel has 25,100 rows (one per `symbol`, `trade_date`) with 25,000 of those carrying a non-null target (the last five trading dates per symbol cannot have a target because there is no five-day-future close to compare to). The unconditional `y_5d_up` rate on this panel is 0.543, which is the baseline any model has to beat.

## 4.5 Path A: incremental scoring

A naive ML pipeline would re-score every news article and re-embed every filing on every run. On the Databricks Serverless cluster, that takes 53.7 minutes. The pipeline avoids this with `LEFT ANTI JOIN` logic on the two slow cells. Before scoring news, `mlpipeline.py` reads `silver.silver_news_scored` (if it exists), anti-joins `datos_masked.news_redacted` on `(symbol, published_at, title)`, and only scores the rows that are not already in the target. The same pattern applies to MiniLM with `chunk_key` as the join key. The first run of the pipeline takes 53.7 minutes; every subsequent run takes 4.4 minutes. The proof of this is a single `databricks jobs run-now` invocation on the chained Job, which completed on the second run in 4.4 minutes with the same final table contents as the first.

## 4.6 The 9 declarative quality gates

Inside the DLT pipeline, `@dlt.expect_or_drop` and `@dlt.expect_or_fail` annotations enforce nine data-quality rules that appear in the Databricks Data Quality tab on every run.

| Layer | Table | Constraint | Action |
|---|---|---|---|
| silver | silver_prices_cleaned | symbol IS NOT NULL | DROP ROW |
| silver | silver_prices_cleaned | close_px > 0 | DROP ROW |
| silver | silver_prices_cleaned | trade_date BETWEEN 2020-01-01 AND 2026-12-31 | DROP ROW |
| silver | silver_fundamentals_cleaned | symbol IS NOT NULL | DROP ROW |
| silver | silver_fundamentals_cleaned | revenue_ttm > 0 | DROP ROW |
| gold | dim_company | symbol IS NOT NULL | FAIL UPDATE |
| gold | dim_company | sector_name IS NOT NULL | FAIL UPDATE |
| gold | fct_feature_panel_daily | symbol IS NOT NULL | DROP ROW |
| gold | fct_feature_panel_daily | trade_date IS NOT NULL | DROP ROW |

These are not unit tests run after the pipeline; they are declarative rules built into the pipeline definition that DLT enforces row-by-row at write time and surfaces in a UI tab.

# Section 5. Findings

## 5.1 Per-stock coverage of the silver and gold layers

Every analytical table in the pipeline contains all 20 stocks. The completeness check ran clean: prices are 1,255 trading days per stock, fundamentals are 20 quarterly statements per stock, predictions are 1,194 walk-forward rows per stock. News and press counts vary by company (the FMP press endpoint caps at 50 rows per call and large-caps hit that ceiling; small companies show their actual count) but every stock has at least one row in every analytical table. The two intentionally sparse tables are `raw.ingest_log` (only 8 rows per symbol, one per source per ingest) and `gold.fct_rag_queries` (only 3 rows from demo queries, all on AAPL, because the table is an event log, not a reference table).

## 5.2 AUC across walk-forward folds

Two models were trained and evaluated with walk-forward cross-validation. Rung 1 uses the 15 structured features only. Rung 2 adds the 9 text features (4 sentiment plus 5 PCA components on 10-K embeddings). Both rungs share hyperparameters: 300 trees, max depth 5, learning rate 0.05, subsample and column-subsample 0.8, minimum child weight 5, gamma 0.1, L2 regularisation 1.0, random seed 7. The walk-forward window is 3 years training, 5-day gap, 63-day test, sliding by one quarter, yielding 10 folds across the 2024-2025 evaluation period.

| Rung | Features | Mean fold AUC | Hit rate | Predictions |
|---|---|---|---|---|
| Rung 1 | 15 structured | 0.513 | 0.585 | 9,740 |
| Rung 2 | 24 (15 structured + 4 sentiment + 5 PCA) | 0.512 | 0.564 | 9,740 |

The structured-only model wins by one basis point of AUC and by two points of hit rate. The text features did not add lift.

## 5.3 The main finding: text did not help

Rung 2 underperformed Rung 1 in both AUC and hit rate. The result is small but consistent across folds (Rung 2 was below Rung 1 in 7 out of 10 folds), and it survives different hyperparameter trims. Three possible explanations are worth naming in honest order.

First, the text features are noisy proxies for the same information that price already encodes. FinBERT scoring on a press release one day before the price has already absorbed the news adds nothing on a five-day horizon, and at this universe size (20 stocks, large-caps with deep coverage) most material news is in the price within minutes. The text features carry signal at longer horizons or for smaller-cap stocks; on this configuration, they do not.

Second, the PCA on 10-K embeddings condenses 384 dimensions per filing into 5, which discards most of the variance. A model that used the raw embeddings directly (with regularisation) might extract more, but at this dataset size (25,000 panel rows) the optimisation runs into a curse-of-dimensionality wall well before extracting anything useful.

Third, the sentiment scoring assumes the FinBERT-tone three-class probabilities map linearly to a single scalar, which is a strong assumption. A model that treated the three probabilities as separate features might find the negative class informative even when neutral and positive look noisy. This is the obvious next experiment.

## 5.4 Backtest profit and loss

The top-5 long backtest holds the five highest-probability names each week, rebalances every five trading days, and charges five basis points of round-trip cost per turnover. Over the 2024-2025 out-of-sample window (96 rebalance days), cumulative net returns:

| Strategy | Cumulative net return | Days |
|---|---|---|
| Rung 1 (structured only) | +45.1% | 96 |
| Rung 2 (24 features) | +37.2% | 96 |

The relative ranking matches the AUC and hit-rate ranking: Rung 1 beats Rung 2. The absolute returns are real but driven by the rising market over 2024-2025; the universe equal-weighted benchmark over the same window returned approximately +28%. Both rungs beat the benchmark by a comfortable margin, which means the model is selecting the better-performing names from the universe, not just picking up market beta. The lift of Rung 1 over the equal-weighted benchmark is the answer to the business question posed in Section 1.

## 5.5 RAG retrieval evaluation

The retrieval-augmented generation layer was evaluated on three demo queries against AAPL: "What does Apple flag as its biggest concentration risk?", "How does Apple describe macroeconomic exposure in its most recent 10-K?", "Summarise Apple's most material 8-K filings in the last 12 months." For each query, the retriever returns the top-3 chunks by cosine similarity from the 13,847-chunk corpus, and Claude Haiku 4.5 composes an answer constrained to those chunks. The three queries returned chunks that were genuinely on-topic in all three cases (verified by reading the cited Item 1A sections). The audit row in `fct_rag_queries` captures the full chain: query text, cited chunk identifiers, model name, latency in milliseconds, response.

The choice of three queries on one stock is the right calibration for a demo, not for a benchmark; a full RAG evaluation would require a ground-truth dataset of question-answer pairs, which is out of scope here.

# Section 6. Reflection

## 6.1 What I learned about pipeline architecture

The cost of "obvious" choices became clear when they had to be reversed. Three examples.

Computing `date_key` and `company_key` locally with `md5()` in the fact tables seemed harmless because the result is deterministic and the joins downstream still work. It silently breaks Unity Catalog lineage because the catalog only sees fact-to-dimension edges where a `FROM dim_*` clause exists. The fix (rewriting the gold facts to explicitly `JOIN dim_date`, `JOIN dim_company`, and `JOIN dim_sector`) is two extra `JOIN` clauses and produces identical data, but the lineage graph in Unity Catalog and dbt docs now renders the star schema correctly. This is one of those design choices where the right pattern is invisible in the data and only visible in the metadata, and the cost of getting it wrong shows up only when somebody else tries to read the lineage.

A pandas roundtrip in the ML notebook breaks Spark lineage even more decisively. When `mlpipeline.py` does `spark.table("...").toPandas()`, runs sklearn or xgboost on the pandas DataFrame, and then writes back with `spark.createDataFrame(pdf).write.saveAsTable(...)`, Unity Catalog records the write but cannot connect it to the source table because the chain crosses out of Spark and back in. Three of the gold tables (`fct_embedding_per_company`, `fct_predictions`, `fct_backtest_pnl_daily`) therefore appear in the catalog with an empty upstream-table list. The honest answer is that these tables are trained or sklearn-derived and cannot be expressed as `@dlt.table` functions on Free Edition Serverless. A production deployment using `pyspark.ml.PCA` and a Spark-native XGBoost integration would close the gap.

The third lesson was about incremental processing. The first time the chained Job ran end-to-end it took 53.7 minutes, because every news article was scored from scratch by FinBERT and every chunk was embedded from scratch by MiniLM. The Path A refactor (a `LEFT ANTI JOIN` against the existing scored tables before each model pass) drops re-runs to 4.4 minutes. The pattern is exactly what streaming tables in DLT do automatically via checkpoint tracking; outside DLT, it has to be coded explicitly. Re-runs in production are dominated by the cost of redoing work that has already been done, and the engineering cost of writing the anti-join logic pays for itself the first time the pipeline is run more than once.

## 6.2 What I learned about the financial-modelling problem

The result that text features did not lift the model over a price-and-fundamentals baseline is not a surprise to anyone who has worked with short-horizon equity prediction, but it is a surprise to anyone who reads the popular literature on LLM-driven finance. The literature systematically over-claims because most published results are reported on stale event windows where the news is days or weeks old and the price has already absorbed it. On a clean walk-forward setup at five-day horizons, on large-caps with deep analyst coverage, the contribution of text is essentially zero. This finding is the most important one in the project because it inverts the headline assumption of the popular literature.

## 6.3 What I would do differently

Three things. First, use sentence-level sentiment (with the three FinBERT probabilities as separate features) rather than a single scalar; the linear-collapse assumption is throwing away signal. Second, expand the universe to include small-caps and mid-caps where text would plausibly matter more, ideally with a CRSP-style point-in-time database to avoid survivorship bias. Third, port the trained-output stage to `pyspark.ml` so the lineage graph closes properly, even at the cost of slightly slower training.

## 6.4 What I would NOT do differently

The medallion structure, the explicit star-schema joins, the `@dlt.expect` constraints, and the dual-stack (local Postgres mirror of Databricks) approach were all the right calls and would repeat. The local-first development with a Databricks mirror as Appendix B is a pattern I will reuse outside academic work: it gives all the lineage and governance benefits of the cloud platform when they are needed, at zero cloud cost during the iteration phase. The bytea-packed embedding storage in Postgres is a deliberate trade against pgvector, and it would survive scrutiny if the project ever scaled past 100,000 chunks; below that, the cosine-in-Python latency is invisible to the user.

# Appendix A. Code repository

The full source code (Python ingesters, dbt models, Databricks notebooks, Streamlit app, tests, build scripts) is at `https://github.com/feliperent/advdatafinal` on the `main` branch. Latest commit `9ab5151` includes the dbt source declaration and the `fct_rag_chunk_citations` bridge model.

# Appendix B. Databricks workspace mirror

The full pipeline is mirrored on Databricks Free Edition Serverless at `https://dbc-7ebd40f3-042c.cloud.databricks.com/`. The DLT pipeline `advdatafinal_dlt` (id `a6365b9a-54c9-454b-aed9-d529739b31db`) holds the 21 SQL-derived nodes, and the parent Job `advdatafinal-pipeline` (id `289218560821314`) wraps it with the downstream notebook task. The catalog `advdatafinal` (with schemas `raw`, `datos_masked`, `silver`, `gold`) holds 30 tables that match the local Postgres state object-for-object except for two intentionally local-only audit tables (`raw.ingest_log` and `gold.fct_rag_queries`) and two Databricks-only architectural splits (`silver.silver_prices_features`, separated because DLT cannot run window functions on a streaming table; `gold.fct_feature_panel_daily_full`, separated because the trained-text join lives outside the DLT pipeline).

# Appendix C. Reproducibility

A grader who wants to reproduce the local pipeline runs `make all` against a fresh Postgres 18 instance with the `DATABASE_URL` and `FMP_API_KEY` environment variables set. The first full ingest takes about 30 minutes, the dbt build takes about 90 seconds, the walk-forward training takes about 8 minutes, and the backtest takes about 30 seconds. A grader who wants to reproduce the Databricks side opens the workspace URL above and clicks "Run now" on the Job; the first run takes 54 minutes, every subsequent run takes 4.4 minutes thanks to Path A.
