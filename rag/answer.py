# Claude Haiku RAG generator: question + top-k chunks + SHAP -> explanatory paragraph with citations.
# Audit trail: every call writes one row to gold.fct_rag_queries.
from __future__ import annotations

import json
import os
import time
from typing import Any

import anthropic

from ingest.common import pg_conn
from rag.retrieve import retrieve

CLIENT = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-haiku-4-5-20251001"

PROMPT_TEMPLATE = """You are a financial analyst explaining a model prediction. The model predicted {predicted_class} for {symbol} on {date} with probability {prob_up:.2f}.

Top features driving this prediction (SHAP attributions, larger absolute value = more important):
{shap_block}

Below are the {n_chunks} most relevant text chunks from {symbol}'s SEC filings and recent communications, retrieved by semantic similarity to the question. Each chunk carries an ID like [10K-AAPL-2024-09-29-000079-c07].

{chunks_block}

User question: {question}

Instructions:
1. Write ONE paragraph (4-6 sentences) explaining the prediction.
2. Reference at least 2 chunk IDs inline like [10K-AAPL-2024-09-29-000079-c07].
3. Use plain language. Do not speculate beyond what the chunks and SHAP attributions support.
4. If the chunks contradict the prediction, say so honestly.
"""

def _format_shap(shap_json: dict[str, float] | None) -> str:
    if not shap_json:
        return "  (no SHAP attribution available for this rung/row)"
    items = sorted(shap_json.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
    return "\n".join(f"  - {k}: {v:+.4f}" for k, v in items)

def _format_chunks(chunks_df) -> str:
    return "\n\n".join(
        f"[{r.chunk_key}]: {(r.body_chunk or '')[:600]}" for _, r in chunks_df.iterrows()
    )

def _fetch_prediction_context(symbol: str, date: str, rung: int = 2) -> dict[str, Any]:
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.prob_up::float, p.predicted_class, p.shap_json
            FROM gold.fct_predictions p
            JOIN gold.dim_company c USING (company_key)
            WHERE c.symbol = %s AND p.date_key = md5(%s) AND p.model_rung = %s
            LIMIT 1
            """,
            (symbol, str(date), rung),
        )
        row = cur.fetchone()
    if not row:
        return {"prob_up": 0.5, "predicted_class": 0, "shap_json": None}
    return {"prob_up": float(row[0]), "predicted_class": int(row[1]), "shap_json": row[2]}

def _template_fallback(question: str, symbol: str, as_of_date: str, ctx: dict, chunks) -> str:
    # Template summary used when the LLM API is unavailable.
    direction = "UP" if ctx["predicted_class"] else "DOWN"
    top3 = chunks.head(3)
    cites = ", ".join(f"[{r.chunk_key}]" for _, r in top3.iterrows())
    quotes = "\n\n".join(
        f"[{r.chunk_key}] sim={r.similarity:.3f}: {(r.body_chunk or '')[:280]}..."
        for _, r in top3.iterrows()
    )
    return (
        f"The Rung-{ctx.get('rung', 2)} model predicts {symbol} {direction} on {as_of_date} "
        f"with probability {ctx['prob_up']:.2f}. The most semantically relevant passages "
        f"from {symbol}'s filings retrieved for the question '{question}' are {cites}. "
        f"Top excerpts:\n\n{quotes}\n\n"
        f"(Note: this is the template-fallback summary; the Claude-Haiku-generated paragraph "
        f"is available when the Anthropic API has credit. The retrieval, citations, "
        f"and as_of_date guard are all live.)"
    )

def answer(
    question: str,
    symbol: str,
    as_of_date: str,
    rung: int = 2,
    top_k: int = 8,
) -> tuple[str, list[str], dict[str, Any]]:
    # Return (paragraph, citation_keys, usage_dict).
    ctx = _fetch_prediction_context(symbol, as_of_date, rung=rung)
    ctx["rung"] = rung
    chunks = retrieve(question, symbol=symbol, as_of_date_max=as_of_date, top_k=top_k)

    if chunks.empty:
        msg = f"No filings or news available for {symbol} on or before {as_of_date}."
        return msg, [], {"tokens_in": 0, "tokens_out": 0, "latency_ms": 0, "source": "no-retrieval"}

    prompt = PROMPT_TEMPLATE.format(
        symbol=symbol,
        date=as_of_date,
        prob_up=ctx["prob_up"],
        predicted_class="UP" if ctx["predicted_class"] else "DOWN",
        shap_block=_format_shap(ctx["shap_json"]),
        n_chunks=len(chunks),
        chunks_block=_format_chunks(chunks),
        question=question,
    )

    t0 = time.time()
    paragraph: str
    usage: dict[str, Any]
    try:
        resp = CLIENT.messages.create(
            model=MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        paragraph = resp.content[0].text
        usage = {
            "tokens_in": resp.usage.input_tokens,
            "tokens_out": resp.usage.output_tokens,
            "source": "claude-haiku",
        }
    except Exception as e:
        # Graceful fallback: Anthropic credit exhaustion, network error, etc.
        paragraph = _template_fallback(question, symbol, as_of_date, ctx, chunks)
        usage = {"tokens_in": 0, "tokens_out": 0, "source": "template-fallback", "error": str(e)[:200]}

    latency_ms = int((time.time() - t0) * 1000)
    usage["latency_ms"] = latency_ms
    citations = list(chunks["chunk_key"])

    # Audit log
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS gold.fct_rag_queries (
              query_id    bigserial PRIMARY KEY,
              ts          timestamptz NOT NULL DEFAULT now(),
              question    text NOT NULL,
              symbol      text NOT NULL,
              as_of_date  date NOT NULL,
              rung        smallint,
              top_k_chunk_ids text[] NOT NULL,
              llm_model   text NOT NULL,
              tokens_in   integer,
              tokens_out  integer,
              response    text NOT NULL,
              latency_ms  integer
            )
            """
        )
        cur.execute(
            """
            INSERT INTO gold.fct_rag_queries
              (question, symbol, as_of_date, rung, top_k_chunk_ids, llm_model, tokens_in, tokens_out, response, latency_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (question, symbol, as_of_date, rung, citations,
             MODEL if usage.get("source") == "claude-haiku" else "template-fallback",
             usage.get("tokens_in", 0), usage.get("tokens_out", 0), paragraph, latency_ms),
        )

    return paragraph, citations, usage

if __name__ == "__main__":
    para, cites, usage = answer(
        "What does Apple say about supply chain concentration risk?",
        symbol="AAPL", as_of_date="2025-12-01", rung=2,
    )
    print(para)
    print("\nCitations:", cites)
    print("Usage:", usage)
