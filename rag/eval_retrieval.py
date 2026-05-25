"""Evaluate retrieval quality on rag/eval_set.jsonl. Reports MRR@5 and P@5.

A query is 'correct' if the top-K retrieved chunks contain at least one chunk
whose chunk_key contains the expected_substring (e.g. '10K-AAPL-' for an AAPL-related question)."""
from __future__ import annotations

import json
from pathlib import Path

from rag.retrieve import retrieve

EVAL_FILE = Path(__file__).parent / "eval_set.jsonl"
TOP_K = 5


def main() -> None:
    queries = [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]
    print(f"Evaluating retrieval on {len(queries)} queries (top_k={TOP_K})")

    mrr_total = 0.0
    p_total = 0.0
    detail = []

    for q in queries:
        chunks = retrieve(
            q["question"],
            symbol=q.get("symbol"),
            as_of_date_max=q.get("as_of_date"),
            top_k=TOP_K,
        )
        match_ranks = []
        match_count = 0
        for i, ck in enumerate(chunks["chunk_key"].tolist()):
            if q["expected_substring"] in ck:
                if not match_ranks:
                    match_ranks.append(i + 1)
                match_count += 1

        rr = 1.0 / match_ranks[0] if match_ranks else 0.0
        precision = match_count / TOP_K
        mrr_total += rr
        p_total += precision
        detail.append((q["symbol"], q["question"][:50], rr, precision, match_count))

    mrr5 = mrr_total / len(queries)
    p5 = p_total / len(queries)
    print(f"\nMRR@{TOP_K} = {mrr5:.3f}    P@{TOP_K} = {p5:.3f}    over {len(queries)} queries\n")

    print(f"{'symbol':6}  {'question':52}  {'RR':>6}  {'P@5':>6}  {'hits':>5}")
    print("-" * 88)
    for sym, q, rr, p, n in detail:
        print(f"{sym:6}  {q:52}  {rr:>6.3f}  {p:>6.3f}  {n:>5}")


if __name__ == "__main__":
    main()
