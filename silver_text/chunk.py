"""Deterministic 500-token chunker with 50-token overlap. Uses tiktoken for accurate token counting."""
from __future__ import annotations

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50


def chunk_text(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    tokens = ENC.encode(text)
    chunks: list[str] = []
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    for i in range(0, len(tokens), step):
        slice_tokens = tokens[i : i + CHUNK_TOKENS]
        if len(slice_tokens) < 50:  # drop tiny tails
            break
        chunks.append(ENC.decode(slice_tokens))
    return chunks
