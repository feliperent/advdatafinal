"""sentence-transformers/all-MiniLM-L6-v2 embedding wrapper. 384-dim, L2-normalised."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str], show_progress: bool = True) -> np.ndarray:
    """Return (n, 384) L2-normalised embeddings."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return get_model().encode(texts, normalize_embeddings=True, show_progress_bar=show_progress)
