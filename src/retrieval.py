"""
Retrieval helpers: query embedding + index search.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


class QueryEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        vec = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=False)
        vec = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
        vec = vec / norm
        return vec[0]


def retrieve_top_k(query: str, index, embedder: QueryEmbedder, top_k: int) -> List[Dict]:
    query_emb = embedder.embed(query)
    return index.retrieve(query_emb, top_k=top_k)
