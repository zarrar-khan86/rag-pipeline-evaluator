"""
Global corpus building and vector index utilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.data_loader import load_examples


@dataclass
class IndexedChunk:
    chunk_id: int
    text: str
    source_title: str
    source_url: str
    interaction_id: Optional[str]


class SimpleVectorIndex:
    """In-memory cosine-similarity index over normalized embeddings."""

    def __init__(self, embeddings: np.ndarray, chunks: List[IndexedChunk], model_name: str):
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D matrix")
        if embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings and chunks length mismatch")

        self.embeddings = embeddings.astype(np.float32)
        self.chunks = chunks
        self.model_name = model_name

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]

        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q) + 1e-12
        q = q / q_norm

        scores = self.embeddings @ q
        top_k = max(1, min(top_k, len(self.chunks)))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        results: List[Dict] = []
        for rank, idx in enumerate(top_idx, start=1):
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.text,
                    "score": float(scores[idx]),
                    "source_title": chunk.source_title,
                    "source_url": chunk.source_url,
                    "interaction_id": chunk.interaction_id,
                }
            )
        return results


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / norms


def _build_chunk_records(dataset_path: str) -> List[IndexedChunk]:
    chunks: List[IndexedChunk] = []
    chunk_id = 0

    for example in load_examples(path=dataset_path):
        interaction_id = example.get("interaction_id")
        for sr in example.get("search_results", []):
            snippet = (sr.get("page_snippet") or "").strip()
            if not snippet:
                continue

            chunks.append(
                IndexedChunk(
                    chunk_id=chunk_id,
                    text=snippet,
                    source_title=sr.get("page_name") or "Unknown source",
                    source_url=sr.get("page_url") or "",
                    interaction_id=interaction_id,
                )
            )
            chunk_id += 1

    if not chunks:
        raise ValueError("No non-empty snippets found in dataset.")

    return chunks


def _get_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def build_index(
    dataset_path: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    index_path: Optional[str] = None,
    batch_size: int = 256,
) -> SimpleVectorIndex:
    """
    Build a global snippet corpus and vector index from the full dataset.
    """
    chunks = _build_chunk_records(dataset_path)
    embedder = _get_embedder(embedding_model)

    texts = [c.text for c in chunks]
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = _normalize_rows(np.asarray(embeddings, dtype=np.float32))

    index = SimpleVectorIndex(embeddings=embeddings, chunks=chunks, model_name=embedding_model)

    if index_path:
        save_index(index, index_path)

    return index


def save_index(index: SimpleVectorIndex, index_path: str) -> None:
    base = Path(index_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(base) + ".npz",
        embeddings=index.embeddings,
    )

    meta = {
        "model_name": index.model_name,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source_title": c.source_title,
                "source_url": c.source_url,
                "interaction_id": c.interaction_id,
            }
            for c in index.chunks
        ],
    }
    with open(str(base) + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def load_index(index_path: str) -> SimpleVectorIndex:
    base = Path(index_path)
    npz_file = Path(str(base) + ".npz")
    json_file = Path(str(base) + ".json")

    if not npz_file.exists() or not json_file.exists():
        raise FileNotFoundError(
            f"Index files not found. Expected {npz_file} and {json_file}."
        )

    data = np.load(npz_file)
    embeddings = data["embeddings"].astype(np.float32)

    with open(json_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    chunks = [
        IndexedChunk(
            chunk_id=item["chunk_id"],
            text=item["text"],
            source_title=item.get("source_title") or "Unknown source",
            source_url=item.get("source_url") or "",
            interaction_id=item.get("interaction_id"),
        )
        for item in meta["chunks"]
    ]

    return SimpleVectorIndex(
        embeddings=embeddings,
        chunks=chunks,
        model_name=meta.get("model_name", "all-MiniLM-L6-v2"),
    )


def embed_text(text: str, embedding_model: str) -> np.ndarray:
    embedder = _get_embedder(embedding_model)
    emb = embedder.encode([text], convert_to_numpy=True, normalize_embeddings=False)
    emb = np.asarray(emb, dtype=np.float32)
    emb = _normalize_rows(emb)
    return emb[0]
