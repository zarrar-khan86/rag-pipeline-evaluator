"""
RAG Fusion: multiple queries, retrieve from global index for each, merge ranked lists (e.g. RRF), generate answer.
Do not remove or rename this file.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from src.retrieval import retrieve_top_k


def _query_variants(query: str) -> List[str]:
	return [
		query,
		f"detailed factual answer for: {query}",
		f"key entities and evidence for: {query}",
		f"short answer with supporting facts: {query}",
	]


def _rrf_fuse(ranked_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
	# Reciprocal Rank Fusion score: sum(1 / (k + rank_i(d))).
	fused = defaultdict(float)
	chunk_payload = {}

	for ranked in ranked_lists:
		for item in ranked:
			chunk_id = item["chunk_id"]
			rank = max(1, int(item.get("rank", 1)))
			fused[chunk_id] += 1.0 / (k + rank)
			chunk_payload[chunk_id] = item

	ordered_ids = sorted(fused.keys(), key=lambda cid: fused[cid], reverse=True)
	out = []
	for idx, chunk_id in enumerate(ordered_ids, start=1):
		base = dict(chunk_payload[chunk_id])
		base["rank"] = idx
		base["fusion_score"] = float(fused[chunk_id])
		out.append(base)
	return out


def run(query, index, embedder, top_k, generator):
	variants = _query_variants(query)
	ranked_lists = [retrieve_top_k(v, index, embedder, top_k=top_k) for v in variants]
	fused = _rrf_fuse(ranked_lists)
	retrieved = fused[:top_k]

	answer = generator.generate(query=query, retrieved_chunks=retrieved)
	return {
		"pipeline": "rag_fusion",
		"retrieved_chunks": retrieved,
		"answer": answer,
		"meta": {"query_variants": variants},
	}
