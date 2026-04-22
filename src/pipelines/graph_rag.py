"""
Graph RAG: graph-augmented retrieval over the corpus (e.g. entity/relation graph or similarity graph), then generate.
Do not remove or rename this file.
"""

from __future__ import annotations

import re
from collections import defaultdict

from src.retrieval import retrieve_top_k


def _extract_entities(text):
	# Lightweight entity proxy: title-cased multiword phrases and capitalized tokens.
	candidates = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
	return {c.strip() for c in candidates if len(c.strip()) > 2}


def _expand_by_entity_overlap(seed_chunks, all_chunks, top_k):
	seed_entities = set()
	for c in seed_chunks:
		seed_entities.update(_extract_entities(c.get("chunk_text", "")))

	overlap_scores = defaultdict(float)
	payloads = {}
	for c in all_chunks:
		entities = _extract_entities(c.get("chunk_text", ""))
		overlap = len(seed_entities.intersection(entities))
		if overlap <= 0:
			continue
		chunk_id = c["chunk_id"]
		overlap_scores[chunk_id] = overlap + float(c.get("score", 0.0))
		payloads[chunk_id] = c

	ranked_ids = sorted(overlap_scores.keys(), key=lambda cid: overlap_scores[cid], reverse=True)
	expanded = []
	for cid in ranked_ids[:top_k]:
		item = dict(payloads[cid])
		item["graph_score"] = float(overlap_scores[cid])
		expanded.append(item)
	return expanded


def run(query, index, embedder, top_k, generator):
	# Seed retrieval from vector index.
	seed = retrieve_top_k(query, index, embedder, top_k=max(top_k, 5))
	# Build graph neighborhood via entity overlap against a wider candidate set.
	broader = retrieve_top_k(query, index, embedder, top_k=min(30, max(10, top_k * 5)))
	graph_augmented = _expand_by_entity_overlap(seed, broader, top_k=top_k)
	retrieved = graph_augmented if graph_augmented else seed[:top_k]

	answer = generator.generate(query=query, retrieved_chunks=retrieved)
	return {
		"pipeline": "graph_rag",
		"retrieved_chunks": retrieved,
		"answer": answer,
		"meta": {"seed_count": len(seed), "graph_selected": len(graph_augmented)},
	}
