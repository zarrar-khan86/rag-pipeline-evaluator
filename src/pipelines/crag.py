"""
CRAG (Corrective RAG): assess retrieval confidence; use or correct retrieval based on it, then generate.
Do not remove or rename this file.
"""

from __future__ import annotations

from statistics import mean

from src.retrieval import retrieve_top_k


def _confidence_from_scores(chunks):
	if not chunks:
		return 0.0
	top_score = float(chunks[0].get("score", 0.0))
	avg_score = mean(float(c.get("score", 0.0)) for c in chunks)
	# Weighted confidence favoring the strongest match while considering list quality.
	conf = 0.7 * top_score + 0.3 * avg_score
	return float(conf)


def run(query, index, embedder, top_k, generator):
	retrieved = retrieve_top_k(query, index, embedder, top_k=top_k)
	confidence = _confidence_from_scores(retrieved)

	# If confidence is low, use stricter evidence selection (top-1).
	if confidence < 0.40 and retrieved:
		corrected_retrieval = retrieved[:1]
	else:
		corrected_retrieval = retrieved

	allow_no_context = confidence < 0.25
	answer = generator.generate(
		query=query,
		retrieved_chunks=corrected_retrieval,
		require_citations=True,
		allow_no_context=allow_no_context,
	)
	return {
		"pipeline": "crag",
		"retrieved_chunks": corrected_retrieval,
		"answer": answer,
		"meta": {
			"retrieval_confidence": confidence,
			"used_low_confidence_fallback": confidence < 0.40,
		},
	}
