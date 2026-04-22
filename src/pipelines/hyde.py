"""
HyDE: generate hypothetical document, retrieve from global index by similarity to it, generate final answer.
Do not remove or rename this file.
"""

from __future__ import annotations


def run(query, index, embedder, top_k, generator):
	hypothetical_doc = generator.generate_hypothetical_document(query)
	hyp_embedding = embedder.embed(hypothetical_doc)
	retrieved = index.retrieve(hyp_embedding, top_k=top_k)

	answer = generator.generate(query=query, retrieved_chunks=retrieved)
	return {
		"pipeline": "hyde",
		"hypothetical_document": hypothetical_doc,
		"retrieved_chunks": retrieved,
		"answer": answer,
		"meta": {},
	}
