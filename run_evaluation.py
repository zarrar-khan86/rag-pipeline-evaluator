"""
Run all 4 pipelines on the dev set (or a subset), compute accuracy per pipeline, print or save results.
Do not remove or rename this file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.corpus import build_index, load_index
from src.data_loader import load_examples
from src.evaluation import evaluate_pipeline_predictions, is_correct
from src.generation import AnswerGenerator
from src.pipelines import PIPELINES
from src.retrieval import QueryEmbedder


def _load_config(path: str = "config/config.yaml") -> dict:
	cfg_path = Path(path)
	if not cfg_path.exists():
		cfg_path = Path("config/config.example.yaml")
	with open(cfg_path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N samples")
	parser.add_argument(
		"--save-json",
		type=str,
		default="evaluation_results.json",
		help="Output JSON file path",
	)
	parser.add_argument(
		"--rebuild-index",
		action="store_true",
		help="Force rebuilding index even if saved files exist",
	)
	args = parser.parse_args()

	cfg = _load_config()
	dataset_path = cfg.get("dataset_path", "dataset/crag_task_1_and_2_dev_v4.jsonl")
	embedding_model = cfg.get("embedding_model", "all-MiniLM-L6-v2")
	generation_model = cfg.get("generation_model", "fallback")
	top_k = int(cfg.get("top_k", 3))
	index_path = cfg.get("index_path", "index/crag_snippet_index")

	npz_exists = Path(index_path + ".npz").exists()
	json_exists = Path(index_path + ".json").exists()
	if not args.rebuild_index and npz_exists and json_exists:
		index = load_index(index_path)
	else:
		index = build_index(
			dataset_path=dataset_path,
			embedding_model=embedding_model,
			index_path=index_path,
		)

	embedder = QueryEmbedder(embedding_model)
	generator = AnswerGenerator(model_name=generation_model)

	rows = []
	detail_rows = []

	for ex in load_examples(path=dataset_path, limit=args.limit):
		query = ex["query"]
		answer = ex["answer"]
		alt_ans = ex["alt_ans"]

		for pipeline_name, runner in PIPELINES.items():
			result = runner(query=query, index=index, embedder=embedder, top_k=top_k, generator=generator)
			prediction = result.get("answer", "")
			correct = is_correct(prediction, answer, alt_ans)
			retrieved = result.get("retrieved_chunks", [])

			rows.append(
				{
					"pipeline": pipeline_name,
					"prediction": prediction,
					"answer": answer,
					"alt_ans": alt_ans,
				}
			)
			detail_rows.append(
				{
					"interaction_id": ex.get("interaction_id"),
					"query": query,
					"pipeline": pipeline_name,
					"prediction": prediction,
					"gold_answer": answer,
					"is_correct": correct,
					"retrieval_scores": [float(c.get("score", 0.0)) for c in retrieved],
					"retrieval_confidence": result.get("meta", {}).get("retrieval_confidence"),
				}
			)

	acc = evaluate_pipeline_predictions(rows)

	print("\n=== Accuracy by Pipeline ===")
	for name, score in sorted(acc.items(), key=lambda x: x[1], reverse=True):
		print(f"{name:12s} : {score:.4f}")

	payload = {
		"config": {
			"dataset_path": dataset_path,
			"embedding_model": embedding_model,
			"generation_model": generation_model,
			"top_k": top_k,
			"limit": args.limit,
		},
		"accuracy": acc,
		"details": detail_rows,
	}

	out_path = Path(args.save_json)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=False, indent=2)

	print(f"\nSaved detailed results to: {out_path}")


if __name__ == "__main__":
	main()
