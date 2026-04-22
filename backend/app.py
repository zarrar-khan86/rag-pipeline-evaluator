from __future__ import annotations

from pathlib import Path
import sys

# Allow running this file directly: `python backend/app.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.corpus import build_index, load_index
from src.data_loader import load_examples
from src.generation import AnswerGenerator
from src.pipelines import PIPELINES
from src.retrieval import QueryEmbedder


app = Flask(__name__)
CORS(app)


def _load_config(path: str = "config/config.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        cfg_path = Path("config/config.example.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = _load_config()
DATASET_PATH = CFG.get("dataset_path", "dataset/crag_task_1_and_2_dev_v4.jsonl")
EMBEDDING_MODEL = CFG.get("embedding_model", "all-MiniLM-L6-v2")
GENERATION_MODEL = CFG.get("generation_model", "fallback")
TOP_K = int(CFG.get("top_k", 3))
INDEX_PATH = CFG.get("index_path", "index/crag_snippet_index")


def _init_runtime():
    npz_exists = Path(INDEX_PATH + ".npz").exists()
    json_exists = Path(INDEX_PATH + ".json").exists()
    if npz_exists and json_exists:
        idx = load_index(INDEX_PATH)
    else:
        idx = build_index(
            dataset_path=DATASET_PATH,
            embedding_model=EMBEDDING_MODEL,
            index_path=INDEX_PATH,
        )

    emb = QueryEmbedder(EMBEDDING_MODEL)
    gen = AnswerGenerator(model_name=GENERATION_MODEL)
    return idx, emb, gen


INDEX, EMBEDDER, GENERATOR = _init_runtime()


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/samples")
def samples():
    limit = int(request.args.get("limit", 10))
    rows = []
    for ex in load_examples(path=DATASET_PATH, limit=limit):
        rows.append(
            {
                "interaction_id": ex.get("interaction_id"),
                "query": ex.get("query", ""),
            }
        )
    return jsonify({"samples": rows})


@app.post("/api/query")
def query_api():
    payload = request.get_json(force=True)
    query = (payload.get("query") or "").strip()
    pipeline_name = payload.get("pipeline", "rag_fusion")
    top_k = int(payload.get("top_k", TOP_K))

    if not query:
        return jsonify({"error": "query is required"}), 400
    if pipeline_name not in PIPELINES:
        return jsonify({"error": f"unknown pipeline: {pipeline_name}"}), 400

    runner = PIPELINES[pipeline_name]
    result = runner(query=query, index=INDEX, embedder=EMBEDDER, top_k=top_k, generator=GENERATOR)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
