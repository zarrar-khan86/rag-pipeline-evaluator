# RAG in the Wild 

rag-pipeline-evaluator is an evaluation framework designed to test and benchmark advanced Retrieval-Augmented Generation (RAG) strategies. Built to handle a noisy, pre-crawled web corpus where relevance is not guaranteed, this repository implements and compares four distinct retrieval pipelines—RAG Fusion, HyDE, Corrective RAG (CRAG), and Graph RAG. The goal of this project is to determine the most robust, hallucination-resistant architecture for factual question-answering systems.

## Requirements

- Python 3.9+
- Node.js 18+ (for the React frontend)

---

## Setup

### Python (backend and pipelines)

```bash
pip install -r requirements.txt
```

Copy `config/config.example.yaml` to `config/config.yaml` and set:

- `dataset_path` — path to `dataset/crag_task_1_and_2_dev_v4.jsonl`
- `embedding_model` — e.g. `all-MiniLM-L6-v2`
- `generation_model` — model name for the LLM you use for answer generation 
- `top_k` — number of chunks to retrieve per query




### Frontend (React)

```bash
cd frontend
npm install
```

---

## Dataset

This project uses the **CRAG Task 1 & 2 dev v4** dataset. 
Download the dataset and place it in the `dataset/` folder yourself.

- **Download (Task 1 & 2, compressed):** [crag_task_1_and_2_dev_v4.jsonl.bz2](https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2)
- Decompress the file (e.g. with 7-Zip or `bzip2 -d`), then put the resulting `crag_task_1_and_2_dev_v4.jsonl` inside the `dataset/` folder.
- **Path after setup:** `dataset/crag_task_1_and_2_dev_v4.jsonl`
- **Format:** One JSON object per line. Fields: `query`, `answer`, `alt_ans`, `search_results` (list of up to 5 items; each has `page_snippet`).
- **Schema:** See `docs/dataset.md`.

All `page_snippet` texts from all rows form the global corpus. Build one embedding index from this corpus; all four pipelines retrieve from it.

---

## Running the project

**Evaluation (run from project root):**

```bash
python run_evaluation.py
```

Optional flags:

```bash
python run_evaluation.py --limit 200 --save-json outputs/eval.json
python run_evaluation.py --rebuild-index
```

This will print accuracy per pipeline and save detailed records (including retrieval scores/confidence) to JSON.

**Backend API (run from project root):**

```bash
python backend/app.py
```

API endpoints:

- `GET /api/health`
- `GET /api/samples?limit=12`
- `POST /api/query` with JSON body: `{"query": "...", "pipeline": "rag_fusion|hyde|crag|graph_rag", "top_k": 3}`

**Frontend (run from project root):**

```bash
cd frontend
npm run dev
```

Open the URL shown (e.g. http://localhost:3000). The frontend proxies `/api/*` to `http://localhost:8000`, so keep the backend running.

---

## Folder structure

Do not change the folder structure. Required layout and the full case-study description are in `ASSIGNMENT.md`.
