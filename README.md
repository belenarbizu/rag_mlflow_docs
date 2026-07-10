# MLflow Docs RAG

Question-answering system over the official MLflow documentation (`classic-ml` section), built with a fully local and open source RAG (Retrieval-Augmented Generation) pipeline.

## Architecture

```
Documents (.mdx) → Cleaning + Chunking → Embeddings → Qdrant (vector DB)
                                                              ↓
                        User → FastAPI /query → Retrieval → LLM (Ollama) → Answer + sources
```

## Stack

| Component | Tool |
|---|---|
| Chunking | LangChain text splitters |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | Qdrant |
| LLM | Mistral 7B (via Ollama, local) |
| API | FastAPI |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision/recall) |
| Containers | Docker / Docker Compose |

## Why this project

This project complements a previous classic MLOps project (churn prediction with experiment tracking, drift monitoring, and deployment) by tackling a GenAI use case: indexing real technical documentation and serving answers grounded in verifiable sources, while systematically evaluating quality and failure modes rather than assuming the system "just works".

## Project structure

```
.
├── Dockerfile
├── docker-compose.yml     # spins up Qdrant + API together
├── conftest.py            # pytest path configuration
├── requirements.txt
├── .env.example           # environment variables template (copy to .env)
├── src/
│   ├── load_and_chunk.py  # loads .mdx/.md files, cleans and splits into chunks
│   ├── embed_and_index.py # generates embeddings and uploads them to Qdrant
│   ├── retriever.py       # semantic search over Qdrant
│   ├── generator.py       # builds the prompt and calls the LLM via Ollama
│   ├── main.py            # FastAPI app (POST /query endpoint)
│   └── run_ragas_eval.py  # automated evaluation with RAGAS
├── tests/
│   └── test_pipeline.py   # 42 pytest tests (chunking, retrieval, generator, API)
└── eval_dataset.json      # questions + reference answers for evaluation
```

> `classic-ml/` (source documents) and `.env` are not included in this repo — see setup instructions below.

## Getting started

**Requirements**: Python 3.10+, Docker, [Ollama](https://ollama.com/download)

### 1. Download the MLflow documentation

The source documents are **not included in this repository**. Clone the MLflow repo and copy the `classic-ml` folder locally:

```bash
git clone --depth 1 https://github.com/mlflow/mlflow.git
```

Then copy `mlflow/docs/docs/classic-ml/` into the root of this project so the structure looks like:

```
rag-mlflow-docs/
├── classic-ml/        ← paste it here
├── src/
├── ...
```

### 2. Set up environment variables

`.env` is not tracked by git. Create it from the provided template:

```bash
cp .env.example .env
```

The `.env.example` file already contains the correct default values — no changes needed to run the project locally:

```
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=mlflow_docs
OLLAMA_MODEL=mistral
OLLAMA_HOST=http://host.docker.internal:11434
```

### 3. Pull the local model

```bash
ollama pull mistral
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Index the documentation

```bash
python src/load_and_chunk.py --input ./classic-ml --output chunks.jsonl
python src/embed_and_index.py --input chunks.jsonl --collection mlflow_docs
```

### 6. Start the full stack

```bash
docker compose up --build
```

This starts both Qdrant and the FastAPI app in Docker. The API is available at `http://localhost:8000/docs` (Swagger UI).

```powershell
# Windows PowerShell
Invoke-WebRequest -Uri http://localhost:8000/query `
  -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"question": "What is MLflow Model Registry used for?"}' `
  -UseBasicParsing
```

```bash
# Linux / macOS
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is MLflow Model Registry used for?"}'
```

## Tests

```bash
pytest tests/ -v
```

42 tests covering chunking, MDX cleaning, retrieval (mocked Qdrant), generator (mocked Ollama), and the FastAPI endpoints.

## Evaluation

The system was evaluated with [RAGAS](https://docs.ragas.io/) on a set of 6 questions with reference answers, measuring faithfulness (groundedness / hallucination), answer relevancy, and context precision/recall.

**Summary results**: faithfulness 0.77 · answer relevancy 0.93 · context recall 0.75

To run the evaluation yourself:

```bash
python src/run_ragas_eval.py --dataset eval_dataset.json --output ragas_results.csv
```

> The evaluation uses `llama3.1:8b` as judge model and runs best with a GPU. It was tested on Google Colab (T4).

## Design decisions

- **Only `classic-ml` was indexed**, not the full MLflow documentation, as a deliberate scope for an evaluable MVP.
- **Fully local models** (embeddings and LLM via Ollama) to avoid paid API dependencies and keep the project reproducible without API keys.
- **Custom MDX cleaning**: frontmatter, JSX imports, UI components (`<Tabs>`, etc.) and broken image links are stripped out so the LLM receives clean, useful text rather than documentation markup.
- Evaluation with a local LLM judge exposed real limitations — documented honestly rather than hidden, as part of a rigorous evaluation process.

## Next steps

- Query observability in production (Langfuse)
- Expand the indexed corpus and the evaluation dataset
