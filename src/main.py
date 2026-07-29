"""
main.py

API del RAG sobre la documentacion de MLflow.
Cada query queda registrada en Langfuse con:
  - la pregunta recibida
  - los chunks recuperados (retrieval span)
  - la llamada al LLM (generation span)
  - la respuesta final y las fuentes

Variables de entorno (ver .env.example):
    QDRANT_URL              - URL de Qdrant (default: http://localhost:6333)
    QDRANT_COLLECTION       - nombre de la coleccion (default: mlflow_docs)
    OLLAMA_MODEL            - modelo LLM (default: mistral)
    OLLAMA_HOST             - URL de Ollama (default: http://localhost:11434)
    LANGFUSE_PUBLIC_KEY     - clave publica de Langfuse
    LANGFUSE_SECRET_KEY     - clave secreta de Langfuse
    LANGFUSE_HOST           - host de Langfuse (default: https://cloud.langfuse.com)
"""

import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from langfuse import get_client, observe, propagate_attributes
from pydantic import BaseModel

from src.generator import Generator
from src.retriever import Retriever

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "mlflow_docs")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")

app = FastAPI(title="MLflow Docs RAG")
lf = get_client()  # lee LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST del .env


def init_retriever(retries: int = 5, delay: int = 3) -> Retriever:
    """Intenta conectar a Qdrant con reintentos."""
    for attempt in range(1, retries + 1):
        try:
            r = Retriever(qdrant_url=QDRANT_URL, collection=COLLECTION)
            r.client.get_collection(COLLECTION)
            print(f"Conectado a Qdrant en {QDRANT_URL} (intento {attempt})")
            return r
        except Exception as e:
            print(f"Qdrant no disponible (intento {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError(f"No se pudo conectar a Qdrant en {QDRANT_URL} tras {retries} intentos")


retriever = init_retriever()
generator = Generator(model=OLLAMA_MODEL)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@observe(name="retrieval", as_type="retriever", capture_input=True, capture_output=True)
def run_retrieval(question: str, top_k: int) -> list[dict]:
    """Busca los chunks mas relevantes en Qdrant. El decorador registra input/output en Langfuse."""
    return retriever.search(question, top_k=top_k)


@observe(name="generation", as_type="generation", capture_input=True, capture_output=True)
def run_generation(question: str, chunks: list[dict]) -> str:
    """Llama al LLM con el contexto recuperado. El decorador registra input/output en Langfuse."""
    return generator.answer(question, chunks)


@app.post("/query", response_model=QueryResponse)
@observe(name="rag-query", capture_input=True, capture_output=True)
def query(request: QueryRequest):
    """
    Endpoint principal del RAG.
    Cada llamada genera una traza en Langfuse con los spans de retrieval y generacion.
    """
    with propagate_attributes(tags=["rag", "mlflow-docs"]):
        chunks = run_retrieval(request.question, top_k=request.top_k)
        answer = run_generation(request.question, chunks)

    sources = sorted(set(c["source"] for c in chunks))
    return QueryResponse(answer=answer, sources=sources)
