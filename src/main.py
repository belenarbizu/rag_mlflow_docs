"""
main.py

API del RAG sobre la documentacion de MLflow.

Variables de entorno (ver .env.example):
    QDRANT_URL          - URL de Qdrant (default: http://localhost:6333)
    QDRANT_COLLECTION   - nombre de la coleccion (default: mlflow_docs)
    OLLAMA_MODEL        - modelo LLM (default: mistral)
    OLLAMA_HOST         - URL de Ollama (default: http://localhost:11434)
"""

import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from src.generator import Generator
from src.retriever import Retriever

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "mlflow_docs")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")

app = FastAPI(title="MLflow Docs RAG")


def init_retriever(retries: int = 5, delay: int = 3) -> Retriever:
    """Intenta conectar a Qdrant con reintentos, por si la API arranca antes que Qdrant."""
    for attempt in range(1, retries + 1):
        try:
            r = Retriever(qdrant_url=QDRANT_URL, collection=COLLECTION)
            # Prueba real de conexion
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


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    chunks = retriever.search(request.question, top_k=request.top_k)
    answer = generator.answer(request.question, chunks)
    sources = sorted(set(c["source"] for c in chunks))
    return QueryResponse(answer=answer, sources=sources)
