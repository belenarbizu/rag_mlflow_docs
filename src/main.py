"""
main.py

API del RAG sobre la documentacion de MLflow.

Uso:
    uvicorn main:app --reload --port 8000
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from generator import Generator
from retriever import Retriever

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "mlflow_docs")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")

app = FastAPI(title="MLflow Docs RAG")

retriever = Retriever(qdrant_url=QDRANT_URL, collection=COLLECTION)
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
