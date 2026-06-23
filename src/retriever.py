"""
retriever.py

Encapsula la logica de busqueda: recibe una pregunta en texto,
la convierte a embedding y devuelve los chunks mas similares
desde Qdrant.
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Retriever:
    def __init__(self, qdrant_url: str, collection: str):
        self.client = QdrantClient(url=qdrant_url)
        self.collection = collection
        self.model = SentenceTransformer(MODEL_NAME)

    def search(self, query: str, top_k: int = 5):
        query_vector = self.model.encode(query).tolist()
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
        ).points

        return [
            {
                "score": r.score,
                "text": r.payload["text"],
                "source": r.payload["source"],
                "title": r.payload["title"],
            }
            for r in results
        ]
