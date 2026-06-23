"""
embed_and_index.py

Lee chunks.jsonl, genera un embedding por chunk con sentence-transformers
(modelo local, gratuito, corre en CPU) y los sube a una coleccion de Qdrant.

Requisitos:
    pip install qdrant-client sentence-transformers

Antes de correr esto, levanta Qdrant:
    docker compose up -d

Uso:
    python embed_and_index.py --input chunks.jsonl --collection mlflow_docs
"""

import argparse
import json

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensiones, rapido y gratuito
BATCH_SIZE = 64


def load_chunks(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="chunks.jsonl")
    parser.add_argument("--collection", default="mlflow_docs")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    args = parser.parse_args()

    print(f"Cargando modelo de embeddings: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    vector_size = model.get_sentence_embedding_dimension()

    print(f"Cargando chunks desde: {args.input}")
    chunks = load_chunks(args.input)
    print(f"Total de chunks: {len(chunks)}")

    client = QdrantClient(url=args.qdrant_url)

    # Crea (o recrea) la coleccion con el tamano de vector correcto
    client.recreate_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Coleccion '{args.collection}' creada en Qdrant (dim={vector_size})")

    # Procesa en batches para no saturar memoria
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False)

        points = [
            PointStruct(
                id=start + i,
                vector=embeddings[i].tolist(),
                payload={
                    "text": batch[i]["text"],
                    "source": batch[i]["source"],
                    "title": batch[i]["title"],
                    "chunk_id": batch[i]["chunk_id"],
                },
            )
            for i in range(len(batch))
        ]
        client.upsert(collection_name=args.collection, points=points)
        print(f"  Subidos {start + len(batch)}/{len(chunks)}")

    print("Listo. Coleccion indexada en Qdrant.")


if __name__ == "__main__":
    main()
