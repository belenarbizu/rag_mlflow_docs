"""
run_ragas_eval.py

Corre el pipeline RAG completo (retriever + generator) sobre un set
de preguntas de evaluacion, y mide la calidad con RAGAS usando:

  - Faithfulness        -> la respuesta se basa en el contexto? (mide alucinacion)
  - AnswerRelevancy      -> la respuesta es relevante a la pregunta?
  - LLMContextPrecision  -> los chunks recuperados son relevantes?
  - LLMContextRecall     -> el contexto recuperado cubre lo que pedia la pregunta?

El "juez" que evalua estas metricas es un LLM (aqui, el mismo Ollama local).

Requisitos (version pineada por incompatibilidad conocida entre ragas
0.4.3 y versiones recientes de langchain-community):
    pip install "ragas==0.4.3" "langchain-community==0.3.27"

Uso:
    python run_ragas_eval.py --dataset eval_dataset.json --output ragas_results.csv
"""

import argparse
import json

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from generator import Generator
from retriever import Retriever

JUDGE_MODEL = "llama3.2:3b"  # modelo mas liviano que mistral; mas rapido en CPU como juez


def build_ragas_dataset(eval_items, retriever, generator):
    rows = []
    for item in eval_items:
        question = item["question"]
        chunks = retriever.search(question, top_k=5)
        contexts = [c["text"] for c in chunks]
        answer = generator.answer(question, chunks)

        rows.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": item["ground_truth"],
        })
        print(f"  Procesada: {question[:60]}...")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval_dataset.json")
    parser.add_argument("--output", default="ragas_results.csv")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="mlflow_docs")
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        eval_items = json.load(f)

    print(f"Preguntas de evaluacion: {len(eval_items)}")

    retriever = Retriever(qdrant_url=args.qdrant_url, collection=args.collection)
    generator = Generator(model="mistral")

    print("Generando respuestas con el pipeline RAG...")
    rows = build_ragas_dataset(eval_items, retriever, generator)

    dataset = EvaluationDataset.from_list(rows)

    print(f"Cargando modelo juez ({JUDGE_MODEL}) para evaluar...")
    judge_llm = LangchainLLMWrapper(
        ChatOllama(model=JUDGE_MODEL, request_timeout=300.0, temperature=0.0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
    ]

    # max_workers=1 evita saturar a Ollama con peticiones en paralelo (es lo que
    # causaba los TimeoutError); timeout alto le da tiempo a un modelo en CPU.
    run_config = RunConfig(timeout=300, max_workers=1, max_retries=2)

    print("Calculando metricas (esto puede tardar varios minutos en CPU)...")
    result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)

    df = result.to_pandas()
    df.to_csv(args.output, index=False)

    print("\n=== Resultados promedio ===")
    for col in ["faithfulness", "answer_relevancy",
                "llm_context_precision_with_reference", "llm_context_recall"]:
        if col in df.columns:
            print(f"  {col}: {df[col].mean():.2f}")

    print(f"\nResultados detallados guardados en: {args.output}")


if __name__ == "__main__":
    main()
