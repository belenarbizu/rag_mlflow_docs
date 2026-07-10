"""
generator.py

Construye el prompt con el contexto recuperado y llama a un modelo
local servido por Ollama.

Variables de entorno:
    OLLAMA_MODEL  - nombre del modelo (default: mistral)
    OLLAMA_HOST   - URL del servidor Ollama (default: http://localhost:11434)
                    En Docker usar: http://host.docker.internal:11434
"""

import os

SYSTEM_PROMPT = """Eres un asistente experto en MLflow. Respondes preguntas \
usando UNICAMENTE la informacion del contexto proporcionado, extraido de la \
documentacion oficial de MLflow.

Reglas:
- Si el contexto no tiene informacion suficiente para responder, dilo \
claramente en vez de inventar una respuesta.
- Cuando uses informacion de un fragmento, menciona de que archivo viene \
(usa el campo "source").
- Se conciso y tecnico, como si hablaras con otro desarrollador.
"""


class Generator:
    def __init__(self, model: str = "mistral"):
        self.model = model
        self.ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def build_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, start=1):
            parts.append(
                f"[Fragmento {i} | fuente: {c['source']}]\n{c['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def answer(self, question: str, chunks: list[dict]) -> str:
        import ollama

        context = self.build_context(chunks)
        user_message = (
            f"Contexto:\n{context}\n\n"
            f"Pregunta del usuario: {question}"
        )

        client = ollama.Client(host=self.ollama_host)
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response["message"]["content"]
