"""
generator.py

Construye el prompt con el contexto recuperado y llama a un modelo
local servido por Ollama (gratuito, corre en tu propia maquina).

Requisitos:
    1. Instalar Ollama: https://ollama.com/download
    2. Descargar un modelo, por ejemplo:
         ollama pull llama3.2:3b
    3. pip install ollama
"""

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
    def __init__(self, model="mistral"):
        self.model = model
 
    def build_context(self, chunks):
        parts = [f"[Fragmento {i+1} | fuente: {c['source']}]\n{c['text']}" for i, c in enumerate(chunks)]
        return "\n\n---\n\n".join(parts)
 
    def answer(self, question, chunks):
        import ollama  # import lazy: permite mockear en tests sin tener ollama instalado
        context = self.build_context(chunks)
        user_message = f"Contexto:\n{context}\n\nPregunta del usuario: {question}"
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response["message"]["content"]
