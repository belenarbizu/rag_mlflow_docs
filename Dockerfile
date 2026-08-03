FROM python:3.11-slim

WORKDIR /app

# Instala dependencias primero (aprovecha cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el codigo fuente
COPY src/ ./src/

# Arranca uvicorn desde src/ para que los imports relativos funcionen
WORKDIR /app/src
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]