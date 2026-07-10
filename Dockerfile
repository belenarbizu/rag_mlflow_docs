FROM python:3.11-slim

WORKDIR /app

# Instala dependencias primero (aprovecha cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el codigo fuente
COPY src/ ./src/

# Expone el puerto de la API
EXPOSE 8000

# Arranca con uvicorn apuntando al modulo en src/
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]