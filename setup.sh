#!/bin/bash
set -e

VENV_DIR="venv"

echo "Iniciando configuración del entorno..."

# Crear el entorno virtual si no existe
if [ ! -d "$VENV_DIR" ]; then
    echo "Creando entorno virtual en '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
fi

# Activar el entorno, actualizar pip e instalar dependencias
echo "Activando entorno y actualizando pip..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo "Instalando dependencias desde requirements.txt..."
    pip install -r requirements.txt
    echo "Configuración completada con éxito."
else
    echo "Aviso: No se encontró requirements.txt, se creó el entorno vacío."
fi

echo "Para activar el entorno manualmente, usa: source $VENV_DIR/bin/activate"
