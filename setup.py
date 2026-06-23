import os
import subprocess
import sys

def main():
    venv_dir = "venv"
    req_file = "requirements.txt"

    print(f"--- Creando entorno virtual en '{venv_dir}' ---")
    subprocess.check_call([sys.executable, "-m", "venv", venv_dir])

    # Definir el ejecutable de python dentro del entorno según el sistema operativo
    if os.name == "nt":  # Windows
        python_venv = os.path.join(venv_dir, "Scripts", "python.exe")
    else:  # Unix/macOS
        python_venv = os.path.join(venv_dir, "bin", "python")

    print("--- Actualizando pip ---")
    subprocess.check_call([python_venv, "-m", "pip", "install", "--upgrade", "pip"])

    if os.path.exists(req_file):
        print(f"--- Instalando dependencias desde {req_file} ---")
        subprocess.check_call([python_venv, "-m", "pip", "install", "-r", req_file])
        print("--- Configuración completada ---")
    else:
        print(f"Aviso: No se encontró '{req_file}'.")

if __name__ == "__main__":
    main()
