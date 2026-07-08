"""
conftest.py

Configura el sys.path para que pytest encuentre los modulos en src/
sin necesidad de instalar el paquete.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))