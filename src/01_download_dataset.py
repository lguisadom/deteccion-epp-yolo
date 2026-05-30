"""
Descarga el dataset Construction Site Safety desde Roboflow Universe.

La clave de API se lee desde el archivo .env (variable ROBOFLOW_API_KEY) y el dataset se
guarda en datasets/.

Dataset: roboflow-universe-projects/construction-site-safety (v27, formato YOLOv8).
Clases (10): Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone,
             Safety Vest, machinery, vehicle.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# --- Configuración del dataset (pública, de Roboflow Universe) ---
WORKSPACE = "roboflow-universe-projects"
PROJECT = "construction-site-safety"
VERSION = 27
FORMAT = "yolov8"

# Carpeta raíz del proyecto (un nivel arriba de src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = PROJECT_ROOT / "datasets"


def main() -> None:
    # Cargar variables del archivo .env
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key or api_key == "tu_api_key_aqui":
        sys.exit(
            "ERROR: falta ROBOFLOW_API_KEY. Define la variable en un archivo .env "
            "(ver .env.example) con tu clave de API de Roboflow."
        )

    # Importar aquí para que el mensaje de error de arriba salga aunque falte la lib
    from roboflow import Roboflow

    DOWNLOAD_DIR.mkdir(exist_ok=True)

    print(f"Descargando {WORKSPACE}/{PROJECT} v{VERSION} (formato {FORMAT})...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    dataset = version.download(FORMAT, location=str(DOWNLOAD_DIR / PROJECT))

    print("\nListo. Dataset descargado en:")
    print(f"  {dataset.location}")
    print("\nRevisa el archivo data.yaml dentro de esa carpeta.")


if __name__ == "__main__":
    main()
