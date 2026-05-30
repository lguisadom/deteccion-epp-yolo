"""
Entrena un modelo YOLOv8 para la detección de EPP en obras de construcción.

Las decisiones de entrenamiento se definen y documentan en el bloque CONFIG de abajo. Los
resultados se guardan en outputs/training/<name>/: pesos (best.pt / last.pt), curvas de
pérdida, matriz de confusión y curvas PR.
"""

from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "datasets" / "construction-site-safety" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"

# =====================================================================================
# CONFIG — decisiones de entrenamiento documentadas
# =====================================================================================
CONFIG = {
    # --- Modelo ---
    "model": "yolov8s.pt",   # YOLOv8 "small", preentrenado en COCO (transfer learning).
                             #   YOLOv8 es ANCHOR-FREE: no se configuran anclas (anchors).

    # --- Datos ---
    "data": str(DATA_YAML),

    # --- Hiperparámetros de entrenamiento ---
    "epochs": 60,            # pasadas completas por el dataset
    "imgsz": 640,            # resolución de entrada (estándar YOLO)
    "batch": 8,              # tamaño de lote — cabe en 8 GB de VRAM (RTX 4070 Laptop)
    "device": 0,             # GPU 0  (usar 'cpu' si no hubiera GPU)
    "patience": 20,          # early stopping: para si no mejora en 20 épocas
    "seed": 0,               # reproducibilidad

    # --- Data augmentation (entregable: documentar mosaic on/off) ---
    "close_mosaic": 10,      # apaga el augmentation "mosaic" en las últimas 10 épocas
                             #   (mejora la precisión fina al final del entrenamiento)

    # --- Dónde guardar ---
    "project": str(OUTPUT_DIR),
    "name": "yolov8s_baseline",
    "exist_ok": True,        # sobrescribe si ya existe la carpeta
}
# =====================================================================================


def main() -> None:
    print(f"Dataset: {DATA_YAML}")
    print(f"Modelo : {CONFIG['model']}  |  epochs={CONFIG['epochs']}  "
          f"batch={CONFIG['batch']}  imgsz={CONFIG['imgsz']}")

    model = YOLO(CONFIG["model"])
    results = model.train(**CONFIG)

    print("\nEntrenamiento terminado.")
    print(f"Resultados en: {OUTPUT_DIR / CONFIG['name']}")
    print("  - weights/best.pt  -> el mejor modelo (usar este para validar/predecir)")
    print("  - results.png, confusion_matrix.png, *_curve.png -> figuras para el informe")


if __name__ == "__main__":
    main()
