"""
Evalúa el modelo entrenado sobre el conjunto de prueba (test): mAP@0.5 y mAP@0.5:0.95
globales y por clase, más la matriz de confusión.

Genera:
    - Tabla por clase impresa en consola.
    - outputs/metrics/test_metrics_per_class.csv
    - outputs/validation/test/ (matriz de confusión, curvas PR, etc.)
"""

from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "datasets" / "construction-site-safety" / "data.yaml"
WEIGHTS = PROJECT_ROOT / "outputs" / "training" / "yolov8s_baseline" / "weights" / "best.pt"
VAL_DIR = PROJECT_ROOT / "outputs" / "validation"
SPLIT = "test"  # evaluar sobre el conjunto de prueba (no visto en entrenamiento)


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(f"No existe el modelo: {WEIGHTS}\nEntrena primero con: python src/04_train.py")

    print(f"Evaluando {WEIGHTS.name} sobre el split '{SPLIT}'...\n")
    model = YOLO(str(WEIGHTS))
    metrics = model.val(
        data=str(DATA_YAML),
        split=SPLIT,
        project=str(VAL_DIR),
        name=SPLIT,
        exist_ok=True,
        plots=True,   # genera matriz de confusión y curvas PR
    )

    # --- Métricas globales ---
    print("\n=== Métricas globales (test) ===")
    print(f"  mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")

    # --- Desglose por clase ---
    names = metrics.names
    print("\n=== Desglose por clase ===")
    header = f"{'clase':<16}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for i, c in enumerate(metrics.box.ap_class_index):
        name = names[c]
        p = metrics.box.p[i]
        r = metrics.box.r[i]
        ap50 = metrics.box.ap50[i]
        ap = metrics.box.maps[c]  # mAP50-95 de esta clase
        rows.append((name, p, r, ap50, ap))
        print(f"{name:<16}{p:>8.3f}{r:>8.3f}{ap50:>9.3f}{ap:>10.3f}")

    # --- Guardar CSV ---
    out_metrics = PROJECT_ROOT / "outputs" / "metrics"
    out_metrics.mkdir(parents=True, exist_ok=True)
    csv_path = out_metrics / "test_metrics_per_class.csv"
    with csv_path.open("w") as f:
        f.write("class_name,precision,recall,mAP50,mAP50-95\n")
        for name, p, r, ap50, ap in rows:
            f.write(f"{name},{p:.4f},{r:.4f},{ap50:.4f},{ap:.4f}\n")
    print(f"\nCSV por clase: {csv_path}")
    print(f"Matriz de confusión y curvas en: {VAL_DIR / SPLIT}")


if __name__ == "__main__":
    main()
