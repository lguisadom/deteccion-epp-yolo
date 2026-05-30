"""
Demostración del verificador de cumplimiento sobre imágenes del conjunto de prueba.
Conecta el modelo entrenado (best.pt) con el módulo compliance_checker.

Flujo: imagen -> YOLO detecta objetos -> compliance_checker decide el cumplimiento por
persona -> se dibujan cajas (verde = cumple, rojo = no) y un banner con la tasa del fotograma.

Genera:
    - outputs/compliance/images/ (imágenes anotadas)
    - outputs/metrics/compliance_per_image.csv
    - report/figures/demo_cumplimiento.png (mosaico para el informe)
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

from compliance_checker import PPEComplianceChecker, Detection, annotate_frame

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = PROJECT_ROOT / "outputs" / "training" / "yolov8s_baseline" / "weights" / "best.pt"
TEST_IMAGES = PROJECT_ROOT / "datasets" / "construction-site-safety" / "test" / "images"
OUT_DIR = PROJECT_ROOT / "outputs" / "compliance" / "images"
CSV_PATH = PROJECT_ROOT / "outputs" / "metrics" / "compliance_per_image.csv"
MONTAGE_PATH = PROJECT_ROOT / "report" / "figures" / "demo_cumplimiento.png"

CONF = 0.25          # umbral de confianza para las detecciones
MAX_ANNOTATED = 20   # cuántas imágenes (con personas) guardar anotadas
N_MONTAGE = 6        # cuántas poner en el mosaico del informe


def detections_from_result(result, names) -> list:
    """Convierte el resultado de YOLO en una lista de Detection del compliance_checker."""
    dets = []
    for box in result.boxes:
        cls_id = int(box.cls)
        dets.append(Detection(
            cls_name=names[cls_id],
            conf=float(box.conf),
            xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),
        ))
    return dets


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(f"No existe el modelo: {WEIGHTS}\nEntrena primero: python src/04_train.py")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    MONTAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(WEIGHTS))
    checker = PPEComplianceChecker(min_conf=CONF)

    images = sorted(TEST_IMAGES.glob("*.jpg"))
    print(f"Procesando {len(images)} imágenes de test...\n")

    rows = []           # (nombre, n_personas, n_cumplen, tasa)
    annotated_paths = []
    for img_path in images:
        result = model(str(img_path), conf=CONF, verbose=False)[0]
        dets = detections_from_result(result, model.names)
        statuses, rate = checker.check_frame(dets)

        n_persons = len(statuses)
        if n_persons == 0:
            continue  # solo nos interesan frames con personas para la demo de cumplimiento

        n_ok = sum(1 for st in statuses if st.compliant)
        rows.append((img_path.name, n_persons, n_ok, rate))

        if len(annotated_paths) < MAX_ANNOTATED:
            img = cv2.imread(str(img_path))
            img = annotate_frame(img, statuses, rate)
            out = OUT_DIR / img_path.name
            cv2.imwrite(str(out), img)
            annotated_paths.append(out)

    # --- CSV resumen ---
    with CSV_PATH.open("w") as f:
        f.write("imagen,personas,cumplen,tasa_cumplimiento\n")
        for name, n, ok, rate in rows:
            f.write(f"{name},{n},{ok},{rate:.3f}\n")

    # --- Estadísticas globales ---
    total_persons = sum(r[1] for r in rows)
    total_ok = sum(r[2] for r in rows)
    print(f"Imágenes con personas : {len(rows)}")
    print(f"Personas detectadas   : {total_persons}")
    print(f"Personas que cumplen  : {total_ok}")
    if total_persons:
        print(f"Tasa global de cumplimiento: {total_ok / total_persons:.1%}")
    print(f"\nImágenes anotadas en : {OUT_DIR}")
    print(f"CSV por imagen       : {CSV_PATH}")

    # --- Mosaico para el informe ---
    sample = annotated_paths[:N_MONTAGE]
    if sample:
        cols = 3
        rows_n = (len(sample) + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(15, 5 * rows_n))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, p in zip(axes, sample):
            im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            ax.imshow(im)
            ax.axis("off")
        for ax in axes[len(sample):]:
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(MONTAGE_PATH, dpi=110)
        print(f"Mosaico informe      : {MONTAGE_PATH}")


if __name__ == "__main__":
    main()
