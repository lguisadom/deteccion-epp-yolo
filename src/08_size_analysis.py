"""
Análisis de robustez ante objetos pequeños: recall en personas según su tamaño.

El análisis de oclusión (07) sugirió que el factor real de dificultad es el tamaño del objeto,
no la oclusión. Aquí se confirma de forma directa: las personas (ground truth) se agrupan por
tamaño relativo de su caja (pequeña/mediana/grande) y se mide el recall en cada grupo.

Tamaño relativo = área de la caja / área de la imagen.
Una persona se considera detectada si existe una predicción Person con IoU >= 0.5.

Genera:
    - outputs/metrics/size_person_recall.csv
    - report/figures/size_recall.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "construction-site-safety"
DATA_YAML = DATASET_DIR / "data.yaml"
WEIGHTS = PROJECT_ROOT / "outputs" / "training" / "yolov8s_baseline" / "weights" / "best.pt"
TEST_IMAGES = DATASET_DIR / "test" / "images"
TEST_LABELS = DATASET_DIR / "test" / "labels"

CONF = 0.25
IOU_MATCH = 0.5
GROUPS = ["pequeña", "mediana", "grande"]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(f"No existe el modelo: {WEIGHTS}")

    names = yaml.safe_load(DATA_YAML.read_text())["names"]
    person_id = names.index("Person")
    model = YOLO(str(WEIGHTS))

    persons = []  # (tamaño_relativo, detected)
    for img in sorted(TEST_IMAGES.glob("*.jpg")):
        label = TEST_LABELS / (img.stem + ".txt")
        if not label.exists():
            continue
        result = model(str(img), conf=CONF, verbose=False)[0]
        H, W = result.orig_shape

        pred_persons = [tuple(b.xyxy[0].tolist()) for b in result.boxes
                        if int(b.cls) == person_id]

        for line in label.read_text().splitlines():
            p = line.split()
            if len(p) < 5 or int(p[0]) != person_id:
                continue
            xc, yc, w, h = map(float, p[1:5])
            rel_size = w * h  # área relativa (w,h ya están normalizados 0-1)
            box = ((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H)
            detected = any(iou(box, pred) >= IOU_MATCH for pred in pred_persons)
            persons.append((rel_size, detected))

    persons.sort(key=lambda x: x[0])
    n = len(persons)
    cuts = [0, n // 3, 2 * n // 3, n]
    print(f"Total de personas (GT) en test: {n}\n")
    print(f"{'tamaño':<10}{'personas':>9}{'rango área':>18}{'recall':>9}")
    print("-" * 46)

    results = []
    for gi, group in enumerate(GROUPS):
        chunk = persons[cuts[gi]:cuts[gi + 1]]
        sizes = [s for s, _ in chunk]
        rec = sum(1 for _, d in chunk if d) / len(chunk)
        rng = f"{min(sizes):.4f}-{max(sizes):.4f}"
        results.append((group, len(chunk), rng, rec))
        print(f"{group:<10}{len(chunk):>9}{rng:>18}{rec:>9.3f}")

    gain = results[2][3] - results[0][3]
    print(f"\nDiferencia de recall pequeña -> grande: +{gain:.3f} "
          f"({gain / results[0][3] * 100:.1f}% relativo)")

    csv = PROJECT_ROOT / "outputs" / "metrics" / "size_person_recall.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    with csv.open("w") as f:
        f.write("tamano,personas,rango_area_relativa,recall\n")
        for g, ni, rng, rec in results:
            f.write(f"{g},{ni},{rng},{rec:.4f}\n")
    print(f"CSV: {csv}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([r[0] for r in results], [r[3] for r in results],
           color=["#f44336", "#ff9800", "#4caf50"])
    ax.set_ylabel("Recall (personas detectadas)")
    ax.set_xlabel("Tamaño de la persona (área relativa)")
    ax.set_title("Recall en personas según su tamaño (test)")
    ax.set_ylim(0, 1)
    for i, r in enumerate(results):
        ax.text(i, r[3] + 0.02, f"{r[3]:.2f}", ha="center")
    plt.tight_layout()
    figp = PROJECT_ROOT / "report" / "figures" / "size_recall.png"
    fig.savefig(figp, dpi=120)
    print(f"Gráfico: {figp}")


if __name__ == "__main__":
    main()
