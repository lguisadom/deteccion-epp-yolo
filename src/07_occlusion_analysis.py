"""
Análisis de robustez ante la oclusión, medido por persona.

Un primer intento que agrupaba imágenes por solapamiento de cajas resultó confundido: las
imágenes con más solapamiento eran grupos de personas grandes y claras (más fáciles de
detectar). Para aislar la oclusión, se mide por objeto y solo en la clase Person:

  - Oclusión de una persona = máximo IoU de su caja (ground truth) con cualquier otra caja
    de la imagen (mayor IoU = más tapada por otro objeto).
  - Las personas se agrupan en bajo/medio/alto (terciles de oclusión).
  - Se mide el recall en cada grupo: qué fracción de personas detectó el modelo (una persona
    se considera detectada si existe una predicción Person con IoU >= 0.5).

Responde la pregunta de seguridad: ¿se escapan los trabajadores ocluidos?

Genera:
    - outputs/metrics/occlusion_person_recall.csv
    - report/figures/occlusion_recall.png
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
IOU_MATCH = 0.5  # IoU mínimo para considerar una persona "detectada"
GROUPS = ["bajo", "medio", "alto"]


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
    persons = []  # (occlusion, detected)

    for img in sorted(TEST_IMAGES.glob("*.jpg")):
        label = TEST_LABELS / (img.stem + ".txt")
        if not label.exists():
            continue

        result = model(str(img), conf=CONF, verbose=False)[0]
        H, W = result.orig_shape  # alto, ancho en píxeles

        # Cajas GT (todas) en píxeles + cuáles son Person
        gt_boxes, gt_person_idx = [], []
        for line in label.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cid = int(p[0])
            xc, yc, w, h = map(float, p[1:5])
            box = ((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H)
            if cid == person_id:
                gt_person_idx.append(len(gt_boxes))
            gt_boxes.append(box)

        # Predicciones Person (píxeles)
        pred_persons = [tuple(b.xyxy[0].tolist()) for b in result.boxes
                        if int(b.cls) == person_id]

        # Por cada persona GT: oclusión + si fue detectada
        for pi in gt_person_idx:
            pbox = gt_boxes[pi]
            occ = max((iou(pbox, gt_boxes[j]) for j in range(len(gt_boxes)) if j != pi),
                      default=0.0)
            detected = any(iou(pbox, pred) >= IOU_MATCH for pred in pred_persons)
            persons.append((occ, detected))

    if not persons:
        raise SystemExit("No se encontraron personas en el test.")

    # Agrupar por terciles de oclusión
    persons.sort(key=lambda x: x[0])
    n = len(persons)
    cuts = [0, n // 3, 2 * n // 3, n]
    print(f"Total de personas (GT) en test: {n}\n")
    print(f"{'oclusión':<10}{'personas':>9}{'rango IoU':>16}{'recall':>9}")
    print("-" * 44)

    results = []
    for gi, group in enumerate(GROUPS):
        chunk = persons[cuts[gi]:cuts[gi + 1]]
        occs = [o for o, _ in chunk]
        rec = sum(1 for _, d in chunk if d) / len(chunk)
        rng = f"{min(occs):.3f}-{max(occs):.3f}"
        results.append((group, len(chunk), rng, rec))
        print(f"{group:<10}{len(chunk):>9}{rng:>16}{rec:>9.3f}")

    drop = results[0][3] - results[2][3]
    print(f"\nCaída de recall de oclusión baja -> alta: {drop:.3f} "
          f"({drop / results[0][3] * 100:.1f}% relativo)")

    # CSV
    csv = PROJECT_ROOT / "outputs" / "metrics" / "occlusion_person_recall.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    with csv.open("w") as f:
        f.write("nivel_oclusion,personas,rango_iou,recall\n")
        for g, ni, rng, rec in results:
            f.write(f"{g},{ni},{rng},{rec:.4f}\n")
    print(f"CSV: {csv}")

    # Gráfico
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([r[0] for r in results], [r[3] for r in results],
           color=["#4caf50", "#ff9800", "#f44336"])
    ax.set_ylabel("Recall (personas detectadas)")
    ax.set_xlabel("Nivel de oclusión de la persona")
    ax.set_title("Recall en personas según su nivel de oclusión (test)")
    ax.set_ylim(0, 1)
    for i, r in enumerate(results):
        ax.text(i, r[3] + 0.02, f"{r[3]:.2f}", ha="center")
    plt.tight_layout()
    figp = PROJECT_ROOT / "report" / "figures" / "occlusion_recall.png"
    fig.savefig(figp, dpi=120)
    print(f"Gráfico: {figp}")


if __name__ == "__main__":
    main()
