"""
Verificación visual de las anotaciones: dibuja algunas imágenes de entrenamiento con sus
cajas (ground truth) para confirmar que las etiquetas están bien alineadas con las imágenes.

Genera: outputs/figures/sample_annotations.png (mosaico de imágenes anotadas).
"""

import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "construction-site-safety"
DATA_YAML = DATASET_DIR / "data.yaml"
N_SAMPLES = 6
SEED = 42  # fijo para reproducibilidad

# Un color por clase (BGR), generado de forma determinista
def color_for(class_id: int) -> tuple:
    rng = random.Random(class_id * 7 + 13)
    return (rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255))


def draw_boxes(img, label_path: Path, names: list):
    """Dibuja las cajas YOLO (normalizadas) sobre la imagen."""
    h, w = img.shape[:2]
    if not label_path.exists():
        return img
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        color = color_for(cid)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, names[cid], (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def main() -> None:
    names = yaml.safe_load(DATA_YAML.read_text())["names"]
    images_dir = DATASET_DIR / "train" / "images"
    labels_dir = DATASET_DIR / "train" / "labels"

    all_images = sorted(images_dir.glob("*.jpg"))
    random.Random(SEED).shuffle(all_images)
    samples = all_images[:N_SAMPLES]

    cols = 3
    rows = (N_SAMPLES + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for ax, img_path in zip(axes, samples):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label_path = labels_dir / (img_path.stem + ".txt")
        img = draw_boxes(img, label_path, names)
        ax.imshow(img)
        ax.set_title(img_path.name[:30], fontsize=8)
        ax.axis("off")

    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.tight_layout()
    out = PROJECT_ROOT / "outputs" / "figures" / "sample_annotations.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"Mosaico guardado en: {out}")


if __name__ == "__main__":
    main()
