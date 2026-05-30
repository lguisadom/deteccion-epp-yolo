"""
Cuantifica el desbalance de clases del dataset: cuenta las instancias (objetos) de cada
clase en cada partición (train/valid/test).

Genera:
    - Tabla por clase impresa en consola, con el ratio de desbalance.
    - outputs/metrics/class_distribution.csv
    - outputs/figures/class_distribution.png
"""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana (guardamos a archivo, no mostramos)
import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "construction-site-safety"
DATA_YAML = DATASET_DIR / "data.yaml"
SPLITS = ["train", "valid", "test"]


def count_instances_per_class(labels_dir: Path, num_classes: int) -> Counter:
    """Cuenta las instancias de cada clase en una carpeta de labels YOLO."""
    counts: Counter = Counter()
    if not labels_dir.is_dir():
        return counts
    for txt in labels_dir.glob("*.txt"):
        for line in txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            class_id = int(line.split()[0])  # primer número = id de clase
            counts[class_id] += 1
    return counts


def main() -> None:
    names = yaml.safe_load(DATA_YAML.read_text())["names"]
    num_classes = len(names)

    # Conteo por split
    per_split = {}
    for split in SPLITS:
        labels_dir = DATASET_DIR / split / "labels"
        per_split[split] = count_instances_per_class(labels_dir, num_classes)

    # --- Tabla en consola ---
    header = f"{'id':>2}  {'clase':<16}{'train':>8}{'valid':>8}{'test':>8}{'TOTAL':>9}"
    print(header)
    print("-" * len(header))
    totals = Counter()
    rows = []
    for cid, name in enumerate(names):
        tr = per_split["train"][cid]
        va = per_split["valid"][cid]
        te = per_split["test"][cid]
        tot = tr + va + te
        totals[cid] = tot
        rows.append((cid, name, tr, va, te, tot))
        print(f"{cid:>2}  {name:<16}{tr:>8}{va:>8}{te:>8}{tot:>9}")

    grand_total = sum(totals.values())
    print("-" * len(header))
    print(f"{'':>2}  {'TOTAL':<16}{'':>8}{'':>8}{'':>8}{grand_total:>9}")

    # Métrica de desbalance: ratio entre la clase más y menos frecuente
    sorted_tot = sorted(totals.values(), reverse=True)
    if sorted_tot[-1] > 0:
        ratio = sorted_tot[0] / sorted_tot[-1]
        print(f"\nDesbalance (clase mayoritaria / minoritaria): {ratio:.1f}x")

    # --- Guardar CSV ---
    out_metrics = PROJECT_ROOT / "outputs" / "metrics"
    out_metrics.mkdir(parents=True, exist_ok=True)
    csv_path = out_metrics / "class_distribution.csv"
    with csv_path.open("w") as f:
        f.write("class_id,class_name,train,valid,test,total\n")
        for cid, name, tr, va, te, tot in rows:
            f.write(f"{cid},{name},{tr},{va},{te},{tot}\n")
    print(f"\nCSV guardado en: {csv_path}")

    # --- Guardar gráfico de barras (total por clase) ---
    out_figures = PROJECT_ROOT / "outputs" / "figures"
    out_figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [r[1] for r in rows]
    values = [r[5] for r in rows]
    ax.bar(labels, values, color="steelblue")
    ax.set_ylabel("Número de instancias (total)")
    ax.set_title("Distribución de clases — Construction Site Safety")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig_path = out_figures / "class_distribution.png"
    fig.savefig(fig_path, dpi=120)
    print(f"Gráfico guardado en: {fig_path}")


if __name__ == "__main__":
    main()
