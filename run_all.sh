#!/usr/bin/env bash
# =====================================================================================
# Pipeline completo de extremo a extremo — Detección de cumplimiento de EPP (YOLOv8)
#
# Uso:   bash run_all.sh
#
# Requisitos previos:
#   1. Entorno con dependencias instaladas:  pip install -r requirements.txt
#   2. Archivo .env con ROBOFLOW_API_KEY (copiar de .env.example).
#
# Variable opcional: PYTHON (intérprete a usar). Por defecto: python
# =====================================================================================
set -e  # aborta si cualquier paso falla

PYTHON="${PYTHON:-python}"

echo ">>> [1/8] Descargando dataset..."
$PYTHON src/01_download_dataset.py

echo ">>> [2/8] Analizando dataset (desbalance)..."
$PYTHON src/02_analyze_dataset.py

echo ">>> [3/8] Sanity check de anotaciones..."
$PYTHON src/03_visualize_samples.py

echo ">>> [4/8] Entrenando YOLOv8 (~30 min en GPU)..."
$PYTHON src/04_train.py

echo ">>> [5/8] Evaluando en test (mAP por clase + matriz de confusión)..."
$PYTHON src/05_validate.py

echo ">>> [6/8] Demo del verificador de cumplimiento sobre imágenes..."
$PYTHON src/06_predict_images.py

echo ">>> [7/8] Análisis de oclusión (recall por persona)..."
$PYTHON src/07_occlusion_analysis.py

echo ">>> [8/8] Análisis de objetos pequeños (recall por tamaño)..."
$PYTHON src/08_size_analysis.py

echo ">>> Pipeline completo. Resultados en outputs/ y report/figures/."
