# Detección de Cumplimiento de EPP en Obras de Construcción con YOLOv8

Sistema de visión por computadora que detecta trabajadores y su Equipo de Protección Personal
(EPP) en imágenes de obras, y **verifica el cumplimiento** (casco + chaleco) por persona,
reportando una tasa de cumplimiento por fotograma.

> Examen Parcial - Redes Neuronales y Aprendizaje Profundo. Pregunta 3.

## Revisión de la literatura

La detección automática de EPP en obras de construcción es un problema activo en visión por
computadora. Este trabajo se apoya en las siguientes referencias:

**Fundamentos del detector**

- **Redmon et al. (CVPR 2016)** — *You Only Look Once: Unified, Real-Time Object Detection.*
  Propuso el paradigma YOLO de detección en un solo paso (*single-stage*), base de la familia de
  modelos que usamos. Introdujo la idea de tratar la detección como un problema de regresión directa
  sobre una grilla de la imagen, logrando velocidad en tiempo real.

- **Lin et al. (ICCV 2017)** — *Focal Loss for Dense Object Detection.*
  Identificó el desbalance extremo entre objetos de fondo y objetos de interés como la causa
  principal del bajo desempeño en detectores densos. Propuso la *Focal Loss*, que pondera más los
  ejemplos difíciles durante el entrenamiento. Relevante para nuestro trabajo porque el dataset
  presenta un desbalance de 6.2x entre clases, y las clases de incumplimiento (las más escasas)
  son las más débiles. Se cita como trabajo futuro para mejorar esas clases.

**Aplicaciones de detección de EPP en construcción**

- **Wang et al. — *Sensors* (2021)** — *Fast Personal Protective Equipment Detection for Real
  Construction Sites Using Deep Learning Approaches.*
  Evaluó detectores de aprendizaje profundo para EPP en obras reales, reportando que las caídas
  desde altura concentran más de la mitad de las muertes en el sector y que el casco reduce hasta
  un 95% el riesgo de lesión cerebral grave. Sustenta la motivación del problema en este trabajo.

- **Hayat y Morgado-Dias — *Applied Sciences* (2022)** — *Deep Learning-Based Automatic Safety
  Helmet Detection System for Construction Safety.*
  Propuso un sistema de detección automática de cascos mediante aprendizaje profundo para entornos
  de construcción. Complementa la base estadística de siniestralidad y valida la viabilidad del
  enfoque con redes convolucionales profundas para este dominio.

- **Barlybayev et al. — *Cogent Engineering* (2024)** — *Personal Protective Equipment Detection
  Using YOLOv8 Architecture on Object Detection Benchmark Datasets.*
  Evaluó YOLOv8 específicamente para detección de EPP sobre datasets de referencia, confirmando su
  idoneidad para este dominio y aportando comparaciones de desempeño entre variantes del modelo.
  Es el trabajo más directamente comparable al nuestro en arquitectura y tarea.

- **Wei et al. — *Scientific Reports* (2024)** — *Research on Helmet Wearing Detection Method
  Based on Deep Learning.*
  Estudió métodos de detección del uso de casco mediante aprendizaje profundo, incluyendo
  estrategias para mejorar la detección en condiciones de oclusión y escala variable, dos de las
  limitaciones que identificamos en nuestro análisis de robustez.

**Posición de este trabajo respecto a la literatura**

La mayoría de los trabajos citados se enfocan en la detección de objetos individuales (casco,
chaleco) sin un módulo explícito de verificación de cumplimiento por persona. Este trabajo aporta
esa capa: un verificador basado en reglas geométricas que asocia el EPP detectado a cada persona y
emite un veredicto individual, produciendo una tasa de cumplimiento por fotograma que los trabajos
anteriores no reportan directamente.

## Dataset

*Construction Site Safety* (Roboflow Universe, versión 27, CC BY 4.0): 10 clases y 2603 / 114 / 82
imágenes (train / val / test, partición ya provista). Presenta un fuerte desbalance de clases (6.2x
entre la más y la menos frecuente: `Person` con 10 031 instancias frente a `vehicle` con 1617), lo
que condiciona el aprendizaje de las clases de incumplimiento, que son justamente las más escasas.
El dataset no se incluye en el repositorio; lo descarga `src/01_download_dataset.py`.

![Distribución de instancias por clase](report/figures/distribucion_clases.png)

## Modelo y entrenamiento

Detector **YOLOv8s** (variante *small*, anchor-free) con *transfer learning* desde COCO. Ajuste
fino (fine-tuning) durante 60 épocas a 640 px de entrada, con data augmentation (mosaic) y early
stopping. Las curvas muestran un entrenamiento estable, sin sobreajuste. Los pesos entrenados
(`best.pt`) no se versionan (son pesados): se generan al ejecutar el entrenamiento.

![Curvas de entrenamiento](report/figures/curvas_entrenamiento.png)

## Resultados principales (conjunto de prueba)

| Métrica | Valor |
|---|---|
| mAP@0.5 | 0.767 |
| mAP@0.5:0.95 | 0.492 |
| Precision | 0.907 |
| Recall | 0.710 |

Clases con mejor desempeño: Safety Vest (mAP50 0.89), Person (0.88), Hardhat (0.87).
Debilidad principal: **objetos pequeños** (Safety Cone 0.46; recall en personas pequeñas 0.64
vs 1.00 en grandes).

![Matriz de confusión normalizada (test)](report/figures/matriz_confusion_test.png)

![Curva precision-recall por clase (test)](report/figures/curva_PR_test.png)

## Verificador de cumplimiento

A partir de las detecciones de YOLO, un módulo basado en reglas (`src/compliance_checker.py`)
asocia el EPP a cada persona y decide:

> Una persona **cumple** si tiene casco (Hardhat) en su región de cabeza **y** chaleco (Safety
> Vest) en su torso, sin violaciones (NO-Hardhat / NO-Safety Vest). Criterio conservador
> (pro-seguridad): EPP faltante = no conforme.

El detector y el verificador son piezas separadas: el primero detecta objetos, el segundo decide
cumplimiento. Esto mantiene el sistema interpretable y permite ajustar la lógica sin reentrenar.
Salida: cajas verde (cumple) / rojo (no cumple) + tasa de cumplimiento del fotograma.

![Demostración del verificador sobre imágenes de prueba](report/figures/demo_cumplimiento.png)

## Material de demostración

El repositorio incluye la evidencia completa del verificador:

- **Imágenes anotadas** del conjunto de prueba en `outputs/compliance/images/` (cajas verdes para
  quien cumple, rojas para quien no, con la tasa de cumplimiento por imagen).
- **Tabla de cumplimiento por imagen:** `outputs/metrics/compliance_per_image.csv`.
- **Videos anotados** de obras (vía `src/09_predict_video.py`), cada uno con su tabla de tasa de
  cumplimiento por fotograma.

Permiten revisar casos de cumplimiento, de incumplimiento correctamente detectado y de falsos
positivos por equipo no detectado.

### Análisis complementario a nivel de imagen (oclusión)

Además del análisis por persona, se estimó la oclusión a nivel de imagen y se midió el mAP@0.5 por
nivel (bajo / medio / alto):

![mAP@0.5 por nivel de oclusión estimada de la imagen](report/figures/occlusion_map.png)

## Robustez: hallazgo clave (oclusión vs tamaño)

Se evaluó la robustez ante oclusión partiendo el conjunto de prueba por nivel de solapamiento de
cajas. El recall **no cae** con la oclusión (0.69 / 0.91 / 0.90), un resultado contraintuitivo.
La causa es una **variable de confusión**: el solapamiento de cajas está correlacionado con el
tamaño (las personas muy solapadas suelen ser grandes y cercanas, fáciles de detectar). Al aislar
el tamaño, el patrón es claro: el recall sube de 0.64 (personas pequeñas) a 1.00 (grandes). El
factor real de dificultad es el **tamaño del objeto**, no la oclusión medida por solapamiento.

![Recall por tamaño relativo de la persona (test)](report/figures/size_recall.png)

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── run_all.sh                 # pipeline completo de extremo a extremo
├── .env.example               # plantilla para la API key de Roboflow
├── configs/
├── src/
│   ├── 01_download_dataset.py   # descarga el dataset (Roboflow)
│   ├── 02_analyze_dataset.py    # desbalance de clases
│   ├── 03_visualize_samples.py  # sanity check de anotaciones
│   ├── 04_train.py              # entrena YOLOv8
│   ├── 05_validate.py           # mAP por clase + matriz de confusión
│   ├── 06_predict_images.py     # demo del verificador (imágenes)
│   ├── 07_occlusion_analysis.py # robustez ante oclusión
│   ├── 08_size_analysis.py      # robustez ante objetos pequeños
│   ├── 09_predict_video.py      # demo del verificador sobre video
│   └── compliance_checker.py    # módulo: lógica de cumplimiento
├── notebooks/                 # cuadernos de Google Colab
├── outputs/                   # métricas, figuras, imágenes anotadas (generado)
└── report/figures/            # figuras seleccionadas para el informe
```

## Requisitos

- Python 3.12, GPU NVIDIA recomendada (entrenamiento). Probado en RTX 4070 (8 GB), CUDA.

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configuración (clave de Roboflow)

Para descargar el dataset necesitas una API key de Roboflow (gratuita):

```bash
cp .env.example .env
# edita .env y coloca tu ROBOFLOW_API_KEY
```

## Ejecución

```bash
bash run_all.sh
```

El script ejecuta el pipeline completo en orden: descarga, análisis, entrenamiento, evaluación,
verificador sobre imágenes y análisis de robustez. También puedes correr cada script por separado:

```bash
python src/01_download_dataset.py
# ...
python src/09_predict_video.py data/videos/<archivo>.mp4
```

## Ejecución en Google Colab

En `notebooks/` hay cuadernos listos para Colab (activar GPU en *Entorno de ejecución -> Cambiar
tipo de entorno*):

- `colab_epp_yolo.ipynb`: clona el repositorio y ejecuta el pipeline.
- `colab_epp_yolo_autocontenido.ipynb`: incluye todo el código embebido en el cuaderno.
- `colab_epp_yolo_autocontenido_con_prueba.ipynb`: igual que el anterior, más una celda para
  probar el modelo con una imagen propia.

## Limitaciones

- El detector solo reconoce las 10 clases anotadas en el dataset (no incluye arneses, por ejemplo).
- El verificador usa reglas geométricas (regiones de cabeza/torso); puede fallar con poses
  inusuales, perspectivas extremas u oclusión severa.
- Las clases de incumplimiento (NO-) y los objetos pequeños son los más débiles.
- El análisis se basa en una sola corrida de entrenamiento (semilla fija); no se reportan
  intervalos de confianza sobre múltiples corridas.
